"""NVIDIA chat-completions adapter with validation, retries and token handling."""

import asyncio
import json
import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

from ..config import Settings
from ..constants import (
    CUSTOM_LANGUAGE_RECOGNITION_JSON_RETRY_PROMPT,
    CUSTOM_LANGUAGE_RECOGNITION_PROMPT,
    JSON_RETRY_PROMPT,
    REPETITION_RETRY_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)

logger = logging.getLogger('localizer')
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}')
LANGUAGE_CODE = re.compile(r'^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$')
T = TypeVar('T')


class InvalidModelJSONError(ValueError):
    """Raised when NVIDIA returns text that cannot be parsed as JSON."""


class UnchangedTranslationError(ValueError):
    """Raised when the model explicitly returns the original source strings unchanged."""


class RepetitionLoopError(ValueError):
    """Raised when a truncated model response degenerates into repeated text."""


class UnsupportedTargetLanguageError(RuntimeError):
    """Raised when repeated completed model responses decline the requested language."""


@dataclass(frozen=True)
class CustomLanguageRecognition:
    """Validated recognition result for a manually entered target-language name."""

    language_recognized: bool
    language: str | None
    language_code: str | None


def _strip_markdown_json_fence(content: str) -> str:
    """Remove a complete Markdown code fence around model JSON when present.

    :param content: Raw model text.
    :return: Model text without a surrounding Markdown code fence.
    :raises ValueError: If a Markdown fence starts but is incomplete.
    """
    content = content.strip()
    if content.startswith('```'):
        lines = content.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith('```'):
            raise ValueError('Incomplete Markdown JSON fence')
        content = '\n'.join(lines[1:-1]).strip()
    return content


def _load_model_json(content: str):
    """Parse JSON returned by the model after tolerating a complete code fence.

    :param content: Raw text returned in ``message.content``.
    :return: Parsed JSON value.
    :raises InvalidModelJSONError: If the model output is not valid JSON.
    :raises ValueError: If a surrounding Markdown fence is incomplete.
    """
    content = _strip_markdown_json_fence(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidModelJSONError('Model returned invalid JSON') from exc


def _validate_translations(translations, originals: list[str]) -> list[str]:
    """Validate translation count, item types and placeholder preservation.

    :param translations: Parsed candidate translation array.
    :param originals: Ordered source strings sent to the model.
    :return: Validated translated strings in source order.
    :raises UnchangedTranslationError: If every returned string is unchanged.
    :raises ValueError: If the structure, count or placeholders are invalid.
    """
    if (
        not isinstance(translations, list)
        or len(translations) != len(originals)
        or any(not isinstance(item, str) for item in translations)
    ):
        raise ValueError(f'Expected an array of exactly {len(originals)} strings')

    for original, translated in zip(originals, translations):
        if sorted(PLACEHOLDER.findall(original)) != sorted(PLACEHOLDER.findall(translated)):
            raise ValueError('Placeholder mismatch')

    if translations == originals:
        raise UnchangedTranslationError('Model returned the original source strings unchanged')

    return translations


def _has_repetition_loop(
    content: str,
    min_repeats: int = 10,
    max_phrase_tokens: int = 4,
) -> bool:
    """Detect a short phrase repeated consecutively enough to indicate degeneration.

    :param content: Raw model output to inspect.
    :param min_repeats: Minimum consecutive repetitions required for detection.
    :param max_phrase_tokens: Maximum repeated phrase length in word tokens.
    :return: True when a likely repetition loop is found.
    """
    tokens = re.findall(r'\w+', content.casefold(), flags=re.UNICODE)
    if len(tokens) < min_repeats:
        return False

    for phrase_size in range(1, max_phrase_tokens + 1):
        required_tokens = phrase_size * min_repeats
        if len(tokens) < required_tokens:
            break

        max_start = len(tokens) - required_tokens
        for start in range(max_start + 1):
            phrase = tokens[start:start + phrase_size]
            repeated = True
            for repeat_index in range(1, min_repeats):
                offset = start + repeat_index * phrase_size
                if tokens[offset:offset + phrase_size] != phrase:
                    repeated = False
                    break
            if repeated:
                return True

    return False


def parse_translation(content: str, originals: list[str]) -> list[str]:
    """Parse and validate an ordered translated JSON string array.

    :param content: Raw text returned in the model's ``message.content`` field.
    :param originals: Ordered source strings sent to the model.
    :return: Validated translated strings in the same order as ``originals``.
    :raises InvalidModelJSONError: If the model output is not valid JSON.
    :raises ValueError: If the JSON has an unsafe or unexpected structure.
    """
    return _validate_translations(_load_model_json(content), originals)


def parse_custom_language_recognition(content: str) -> CustomLanguageRecognition:
    """Parse a language-recognition result from a manual language name.

    :param content: Raw text returned in the model response.
    :return: Validated recognition result with canonical language name and code.
    :raises InvalidModelJSONError: If the model output is not valid JSON.
    :raises ValueError: If the JSON object has an unsafe or unexpected structure.
    """
    parsed = _load_model_json(content)
    if not isinstance(parsed, dict):
        raise ValueError('Expected a JSON object for custom-language recognition')

    recognized = parsed.get('language_recognized')
    if not isinstance(recognized, bool):
        raise ValueError('language_recognized must be a boolean')

    language = parsed.get('language')
    language_code = parsed.get('language_code')

    if not recognized:
        if language is not None or language_code is not None:
            raise ValueError(
                'Unrecognized language must return language=null and language_code=null'
            )
        return CustomLanguageRecognition(
            language_recognized=False,
            language=None,
            language_code=None,
        )

    if not isinstance(language, str) or not language.strip():
        raise ValueError('Recognized language must include a canonical language name')
    if (
        not isinstance(language_code, str)
        or not LANGUAGE_CODE.fullmatch(language_code.strip())
    ):
        raise ValueError('Recognized language must include a valid standard language code')

    return CustomLanguageRecognition(
        language_recognized=True,
        language=language.strip(),
        language_code=language_code.strip(),
    )


def build_translation_prompt(
    values: list[str],
    language: str,
    json_retry: bool = False,
    repetition_retry: bool = False,
) -> str:
    """Build the prompt for translating trusted target-language strings.

    :param values: Ordered source strings that must be translated.
    :param language: Trusted target-language name and optional standard code.
    :param json_retry: Whether to append a stricter invalid-JSON correction instruction.
    :param repetition_retry: Whether to append repetition-loop recovery instructions.
    :return: Fully formatted user prompt ready for the NVIDIA chat-completions request.
    """
    prompt = USER_PROMPT.format(
        language=language,
        values=json.dumps(values, ensure_ascii=False),
    )
    if json_retry:
        prompt += JSON_RETRY_PROMPT
    if repetition_retry:
        prompt += REPETITION_RETRY_PROMPT
    return prompt


def build_language_recognition_prompt(
    language: str,
    json_retry: bool = False,
) -> str:
    """Build the prompt for recognizing a manually entered language name.

    :param language: User-provided target-language name.
    :param json_retry: Whether to append a stricter invalid-JSON correction instruction.
    :return: Fully formatted recognition prompt ready for NVIDIA.
    """
    prompt = CUSTOM_LANGUAGE_RECOGNITION_PROMPT.format(language=language)
    if json_retry:
        prompt += CUSTOM_LANGUAGE_RECOGNITION_JSON_RETRY_PROMPT
    return prompt


def _is_unsupported_translation(
    operation: str,
    unchanged_response_count: int,
    other_model_response_failure_count: int,
) -> bool:
    """Decide whether retries indicate a recognized but unsupported target language.

    :param operation: Current NVIDIA operation name.
    :param unchanged_response_count: Completed model responses that returned the source unchanged.
    :param other_model_response_failure_count: Completed model responses that failed for another reason.
    :return: True when the model repeatedly declined only the requested translation.
    """
    return (
        operation == 'translation'
        and unchanged_response_count >= 2
        and other_model_response_failure_count == 0
    )

def build_request_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    enable_thinking: bool,
) -> dict:
    """Build the OpenAI-compatible NVIDIA chat-completions request body.

    :param model: NVIDIA model identifier.
    :param prompt: Fully formatted user prompt containing the target language and strings.
    :param max_tokens: Maximum number of output tokens allowed for this attempt.
    :param enable_thinking: Whether the model's reasoning/thinking mode should be enabled.
    :return: JSON-serializable request body for NVIDIA's chat-completions endpoint.
    """
    return {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0,
        'max_tokens': max_tokens,
        'chat_template_kwargs': {'enable_thinking': enable_thinking},
    }


def _fallback_model_available(settings: Settings) -> bool:
    """Return whether a usable fallback NVIDIA model is configured.

    :param settings: Validated runtime NVIDIA settings.
    :return: True when a non-placeholder fallback model identifier is available.
    """
    return bool(
        settings.nvidia_fallback_model_id
        and not settings.nvidia_fallback_model_id.startswith('your-')
    )


async def _request_with_retries(
    settings: Settings,
    prompt_builder: Callable[[bool], str],
    parser: Callable[[str], T],
    operation: str,
    force_fallback_model: bool = False,
) -> T:
    """Run one NVIDIA operation with validation, retries and fallback-model switching.

    :param settings: Validated runtime settings, including credentials and retry limits.
    :param prompt_builder: Callable that builds the prompt from the JSON-retry flag.
    :param parser: Callable that parses and validates the model raw text response.
    :param operation: Short operation name used in logs and terminal errors.
    :param force_fallback_model: Whether every attempt should use the configured fallback model.
    :return: Parsed and validated model result.
    :raises RepetitionLoopError: If a translation response degenerates into repeated text.
    :raises UnsupportedTargetLanguageError: If repeated translation responses return the source unchanged.
    :raises RuntimeError: If configuration is missing or all attempts fail.
    """
    if not settings.nvidia_api_key or settings.nvidia_api_key.startswith('your-'):
        raise RuntimeError('NVIDIA_API_KEY is not configured')
    if not settings.nvidia_model_id or settings.nvidia_model_id.startswith('your-'):
        raise RuntimeError('NVIDIA_MODEL_ID is not configured')

    url = settings.nvidia_base_url.rstrip('/') + '/chat/completions'
    max_tokens = settings.nvidia_max_tokens
    last_error: Exception = RuntimeError(f'{operation} did not complete')
    json_retry_required = False
    unchanged_response_count = 0
    other_model_response_failure_count = 0

    async with httpx.AsyncClient(timeout=settings.nvidia_timeout_seconds) as client:
        for attempt in range(settings.nvidia_retries + 1):
            fallback_available = _fallback_model_available(settings)
            use_fallback = bool(
                fallback_available
                and (force_fallback_model or attempt >= 3)
            )
            model = settings.nvidia_fallback_model_id if use_fallback else settings.nvidia_model_id
            raw_content: str | None = None

            try:
                payload = build_request_payload(
                    model=model,
                    prompt=prompt_builder(json_retry_required),
                    max_tokens=max_tokens,
                    enable_thinking=settings.nvidia_enable_thinking,
                )
                response = await client.post(
                    url,
                    headers={'Authorization': f'Bearer {settings.nvidia_api_key}'},
                    json=payload,
                )
                response.raise_for_status()

                choice = response.json()['choices'][0]
                raw_content = choice['message']['content']
                if not isinstance(raw_content, str):
                    raise ValueError('Response content is not text')

                reason = choice.get('finish_reason')
                if reason == 'length':
                    logger.warning(
                        'NVIDIA raw truncated response, operation=%s model=%s: %r',
                        operation,
                        model,
                        raw_content[:settings.nvidia_raw_response_log_chars],
                    )
                    if operation == 'translation' and _has_repetition_loop(raw_content):
                        logger.warning(
                            'NVIDIA repetition loop detected, operation=%s model=%s',
                            operation,
                            model,
                        )
                        raise RepetitionLoopError(
                            'Truncated response entered a repetition loop'
                        )

                    max_tokens = min(max_tokens * 2, settings.nvidia_max_tokens_cap)
                    raise ValueError('Truncated response (finish_reason=length)')
                if reason not in ('stop', None):
                    raise ValueError(f'Unexpected finish_reason={reason}')

                return parser(raw_content)

            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc

                if isinstance(exc, UnchangedTranslationError):
                    unchanged_response_count += 1
                elif raw_content is not None:
                    other_model_response_failure_count += 1

                if isinstance(exc, InvalidModelJSONError):
                    json_retry_required = True

                if isinstance(exc, InvalidModelJSONError) and raw_content is not None:
                    logger.warning(
                        'NVIDIA raw response after invalid JSON, operation=%s model=%s: %r',
                        operation,
                        model,
                        raw_content[:settings.nvidia_raw_response_log_chars],
                    )

                logger.warning(
                    'NVIDIA %s attempt %d/%d failed, model=%s: %s',
                    operation,
                    attempt + 1,
                    settings.nvidia_retries + 1,
                    model,
                    str(exc)[:300],
                )

                if isinstance(exc, RepetitionLoopError):
                    raise

                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in (400, 401, 403, 404, 410)
                ):
                    break

                if attempt < settings.nvidia_retries:
                    delay = min(2 ** attempt, settings.nvidia_retry_max_delay_seconds)
                    await asyncio.sleep(
                        delay + random.uniform(0, settings.nvidia_retry_jitter_seconds)
                    )

    if _is_unsupported_translation(
        operation,
        unchanged_response_count,
        other_model_response_failure_count,
    ):
        raise UnsupportedTargetLanguageError(
            f'{operation} unsupported after {unchanged_response_count} unchanged responses'
        ) from last_error

    raise RuntimeError(f'NVIDIA {operation} failed: {type(last_error).__name__}') from last_error


async def _translate_batch(
    values: list[str],
    language: str,
    settings: Settings,
    repetition_retry: bool = False,
    force_fallback_model: bool = False,
) -> list[str]:
    """Translate one batch without changing its boundaries.

    :param values: Ordered English strings in this translation batch.
    :param language: Trusted target-language name and optional standard code.
    :param settings: Validated runtime NVIDIA and retry settings.
    :param repetition_retry: Whether to tell the model a parent batch entered a repetition loop.
    :param force_fallback_model: Whether to use only the configured fallback model.
    :return: Translated strings in exactly the same order as the batch.
    :raises RepetitionLoopError: If this batch degenerates into a repetition loop.
    :raises UnsupportedTargetLanguageError: If the model repeatedly declines the target language.
    :raises RuntimeError: If all attempts fail.
    """
    return await _request_with_retries(
        settings=settings,
        prompt_builder=lambda json_retry: build_translation_prompt(
            values,
            language,
            json_retry=json_retry,
            repetition_retry=repetition_retry,
        ),
        parser=lambda raw: parse_translation(raw, values),
        operation='translation',
        force_fallback_model=force_fallback_model,
    )


async def _translate_values_recursive(
    values: list[str],
    language: str,
    settings: Settings,
    repetition_retry: bool = False,
) -> list[str]:
    """Translate adaptively, splitting only batches that enter repetition loops.

    :param values: Ordered English strings to translate in the current batch.
    :param language: Trusted target-language name and optional standard code.
    :param settings: Validated runtime NVIDIA and retry settings.
    :param repetition_retry: Whether this batch follows a parent repetition-loop failure.
    :return: Translated strings preserving the original global order.
    :raises RepetitionLoopError: If a single-item batch loops and no fallback can recover it.
    :raises UnsupportedTargetLanguageError: If the model repeatedly declines the target language.
    :raises RuntimeError: If all attempts fail.
    """
    try:
        return await _translate_batch(
            values,
            language,
            settings,
            repetition_retry=repetition_retry,
        )
    except RepetitionLoopError:
        if len(values) <= 1:
            if not _fallback_model_available(settings):
                raise

            logger.warning(
                'NVIDIA repetition loop on single-item batch; switching to fallback model'
            )
            return await _translate_batch(
                values,
                language,
                settings,
                repetition_retry=True,
                force_fallback_model=True,
            )

        midpoint = len(values) // 2
        left_values = values[:midpoint]
        right_values = values[midpoint:]
        logger.warning(
            'NVIDIA repetition loop recovery: splitting translation batch size=%d into %d+%d',
            len(values),
            len(left_values),
            len(right_values),
        )

        left_translated = await _translate_values_recursive(
            left_values,
            language,
            settings,
            repetition_retry=True,
        )
        right_translated = await _translate_values_recursive(
            right_values,
            language,
            settings,
            repetition_retry=True,
        )
        return left_translated + right_translated


async def translate_values(
    values: list[str],
    language: str,
    settings: Settings,
) -> list[str]:
    """Translate strings into a trusted target language with adaptive loop recovery.

    :param values: Ordered English source strings to translate.
    :param language: Trusted target-language name and optional standard code.
    :param settings: Validated runtime settings, including credentials and retry limits.
    :return: Translated strings in exactly the same order as values.
    :raises RepetitionLoopError: If recursive splitting reaches an unrecoverable single-item loop.
    :raises UnsupportedTargetLanguageError: If repeated completed responses decline the target language.
    :raises RuntimeError: If configuration is missing or all translation attempts fail.
    """
    return await _translate_values_recursive(values, language, settings)


async def recognize_custom_language(
    language: str,
    settings: Settings,
) -> CustomLanguageRecognition:
    """Recognize a manually entered language before any UI translation is attempted.

    :param language: Manually entered target-language name.
    :param settings: Validated runtime settings, including credentials and retry limits.
    :return: Recognition status plus canonical English name and standard language code.
    :raises RuntimeError: If configuration is missing or all recognition attempts fail.
    """
    return await _request_with_retries(
        settings=settings,
        prompt_builder=lambda json_retry: build_language_recognition_prompt(
            language,
            json_retry=json_retry,
        ),
        parser=parse_custom_language_recognition,
        operation='language recognition',
    )
