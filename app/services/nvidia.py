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
    SYSTEM_PROMPT,
    USER_PROMPT,
)

logger = logging.getLogger('localizer')
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}')
LANGUAGE_CODE = re.compile(r'^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$')
T = TypeVar('T')

class InvalidModelJSONError(ValueError):
    """Raised when NVIDIA returns text that cannot be parsed as JSON."""


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

    return translations


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
) -> str:
    """Build the prompt for translating trusted target-language strings.

    :param values: Ordered source strings that must be translated.
    :param language: Trusted target-language name and optional standard code.
    :param json_retry: Whether to append a stricter invalid-JSON correction instruction.
    :return: Fully formatted user prompt ready for the NVIDIA chat-completions request.
    """
    prompt = USER_PROMPT.format(
        language=language,
        values=json.dumps(values, ensure_ascii=False),
    )
    if json_retry:
        prompt += JSON_RETRY_PROMPT
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


async def _request_with_retries(
    settings: Settings,
    prompt_builder: Callable[[bool], str],
    parser: Callable[[str], T],
    operation: str,
) -> T:
    """Run one NVIDIA operation with validation, retries and fallback-model switching.

    :param settings: Validated runtime settings, including credentials and retry limits.
    :param prompt_builder: Callable that builds the prompt and receives the JSON-retry flag.
    :param parser: Callable that parses and validates the model raw text response.
    :param operation: Short operation name used in logs and terminal errors.
    :return: Parsed and validated model result.
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

    async with httpx.AsyncClient(timeout=settings.nvidia_timeout_seconds) as client:
        for attempt in range(settings.nvidia_retries + 1):
            use_fallback = (
                attempt >= 3
                and settings.nvidia_fallback_model_id
                and not settings.nvidia_fallback_model_id.startswith('your-')
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
                reason = choice.get('finish_reason')
                if reason == 'length':
                    max_tokens = min(max_tokens * 2, settings.nvidia_max_tokens_cap)
                    raise ValueError('Truncated response (finish_reason=length)')
                if reason not in ('stop', None):
                    raise ValueError(f'Unexpected finish_reason={reason}')

                raw_content = choice['message']['content']
                if not isinstance(raw_content, str):
                    raise ValueError('Response content is not text')

                return parser(raw_content)

            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc

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

    raise RuntimeError(f'NVIDIA {operation} failed: {type(last_error).__name__}') from last_error


async def translate_values(
    values: list[str],
    language: str,
    settings: Settings,
) -> list[str]:
    """Translate strings into a trusted target language.

    :param values: Ordered English source strings to translate.
    :param language: Trusted target-language name and optional standard code.
    :param settings: Validated runtime settings, including credentials and retry limits.
    :return: Translated strings in exactly the same order as values.
    :raises RuntimeError: If configuration is missing or all translation attempts fail.
    """
    return await _request_with_retries(
        settings=settings,
        prompt_builder=lambda json_retry: build_translation_prompt(
            values,
            language,
            json_retry=json_retry,
        ),
        parser=lambda raw: parse_translation(raw, values),
        operation='translation',
    )


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
