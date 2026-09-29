"""NVIDIA chat-completions adapter with validation, retries and token handling."""

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass

import httpx

from ..config import Settings
from ..constants import (
    CUSTOM_LANGUAGE_JSON_RETRY_PROMPT,
    CUSTOM_LANGUAGE_USER_PROMPT,
    JSON_RETRY_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)

logger = logging.getLogger('localizer')
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}')
LANGUAGE_CODE = re.compile(r'^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$')


class InvalidModelJSONError(ValueError):
    """Raised when NVIDIA returns text that cannot be parsed as JSON."""


@dataclass(frozen=True)
class CustomLanguageTranslation:
    """Validated result for a manually entered target-language name."""

    language_recognized: bool
    language: str | None
    language_code: str | None
    translations: list[str]


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


def parse_custom_language_translation(
    content: str,
    originals: list[str],
) -> CustomLanguageTranslation:
    """Parse a combined language-recognition and translation result.

    :param content: Raw text returned in the model's ``message.content`` field.
    :param originals: Ordered source strings sent to the model.
    :return: Validated recognition result with normalized language, language code and translations.
    :raises InvalidModelJSONError: If the model output is not valid JSON.
    :raises ValueError: If the JSON object has an unsafe or unexpected structure.
    """
    parsed = _load_model_json(content)
    if not isinstance(parsed, dict):
        raise ValueError('Expected a JSON object for custom-language translation')

    recognized = parsed.get('language_recognized')
    if not isinstance(recognized, bool):
        raise ValueError('language_recognized must be a boolean')

    language = parsed.get('language')
    language_code = parsed.get('language_code')
    translations = parsed.get('translations')

    if not recognized:
        if language is not None or language_code is not None or translations != []:
            raise ValueError(
                'Unrecognized language must return language=null, language_code=null '
                'and translations=[]'
            )
        return CustomLanguageTranslation(
            language_recognized=False,
            language=None,
            language_code=None,
            translations=[],
        )

    if not isinstance(language, str) or not language.strip():
        raise ValueError('Recognized language must include a normalized language name')
    if (
        not isinstance(language_code, str)
        or not LANGUAGE_CODE.fullmatch(language_code.strip())
    ):
        raise ValueError('Recognized language must include a valid standard language code')

    return CustomLanguageTranslation(
        language_recognized=True,
        language=language.strip(),
        language_code=language_code.strip(),
        translations=_validate_translations(translations, originals),
    )


def build_translation_prompt(
    values: list[str],
    language: str,
    json_retry: bool = False,
    validate_custom_language: bool = False,
) -> str:
    """Build the prompt for translation and optional custom-language validation.

    :param values: Ordered source strings that must be translated.
    :param language: Target language name supplied by the dropdown or user.
    :param json_retry: Whether to append a stricter invalid-JSON correction instruction.
    :param validate_custom_language: Whether the model must validate a manually entered language.
    :return: Fully formatted user prompt ready for the NVIDIA chat-completions request.
    """
    template = CUSTOM_LANGUAGE_USER_PROMPT if validate_custom_language else USER_PROMPT
    prompt = template.format(
        language=language,
        values=json.dumps(values, ensure_ascii=False),
    )
    if json_retry:
        prompt += (
            CUSTOM_LANGUAGE_JSON_RETRY_PROMPT
            if validate_custom_language
            else JSON_RETRY_PROMPT
        )
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


async def _translate(
    values: list[str],
    language: str,
    settings: Settings,
    validate_custom_language: bool,
):
    """Execute NVIDIA translation with retries and optional language recognition.

    :param values: Ordered English source strings to translate.
    :param language: Target language label or manually entered language name.
    :param settings: Validated runtime settings, including credentials and retry limits.
    :param validate_custom_language: Whether model output must include language recognition.
    :return: A translated string list or ``CustomLanguageTranslation``.
    :raises RuntimeError: If configuration is missing or all translation attempts fail.
    """
    if not settings.nvidia_api_key or settings.nvidia_api_key.startswith('your-'):
        raise RuntimeError('NVIDIA_API_KEY is not configured')
    if not settings.nvidia_model_id or settings.nvidia_model_id.startswith('your-'):
        raise RuntimeError('NVIDIA_MODEL_ID is not configured')

    url = settings.nvidia_base_url.rstrip('/') + '/chat/completions'
    base_prompt = build_translation_prompt(
        values,
        language,
        validate_custom_language=validate_custom_language,
    )
    max_tokens = settings.nvidia_max_tokens
    last_error: Exception = RuntimeError('Translation did not complete')
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
                prompt = (
                    build_translation_prompt(
                        values,
                        language,
                        json_retry=True,
                        validate_custom_language=validate_custom_language,
                    )
                    if json_retry_required
                    else base_prompt
                )
                payload = build_request_payload(
                    model=model,
                    prompt=prompt,
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

                if validate_custom_language:
                    return parse_custom_language_translation(raw_content, values)
                return parse_translation(raw_content, values)

            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc

                if isinstance(exc, InvalidModelJSONError):
                    json_retry_required = True

                if isinstance(exc, InvalidModelJSONError) and raw_content is not None:
                    logger.warning(
                        'NVIDIA raw response after invalid JSON, model=%s: %r',
                        model,
                        raw_content[:settings.nvidia_raw_response_log_chars],
                    )

                logger.warning(
                    'NVIDIA attempt %d/%d failed, model=%s: %s',
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

    raise RuntimeError(f'NVIDIA translation failed: {type(last_error).__name__}') from last_error


async def translate_values(values: list[str], language: str, settings: Settings) -> list[str]:
    """Translate a trusted predefined target language.

    :param values: Ordered English source strings to translate.
    :param language: Trusted target language display name from the predefined list.
    :param settings: Validated runtime settings, including credentials and retry limits.
    :return: Translated strings in exactly the same order as ``values``.
    :raises RuntimeError: If configuration is missing or all translation attempts fail.
    """
    return await _translate(
        values,
        language,
        settings,
        validate_custom_language=False,
    )


async def translate_custom_language(
    values: list[str],
    language: str,
    settings: Settings,
) -> CustomLanguageTranslation:
    """Validate a manually entered language name and translate in the same request.

    :param values: Ordered English source strings to translate.
    :param language: Manually entered target-language name.
    :param settings: Validated runtime settings, including credentials and retry limits.
    :return: Recognition status, normalized language name and translations.
    :raises RuntimeError: If configuration is missing or all translation attempts fail.
    """
    return await _translate(
        values,
        language,
        settings,
        validate_custom_language=True,
    )
)