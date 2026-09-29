"""NVIDIA chat-completions adapter with validation, retries and token handling."""

import asyncio
import json
import logging
import random
import re

import httpx

from ..config import Settings
from ..constants import JSON_RETRY_PROMPT, SYSTEM_PROMPT, USER_PROMPT

logger = logging.getLogger('localizer')
PLACEHOLDER = re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}')


class InvalidModelJSONError(ValueError):
    """Raised when NVIDIA returns text that cannot be parsed as JSON."""


def parse_translation(content: str, originals: list[str]) -> list[str]:
    """Parse and validate an ordered translated JSON string array.

    Markdown JSON fences are tolerated because some models add them despite the
    prompt. The parsed array must have exactly the same number of strings as the
    source array, and source placeholders must be preserved in each item.

    :param content: Raw text returned in the model's ``message.content`` field.
    :param originals: Ordered source strings sent to the model.
    :return: Validated translated strings in the same order as ``originals``.
    :raises InvalidModelJSONError: If the model output is not valid JSON.
    :raises ValueError: If the JSON has an unsafe or unexpected structure.
    """
    content = content.strip()
    if content.startswith('```'):
        lines = content.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith('```'):
            raise ValueError('Incomplete Markdown JSON fence')
        content = '\n'.join(lines[1:-1]).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidModelJSONError('Model returned invalid JSON') from exc

    if (
        not isinstance(parsed, list)
        or len(parsed) != len(originals)
        or any(not isinstance(item, str) for item in parsed)
    ):
        raise ValueError(f'Expected an array of exactly {len(originals)} strings')

    for original, translated in zip(originals, parsed):
        if sorted(PLACEHOLDER.findall(original)) != sorted(PLACEHOLDER.findall(translated)):
            raise ValueError('Placeholder mismatch')

    return parsed


def build_translation_prompt(
    values: list[str],
    language: str,
    json_retry: bool = False,
) -> str:
    """Build the user prompt for an initial translation or JSON-correction retry.

    The retry instruction is appended only after a model response failed JSON
    parsing. It asks the model to keep the requested translations while making
    the complete JSON payload valid, including escaped internal double quotes.

    :param values: Ordered source strings that must be translated.
    :param language: Target language display name, for example ``Deutsch`` or ``עברית``.
    :param json_retry: Whether to append the stricter invalid-JSON correction instruction.
    :return: Fully formatted user prompt ready for the NVIDIA chat-completions request.
    """
    prompt = USER_PROMPT.format(
        language=language,
        values=json.dumps(values, ensure_ascii=False),
    )
    if json_retry:
        prompt += JSON_RETRY_PROMPT
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


async def translate_values(values: list[str], language: str, settings: Settings) -> list[str]:
    """Translate ordered values with NVIDIA and validate the returned array.

    Transient failures and malformed responses are retried. A response ending
    with ``finish_reason == 'length'`` is retried with a larger token budget.
    After several failures a configured fallback NVIDIA model may be used.
    Invalid JSON responses are logged as raw model text, truncated to the
    configured diagnostic limit. After such a parse failure, later attempts add
    an explicit JSON-validity and internal-quote escaping correction instruction.
    API keys and request headers are never logged.

    :param values: Ordered English source strings to translate.
    :param language: Target language display name, for example ``Deutsch`` or ``עברית``.
    :param settings: Validated runtime settings, including credentials and retry limits.
    :return: Translated strings in exactly the same order as ``values``.
    :raises RuntimeError: If configuration is missing or all translation attempts fail.
    """
    if not settings.nvidia_api_key or settings.nvidia_api_key.startswith('your-'):
        raise RuntimeError('NVIDIA_API_KEY is not configured')
    if not settings.nvidia_model_id or settings.nvidia_model_id.startswith('your-'):
        raise RuntimeError('NVIDIA_MODEL_ID is not configured')

    url = settings.nvidia_base_url.rstrip('/') + '/chat/completions'
    base_prompt = build_translation_prompt(values, language)
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
                    build_translation_prompt(values, language, json_retry=True)
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
