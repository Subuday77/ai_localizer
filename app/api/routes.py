"""HTTP endpoints; translation business logic lives in services/."""

import asyncio
import logging
import secrets
import time

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..constants import (
    FALLBACK_MESSAGE,
    LANGUAGE_NOT_RECOGNIZED_MESSAGE,
    UNSUPPORTED_LANGUAGE_MESSAGE,
    LANGUAGES,
    MOCK_DICTIONARY,
)
from ..schemas import TranslateRequest, TranslateResponse
from ..services.openai import (
    UnsupportedTargetLanguageError,
    recognize_custom_language,
    translate_values,
)

router = APIRouter()
logger = logging.getLogger('localizer')


@router.get('/health', tags=['System'])
def health() -> dict[str, str]:
    """Report process availability without contacting OpenAI.

    :return: A small status payload indicating that the backend process is alive.
    """
    return {'status': 'ok'}


@router.get('/languages', tags=['Localization'])
def languages() -> dict[str, str]:
    """Return predefined language choices for the future frontend dropdown.

    :return: Mapping of language codes to native display names.
    """
    return LANGUAGES


@router.post('/translate', response_model=TranslateResponse, tags=['Localization'])
async def translate(
    request: TranslateRequest,
    settings: Settings = Depends(get_settings),
) -> TranslateResponse:
    """Translate values while emitting content-free lifecycle diagnostics.

    Request identifiers correlate entry, completion and failure. Never log
    dictionary values, keys, prompts, credentials or custom language names.
    """
    request_id = secrets.token_hex(6)
    started = time.monotonic()
    custom_language = bool(request.language_name)
    target_code = (request.language_code or '').lower().strip()
    log_target = target_code if target_code in LANGUAGES else (
        'custom' if custom_language else 'unknown'
    )
    keys_count = len(request.dictionary)
    total_chars = sum(len(value) for value in request.dictionary.values())
    logger.info(
        'Translation request started: id=%s target=%s custom=%s keys=%d chars=%d',
        request_id, log_target, str(custom_language).lower(), keys_count, total_chars,
    )

    def finished(result: TranslateResponse, outcome: str) -> TranslateResponse:
        logger.info(
            'Translation request completed: id=%s target=%s outcome=%s '
            'fallback=%s keys=%d duration_ms=%d',
            request_id, log_target, outcome, str(result.fallback).lower(),
            keys_count, int((time.monotonic() - started) * 1000),
        )
        return result

    language = request.language_name or LANGUAGES.get(
        target_code, request.language_code,
    )
    if target_code == 'en' and not custom_language:
        return finished(TranslateResponse(
            dictionary=request.dictionary,
            language=language,
            language_recognized=True,
            fallback=False,
        ), 'english_passthrough')

    keys = list(request.dictionary.keys())
    values = list(request.dictionary.values())
    response_language = language
    language_recognized_state: bool | None = True if not custom_language else None

    try:
        if custom_language:
            recognition = await recognize_custom_language(language, settings)
            if not recognition.language_recognized:
                logger.info('Translation language not recognized: id=%s', request_id)
                return finished(TranslateResponse(
                    dictionary=request.dictionary,
                    language=language,
                    language_recognized=False,
                    fallback=False,
                    error=LANGUAGE_NOT_RECOGNIZED_MESSAGE,
                ), 'unrecognized')
            logger.info(
                'Translation language recognized: id=%s normalized_code=%s',
                request_id, recognition.language_code,
            )
            response_language = recognition.language or language
            language_recognized_state = True
            target_language = (
                f'{recognition.language} (language code: {recognition.language_code})'
            )
            translated = await translate_values(values, target_language, settings)
        else:
            translated = await translate_values(values, language, settings)
            response_language = language

        return finished(TranslateResponse(
            dictionary=dict(zip(keys, translated)),
            language=response_language,
            language_recognized=True,
            fallback=False,
        ), 'translated')
    except asyncio.CancelledError:
        logger.warning(
            'Translation request cancelled: id=%s target=%s duration_ms=%d',
            request_id, log_target, int((time.monotonic() - started) * 1000),
        )
        raise
    except UnsupportedTargetLanguageError:
        logger.warning(
            'Translation unsupported: id=%s target=%s duration_ms=%d',
            request_id, log_target, int((time.monotonic() - started) * 1000),
        )
        return finished(TranslateResponse(
            dictionary=request.dictionary,
            language=response_language,
            language_recognized=language_recognized_state,
            fallback=True,
            error=UNSUPPORTED_LANGUAGE_MESSAGE,
        ), 'unsupported_fallback')
    except Exception as exc:
        logger.exception(
            'Translation request failed: id=%s target=%s error_type=%s',
            request_id, log_target, type(exc).__name__,
        )
        return finished(TranslateResponse(
            dictionary=request.dictionary,
            language=response_language,
            language_recognized=language_recognized_state,
            fallback=True,
            error=FALLBACK_MESSAGE,
        ), 'technical_fallback')


@router.post('/demo', response_model=TranslateResponse, tags=['Demo'])
async def demo(
    language_code: str = 'de',
    settings: Settings = Depends(get_settings),
) -> TranslateResponse:
    """Translate the built-in ten-entry mock dictionary for local testing.

    :param language_code: Target language code from ``LANGUAGES``; defaults to German.
    :param settings: Runtime OpenAI and retry configuration injected by FastAPI.
    :return: The same response structure used by the production translation endpoint.
    """
    return await translate(
        TranslateRequest(dictionary=MOCK_DICTIONARY, language_code=language_code),
        settings,
    )
