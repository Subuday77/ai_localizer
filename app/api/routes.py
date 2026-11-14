"""HTTP endpoints; translation business logic lives in services/."""

import asyncio
import logging
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError

from ..contexts.repository import (
    ContextNotFoundError, ContextUnavailableError, InvalidContextDataError,
    build_context_prompt, list_public_contexts, resolve_context,
)

from ..auth import require_api_key

from ..config import Settings, get_settings
from ..constants import (
    FALLBACK_MESSAGE,
    LANGUAGE_NOT_RECOGNIZED_MESSAGE,
    UNSUPPORTED_LANGUAGE_MESSAGE,
    LANGUAGES,
    MOCK_DICTIONARY,
)
from ..schemas import TranslateRequest, TranslateResponse
from ..quality.controller import QualityConfigurationError, require_independent_controller
from ..quality.pipeline import VerificationFailed, verify_translation
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


@router.get('/contexts', tags=['Localization'])
def contexts(settings: Settings = Depends(get_settings)) -> dict:
    """List public published contexts without invoking any AI model."""
    try:
        return {'contexts': list_public_contexts(
            settings.localizer_database_url, settings.localizer_database_password,
        )}
    except ContextUnavailableError:
        raise HTTPException(status_code=503, detail='Context catalog is not configured')
    except (SQLAlchemyError, InvalidContextDataError):
        logger.exception('Context catalog database unavailable')
        raise HTTPException(status_code=503, detail='Context catalog unavailable')


@router.post('/translate', response_model=TranslateResponse, tags=['Localization'], dependencies=[Depends(require_api_key)])
async def translate(
    request: TranslateRequest,
    settings: Settings = Depends(get_settings),
) -> TranslateResponse:
    """Translate values while emitting content-free lifecycle diagnostics.

    Request identifiers correlate entry, completion and failure. Never log
    dictionary values, keys, prompts, credentials or custom language names.
    """
    if request.quality_mode == 'verified':
        # Verified is deliberately internal-only; no anonymous or public-demo access.
        if not settings.localizer_require_api_key or not settings.localizer_qa_verified_enabled:
            raise HTTPException(status_code=403, detail='Verified mode is not enabled for this API')
        try:
            require_independent_controller(settings)
        except QualityConfigurationError:
            raise HTTPException(status_code=503, detail='Verified quality configuration is incomplete')
        if request.language_code == 'en' and not request.language_name:
            raise HTTPException(status_code=422, detail='Verified mode requires a target translation')
    request_id = secrets.token_hex(6)
    started = time.monotonic()
    custom_language = bool(request.language_name)
    target_code = (request.language_code or '').lower().strip()
    log_target = target_code if target_code in LANGUAGES else (
        'custom' if custom_language else 'unknown'
    )
    resolved_context = None
    if request.context_id:
        try:
            resolved_context = await run_in_threadpool(
                resolve_context,
                settings.localizer_database_url,
                request.context_id,
                request.context_version,
                settings.localizer_database_password,
            )
        except ContextNotFoundError:
            raise HTTPException(status_code=422, detail='Unknown or unpublished context/version')
        except ContextUnavailableError:
            raise HTTPException(status_code=503, detail='Context catalog is not configured')
        except (SQLAlchemyError, InvalidContextDataError):
            logger.exception('Context resolution failed: id=%s', request_id)
            raise HTTPException(status_code=503, detail='Context catalog unavailable')

    keys_count = len(request.dictionary)
    total_chars = sum(len(value) for value in request.dictionary.values())
    logger.info(
        'Translation request started: id=%s target=%s custom=%s keys=%d chars=%d',
        request_id, log_target, str(custom_language).lower(), keys_count, total_chars,
    )

    def finished(result: TranslateResponse, outcome: str) -> TranslateResponse:
        if resolved_context is not None:
            result.context_id = resolved_context.context_id
            result.context_version = resolved_context.version
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
    quality_report = None

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
            context_prompt = (
                build_context_prompt(resolved_context, recognition.language_code)
                if resolved_context else None
            )
            if context_prompt is None:
                translated = await translate_values(values, target_language, settings)
            else:
                translated = await translate_values(
                    values, target_language, settings, context_prompt=context_prompt,
                )
        else:
            context_prompt = (
                build_context_prompt(resolved_context, target_code)
                if resolved_context else None
            )
            if context_prompt is None:
                translated = await translate_values(values, language, settings)
            else:
                translated = await translate_values(
                    values, language, settings, context_prompt=context_prompt,
                )
            response_language = language

        if request.quality_mode == 'verified':
            try:
                translated, quality_report = await verify_translation(
                    values, translated,
                    target_language if custom_language else language,
                    settings, resolved_context, context_prompt,
                )
            except VerificationFailed as exc:
                logger.warning('Independent verification failed: id=%s', request_id)
                raise HTTPException(
                    status_code=422,
                    detail={'status': 'verification_failed',
                            'quality_report': exc.report.model_dump()},
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Independent verification unavailable: id=%s', request_id)
                raise HTTPException(status_code=503, detail='Independent verification unavailable')

        return finished(TranslateResponse(
            dictionary=dict(zip(keys, translated)),
            language=response_language,
            language_recognized=True,
            fallback=False,
            quality_report=quality_report,
        ), 'translated')
    except asyncio.CancelledError:
        logger.warning(
            'Translation request cancelled: id=%s target=%s duration_ms=%d',
            request_id, log_target, int((time.monotonic() - started) * 1000),
        )
        raise
    except HTTPException:
        raise
    except UnsupportedTargetLanguageError:
        if request.quality_mode == 'verified':
            raise HTTPException(status_code=422, detail='Target language not reliably supported')
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
        if request.quality_mode == 'verified':
            logger.exception('Verified translation failed: id=%s', request_id)
            raise HTTPException(status_code=503, detail='Verified translation unavailable')
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


@router.post('/demo', response_model=TranslateResponse, tags=['Demo'], dependencies=[Depends(require_api_key)])
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
