"""HTTP endpoints; translation business logic lives in services/."""

import logging

from fastapi import APIRouter, Depends

from ..config import Settings, get_settings
from ..constants import (
    FALLBACK_MESSAGE,
    LANGUAGE_NOT_RECOGNIZED_MESSAGE,
    LANGUAGES,
    MOCK_DICTIONARY,
)
from ..schemas import TranslateRequest, TranslateResponse
from ..services.nvidia import translate_custom_language, translate_values

router = APIRouter()
logger = logging.getLogger('localizer')


@router.get('/health', tags=['System'])
def health() -> dict[str, str]:
    """Report process availability without contacting NVIDIA.

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
    """Translate dictionary values while preserving every original key.

    A custom ``language_name`` takes precedence over ``language_code``. Predefined
    dropdown languages are trusted. A manually entered language name is validated
    by the same NVIDIA request that performs the translation. Minor spelling errors
    are accepted when the intended real language is unambiguous.

    :param request: Source dictionary and target-language selection supplied by the client.
    :param settings: Runtime NVIDIA and retry configuration injected by FastAPI.
    :return: Translation result, normalized language, recognition state and fallback state.
    """
    custom_language = bool(request.language_name)
    language = request.language_name or LANGUAGES.get(
        (request.language_code or '').lower(),
        request.language_code,
    )

    if (request.language_code or '').lower() == 'en' and not custom_language:
        return TranslateResponse(
            dictionary=request.dictionary,
            language=language,
            language_recognized=True,
            fallback=False,
        )

    keys = list(request.dictionary.keys())
    values = list(request.dictionary.values())

    try:
        if custom_language:
            result = await translate_custom_language(values, language, settings)
            if not result.language_recognized:
                logger.info('Custom language was not recognized: %r', language)
                return TranslateResponse(
                    dictionary=request.dictionary,
                    language=language,
                    language_recognized=False,
                    fallback=False,
                    error=LANGUAGE_NOT_RECOGNIZED_MESSAGE,
                )
            logger.info(
                'Custom language recognized: input=%r normalized=%r code=%r',
                language,
                result.language,
                result.language_code,
            )
            translated = result.translations
            response_language = result.language or language
        else:
            translated = await translate_values(values, language, settings)
            response_language = language

        return TranslateResponse(
            dictionary=dict(zip(keys, translated)),
            language=response_language,
            language_recognized=True,
            fallback=False,
        )
    except Exception as exc:
        logger.error('English fallback for language=%s: %s', language, exc)
        return TranslateResponse(
            dictionary=request.dictionary,
            language=language,
            language_recognized=None if custom_language else True,
            fallback=True,
            error=FALLBACK_MESSAGE,
        )


@router.post('/demo', response_model=TranslateResponse, tags=['Demo'])
async def demo(
    language_code: str = 'de',
    settings: Settings = Depends(get_settings),
) -> TranslateResponse:
    """Translate the built-in ten-entry mock dictionary for local testing.

    :param language_code: Target language code from ``LANGUAGES``; defaults to German.
    :param settings: Runtime NVIDIA and retry configuration injected by FastAPI.
    :return: The same response structure used by the production translation endpoint.
    """
    return await translate(
        TranslateRequest(dictionary=MOCK_DICTIONARY, language_code=language_code),
        settings,
    )
