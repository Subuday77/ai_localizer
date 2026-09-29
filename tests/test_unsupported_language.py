"""Focused tests for unsupported-language fallback semantics."""

import asyncio

from app.api import routes
from app.config import Settings
from app.constants import FALLBACK_MESSAGE, UNSUPPORTED_LANGUAGE_MESSAGE
from app.schemas import TranslateRequest
from app.services.nvidia import CustomLanguageRecognition, UnsupportedTargetLanguageError


def test_recognized_but_unsupported_language_gets_specific_message(monkeypatch) -> None:
    """Return a specific fallback message after successful language recognition.

    :param monkeypatch: Pytest fixture used to replace NVIDIA service calls.
    :return: None.
    :raises AssertionError: If unsupported languages are reported as generic provider failures.
    """
    async def fake_recognize(language: str, settings: Settings) -> CustomLanguageRecognition:
        """Return a deterministic recognized Chukchi language result.

        :param language: User-entered target-language name.
        :param settings: Runtime settings supplied by the route.
        :return: Recognized Chukchi metadata.
        """
        return CustomLanguageRecognition(True, 'Chukchi', 'ckt')

    async def fake_translate(values: list[str], language: str, settings: Settings) -> list[str]:
        """Simulate repeated model refusal to translate the recognized language.

        :param values: Ordered source strings.
        :param language: Normalized target-language description.
        :param settings: Runtime settings supplied by the route.
        :return: Never returns because the simulated target language is unsupported.
        :raises UnsupportedTargetLanguageError: Always, to model exhausted unchanged responses.
        """
        raise UnsupportedTargetLanguageError('translation unsupported after 5 unchanged responses')

    monkeypatch.setattr(routes, 'recognize_custom_language', fake_recognize)
    monkeypatch.setattr(routes, 'translate_values', fake_translate)

    request = TranslateRequest(
        dictionary={'title': 'Welcome'},
        language_name='Чукотский',
    )
    result = asyncio.run(routes.translate(request, Settings()))

    assert result.dictionary == {'title': 'Welcome'}
    assert result.language == 'Chukchi'
    assert result.language_recognized is True
    assert result.fallback is True
    assert result.error == UNSUPPORTED_LANGUAGE_MESSAGE


def test_recognized_language_keeps_generic_message_for_technical_failure(monkeypatch) -> None:
    """Keep provider failures distinct from unsupported-language fallback.

    :param monkeypatch: Pytest fixture used to replace NVIDIA service calls.
    :return: None.
    :raises AssertionError: If a technical failure is mislabeled as unsupported.
    """
    async def fake_recognize(language: str, settings: Settings) -> CustomLanguageRecognition:
        """Return a deterministic recognized Chukchi language result.

        :param language: User-entered target-language name.
        :param settings: Runtime settings supplied by the route.
        :return: Recognized Chukchi metadata.
        """
        return CustomLanguageRecognition(True, 'Chukchi', 'ckt')

    async def fake_translate(values: list[str], language: str, settings: Settings) -> list[str]:
        """Simulate a provider-side technical failure.

        :param values: Ordered source strings.
        :param language: Normalized target-language description.
        :param settings: Runtime settings supplied by the route.
        :return: Never returns because the simulated provider fails.
        :raises RuntimeError: Always, to model a technical NVIDIA failure.
        """
        raise RuntimeError('provider unavailable')

    monkeypatch.setattr(routes, 'recognize_custom_language', fake_recognize)
    monkeypatch.setattr(routes, 'translate_values', fake_translate)

    request = TranslateRequest(
        dictionary={'title': 'Welcome'},
        language_name='Чукотский',
    )
    result = asyncio.run(routes.translate(request, Settings()))

    assert result.dictionary == {'title': 'Welcome'}
    assert result.language == 'Chukchi'
    assert result.language_recognized is True
    assert result.fallback is True
    assert result.error == FALLBACK_MESSAGE
