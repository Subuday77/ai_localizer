"""Tests for NVIDIA response validation."""

import asyncio
import logging

import pytest

from app.config import Settings
from app.services import nvidia

from app.services.nvidia import (
    UnchangedTranslationError,
    _is_unsupported_translation,
    build_translation_prompt,
    parse_translation,
)


def test_parse_translation_rejects_fully_unchanged_output() -> None:
    """Reject a response when every translated string equals its source string.

    :return: None.
    :raises AssertionError: If unchanged source strings are incorrectly accepted.
    """
    originals = ['Welcome', 'Home', 'Contact']

    with pytest.raises(UnchangedTranslationError, match='original source strings unchanged'):
        parse_translation('["Welcome", "Home", "Contact"]', originals)


def test_parse_translation_allows_partially_unchanged_output() -> None:
    """Allow legitimate translations where some individual strings remain unchanged.

    :return: None.
    :raises AssertionError: If a partially translated response is incorrectly rejected.
    """
    originals = ['Welcome', 'Home', 'Contact']

    result = parse_translation('["Bienvenido", "Home", "Contacto"]', originals)

    assert result == ['Bienvenido', 'Home', 'Contacto']


def test_translation_prompt_requires_exact_target_language() -> None:
    """Require the model to avoid substituting a different language.

    :return: None.
    :raises AssertionError: If the strict target-language fallback instruction disappears.
    """
    prompt = build_translation_prompt(['Welcome', 'Home'], 'Chukchi (language code: ckt)')

    assert 'strictly into Chukchi (language code: ckt)' in prompt
    assert 'do not substitute Russian, English' in prompt
    assert 'return the original input array unchanged' in prompt


def test_unsupported_translation_requires_repeated_clean_declines() -> None:
    """Classify unsupported languages only after repeated unchanged translation responses.

    :return: None.
    :raises AssertionError: If technical or malformed failures are misclassified as unsupported.
    """
    assert _is_unsupported_translation('translation', 2, 0) is True
    assert _is_unsupported_translation('translation', 5, 0) is True
    assert _is_unsupported_translation('translation', 1, 0) is False
    assert _is_unsupported_translation('translation', 5, 1) is False
    assert _is_unsupported_translation('language recognition', 5, 0) is False


def test_truncated_response_logs_raw_content(monkeypatch, caplog) -> None:
    """Log the model text when NVIDIA truncates a response by token length.

    :param monkeypatch: Pytest fixture used to replace NVIDIA HTTP and sleep behavior.
    :param caplog: Pytest fixture used to capture logger output.
    :return: None.
    :raises AssertionError: If truncated model text is not written to the warning log.
    """
    class FakeResponse:
        """Minimal successful response carrying a truncated model result."""

        def raise_for_status(self) -> None:
            """Represent a successful HTTP status.

            :return: None.
            """
            return None

        def json(self) -> dict:
            """Return a truncated NVIDIA-like response body.

            :return: Response mapping with finish_reason set to length.
            """
            return {
                'choices': [
                    {
                        'finish_reason': 'length',
                        'message': {'content': '["partial-yiddish-output"'},
                    }
                ]
            }

    class FakeClient:
        """Async client returning the same truncated response on every attempt."""

        def __init__(self, *args, **kwargs):
            """Accept AsyncClient-compatible constructor arguments.

            :param args: Ignored positional arguments.
            :param kwargs: Ignored keyword arguments.
            :return: None.
            """

        async def __aenter__(self):
            """Enter the fake async client context.

            :return: This fake client.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """Exit the fake async client context without suppressing errors.

            :param exc_type: Exception type from the context, if any.
            :param exc: Exception instance from the context, if any.
            :param tb: Traceback from the context, if any.
            :return: False so exceptions are not suppressed.
            """
            return False

        async def post(self, url: str, headers: dict, json: dict):
            """Return the prepared truncated response.

            :param url: NVIDIA endpoint URL.
            :param headers: Request headers.
            :param json: Request payload.
            :return: Fake truncated response.
            """
            return FakeResponse()

    async def no_sleep(delay: float) -> None:
        """Skip retry delays during the unit test.

        :param delay: Requested retry delay in seconds.
        :return: None.
        """
        return None

    monkeypatch.setattr(nvidia.httpx, 'AsyncClient', FakeClient)
    monkeypatch.setattr(nvidia.asyncio, 'sleep', no_sleep)
    caplog.set_level(logging.WARNING, logger='localizer')

    settings = Settings(
        nvidia_api_key='test-key',
        nvidia_model_id='test-model',
        nvidia_fallback_model_id='',
        nvidia_retry_jitter_seconds=0,
    )

    with pytest.raises(RuntimeError):
        asyncio.run(nvidia.translate_values(['Welcome'], 'Yiddish (language code: yi)', settings))

    assert 'NVIDIA raw truncated response' in caplog.text
    assert 'partial-yiddish-output' in caplog.text
