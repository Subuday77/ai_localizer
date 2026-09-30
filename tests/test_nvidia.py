"""Tests for NVIDIA response validation."""

import asyncio
import logging

import pytest

from app.config import Settings
from app.services import nvidia

from app.services.nvidia import (
    UnchangedTranslationError,
    _has_repetition_loop,
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
    assert 'Each output item must be a concise translation of exactly one input item' in prompt
    assert 'Never pad, extend, or repeat text to fill the response' in prompt


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


def test_repetition_loop_detection_handles_single_and_multiword_patterns() -> None:
    """Detect repeated one-word and short-phrase degeneration without flagging normal text.

    :return: None.
    :raises AssertionError: If obvious repetition loops are missed or normal text is flagged.
    """
    assert _has_repetition_loop(' '.join(['אַזוי'] * 12)) is True
    assert _has_repetition_loop(' '.join(['אַזוי ווייטער'] * 12)) is True
    assert _has_repetition_loop(
        'This is a normal translated sentence with varied words and no repeated loop.'
    ) is False


def test_repetition_retry_prompt_starts_over_without_continuing() -> None:
    """Add explicit recovery instructions after a detected repetition loop.

    :return: None.
    :raises AssertionError: If the corrective repetition prompt is missing.
    """
    prompt = build_translation_prompt(
        ['Welcome', 'Home'],
        'Yiddish (language code: yi)',
        repetition_retry=True,
    )

    assert 'Previous response entered a repetition loop and was truncated' in prompt
    assert 'Start the translation again from the beginning' in prompt
    assert 'do not continue the previous response' in prompt
    assert 'Do not repeat the same word or phrase multiple times' in prompt


def test_repetition_loop_keeps_token_limit_and_switches_fallback_early(monkeypatch) -> None:
    """Keep the token cap after degeneration and switch models after two primary loops.

    :param monkeypatch: Pytest fixture used to replace NVIDIA HTTP and sleep behavior.
    :return: None.
    :raises AssertionError: If loop recovery grows tokens or delays fallback switching.
    """
    payloads: list[dict] = []
    loop_content = '["' + ' '.join(['אַזוי ווייטער'] * 12) + '"]'
    responses = [
        {
            'choices': [
                {
                    'finish_reason': 'length',
                    'message': {'content': loop_content},
                }
            ]
        },
        {
            'choices': [
                {
                    'finish_reason': 'length',
                    'message': {'content': loop_content},
                }
            ]
        },
        {
            'choices': [
                {
                    'finish_reason': 'stop',
                    'message': {'content': '["ברוכים הבאים"]'},
                }
            ]
        },
    ]

    class FakeResponse:
        """Minimal successful HTTP response wrapper used by the loop-recovery test."""

        def __init__(self, body: dict):
            """Store a prepared NVIDIA-like response body.

            :param body: JSON-compatible response body returned by json().
            :return: None.
            """
            self.body = body

        def raise_for_status(self) -> None:
            """Represent a successful HTTP status.

            :return: None.
            """
            return None

        def json(self) -> dict:
            """Return the prepared response body.

            :return: NVIDIA-like JSON response mapping.
            """
            return self.body

    class FakeClient:
        """Async context manager returning prepared responses in order."""

        def __init__(self, *args, **kwargs):
            """Accept AsyncClient-compatible constructor arguments.

            :param args: Ignored positional arguments.
            :param kwargs: Ignored keyword arguments.
            :return: None.
            """

        async def __aenter__(self):
            """Enter the fake asynchronous client context.

            :return: This fake client instance.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """Exit the fake asynchronous client context.

            :param exc_type: Exception type propagated from the context, if any.
            :param exc: Exception instance propagated from the context, if any.
            :param tb: Traceback propagated from the context, if any.
            :return: False so exceptions are not suppressed.
            """
            return False

        async def post(self, url: str, headers: dict, json: dict):
            """Capture a request payload and return the next prepared response.

            :param url: NVIDIA endpoint URL.
            :param headers: Request headers.
            :param json: Request body generated by the adapter.
            :return: The next prepared fake response.
            """
            payloads.append(json)
            return FakeResponse(responses[len(payloads) - 1])

    async def no_sleep(delay: float) -> None:
        """Skip retry waiting during the offline unit test.

        :param delay: Requested retry delay in seconds.
        :return: None.
        """
        return None

    monkeypatch.setattr(nvidia.httpx, 'AsyncClient', FakeClient)
    monkeypatch.setattr(nvidia.asyncio, 'sleep', no_sleep)

    settings = Settings(
        nvidia_api_key='test-key',
        nvidia_model_id='primary-model',
        nvidia_fallback_model_id='fallback-model',
        nvidia_max_tokens=2048,
        nvidia_retry_jitter_seconds=0,
    )
    result = asyncio.run(
        nvidia.translate_values(['Welcome'], 'Yiddish (language code: yi)', settings)
    )

    assert result == ['ברוכים הבאים']
    assert [payload['model'] for payload in payloads] == [
        'primary-model',
        'primary-model',
        'fallback-model',
    ]
    assert [payload['max_tokens'] for payload in payloads] == [2048, 2048, 2048]
    assert 'Previous response entered a repetition loop' not in payloads[0]['messages'][1]['content']
    assert 'Previous response entered a repetition loop' in payloads[1]['messages'][1]['content']
    assert 'Previous response entered a repetition loop' in payloads[2]['messages'][1]['content']


def test_normal_truncation_still_increases_token_limit(monkeypatch) -> None:
    """Increase max_tokens for ordinary truncation that has no repetition loop.

    :param monkeypatch: Pytest fixture used to replace NVIDIA HTTP and sleep behavior.
    :return: None.
    :raises AssertionError: If normal truncation stops increasing the output token cap.
    """
    payloads: list[dict] = []
    responses = [
        {
            'choices': [
                {
                    'finish_reason': 'length',
                    'message': {'content': '["A partial but varied translated response'},
                }
            ]
        },
        {
            'choices': [
                {
                    'finish_reason': 'stop',
                    'message': {'content': '["Willkommen"]'},
                }
            ]
        },
    ]

    class FakeResponse:
        """Minimal successful HTTP response wrapper for normal truncation."""

        def __init__(self, body: dict):
            """Store a prepared response body.

            :param body: JSON-compatible response body.
            :return: None.
            """
            self.body = body

        def raise_for_status(self) -> None:
            """Represent a successful HTTP status.

            :return: None.
            """
            return None

        def json(self) -> dict:
            """Return the prepared response body.

            :return: NVIDIA-like JSON response mapping.
            """
            return self.body

    class FakeClient:
        """Async client returning prepared normal-truncation responses."""

        def __init__(self, *args, **kwargs):
            """Accept AsyncClient-compatible constructor arguments.

            :param args: Ignored positional arguments.
            :param kwargs: Ignored keyword arguments.
            :return: None.
            """

        async def __aenter__(self):
            """Enter the fake asynchronous client context.

            :return: This fake client.
            """
            return self

        async def __aexit__(self, exc_type, exc, tb):
            """Exit the fake asynchronous client context.

            :param exc_type: Exception type from the context, if any.
            :param exc: Exception instance from the context, if any.
            :param tb: Traceback from the context, if any.
            :return: False so exceptions are not suppressed.
            """
            return False

        async def post(self, url: str, headers: dict, json: dict):
            """Capture the payload and return the next response.

            :param url: NVIDIA endpoint URL.
            :param headers: Request headers.
            :param json: Request body generated by the adapter.
            :return: The next prepared response.
            """
            payloads.append(json)
            return FakeResponse(responses[len(payloads) - 1])

    async def no_sleep(delay: float) -> None:
        """Skip retry waiting during the unit test.

        :param delay: Requested retry delay in seconds.
        :return: None.
        """
        return None

    monkeypatch.setattr(nvidia.httpx, 'AsyncClient', FakeClient)
    monkeypatch.setattr(nvidia.asyncio, 'sleep', no_sleep)

    settings = Settings(
        nvidia_api_key='test-key',
        nvidia_model_id='primary-model',
        nvidia_fallback_model_id='fallback-model',
        nvidia_max_tokens=2048,
        nvidia_retry_jitter_seconds=0,
    )
    result = asyncio.run(
        nvidia.translate_values(['Welcome'], 'German (language code: de)', settings)
    )

    assert result == ['Willkommen']
    assert [payload['max_tokens'] for payload in payloads] == [2048, 4096]
