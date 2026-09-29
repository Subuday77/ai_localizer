"""Offline endpoint, prompt and parsing tests; NVIDIA credentials are not required."""

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.constants import MOCK_DICTIONARY
from app.main import app
from app.services import nvidia
from app.services.nvidia import (
    InvalidModelJSONError,
    build_request_payload,
    build_translation_prompt,
    parse_translation,
)


def request(method: str, path: str, **kwargs):
    """Issue a local ASGI request without starting a web server.

    :param method: HTTP method, for example ``GET`` or ``POST``.
    :param path: Application-relative request path.
    :param kwargs: Additional keyword arguments forwarded to ``httpx``.
    :return: The completed ``httpx.Response`` object.
    """

    async def call():
        """Execute the request against the in-process ASGI transport.

        :return: The completed ``httpx.Response`` object.
        """
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url='http://test',
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(call())


def test_health():
    """Verify that the health endpoint reports an available process.

    :return: ``None``; assertions fail the test if the endpoint is unhealthy.
    """
    assert request('GET', '/health').json() == {'status': 'ok'}


def test_languages():
    """Verify that native language labels are exposed by the API.

    :return: ``None``; assertions fail the test if language metadata is wrong.
    """
    assert request('GET', '/languages').json()['de'] == 'Deutsch'


def test_english_no_api_key():
    """Verify that English can be returned without contacting NVIDIA.

    :return: ``None``; assertions fail the test if English takes the AI path.
    """
    data = request(
        'POST',
        '/translate',
        json={'dictionary': MOCK_DICTIONARY, 'language_code': 'en'},
    ).json()
    assert data['dictionary'] == MOCK_DICTIONARY and not data['fallback']


def test_fallback_no_key():
    """Verify English fallback behavior when NVIDIA is not configured.

    :return: ``None``; assertions fail the test if fallback behavior changes.
    """
    data = request(
        'POST',
        '/translate',
        json={'dictionary': MOCK_DICTIONARY, 'language_code': 'de'},
    ).json()
    assert data['dictionary'] == MOCK_DICTIONARY and data['fallback']


def test_parse_placeholders():
    """Verify placeholder preservation during parsed translation validation.

    :return: ``None``; assertions fail the test if placeholders can be lost.
    """
    assert parse_translation('["Hallo {name}"]', ['Hello {name}']) == ['Hallo {name}']
    with pytest.raises(ValueError):
        parse_translation('["Hallo"]', ['Hello {name}'])


def test_invalid_json_uses_specific_exception():
    """Verify that malformed model output can be diagnosed separately.

    :return: ``None``; assertions fail the test if invalid JSON is not distinguishable.
    """
    with pytest.raises(InvalidModelJSONError):
        parse_translation('not-json', ['Hello'])


def test_thinking_is_disabled_in_payload():
    """Verify that the NVIDIA request carries the configured thinking flag.

    :return: ``None``; assertions fail the test if the thinking flag is omitted.
    """
    payload = build_request_payload(
        model='test-model',
        prompt='Translate this',
        max_tokens=512,
        enable_thinking=False,
    )
    assert payload['chat_template_kwargs']['enable_thinking'] is False


def test_translation_prompt_requires_json_quote_escaping():
    """Verify that the normal prompt explicitly requires valid quote escaping.

    :return: ``None``; assertions fail the test if the JSON-safety instruction disappears.
    """
    prompt = build_translation_prompt(['Email address'], 'עברית')
    assert 'escape it correctly for JSON as \\"' in prompt
    assert 'Previous response was invalid JSON' not in prompt


def test_json_retry_adds_stricter_instruction(monkeypatch):
    """Verify that an invalid JSON response strengthens the following retry prompt.

    :param monkeypatch: Pytest fixture used to replace NVIDIA HTTP and sleep behavior.
    :return: ``None``; assertions fail the test if retry prompting or parsing is incorrect.
    """
    payloads: list[dict] = []
    valid_content = json.dumps(['כתובת דוא"ל'], ensure_ascii=False)
    responses = [
        {
            'choices': [
                {
                    'finish_reason': 'stop',
                    'message': {'content': '["כתובת דוא"ל"]'},
                }
            ]
        },
        {
            'choices': [
                {
                    'finish_reason': 'stop',
                    'message': {'content': valid_content},
                }
            ]
        },
    ]

    class FakeResponse:
        """Minimal successful HTTP response wrapper used by the retry test."""

        def __init__(self, body: dict):
            """Store a prepared NVIDIA-like response body.

            :param body: JSON-compatible response body returned by ``json()``.
            :return: ``None``; initializes the fake response in place.
            """
            self.body = body

        def raise_for_status(self) -> None:
            """Represent a successful HTTP status.

            :return: ``None``; no exception is raised for this fake response.
            """
            return None

        def json(self) -> dict:
            """Return the prepared response body.

            :return: NVIDIA-like JSON response mapping.
            """
            return self.body

    class FakeClient:
        """Async context manager that returns prepared NVIDIA responses in order."""

        def __init__(self, *args, **kwargs):
            """Accept the same constructor arguments as ``httpx.AsyncClient``.

            :param args: Ignored positional constructor arguments.
            :param kwargs: Ignored keyword constructor arguments.
            :return: ``None``; initializes the fake client in place.
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
            :return: ``False`` so exceptions are never suppressed.
            """
            return False

        async def post(self, url: str, headers: dict, json: dict):
            """Capture a request payload and return the next prepared response.

            :param url: NVIDIA endpoint URL; accepted for signature compatibility.
            :param headers: Request headers; accepted for signature compatibility.
            :param json: JSON request body generated by the adapter.
            :return: The next prepared fake HTTP response.
            """
            payloads.append(json)
            return FakeResponse(responses[len(payloads) - 1])

    async def no_sleep(delay: float) -> None:
        """Skip retry waiting during the offline unit test.

        :param delay: Requested retry delay in seconds; intentionally ignored.
        :return: ``None`` immediately.
        """
        return None

    monkeypatch.setattr(nvidia.httpx, 'AsyncClient', FakeClient)
    monkeypatch.setattr(nvidia.asyncio, 'sleep', no_sleep)

    settings = Settings(
        nvidia_api_key='test-key',
        nvidia_model_id='test-model',
        nvidia_fallback_model_id='',
        nvidia_retry_jitter_seconds=0,
    )
    result = asyncio.run(nvidia.translate_values(['Email address'], 'עברית', settings))

    assert result == ['כתובת דוא"ל']
    first_prompt = payloads[0]['messages'][1]['content']
    second_prompt = payloads[1]['messages'][1]['content']
    assert 'Previous response was invalid JSON' not in first_prompt
    assert 'Previous response was invalid JSON' in second_prompt
    assert 'ensure the JSON is valid' in second_prompt
    assert 'all internal double quotes are properly escaped' in second_prompt
