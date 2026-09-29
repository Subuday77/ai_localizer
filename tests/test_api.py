"""Offline endpoint, prompt and parsing tests; NVIDIA credentials are not required."""

import asyncio
import json

import httpx
import pytest

from app.api import routes
from app.config import Settings
from app.constants import MOCK_DICTIONARY
from app.main import app
from app.services import nvidia
from app.services.nvidia import (
    CustomLanguageTranslation,
    InvalidModelJSONError,
    build_request_payload,
    build_translation_prompt,
    parse_custom_language_translation,
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
    assert data['dictionary'] == MOCK_DICTIONARY
    assert data['language_recognized'] is True
    assert not data['fallback']


def test_fallback_no_key():
    """Verify English fallback behavior when NVIDIA is not configured.

    :return: ``None``; assertions fail the test if fallback behavior changes.
    """
    data = request(
        'POST',
        '/translate',
        json={'dictionary': MOCK_DICTIONARY, 'language_code': 'de'},
    ).json()
    assert data['dictionary'] == MOCK_DICTIONARY
    assert data['language_recognized'] is True
    assert data['fallback']


def test_custom_language_fallback_has_unknown_recognition_state():
    """Verify technical fallback does not falsely reject a custom language name.

    :return: ``None``; assertions fail if provider failure is reported as an invalid language.
    """
    data = request(
        'POST',
        '/translate',
        json={'dictionary': MOCK_DICTIONARY, 'language_name': 'Deutsch'},
    ).json()
    assert data['dictionary'] == MOCK_DICTIONARY
    assert data['language_recognized'] is None
    assert data['fallback']


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


def test_parse_custom_language_recognized():
    """Verify custom-language parsing accepts a normalized real language.

    :return: ``None``; assertions fail if recognition or normalization is lost.
    """
    content = json.dumps(
        {
            'language_recognized': True,
            'language': 'Deutsch',
            'translations': ['Hallo {name}'],
        },
        ensure_ascii=False,
    )
    result = parse_custom_language_translation(content, ['Hello {name}'])
    assert result == CustomLanguageTranslation(
        language_recognized=True,
        language='Deutsch',
        translations=['Hallo {name}'],
    )


def test_parse_custom_language_unrecognized():
    """Verify custom-language parsing accepts the explicit unrecognized shape.

    :return: ``None``; assertions fail if unknown-language handling is malformed.
    """
    content = json.dumps(
        {
            'language_recognized': False,
            'language': None,
            'translations': [],
        }
    )
    result = parse_custom_language_translation(content, ['Hello'])
    assert result.language_recognized is False
    assert result.language is None
    assert result.translations == []


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


def test_custom_language_prompt_validates_and_allows_typos():
    """Verify manually entered languages are validated in the translation prompt.

    :return: ``None``; assertions fail if recognition rules disappear from the prompt.
    """
    prompt = build_translation_prompt(
        ['Welcome'],
        'Deusch',
        validate_custom_language=True,
    )
    assert 'real human language' in prompt
    assert 'Minor spelling mistakes are acceptable' in prompt
    assert '"language_recognized": true' in prompt
    assert '"language_recognized": false' in prompt


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


def test_custom_language_endpoint_rejects_unknown_without_fallback(monkeypatch):
    """Verify an unrecognized custom language returns English without technical fallback.

    :param monkeypatch: Pytest fixture used to replace the NVIDIA translation call.
    :return: ``None``; assertions fail if unknown-language semantics are incorrect.
    """
    async def fake_translate(values, language, settings):
        """Return a deterministic unrecognized-language result.

        :param values: Source values supplied by the route.
        :param language: User-entered language name.
        :param settings: Injected runtime settings.
        :return: Explicit unrecognized-language result.
        """
        return CustomLanguageTranslation(False, None, [])

    monkeypatch.setattr(routes, 'translate_custom_language', fake_translate)

    data = request(
        'POST',
        '/translate',
        json={'dictionary': MOCK_DICTIONARY, 'language_name': 'Krokozyabrian'},
    ).json()

    assert data['dictionary'] == MOCK_DICTIONARY
    assert data['language'] == 'Krokozyabrian'
    assert data['language_recognized'] is False
    assert data['fallback'] is False
    assert data['error'] == 'Language not recognized'


def test_custom_language_endpoint_uses_normalized_name(monkeypatch):
    """Verify a recognizable typo may be normalized by the model in one request.

    :param monkeypatch: Pytest fixture used to replace the NVIDIA translation call.
    :return: ``None``; assertions fail if normalized language names are discarded.
    """
    translated = {key: f'DE:{value}' for key, value in MOCK_DICTIONARY.items()}

    async def fake_translate(values, language, settings):
        """Return a deterministic recognized-language result.

        :param values: Source values supplied by the route.
        :param language: User-entered language name.
        :param settings: Injected runtime settings.
        :return: Recognized result normalized to ``Deutsch``.
        """
        return CustomLanguageTranslation(
            True,
            'Deutsch',
            [f'DE:{value}' for value in values],
        )

    monkeypatch.setattr(routes, 'translate_custom_language', fake_translate)

    data = request(
        'POST',
        '/translate',
        json={'dictionary': MOCK_DICTIONARY, 'language_name': 'Deusch'},
    ).json()

    assert data['dictionary'] == translated
    assert data['language'] == 'Deutsch'
    assert data['language_recognized'] is True
    assert data['fallback'] is False
    assert data['error'] is None
