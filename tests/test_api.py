"""Offline API contract and OpenAI adapter tests."""

import asyncio
import json

import httpx
import pytest

from app.api import routes
from app.config import Settings, get_settings
from app.constants import MOCK_DICTIONARY
from app.main import app
from app.services.openai import (
    CustomLanguageRecognition,
    InvalidModelJSONError,
    build_request_payload,
    build_translation_prompt,
    parse_custom_language_recognition,
    parse_translation,
)


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(call())


def test_health():
    assert request("GET", "/health").json() == {"status": "ok"}


def test_languages():
    assert request("GET", "/languages").json()["ru"] == "Русский"


def test_english_does_not_use_ai():
    response = request(
        "POST", "/translate",
        json={"dictionary": MOCK_DICTIONARY, "language_code": "en"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["dictionary"] == MOCK_DICTIONARY
    assert data["language_recognized"] is True
    assert data["fallback"] is False


def test_no_key_returns_english_fallback():
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, openai_api_key="", openai_model_id="gpt-5.6"
    )
    try:
        data = request(
            "POST", "/translate",
            json={"dictionary": MOCK_DICTIONARY, "language_code": "de"},
        ).json()
    finally:
        app.dependency_overrides.clear()
    assert data["dictionary"] == MOCK_DICTIONARY
    assert data["fallback"] is True
    assert data["language_recognized"] is True


def test_parse_placeholders():
    assert parse_translation('["Hallo {name}"]', ["Hello {name}"]) == ["Hallo {name}"]
    with pytest.raises(ValueError, match="Placeholder mismatch"):
        parse_translation('["Hallo"]', ["Hello {name}"])


def test_invalid_json_has_distinct_error():
    with pytest.raises(InvalidModelJSONError):
        parse_translation("not-json", ["Hello"])


def test_custom_language_recognition():
    valid = json.dumps({
        "language_recognized": True,
        "language": "German",
        "language_code": "de",
    })
    assert parse_custom_language_recognition(valid) == CustomLanguageRecognition(
        True, "German", "de"
    )
    invalid = json.dumps({
        "language_recognized": False,
        "language": None,
        "language_code": None,
    })
    assert parse_custom_language_recognition(invalid).language_recognized is False


def test_openai_chat_payload():
    payload = build_request_payload(
        model="gpt-5.6",
        prompt="Translate this",
        max_tokens=512,
        reasoning_effort="none",
    )
    assert payload["model"] == "gpt-5.6"
    assert payload["max_completion_tokens"] == 512
    assert payload["reasoning_effort"] == "none"
    assert "chat_template_kwargs" not in payload
    assert "temperature" not in payload


def test_json_translation_prompt():
    prompt = build_translation_prompt(["Hello {name}"], "Русский")
    assert "same order and count" in prompt
    assert "Preserve placeholders" in prompt


def test_custom_language_endpoint_unrecognized(monkeypatch):
    async def fake_recognition(language, settings):
        return CustomLanguageRecognition(False, None, None)

    monkeypatch.setattr(routes, "recognize_custom_language", fake_recognition)
    result = asyncio.run(routes.translate(
        routes.TranslateRequest(dictionary={"title": "Welcome"}, language_name="Xxyz"),
        Settings(_env_file=None, openai_api_key=""),
    ))
    assert result.fallback is False
    assert result.language_recognized is False
    assert result.dictionary == {"title": "Welcome"}


def test_custom_language_endpoint_normalization(monkeypatch):
    async def fake_recognition(language, settings):
        return CustomLanguageRecognition(True, "German", "de")

    async def fake_translation(values, language, settings):
        assert language == "German (language code: de)"
        return ["Willkommen"]

    monkeypatch.setattr(routes, "recognize_custom_language", fake_recognition)
    monkeypatch.setattr(routes, "translate_values", fake_translation)
    result = asyncio.run(routes.translate(
        routes.TranslateRequest(dictionary={"title": "Welcome"}, language_name="Deusch"),
        Settings(_env_file=None, openai_api_key=""),
    ))
    assert result.dictionary == {"title": "Willkommen"}
    assert result.language == "German"
    assert result.language_recognized is True
    assert result.fallback is False


def test_translation_lifecycle_log_is_correlated_and_redacted(monkeypatch, caplog):
    """Start and finish log lines share an id and contain no dictionary content."""
    async def fake_translation(values, language, settings):
        assert language == "Русский"
        return ["СЕКРЕТНЫЙ_ПЕРЕВОД"]

    monkeypatch.setattr(routes, "translate_values", fake_translation)
    caplog.set_level("INFO", logger="localizer")
    result = asyncio.run(routes.translate(
        routes.TranslateRequest(
            dictionary={"SECRET_KEY": "SENSITIVE_SOURCE_TEXT"}, language_code="ru",
        ),
        Settings(_env_file=None, openai_api_key=""),
    ))

    assert result.fallback is False
    starts = [r.message for r in caplog.records if r.message.startswith("Translation request started:")]
    ends = [r.message for r in caplog.records if r.message.startswith("Translation request completed:")]
    assert len(starts) == len(ends) == 1
    request_id = starts[0].split("id=", 1)[1].split()[0]
    assert f"id={request_id}" in ends[0]
    assert "target=ru" in starts[0]
    assert "keys=1" in starts[0]
    assert "outcome=translated" in ends[0]
    assert "duration_ms=" in ends[0]
    for secret in ("SECRET_KEY", "SENSITIVE_SOURCE_TEXT", "СЕКРЕТНЫЙ_ПЕРЕВОД"):
        assert secret not in caplog.text


def test_translation_fallback_and_cancellation_have_distinct_logs(monkeypatch, caplog):
    """A provider failure returns a fallback, whereas cancellation propagates."""
    async def failed_translation(values, language, settings):
        raise RuntimeError("fake-provider-unavailable")

    monkeypatch.setattr(routes, "translate_values", failed_translation)
    caplog.set_level("INFO", logger="localizer")
    request_payload = routes.TranslateRequest(
        dictionary={"title": "Hello"}, language_code="ru",
    )
    result = asyncio.run(routes.translate(
        request_payload, Settings(_env_file=None, openai_api_key=""),
    ))
    assert result.fallback is True
    assert "error_type=RuntimeError" in caplog.text
    assert "outcome=technical_fallback" in caplog.text

    async def cancelled_translation(values, language, settings):
        raise asyncio.CancelledError()

    caplog.clear()
    monkeypatch.setattr(routes, "translate_values", cancelled_translation)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(routes.translate(
            request_payload, Settings(_env_file=None, openai_api_key=""),
        ))
    assert "Translation request cancelled:" in caplog.text
    assert "outcome=technical_fallback" not in caplog.text
