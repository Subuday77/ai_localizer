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
