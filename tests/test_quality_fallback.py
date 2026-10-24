"""Exercise semantic and structural quality failover without external API calls."""

import asyncio
import json
import logging

import pytest

from app.config import Settings
from app.services import openai


class FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": self._content}}
            ]
        }


def fake_responses(monkeypatch, contents: list[str]) -> list[dict]:
    """Provide consecutive Chat Completions responses and capture request payloads."""
    payloads: list[dict] = []
    queue = iter(contents)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            assert url == "https://api.openai.com/v1/chat/completions"
            assert headers["Authorization"] == "Bearer test-key"
            payloads.append(json)
            return FakeResponse(next(queue))

    async def no_sleep(delay):
        pass

    monkeypatch.setattr(openai.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(openai.asyncio, "sleep", no_sleep)
    return payloads


def settings(**kwargs) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_model_id="gpt-6-luna",
        openai_fallback_model_id="gpt-5.4-mini",
        openai_retries=5,
        openai_quality_failures_before_fallback=2,
        openai_retry_jitter_seconds=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "first,second",
    [
        ("not json", "still not json"),
        ('["Здравствуйте {name}", "Лишняя строка"]', '["Привет {name}", "X"]'),
        ('["Привет"]', '["Добрый день"]'),
        ('["Hello {name}"]', '["Hello {name}"]'),
    ],
    ids=["bad-json", "wrong-count", "lost-placeholder", "unchanged-translation"],
)
def test_quality_errors_promote_mini_after_two_consecutive_failures(
    monkeypatch, caplog, first, second
):
    payloads = fake_responses(
        monkeypatch,
        [first, second, '["Привет {name}"]'],
    )
    caplog.set_level(logging.WARNING, logger="localizer")

    result = asyncio.run(
        openai.translate_values(["Hello {name}"], "Русский", settings())
    )

    assert result == ["Привет {name}"]
    assert [payload["model"] for payload in payloads] == [
        "gpt-6-luna", "gpt-6-luna", "gpt-5.4-mini"
    ]
    assert "quality failures=2; switching to fallback" in caplog.text


def test_configurable_threshold_can_switch_after_one_bad_response(monkeypatch):
    payloads = fake_responses(monkeypatch, ["not json", '["Bonjour"]'])
    result = asyncio.run(
        openai.translate_values(
            ["Hello"],
            "Français",
            settings(openai_quality_failures_before_fallback=1),
        )
    )
    assert result == ["Bonjour"]
    assert [payload["model"] for payload in payloads] == [
        "gpt-6-luna", "gpt-5.4-mini"
    ]
    assert "Previous response was invalid JSON" in payloads[1]["messages"][1]["content"]


def test_language_recognition_validation_can_promote_backup(monkeypatch):
    valid = json.dumps({
        "language_recognized": True,
        "language": "Yiddish",
        "language_code": "yi",
    })
    payloads = fake_responses(
        monkeypatch,
        ['{"language_recognized": "yes"}', '{"language": "Yiddish"}', valid],
    )
    result = asyncio.run(openai.recognize_custom_language("Идиш", settings()))
    assert result.language == "Yiddish"
    assert result.language_code == "yi"
    assert [payload["model"] for payload in payloads] == [
        "gpt-6-luna", "gpt-6-luna", "gpt-5.4-mini"
    ]


def test_technical_provider_failure_is_not_mislabeled_unsupported(monkeypatch):
    responses = iter([
        '["Hello"]',
        '["Hello"]',
    ])
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            calls.append(json["model"])
            if len(calls) > 2:
                raise openai.httpx.ConnectError("provider unavailable")
            return FakeResponse(next(responses))

    async def no_sleep(delay):
        pass

    monkeypatch.setattr(openai.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(openai.asyncio, "sleep", no_sleep)

    with pytest.raises(RuntimeError, match="OpenAI translation failed"):
        asyncio.run(openai.translate_values(["Hello"], "Français", settings()))
    assert calls[:3] == ["gpt-6-luna", "gpt-6-luna", "gpt-5.4-mini"]


def test_fallback_is_not_considered_available_when_same_as_primary():
    assert openai._fallback_model_available(
        settings(openai_fallback_model_id="gpt-6-luna")
    ) is False
