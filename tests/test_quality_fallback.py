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
    config = dict(
        openai_api_key="test-key",
        openai_model_id="gpt-6-luna",
        openai_fallback_model_id="gpt-5.4-mini",
        openai_retries=5,
        openai_quality_failures_before_fallback=2,
        openai_retry_jitter_seconds=0,
    )
    config.update(kwargs)
    return Settings(_env_file=None, **config)


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



def test_success_log_reports_primary_model_and_attempt(monkeypatch, caplog):
    """Log a validated first-attempt translation, without leaking its text or key."""
    fake_responses(monkeypatch, ['["Уникальный переведённый текст"]'])
    caplog.set_level(logging.INFO, logger="localizer")
    result = asyncio.run(
        openai.translate_values(["SENSITIVE SOURCE STRING"], "Русский", settings())
    )

    assert result == ["Уникальный переведённый текст"]
    successes = [
        record.message for record in caplog.records
        if "OpenAI translation successful:" in record.message
    ]
    assert len(successes) == 1
    assert "model=gpt-6-luna" in successes[0]
    assert "response_model=gpt-6-luna" in successes[0]
    assert "attempt=1/6" in successes[0]
    assert "fallback=false" in successes[0]
    assert "SENSITIVE SOURCE STRING" not in caplog.text
    assert "Уникальный переведённый текст" not in caplog.text
    assert "test-key" not in caplog.text


def test_success_log_reports_real_fallback_model(monkeypatch, caplog):
    """After two validation failures, successful backup must log fallback=true."""
    fake_responses(monkeypatch, ["wrong json", "wrong json", '["Bonjour"]'])
    caplog.set_level(logging.INFO, logger="localizer")

    assert asyncio.run(
        openai.translate_values(["Hello"], "Français", settings())
    ) == ["Bonjour"]

    successes = [
        record.message for record in caplog.records
        if "OpenAI translation successful:" in record.message
    ]
    assert len(successes) == 1
    assert "model=gpt-5.4-mini" in successes[0]
    assert "attempt=3/6" in successes[0]
    assert "fallback=true" in successes[0]


def test_recognition_logs_the_successful_model(monkeypatch, caplog):
    """Recognition and translation are separately identifiable in Render logs."""
    fake_responses(monkeypatch, [json.dumps({
        "language_recognized": True, "language": "Yiddish", "language_code": "yi"
    })])
    caplog.set_level(logging.INFO, logger="localizer")
    recognized = asyncio.run(openai.recognize_custom_language("Идиш", settings()))

    assert recognized.language == "Yiddish"
    assert "OpenAI language recognition successful:" in caplog.text
    assert "model=gpt-6-luna" in caplog.text
    assert "attempt=1/6" in caplog.text
    assert "fallback=false" in caplog.text


def test_invalid_output_never_logs_false_success(monkeypatch, caplog):
    """Invalid JSON must not be counted as success, even if HTTP returned 200."""
    fake_responses(monkeypatch, ["bad-json"] * 6)
    caplog.set_level(logging.INFO, logger="localizer")

    with pytest.raises(RuntimeError, match="OpenAI translation failed"):
        asyncio.run(
            openai.translate_values(["Hello"], "Français", settings())
        )
    assert "OpenAI translation successful:" not in caplog.text
