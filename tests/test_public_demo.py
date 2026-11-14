"""Public demo isolation, abuse protection and persistent quota regression tests."""

import asyncio
import sqlite3

import httpx
import pytest

from app.api import public_demo
from app.config import Settings, get_settings
from app.main import app


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(call())


@pytest.fixture
def demo_mode(tmp_path):
    settings = Settings(
        _env_file=None,
        localizer_require_api_key=True,
        localizer_api_key_hashes=["0" * 64],
        localizer_demo_enabled=True,
        localizer_demo_db_path=str(tmp_path / "demo.sqlite3"),
        localizer_demo_daily_limit=4,
        localizer_demo_per_ip_hour_limit=2,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield settings
    finally:
        app.dependency_overrides.clear()


def test_demo_disabled_by_default(tmp_path):
    settings = Settings(
        _env_file=None,
        localizer_require_api_key=True,
        localizer_api_key_hashes=["0" * 64],
        localizer_demo_enabled=False,
        localizer_demo_db_path=str(tmp_path / "demo.sqlite3"),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        r = request("POST", "/public-demo/translate", json={"language_code": "en"})
        assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_english_demo_is_fixed_dictionary_and_requires_no_secret(demo_mode):
    r = request("POST", "/public-demo/translate", json={"language_code": "en"})
    assert r.status_code == 200
    assert r.json()["dictionary"] == public_demo.DEMO_ENGLISH_UI
    assert r.json()["fallback"] is False
    assert "brand" in r.json()["dictionary"]


def test_arbitrary_dictionary_and_malicious_language_rejected(demo_mode):
    for payload in (
        {"language_code": "en", "dictionary": {"secret": "translate me"}},
        {"language_code": "xx"},
        {"language_code": "en", "language_name": "Russian; DROP TABLE demo_requests;"},
        {"language_name": "   "},
        {},
    ):
        response = request("POST", "/public-demo/translate", json=payload)
        assert response.status_code == 422
    # Rejected requests do not consume quotas.
    assert request("POST", "/public-demo/translate", json={"language_code": "en"}).status_code == 200


def test_hourly_limit_persists_across_connections_and_restarts(demo_mode):
    payload = {"language_code": "en"}
    assert request("POST", "/public-demo/translate", json=payload).status_code == 200
    assert request("POST", "/public-demo/translate", json=payload).status_code == 200
    blocked = request("POST", "/public-demo/translate", json=payload)
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "3600"
    with sqlite3.connect(demo_mode.localizer_demo_db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM demo_requests").fetchone()[0] == 2


def test_global_daily_limit_applies_to_multiple_clients(demo_mode, monkeypatch):
    # Unique clients get separate hourly allowances, but share a durable global cap.
    for ip in ("203.0.113.1", "203.0.113.2", "203.0.113.3", "203.0.113.4"):
        public_demo._claim_quota(demo_mode, ip)
    with pytest.raises(Exception) as error:
        public_demo._claim_quota(demo_mode, "203.0.113.5")
    assert getattr(error.value, "status_code", None) == 429
    with sqlite3.connect(demo_mode.localizer_demo_db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM demo_requests").fetchone()[0] == 4


def test_unavailable_quota_storage_fails_closed(demo_mode):
    demo_mode.localizer_demo_db_path = "/no-such-demo-directory/state.sqlite3"
    result = request("POST", "/public-demo/translate", json={"language_code": "en"})
    assert result.status_code == 503


def test_private_paid_route_still_requires_bearer(demo_mode):
    result = request(
        "POST", "/translate",
        json={"dictionary": {"hello": "Hello"}, "language_code": "en"},
    )
    assert result.status_code == 401


def test_demo_uses_same_translation_logic_with_no_arbitrary_text(demo_mode, monkeypatch):
    captured = {}
    async def fake_translate(values, language, settings):
        captured["values"] = values
        captured["language"] = language
        return ["ПЕРЕВОД" for _ in values]

    from app.api import routes
    monkeypatch.setattr(routes, "translate_values", fake_translate)
    result = request("POST", "/public-demo/translate", json={"language_code": "ru"})
    assert result.status_code == 200
    assert result.json()["fallback"] is False
    assert len(result.json()["dictionary"]) == len(public_demo.DEMO_ENGLISH_UI)
    assert captured["values"] == list(public_demo.DEMO_ENGLISH_UI.values())
    assert captured["language"] == "Русский"


def test_caddy_ip_header_keeps_hourly_quota_per_visitor(demo_mode):
    payload = {"language_code": "en"}
    for addr in ("203.0.113.9", "203.0.113.9", "203.0.113.10", "203.0.113.10"):
        r = request(
            "POST", "/public-demo/translate",
            json=payload, headers={"X-Demo-Client-IP": addr},
        )
        assert r.status_code == 200
    assert request(
        "POST", "/public-demo/translate",
        json=payload, headers={"X-Demo-Client-IP": "203.0.113.9"},
    ).status_code == 429
