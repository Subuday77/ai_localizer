"""VPS-only paid endpoint authentication contract tests; no OpenAI calls."""

import asyncio
import hashlib

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.main import app

TEST_TOKEN = 'only-a-fixture-not-a-production-api-key'
TEST_HASH = hashlib.sha256(TEST_TOKEN.encode('utf-8')).hexdigest()
TRANSLATION = {
    'dictionary': {'greeting': 'Hello {name}'},
    'language_code': 'en',
}


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(call())


@pytest.fixture
def authenticated_mode():
    settings = Settings(
        _env_file=None,
        localizer_require_api_key=True,
        localizer_api_key_hashes=[TEST_HASH],
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield
    finally:
        app.dependency_overrides.clear()


def test_missing_and_invalid_bearer_token_rejected(authenticated_mode):
    for headers in (
        {},
        {'Authorization': 'Bearer invalid'},
        {'Authorization': 'Basic invalid'},
        {'Authorization': 'Bearer '},
    ):
        result = request('POST', '/translate', json=TRANSLATION, headers=headers)
        assert result.status_code == 401
        assert result.headers['www-authenticate'] == 'Bearer'


def test_valid_bearer_token_allows_english_no_model_call(authenticated_mode):
    response = request(
        'POST', '/translate', json=TRANSLATION,
        headers={'Authorization': f'Bearer {TEST_TOKEN}'},
    )
    assert response.status_code == 200
    assert response.json()['fallback'] is False
    assert response.json()['dictionary'] == TRANSLATION['dictionary']


def test_paid_demo_endpoint_also_requires_key(authenticated_mode):
    assert request('POST', '/demo?language_code=en').status_code == 401
    with_key = request(
        'POST', '/demo?language_code=en',
        headers={'Authorization': f'Bearer {TEST_TOKEN}'},
    )
    assert with_key.status_code == 200


def test_health_and_languages_do_not_require_key(authenticated_mode):
    assert request('GET', '/health').status_code == 200
    assert request('GET', '/languages').status_code == 200


def test_enabled_auth_must_have_valid_hashes():
    with pytest.raises(ValidationError, match='LOCALIZER_API_KEY_HASHES'):
        Settings(_env_file=None, localizer_require_api_key=True, localizer_api_key_hashes=[])
    with pytest.raises(ValidationError, match='LOCALIZER_API_KEY_HASHES'):
        Settings(_env_file=None, localizer_require_api_key=True, localizer_api_key_hashes=['plaintext'])


def test_legacy_render_mode_remains_backward_compatible():
    settings = Settings(
        _env_file=None, localizer_require_api_key=False,
        localizer_api_key_hashes=[],
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = request('POST', '/translate', json=TRANSLATION)
        assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()
