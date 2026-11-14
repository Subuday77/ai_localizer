"""Localizer 2.0 context API, immutable-parent resolution, and compatibility tests."""

import asyncio
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.contexts.models import Base, Context, ContextVersion
from app.contexts.repository import (
    build_context_prompt, list_public_contexts, resolve_context,
)
from app.main import app
from scripts.seed_contexts import seed


def api(method, path, **kwargs):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


@pytest.fixture
def context_db(tmp_path):
    url = f'sqlite:///{tmp_path / "localizer.db"}'
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        assert seed(session) == 6
    yield url
    engine.dispose()


def test_seed_is_idempotent_and_catalog_only_lists_public_published(context_db):
    assert len(list_public_contexts(context_db)) == 6
    engine = create_engine(context_db)
    with Session(engine) as session, session.begin():
        assert seed(session) == 0
        music = session.get(Context, 'music')
        music.is_public = False
        railway = session.get(Context, 'railway')
        railway.published_version = None
    assert [x['id'] for x in list_public_contexts(context_db)] == [
        'electronics', 'military', 'software', 'technology',
    ]
    engine.dispose()


def test_pinned_parent_version_remains_reproducible(context_db):
    original = resolve_context(context_db, 'electronics')
    assert original.version == 1
    assert original.style['tone'] == 'precise and professional'
    assert original.rules[0].preferred == 'проводник'
    engine = create_engine(context_db)
    with Session(engine) as session, session.begin():
        session.add(ContextVersion(
            context_id='technology', version=2, status='published',
            rules=[], glossary=[], style={'tone': 'casual'},
        ))
        session.get(Context, 'technology').published_version = 2
    # Electronics v1 still pins its parent to Technology v1.
    still_original = resolve_context(context_db, 'electronics', 1)
    assert still_original.style['tone'] == 'precise and professional'
    assert resolve_context(context_db, 'technology').style['tone'] == 'casual'
    engine.dispose()


def test_context_prompt_filters_pairs_and_preserves_unambiguous_meaning(context_db):
    selected = resolve_context(context_db, 'electronics')
    russian = build_context_prompt(selected, 'ru')
    assert 'electrical-conductor' in russian
    assert 'проводник' in russian
    assert 'Never mechanically replace' in russian
    german = build_context_prompt(selected, 'de')
    assert 'electrical-conductor' not in german
    assert 'precise and professional' in german


def test_military_rule_is_published_and_only_targets_russian(context_db):
    catalog = list_public_contexts(context_db)
    assert any(item['id'] == 'military' and item['version'] == 1 for item in catalog)
    military = resolve_context(context_db, 'military')
    assert military.version == 1
    russian = build_context_prompt(military, 'ru')
    assert 'military-fruit-salad' in russian
    assert 'наградные планки' in russian
    assert 'фруктовый салат' in russian
    assert 'military-fruit-salad' not in build_context_prompt(military, 'de')


def test_context_api_and_pinned_response(context_db, monkeypatch):
    from app.api import routes

    seen = []

    async def fake_translate(values, language, settings, context_prompt=None):
        seen.append((values, language, context_prompt))
        return ['Проводник']

    monkeypatch.setattr(routes, 'translate_values', fake_translate)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, localizer_database_url=context_db,
        localizer_require_api_key=False,
    )
    try:
        catalog = api('GET', '/contexts')
        assert catalog.status_code == 200
        assert len(catalog.json()['contexts']) == 6
        response = api('POST', '/translate', json={
            'dictionary': {'a': 'Copper is a good conductor.'},
            'language_code': 'ru',
            'context_id': 'electronics',
            'context_version': 1,
        })
        assert response.status_code == 200, response.text
        assert response.json()['dictionary'] == {'a': 'Проводник'}
        assert response.json()['context_id'] == 'electronics'
        assert response.json()['context_version'] == 1
        assert 'electrical-conductor' in seen[0][2]
        # Context IDs are never silently replaced with another context.
        assert api('POST', '/translate', json={
            'dictionary': {'a': 'Hello'}, 'language_code': 'en',
            'context_id': 'does-not-exist',
        }).status_code == 422
        assert api('POST', '/translate', json={
            'dictionary': {'a': 'Hello'}, 'language_code': 'en',
            'context_id': 'electronics', 'context_version': 404,
        }).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_legacy_translation_without_database_still_works():
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, localizer_database_url='', localizer_require_api_key=False,
    )
    try:
        old = api('POST', '/translate', json={
            'dictionary': {'hello': 'Hello'}, 'language_code': 'en',
        })
        assert old.status_code == 200
        assert old.json()['dictionary']['hello'] == 'Hello'
        assert api('GET', '/contexts').status_code == 503
        assert api('POST', '/translate', json={
            'dictionary': {'hello': 'Hello'}, 'language_code': 'en',
            'context_id': 'electronics',
        }).status_code == 503
        assert api('POST', '/translate', json={
            'dictionary': {'hello': 'Hello'}, 'language_code': 'en',
            'context_version': 1,
        }).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_alembic_migration_creates_catalog_tables(tmp_path, monkeypatch):
    """Exercise the shipped migration rather than relying only on create_all()."""
    from alembic import command
    from alembic.config import Config

    url = f'sqlite:///{tmp_path / "migrated.db"}'
    monkeypatch.setenv('LOCALIZER_DATABASE_URL', url)
    monkeypatch.delenv('LOCALIZER_DATABASE_PASSWORD', raising=False)
    get_settings.cache_clear()
    try:
        config = Config(str(Path(__file__).resolve().parents[1] / 'alembic.ini'))
        command.upgrade(config, 'head')
        engine = create_engine(url)
        with Session(engine) as session, session.begin():
            assert seed(session) == 6
        assert len(list_public_contexts(url)) == 6
        engine.dispose()
    finally:
        get_settings.cache_clear()
