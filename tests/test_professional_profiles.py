"""Professional domain profiles: pinned v2 snapshots and multilingual website UI prompts."""

import asyncio
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.contexts.models import Base, Context, ContextVersion
from app.contexts.repository import (
    build_context_prompt,
    list_public_contexts,
    resolve_context,
)
from app.main import app
from scripts.publish_professional_profiles import publish
from scripts.seed_contexts import seed


@pytest.fixture
def profile_db(tmp_path):
    url = f'sqlite:///{tmp_path / "profiles.db"}'
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        assert seed(session) == 6
        assert publish(session) == 6
    yield url
    engine.dispose()


def test_v2_publishes_all_six_and_preserves_v1(profile_db):
    catalog = list_public_contexts(profile_db)
    assert len(catalog) == 6
    assert {item['version'] for item in catalog} == {2}

    v1 = resolve_context(profile_db, 'electronics', 1)
    v2 = resolve_context(profile_db, 'electronics')
    assert v1.version == 1
    assert v1.profiles == ()
    assert v1.style['tone'] == 'precise and professional'
    assert len(v1.rules) == 1 and v1.rules[0].preferred == 'проводник'

    assert v2.version == 2
    assert len(v2.profiles) == 2
    assert 'technology' in v2.profiles[0].specialization
    assert 'electronics' in v2.profiles[1].specialization
    assert v2.style == v1.style
    assert v2.rules == v1.rules
    with Session(create_engine(profile_db)) as session:
        snapshot = session.get(ContextVersion, ('electronics', 2))
        assert snapshot.parent_context_id == 'technology'
        assert snapshot.parent_version == 2
        assert session.get(ContextVersion, ('electronics', 1)).parent_version == 1


def test_specialist_role_defaults_ambiguous_ui_labels_to_domain(profile_db):
    v2 = resolve_context(profile_db, 'electronics')
    german = build_context_prompt(v2, 'de')
    hebrew = build_context_prompt(v2, 'he')
    russian = build_context_prompt(v2, 'ru')
    for prompt in (german, hebrew, russian):
        assert 'professional domain localization assignment' in prompt.lower()
        assert 'Act as an expert website/UI localization translator' in prompt
        assert 'ambiguous or short standalone labels' in prompt
        assert 'electronics and electrical-engineering' in prompt
        assert 'technical systems' in prompt
        assert 'CLEARLY signals a different sense' in prompt
        assert 'Do not invent facts' in prompt
    assert 'electrical-conductor' not in german
    assert 'электроник' not in german
    assert 'electrical-conductor' not in hebrew
    assert 'electrical-conductor' in russian
    assert 'проводник' in russian

    # A pinned v1 request is not upgraded silently to the new specialist role.
    legacy = build_context_prompt(resolve_context(profile_db, 'electronics', 1), 'ru')
    assert 'professional domain localization assignment' not in legacy.lower()
    assert 'Subject-domain context' in legacy


def test_specialists_remain_language_neutral(profile_db):
    for context_id in ('technology', 'electronics', 'software',
                       'railway', 'music', 'military'):
        profile = resolve_context(profile_db, context_id)
        assert profile.version == 2
        assert profile.profiles
        assert 'Russian' not in profile.profiles[-1].specialization
        for target in ('de', 'fr', 'he', 'ja'):
            prompt = build_context_prompt(profile, target)
            assert profile.profiles[-1].specialization in prompt
            assert '"rules":[]' in prompt


def test_idempotent_publish_and_reject_unexpected_snapshots(profile_db):
    engine = create_engine(profile_db)
    with Session(engine) as session, session.begin():
        assert publish(session) == 0
    with Session(engine) as session, session.begin():
        context = session.get(Context, 'music')
        context.published_version = 3
    with Session(engine) as session:
        with pytest.raises(ValueError, match='unknown published version'):
            publish(session)
    engine.dispose()


def test_v2_api_uses_role_by_default_and_v1_can_be_pinned(profile_db, monkeypatch):
    from app.api import routes

    seen = []

    async def fake_translate(values, language, settings, context_prompt=None):
        seen.append((values, language, context_prompt))
        return ['ok']

    monkeypatch.setattr(routes, 'translate_values', fake_translate)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, localizer_database_url=profile_db,
        localizer_require_api_key=False,
    )

    async def call(payload):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            return await client.post('/translate', json=payload)
    try:
        for target in ('ru', 'de', 'he'):
            req = {
                'dictionary': {'label': 'The bus failed.'},
                'language_code': target,
                'context_id': 'electronics',
            }
            v2 = asyncio.run(call(req))
            assert v2.status_code == 200, v2.text
            assert v2.json()['context_version'] == 2
            assert seen[-1][2] and 'professional domain localization assignment' in seen[-1][2].lower()

            v1 = asyncio.run(call({**req, 'context_version': 1}))
            assert v1.status_code == 200, v1.text
            assert v1.json()['context_version'] == 1
            assert 'professional domain localization assignment' not in seen[-1][2].lower()

        no_context = asyncio.run(call({
            'dictionary': {'label': 'The bus failed.'}, 'language_code': 'de',
        }))
        assert no_context.status_code == 200, no_context.text
        assert no_context.json()['context_id'] is None
        assert seen[-1][2] is None
    finally:
        app.dependency_overrides.clear()


def test_actual_alembic_upgrade_preserves_pinned_v1_and_adds_v2(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config

    url = f'sqlite:///{tmp_path / "migration.db"}'
    monkeypatch.setenv('LOCALIZER_DATABASE_URL', url)
    monkeypatch.delenv('LOCALIZER_DATABASE_PASSWORD', raising=False)
    get_settings.cache_clear()
    config = Config(str(Path(__file__).resolve().parents[1] / 'alembic.ini'))
    try:
        # Mimic an existing Localizer 2.0 installation before the profile upgrade.
        command.upgrade(config, '0001_contexts')
        old_engine = create_engine(url)
        with Session(old_engine) as session, session.begin():
            seed(session)
        old_engine.dispose()

        command.upgrade(config, 'head')
        engine = create_engine(url)
        try:
            assert 'domain_profile' in {
                column['name'] for column in inspect(engine).get_columns('context_versions')
            }
            with Session(engine) as session, session.begin():
                assert publish(session) == 6
            assert resolve_context(url, 'military', 1).profiles == ()
            assert 'military' in resolve_context(url, 'military', 2).profiles[-1].specialization
        finally:
            engine.dispose()
    finally:
        get_settings.cache_clear()
