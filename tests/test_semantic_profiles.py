"""V3 semantic profiles: pinned history, role selection, API, and safe publication."""

import asyncio

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.constants import (
    SEMANTIC_ASSIGNMENT_PREFIX,
    SEMANTIC_SYSTEM_PROMPT,
    SPECIALIST_ASSIGNMENT_PREFIX,
    SPECIALIST_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from app.contexts.models import Base, Context, ContextVersion
from app.contexts.repository import build_context_prompt, list_public_contexts, resolve_context
from app.main import app
from app.quality.controller import _build_review_prompt
from scripts.publish_professional_profiles import publish as publish_v2
from scripts.publish_semantic_profiles import publish as publish_v3
from scripts.seed_contexts import seed


@pytest.fixture
def semantic_db(tmp_path):
    url = f'sqlite:///{tmp_path / "semantic.db"}'
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        assert seed(session) == 6
        assert publish_v2(session) == 6
        assert publish_v3(session) == 6
    yield url
    engine.dispose()


def test_v3_preserves_v1_v2_exact_prompt_and_pins_v3_parent(semantic_db):
    for name in ('technology', 'electronics', 'software', 'railway', 'music', 'military'):
        v1, v2, v3 = [resolve_context(semantic_db, name, n) for n in (1, 2, 3)]
        assert v1.version == 1 and not v1.profiles
        assert v2.version == 2 and all(p.interpretation_method is None for p in v2.profiles)
        assert v3.version == 3
        assert v3.profiles[-1].interpretation_method == 'evidence_before_domain'
        assert v3.rules == v2.rules == v1.rules
        assert v3.glossary == v2.glossary == v1.glossary
        assert v3.style == v2.style == v1.style
        assert [p.specialization for p in v3.profiles] == [
            p.specialization for p in v2.profiles
        ]
        for target in ('ru', 'de', 'he'):
            old_v1 = build_context_prompt(v1, target)
            old_v2 = build_context_prompt(v2, target)
            new_v3 = build_context_prompt(v3, target)
            assert old_v1.startswith('\nSubject-domain context')
            assert old_v2.startswith(SPECIALIST_ASSIGNMENT_PREFIX)
            assert old_v2.count('interpretation_method') == 0
            assert new_v3.startswith(SEMANTIC_ASSIGNMENT_PREFIX)
            assert '"interpretation_method":"evidence_before_domain"' in new_v3
            assert 'generic states or verbs compatible with both senses' in new_v3
            assert 'honor that meaning EVEN IF it is outside' in new_v3
            assert 'REQUESTED TARGET LANGUAGE' in new_v3

    with Session(create_engine(semantic_db)) as session:
        for child in ('electronics', 'software'):
            assert session.get(ContextVersion, (child, 2)).parent_version == 2
            assert session.get(ContextVersion, (child, 3)).parent_version == 3
        assert session.get(ContextVersion, ('music', 2)).domain_profile.get(
            'interpretation_method'
        ) is None
        assert session.get(ContextVersion, ('music', 3)).domain_profile[
            'interpretation_method'
        ] == 'evidence_before_domain'
        assert {row['version'] for row in list_public_contexts(semantic_db)} == {3}


def test_v3_does_not_add_per_word_bus_or_note_rules(semantic_db):
    electronics = resolve_context(semantic_db, 'electronics', 3)
    music = resolve_context(semantic_db, 'music', 3)
    assert not any(item.term.casefold() == 'bus' for item in electronics.rules)
    assert not any(item.term.casefold() == 'note' for item in music.rules)
    assert not any(item.term.casefold() in ('bus', 'note') for item in electronics.glossary)


def test_v3_translator_dispatches_to_new_system_role_only(semantic_db, monkeypatch):
    from app.services import openai

    sent = []

    async def fake_request(**kwargs):
        sent.append((kwargs['system_prompt'], kwargs['prompt_builder'](False)))
        return ['translated']

    monkeypatch.setattr(openai, '_request_with_retries', fake_request)
    settings = Settings(_env_file=None)
    for version in (1, 2, 3):
        context = resolve_context(semantic_db, 'electronics', version)
        result = asyncio.run(openai.translate_values(
            ['The bus failed.'], 'Deutsch', settings,
            context_prompt=build_context_prompt(context, 'de'),
        ))
        assert result == ['translated']

    result = asyncio.run(openai.translate_values(
        ['The bus failed.'], 'Deutsch', settings,
    ))
    assert result == ['translated']
    assert [prompt[0] for prompt in sent] == [
        SYSTEM_PROMPT, SPECIALIST_SYSTEM_PROMPT, SEMANTIC_SYSTEM_PROMPT, SYSTEM_PROMPT,
    ]
    assert sent[2][1].count('REQUESTED TARGET LANGUAGE') == 1


def test_v3_controller_uses_matching_semantic_policy(semantic_db):
    v3 = resolve_context(semantic_db, 'music', 3)
    v2 = resolve_context(semantic_db, 'music', 2)
    expected = 'A general predicate compatible with both meanings'
    assert expected in _build_review_prompt(
        ['The note was missing.'], ['Нота отсутствовала.'], 'ru', v3,
    )
    assert expected not in _build_review_prompt(
        ['The note was missing.'], ['Нота отсутствовала.'], 'ru', v2,
    )


def test_v3_idempotent_refuses_changed_snapshot_and_unexpected_version(semantic_db):
    engine = create_engine(semantic_db)
    with Session(engine) as session, session.begin():
        assert publish_v3(session) == 0
    with Session(engine) as session, session.begin():
        context = session.get(Context, 'music')
        context.published_version = 4
    with Session(engine) as session:
        with pytest.raises(ValueError, match='unexpected published version'):
            publish_v3(session)
    engine.dispose()


def test_v3_api_defaults_latest_and_can_pin_both_old_versions(semantic_db, monkeypatch):
    from app.api import routes

    sent = []

    async def fake_translate(values, language, settings, context_prompt=None):
        sent.append((language, context_prompt))
        return ['out']

    monkeypatch.setattr(routes, 'translate_values', fake_translate)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, localizer_database_url=semantic_db,
        localizer_require_api_key=False,
    )

    async def post(body):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test',
        ) as client:
            return await client.post('/translate', json=body)

    try:
        for target in ('ru', 'de', 'he'):
            payload = {
                'dictionary': {'key': 'The bus failed.'},
                'language_code': target,
                'context_id': 'electronics',
            }
            for version, expected in (
                (1, 'Subject-domain context'),
                (2, SPECIALIST_ASSIGNMENT_PREFIX),
                (3, SEMANTIC_ASSIGNMENT_PREFIX),
            ):
                body = {**payload, 'context_version': version}
                res = asyncio.run(post(body))
                assert res.status_code == 200, res.text
                assert res.json()['context_version'] == version
                assert expected in sent[-1][1]

            latest = asyncio.run(post(payload))
            assert latest.status_code == 200, latest.text
            assert latest.json()['context_version'] == 3
            assert sent[-1][1].startswith(SEMANTIC_ASSIGNMENT_PREFIX)

        plain = asyncio.run(post({
            'dictionary': {'key': 'Hello'}, 'language_code': 'de',
        }))
        assert plain.status_code == 200
        assert plain.json()['context_version'] is None
        assert sent[-1][1] is None
    finally:
        app.dependency_overrides.clear()


def test_v3_publish_refuses_missing_v2_and_keeps_transaction_clean(tmp_path):
    url = f'sqlite:///{tmp_path / "no_v2.db"}'
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        seed(session)
    with Session(engine) as session:
        with pytest.raises(ValueError, match='Published v2 context is required'):
            with session.begin():
                publish_v3(session)
    with Session(engine) as session:
        assert session.get(ContextVersion, ('technology', 3)) is None
        assert session.get(Context, 'technology').published_version == 1
    engine.dispose()
