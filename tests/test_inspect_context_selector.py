"""Offline tests for context-aware selector using A's frozen real-format output."""

import asyncio
import json

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.contexts.models import Base
from app.contexts.repository import resolve_context
from scripts.inspect_context_selector import (
    SYSTEM_PROMPT,
    extended_cache_path,
    load_extended_cache,
    generate_missing_controls,
    REVIEW_TARGETS,
    read_candidates,
    parse_selection,
    selection_context,
    select_one,
    user_prompt,
    render_markdown,
)
from scripts.inspect_english_senses import CASES, EXTENDED
from scripts.publish_professional_profiles import publish as publish_v2
from scripts.publish_semantic_profiles import publish as publish_v3
from scripts.seed_contexts import seed


@pytest.fixture
def profiles_db(tmp_path):
    url = f'sqlite:///{tmp_path / "contexts.db"}'
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        assert seed(session) == 6
        assert publish_v2(session) == 6
        assert publish_v3(session) == 6
    yield url
    engine.dispose()


def fake_candidates():
    return [
        {
            'case_id': case_id, 'source': source,
            'evaluation_context_not_sent': context_id,
            'http_status': 200, 'error': None, 'actual_model': 'gpt-6-luna',
            'lexical_inventory': [],
            'interpretations': [
                {'meaning': 'One possible English sentence interpretation.',
                 'kind': 'ordinary', 'evidence': 'A plausible generic reading.'},
                {'meaning': 'A separate specialist English interpretation.',
                 'kind': 'technical', 'evidence': 'An established specialist sense.'},
            ],
        }
        for case_id, source, context_id in CASES
    ]


def test_read_existing_lexical_report_without_rewriting_candidates(tmp_path):
    path = tmp_path / 'lexical.json'
    path.write_text(json.dumps(fake_candidates()), encoding='utf-8')
    cases = read_candidates(path)
    assert len(cases) == 6
    assert cases[0]['source'] == 'The bus failed.'
    assert cases[0]['interpretations'][1]['meaning'].startswith('A separate')
    assert cases[0]['context_id'] == 'electronics'
    assert cases[0]['generator_model'] == 'gpt-6-luna'

    incomplete = fake_candidates()[:-1]
    path.write_text(json.dumps(incomplete), encoding='utf-8')
    with pytest.raises(ValueError, match='Missing original'):
        read_candidates(path)

    duplicated = fake_candidates() + [fake_candidates()[0]]
    path.write_text(json.dumps(duplicated), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate case'):
        read_candidates(path)

    injected = fake_candidates()
    injected[0]['source'] = 'Unexpected modified input'
    path.write_text(json.dumps(injected), encoding='utf-8')
    with pytest.raises(ValueError, match='mismatch'):
        read_candidates(path)


def test_pinned_postgres_context_is_inherited_and_semantics_only(profiles_db):
    electronics = resolve_context(profiles_db, 'electronics', 3)
    payload = selection_context(electronics)
    assert payload['context_id'] == 'electronics'
    assert payload['context_version'] == 3
    assert len(payload['profiles_general_to_specific']) == 2
    assert 'engineering' in payload['profiles_general_to_specific'][-1]['specialization']
    assert payload['conditional_semantic_rules']
    assert 'electrical' in payload['conditional_semantic_rules'][0]['applies_when']
    json_payload = json.dumps(payload)
    assert 'preferred' not in json_payload
    assert 'проводник' not in json_payload
    assert 'target_language' not in json_payload


def test_selector_payload_separates_evidence_and_context_and_hides_target(profiles_db):
    context = resolve_context(profiles_db, 'music', 3)
    source = 'The conductor checked the tickets.'
    case = {
        'case_id': 'tickets',
        'source': source,
        'context_id': 'music',
        'generator_model': 'gpt-6-luna',
        'lexical_inventory': [],
        'interpretations': [
            {'meaning': 'A transport employee inspected passenger tickets.',
             'kind': 'ordinary', 'evidence': 'Ticket checking is usual in transport.'},
            {'meaning': 'An orchestra director looked at event tickets.',
             'kind': 'other', 'evidence': 'It is theoretically possible.'},
        ],
    }
    prompt = user_prompt(case, context)
    body = json.loads(prompt.partition('Input JSON data: ')[2])
    assert body['source_english'] == source
    assert body['candidate_interpretations'][0]['number'] == 1
    assert body['candidate_interpretations'][1]['number'] == 2
    assert body['professional_website_context']['context_id'] == 'music'
    assert body['professional_website_context']['context_version'] == 3
    assert 'music theory' in json.dumps(body).lower()
    assert 'railway' not in json.dumps(body['professional_website_context']).lower()
    assert 'transport employee' in prompt
    assert REVIEW_TARGETS['tickets'] not in prompt
    assert 'select the best candidate based' in SYSTEM_PROMPT.casefold()
    assert 'do not invent an unusual scenario' in SYSTEM_PROMPT.casefold()
    assert 'translate' in SYSTEM_PROMPT.lower()


def test_selection_validation_and_bounds():
    valid = parse_selection(json.dumps({
        'status': 'selected', 'choice': 2, 'basis': 'context_default',
        'reason': 'The engineering context favors a device meaning.',
    }), 2)
    assert valid.choice == 2
    unresolved = parse_selection(json.dumps({
        'status': 'unresolved', 'choice': None, 'basis': 'unresolved',
        'reason': 'Both available candidates remain plausible in the given domain.',
    }), 2)
    assert unresolved.choice is None
    none = parse_selection(json.dumps({
        'status': 'no_candidate', 'choice': None, 'basis': 'unresolved',
        'reason': 'No candidate faithfully describes the original English.',
    }), 2)
    assert none.status == 'no_candidate'

    for bad in (
        {'status': 'selected', 'choice': 3, 'basis': 'context_default',
         'reason': 'Some context evidence for this candidate.'},
        {'status': 'selected', 'choice': 0, 'basis': 'context_default',
         'reason': 'Some context evidence for this candidate.'},
        {'status': 'selected', 'choice': 1, 'basis': 'unresolved',
         'reason': 'The choice contradicts its declared basis.'},
        {'status': 'unresolved', 'choice': 1, 'basis': 'unresolved',
         'reason': 'Ambiguous result should have a null index.'},
        {'status': 'selected', 'choice': 1, 'basis': 'context_default',
         'reason': 'Valid except extraneous data.', 'translation': 'шина'},
    ):
        with pytest.raises((ValueError, ValidationError)):
            parse_selection(json.dumps(bad), 2)


def test_mock_live_selection_never_sends_expected_answer(profiles_db):
    context = resolve_context(profiles_db, 'electronics', 3)
    case = {
        'case_id': 'bus_failed', 'source': 'The bus failed.',
        'context_id': 'electronics', 'generator_model': 'gpt-6-luna',
        'lexical_inventory': [],
        'interpretations': [
            {'meaning': 'Passenger vehicle stopped functioning.', 'kind': 'ordinary',
             'evidence': 'An ordinary vehicle sense.'},
            {'meaning': 'Electronic communication bus stopped working.',
             'kind': 'technical', 'evidence': 'An engineering meaning of bus.'},
        ],
    }

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                'model': 'gpt-6-luna',
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'content': json.dumps({
                        'status': 'selected', 'choice': 2,
                        'basis': 'context_default',
                        'reason': 'The sentence alone is ambiguous; the site scope is electronics.',
                    })},
                }],
            }

    class FakeClient:
        sent = None

        async def post(self, url, headers, json):
            assert url.endswith('/chat/completions')
            assert headers['Authorization'] == 'Bearer test-key'
            self.sent = json
            return FakeResponse()

    client = FakeClient()
    settings = Settings(
        _env_file=None, openai_api_key='test-key',
        openai_model_id='gpt-6-luna',
    )
    row = asyncio.run(select_one(
        client, asyncio.Semaphore(1), settings, case, context,
    ))
    transmitted = ' '.join(m['content'] for m in client.sent['messages'])
    assert REVIEW_TARGETS['bus_failed'] not in transmitted
    assert 'electronics' in transmitted
    assert 'source_english' in transmitted
    assert 'Electronic communication bus' in transmitted
    assert 'Passenger vehicle' in transmitted
    assert row['selected_meaning'] == case['interpretations'][1]['meaning']
    assert row['selection']['choice'] == 2
    assert row['error'] is None
    assert 'Human review target (NOT sent)' in render_markdown([row])


def test_extended_control_set_keeps_adversarial_contexts():
    controls = dict((case[0], (case[1], case[2])) for case in EXTENDED)
    assert len(controls) == 4
    assert controls['orchestra_conductor'][1] == 'railway'
    assert controls['literal_note'][1] == 'music'
    assert controls['literal_bus'][1] == 'electronics'
    assert controls['railway_track'][1] == 'railway'


def test_extended_generator_schema_retry_and_cache_reuse(tmp_path, monkeypatch):
    from scripts import inspect_context_selector as selector

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    seen = []

    async def fake_gen(client, gate, settings, case, *, strict_format=False):
        case_id, source, context_id = case
        seen.append((case_id, strict_format))
        error = (
            'ValueError: Lexical span is not part of the input sentence'
            if case_id in ('literal_bus', 'railway_track') and not strict_format
            else None
        )
        return {
            'case_id': case_id, 'source': source,
            'evaluation_context_not_sent': context_id,
            'requested_model': 'gpt-6-luna', 'actual_model': 'gpt-6-luna',
            'http_status': 200, 'error': error,
            'lexical_inventory': [],
            'interpretations': [] if error else [{
                'meaning': 'A complete English interpretation of the sentence.',
                'kind': 'ordinary',
                'evidence': 'The source words directly support this meaning.',
            }],
        }

    monkeypatch.setattr(selector.httpx, 'AsyncClient', FakeAsyncClient)
    monkeypatch.setattr(selector, 'generate_lexical_candidates', fake_gen)
    model = Settings(_env_file=None, openai_api_key='test-only-key')
    source_path = tmp_path / 'frozen_a.json'
    source_path.write_text('test source fingerprint', encoding='utf-8')
    cache = extended_cache_path(source_path)
    assert cache.parent.name == source_path.parent.name or 'context_selector_candidates_' in cache.name
    base = read_candidates(_write_fake_frozen_candidates(tmp_path))
    result = asyncio.run(generate_missing_controls(base, model, cache))
    assert len(result) == 10
    assert len(seen) == 6  # four first attempts plus two failed-schema retries
    assert ('literal_bus', True) in seen and ('railway_track', True) in seen
    assert cache.is_file()
    cached, errors = load_extended_cache(cache, 'gpt-6-luna')
    assert set(cached) == {case[0] for case in EXTENDED}
    assert errors == {}
    again = asyncio.run(generate_missing_controls(base, model, cache))
    assert len(again) == 10 and len(seen) == 6  # zero new A requests


def _write_fake_frozen_candidates(tmp_path):
    path = tmp_path / 'original_lexical.json'
    path.write_text(json.dumps(fake_candidates()), encoding='utf-8')
    return path


def test_failed_extended_generation_preserves_partial_results(tmp_path, monkeypatch):
    from scripts import inspect_context_selector as selector

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    failed = {'railway_track'}
    seen = []

    async def fake_gen(client, gate, settings, case, *, strict_format=False):
        case_id, source, context_id = case
        seen.append(case_id)
        error = 'ValueError: Invalid model output' if case_id in failed else None
        return {
            'case_id': case_id, 'source': source,
            'evaluation_context_not_sent': context_id,
            'requested_model': 'gpt-6-luna', 'actual_model': 'gpt-6-luna',
            'http_status': 200, 'error': error,
            'lexical_inventory': [],
            'interpretations': [] if error else [{
                'meaning': 'A coherent interpretation supported by the words.',
                'kind': 'ordinary', 'evidence': 'Plain English sentence meaning.',
            }],
        }

    monkeypatch.setattr(selector.httpx, 'AsyncClient', FakeAsyncClient)
    monkeypatch.setattr(selector, 'generate_lexical_candidates', fake_gen)
    model = Settings(_env_file=None, openai_api_key='test-only-key')
    path = _write_fake_frozen_candidates(tmp_path)
    cache = extended_cache_path(path)
    base = read_candidates(path)
    with pytest.raises(ValueError, match='Successful candidates saved'):
        asyncio.run(generate_missing_controls(base, model, cache))
    cached, errors = load_extended_cache(cache, 'gpt-6-luna')
    assert set(cached) == {'orchestra_conductor', 'literal_note', 'literal_bus'}
    assert errors['railway_track'].startswith('ValueError')
    assert seen.count('railway_track') == 2
    failed.clear()
    result = asyncio.run(generate_missing_controls(base, model, cache))
    assert len(result) == 10
    assert seen.count('railway_track') == 3
    assert seen.count('literal_bus') == 1
    cached, errors = load_extended_cache(cache, 'gpt-6-luna')
    assert len(cached) == 4 and errors == {}
