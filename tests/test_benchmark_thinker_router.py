"""Offline contract and decision-path tests for Thinker/Python/Selector benchmark."""

import asyncio
import json

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.contexts.models import Base
from scripts.seed_contexts import seed
from scripts.publish_professional_profiles import publish as publish_v2
from scripts.publish_semantic_profiles import publish as publish_v3
from scripts.benchmark_thinker_router import (
    THINKER_PROMPT, SELECTOR_PROMPT,
    parse_thinker, parse_selector, route, thinker_input, selector_input,
    load_catalog, analyze_case, summary, report_markdown, run_experiment,
)


@pytest.fixture
def db_catalog(tmp_path):
    url = f'sqlite:///{tmp_path / "thinker-contexts.sqlite3"}'
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        assert seed(session) == 6
        assert publish_v2(session) == 6
        assert publish_v3(session) == 6
    settings = Settings(
        _env_file=None, localizer_database_url=url,
        openai_api_key='test-key', openai_model_id='gpt-6-luna',
    )
    yield settings, load_catalog(settings)
    engine.dispose()


def meanings():
    return [
        {
            'meaning': 'A passenger road vehicle stopped operating.',
            'source_fit': 'compatible',
            'source_evidence': 'Failed describes a vehicle no longer functioning.',
            'context_ids': ['railway'],
            'proposed_contexts': [],
        },
        {
            'meaning': 'A shared data communication bus stopped working.',
            'source_fit': 'compatible',
            'source_evidence': 'Failed can describe a data-path malfunction.',
            'context_ids': ['electronics', 'software'],
            'proposed_contexts': [],
        },
    ]


def thinker_json(senses=None):
    return json.dumps({'senses': meanings() if senses is None else senses})


def selector_json(choice=2, mappings=None):
    return json.dumps({
        'status': 'selected', 'choice': choice,
        'basis': 'context_tiebreak',
        'reason': 'Two readings fit English; the domain selects one.',
        'context_mappings': [] if mappings is None else mappings,
    })


def fake_case():
    return ('bus_failed', 'The bus failed.', 'electronics')


def test_thinker_receives_entire_catalog_but_no_selected_site_context(db_catalog):
    _, catalog = db_catalog
    assert len(catalog) == 6
    assert catalog['electronics']['version'] == 3
    assert len(catalog['electronics']['profiles_general_to_specific']) == 2
    data = json.loads(thinker_input('The bus failed.', catalog).split('Input JSON: ')[1])
    assert data['source_english'] == 'The bus failed.'
    assert sorted(data['available_contexts']) == sorted(catalog)
    assert 'site_context_id' not in data
    assert 'human_review_target' not in data
    assert 'translated' not in json.dumps(data).casefold()
    assert 'fruit salad' not in json.dumps(data).casefold()
    assert 'проводник' not in json.dumps(data).casefold()
    assert 'source_fit' in THINKER_PROMPT
    assert 'NOT told which professional context' in THINKER_PROMPT
    assert 'exact_existing_id' in THINKER_PROMPT
    assert 'nearest' in SELECTOR_PROMPT


def test_validate_known_contexts_and_new_proposals(db_catalog):
    _, catalog = db_catalog
    parsed = parse_thinker(thinker_json(), catalog)
    assert len(parsed.senses) == 2
    assert parsed.senses[1].context_ids == ['electronics', 'software']

    unknown = meanings()
    unknown[0]['context_ids'] = ['made_up_id']
    with pytest.raises(ValueError, match='invented context IDs'):
        parse_thinker(thinker_json(unknown), catalog)

    new_context = meanings()
    new_context[0]['context_ids'] = []
    new_context[0]['proposed_contexts'] = [{
        'name': 'Bus passenger transport',
        'description': 'Bus routes, fares, stops and passenger operations.',
        'reason': 'No listed domain addresses road buses specifically.',
    }]
    parsed = parse_thinker(thinker_json(new_context), catalog)
    assert parsed.senses[0].proposed_contexts[0].name == 'Bus passenger transport'

    invalid = {'senses': [dict(meanings()[0], source_fit='likely')]}
    with pytest.raises(ValidationError):
        parse_thinker(json.dumps(invalid), catalog)


def test_router_handles_direct_match_and_ambiguity_without_ai():
    s = meanings()
    outcome = route(s, 'electronics')
    assert outcome == {
        'call_selector': False, 'choice': 2,
        'reason': 'unique_catalog_context_match',
    }
    assert route(s, 'railway')['choice'] == 1
    assert route(s, 'music')['call_selector'] is True
    assert route([s[1]], 'music')['reason'] == 'single_compatible_meaning'

    decisive = meanings()
    decisive[0]['source_fit'] = 'supported'
    # The source-supported reading conflicts with a different domain match.
    assert route(decisive, 'electronics')['reason'] == 'source_and_context_conflict'
    assert route(decisive, 'music')['choice'] == 1
    decisive[1]['source_fit'] = 'supported'
    assert route(decisive, 'electronics')['reason'] == 'competing_supported_meanings'

    novel = meanings()
    novel[0]['proposed_contexts'] = [{
        'name': 'Road buses', 'description': 'Passenger bus services.',
        'reason': 'The catalog lacks dedicated urban transit.',
    }]
    assert route(novel, 'electronics')['reason'] == 'unmatched_proposed_context'
    contradicted = meanings()
    contradicted[0]['source_fit'] = 'contradicted'
    assert route(contradicted, 'electronics')['choice'] == 2
    contradicted[1]['source_fit'] = 'contradicted'
    assert route(contradicted, 'electronics')['call_selector'] is True


def test_selector_validates_nearest_contexts_and_rejects_unknown_ids(db_catalog):
    _, catalog = db_catalog
    names = {'Urban passenger transit'}
    mapping = [{
        'proposed_name': 'Urban passenger transit',
        'nearest_context_ids': ['railway'],
        'reason': 'Both domains cover ticketed passenger transportation.',
    }]
    obj = parse_selector(selector_json(2, mapping), 2, catalog, names)
    assert obj.context_mappings[0].nearest_context_ids == ['railway']
    with pytest.raises(ValueError, match='nonexistent candidate'):
        parse_selector(selector_json(3, mapping), 2, catalog, names)
    invalid = [dict(mapping[0], nearest_context_ids=['fake'])]
    with pytest.raises(ValueError, match='invented'):
        parse_selector(selector_json(2, invalid), 2, catalog, names)
    with pytest.raises(ValueError, match='unrequested'):
        parse_selector(selector_json(2, mapping), 2, catalog, set())
    with pytest.raises(ValidationError):
        parse_selector(json.dumps({
            'status': 'unresolved', 'choice': 1, 'basis': 'unresolved',
            'reason': 'There is no unambiguous reading in the source.',
            'context_mappings': [],
        }), 2, catalog, set())


class MockResponse:
    status_code = 200

    def __init__(self, model_response):
        self.model_response = model_response

    def raise_for_status(self):
        pass

    def json(self):
        return {
            'model': 'gpt-6-luna',
            'choices': [{
                'finish_reason': 'stop',
                'message': {'content': self.model_response},
            }],
        }


class MockClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def post(self, url, headers, json):
        assert url.endswith('/chat/completions')
        assert headers['Authorization'] == 'Bearer test-key'
        self.requests.append(json)
        return MockResponse(self.replies.pop(0))


def test_direct_routing_consumes_one_call_and_selector_avoided(db_catalog):
    settings, catalog = db_catalog
    client = MockClient([thinker_json()])
    row = asyncio.run(analyze_case(
        client, asyncio.Semaphore(1), settings, fake_case(), catalog, 1,
    ))
    assert len(client.requests) == 1
    assert row['final_status'] == 'selected'
    assert row['selected_index'] == 2
    assert row['routing']['reason'] == 'unique_catalog_context_match'
    assert row['selector_attempts'] == 0
    assert row['selected_meaning'] == meanings()[1]['meaning']
    assert row['stage_error'] is None

    sent = json.loads(client.requests[0]['messages'][1]['content'].split('Input JSON: ')[1])
    assert 'site_context_id' not in sent
    assert sent['source_english'] == 'The bus failed.'
    assert 'Electronics' in sent['available_contexts']['electronics']['title']
    assert 'NOT told' in THINKER_PROMPT


def test_escalation_invokes_selector_and_nearest_mapping(db_catalog):
    settings, catalog = db_catalog
    ambiguous = meanings()
    ambiguous[0]['proposed_contexts'] = [{
        'name': 'Urban passenger transit',
        'description': 'Local road bus routes and passenger movement.',
        'reason': 'Only railway appears in the published catalog.',
    }]
    mapping = [{
        'proposed_name': 'Urban passenger transit',
        'nearest_context_ids': ['railway'],
        'reason': 'The closest public transit context concerns railway passengers.',
    }]
    client = MockClient([thinker_json(ambiguous), selector_json(2, mapping)])
    row = asyncio.run(analyze_case(
        client, asyncio.Semaphore(1), settings, fake_case(), catalog, 1,
    ))
    assert len(client.requests) == 2
    assert row['routing']['reason'] == 'unmatched_proposed_context'
    assert row['selector_attempts'] == 1
    assert row['selector_result']['context_mappings'] == mapping
    assert row['selected_index'] == 2
    selector_body = json.loads(client.requests[1]['messages'][1]['content'].split('Input JSON: ')[1])
    assert selector_body['site_context_id'] == 'electronics'
    assert len(selector_body['thinker_senses']) == 2
    assert selector_body['available_contexts']['railway']['version'] == 3
    assert 'Human review target' not in str(selector_body)
    report = report_markdown([row], catalog)
    assert 'Python selected: 0' not in report  # summary uses uppercase for Python selected
    assert 'Selector invoked:** 1' in report
    assert 'Urban passenger transit' in report


def test_recoverable_bad_json_is_retried_and_reported(db_catalog):
    settings, catalog = db_catalog
    client = MockClient(['not JSON', thinker_json()])
    row = asyncio.run(analyze_case(
        client, asyncio.Semaphore(1), settings, fake_case(), catalog, 1,
    ))
    assert len(client.requests) == 2
    assert row['thinker_attempts'] == 2
    assert row['selected_index'] == 2
    assert summary([row])['actual_openai_calls'] == 2
    bad = MockClient(['not JSON', 'not JSON'])
    error_row = asyncio.run(analyze_case(
        bad, asyncio.Semaphore(1), settings, fake_case(), catalog, 1,
    ))
    assert error_row['stage_error'].startswith('thinker:')
    assert error_row['final_status'] == 'error'
    assert summary([error_row])['errors'] == 1


def test_incremental_results_saved_even_when_later_case_fails(db_catalog,
                                                              tmp_path,
                                                              monkeypatch):
    from scripts import benchmark_thinker_router as module
    settings, catalog = db_catalog

    class ContextClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr(module.httpx, 'AsyncClient',
                        lambda **_: ContextClient())

    calls = []

    async def fake_analyze(client, gate, settings, case, cat, trial):
        calls.append(case[0])
        return {
            'case_id': case[0], 'trial': trial, 'source': case[1],
            'site_context_id': case[2],
            'site_context_version': 3,
            'thinker_model': 'gpt-6-luna',
            'thinker_attempts': 1, 'senses': [],
            'routing': None, 'selector_model': None, 'selector_attempts': 0,
            'selector_result': None, 'final_status': 'error',
            'selected_index': None, 'selected_meaning': None,
            'stage_error': 'test error',
            'elapsed_ms': 4,
            'human_review_target_not_sent': 'test target',
        }

    monkeypatch.setattr(module, 'analyze_case', fake_analyze)
    path_json = tmp_path / 'partial.json'
    path_md = tmp_path / 'partial.md'
    results = asyncio.run(run_experiment(
        settings, [fake_case()], catalog, 2, path_json, path_md,
    ))
    assert len(calls) == 2
    assert len(results) == 2
    assert len(json.loads(path_json.read_text())) == 2
    assert 'test error' in path_md.read_text()
    assert summary(results)['errors'] == 2
