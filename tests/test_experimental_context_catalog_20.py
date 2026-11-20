"""Offline tests for the isolated 20-context Thinker experiment."""

import asyncio
import json

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.contexts.models import Base, Context, ContextVersion
from scripts.seed_contexts import seed
from scripts.publish_professional_profiles import publish as publish_v2
from scripts.publish_semantic_profiles import publish as publish_v3
from scripts.experimental_context_catalog_20 import (
    ADDITIONAL_CONTEXTS, ORIGINAL_CONTEXT_IDS, expand_to_20,
)
from scripts.benchmark_thinker_router import (
    THINKER_PROMPT, EXPANDED20_THINKER_RULES,
    analyze_case, expanded_catalog, load_catalog, parse_thinker, route,
    summary, thinker_input, thinker_prompt_for_catalog,
)


@pytest.fixture
def catalog_pair(tmp_path):
    url = f'sqlite:///{tmp_path / "experiment-catalog.db"}'
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        assert seed(session) == 6
        assert publish_v2(session) == 6
        assert publish_v3(session) == 6
    settings = Settings(
        _env_file=None,
        localizer_database_url=url,
        openai_api_key='test-key',
        openai_model_id='gpt-6-luna',
    )
    published = load_catalog(settings)
    yield settings, published, engine
    engine.dispose()


def test_twenty_contexts_are_exactly_six_published_plus_fourteen_temporary(catalog_pair):
    _, published, engine = catalog_pair
    expanded = expand_to_20(published)
    assert len(published) == 6
    assert len(ADDITIONAL_CONTEXTS) == 14
    assert set(published) == ORIGINAL_CONTEXT_IDS
    assert set(expanded) == set(published) | set(ADDITIONAL_CONTEXTS)
    assert len(expanded) == 20
    assert expanded['electronics'] == published['electronics']
    assert expanded['electronics']['version'] == 3
    assert expanded['road_transport']['version'] == 'experiment-only'
    assert expanded['road_transport']['experiment_only'] is True
    assert all(x['profiles_general_to_specific']
               for x in expanded.values())
    assert 'road_transport' not in published

    # A genuine DB check: expansion has NOT published any context or version.
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Context)) == 6
        assert session.scalar(select(func.count()).select_from(ContextVersion)) == 18
    assert set(load_catalog(catalog_pair[0])) == ORIGINAL_CONTEXT_IDS


def test_read_only_expansion_fails_closed_when_real_catalog_has_new_ids(catalog_pair):
    _, published, _ = catalog_pair
    with pytest.raises(ValueError, match='expects six published'):
        expand_to_20(dict(published, road_transport={'title': 'DB context'}))
    with pytest.raises(ValueError, match='expects six published'):
        expand_to_20({k: v for k, v in published.items() if k != 'music'})


def test_thinker_receives_twenty_domain_definitions_but_not_target(catalog_pair):
    _, published, _ = catalog_pair
    expanded = expand_to_20(published)
    parsed = json.loads(thinker_input(
        'The note was missing.', expanded,
    ).partition('Input JSON: ')[2])
    assert len(parsed['available_contexts']) == 20
    assert parsed['available_contexts']['office_administration']['version'] == 'experiment-only'
    assert parsed['available_contexts']['finance']['profiles_general_to_specific']
    assert 'site_context_id' not in parsed
    assert 'human_review_target' not in parsed
    assert thinker_prompt_for_catalog(published) == THINKER_PROMPT
    instruction = thinker_prompt_for_catalog(expanded)
    assert instruction == THINKER_PROMPT + EXPANDED20_THINKER_RULES
    assert 'Only propose a new context if NONE' in instruction
    assert 'never use a vague sentence' in instruction
    assert 'source_fit' in instruction


def _ambiguous_senses():
    return [
        {
            'meaning': 'A passenger vehicle broke down during service.',
            'source_fit': 'compatible',
            'source_evidence': 'The phrase could describe a road vehicle failure.',
            'context_ids': ['road_transport'],
            'proposed_contexts': [],
        },
        {
            'meaning': 'A shared data interconnect ceased functioning.',
            'source_fit': 'compatible',
            'source_evidence': 'The same phrase could describe a hardware interconnect.',
            'context_ids': ['electronics', 'computing_networks'],
            'proposed_contexts': [],
        },
    ]


def test_strict_existing_first_validation_only_applies_to_expanded_trial(catalog_pair):
    _, published, _ = catalog_pair
    expanded = expand_to_20(published)
    senses = _ambiguous_senses()
    record = json.dumps({'senses': senses})
    assert len(parse_thinker(record, expanded).senses) == 2
    assert route(senses, 'electronics')['choice'] == 2
    assert route(senses, 'electronics')['call_selector'] is False
    assert route(senses, 'road_transport')['choice'] == 1

    mixed = _ambiguous_senses()
    mixed[1]['proposed_contexts'] = [{
        'name': 'Communications devices',
        'description': 'Equipment connected by digital interconnects.',
        'reason': 'A much narrower topic than the catalog has provided.',
    }]
    with pytest.raises(ValueError, match='known matching context excludes proposals'):
        parse_thinker(json.dumps({'senses': mixed}), expanded)

    # The old six-context baseline remains untouched, including its
    # earlier permissive schema; we are varying only the experiment mode.
    original = _ambiguous_senses()
    original[0]['context_ids'] = []
    original[0]['proposed_contexts'] = mixed[1]['proposed_contexts']
    original[1]['context_ids'] = ['electronics']
    original[1]['proposed_contexts'] = mixed[1]['proposed_contexts']
    assert len(parse_thinker(json.dumps({'senses': original}), published).senses) == 2

    unknown = _ambiguous_senses()
    unknown[0]['context_ids'] = ['invented-transport-id']
    with pytest.raises(ValueError, match='invented context IDs'):
        parse_thinker(json.dumps({'senses': unknown}), expanded)


def test_known_context_avoids_selector_and_counts_proposals(catalog_pair):
    settings, published, _ = catalog_pair
    catalog = expand_to_20(published)
    model_response = json.dumps({'senses': _ambiguous_senses()})

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                'model': 'gpt-6-luna',
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'content': model_response},
                }],
            }

    class FakeClient:
        def __init__(self):
            self.requests = []

        async def post(self, url, headers, json):
            self.requests.append(json)
            return FakeResponse()

    client = FakeClient()
    row = asyncio.run(analyze_case(
        client, asyncio.Semaphore(1), settings,
        ('bus_failed', 'The bus failed.', 'electronics'),
        catalog, 1,
    ))
    assert len(client.requests) == 1
    assert row['catalog_mode'] == 'expanded20'
    assert row['catalog_size'] == 20
    assert row['final_status'] == 'selected'
    assert row['selected_index'] == 2
    assert row['routing']['reason'] == 'unique_catalog_context_match'
    assert row['selector_attempts'] == 0
    stats = summary([row])
    assert stats['python_selected'] == 1
    assert stats['new_context_proposals'] == 0
    assert stats['cases_with_new_contexts'] == 0
    sent = client.requests[0]
    assert 'Only propose a new context if NONE' in sent['messages'][0]['content']
    user_data = json.loads(sent['messages'][1]['content'].partition('Input JSON: ')[2])
    assert len(user_data['available_contexts']) == 20
    assert 'site_context_id' not in user_data
