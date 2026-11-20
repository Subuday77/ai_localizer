"""Tests for frozen score gates, 37 withheld cases, and offline model paths."""

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
from scripts.experimental_context_catalog_20 import expand_to_20
from scripts.benchmark_thinker_router import load_catalog
from scripts.benchmark_score_cases import (
    FRESH_CASES, USER_CASES, AUTHOR_CASES, REGRESSION_CASES,
    EXTERNAL_SITE_CONTEXTS,
)
from scripts.benchmark_scored_router import (
    THRESHOLDS, THINKER_SCORE_PROMPT, SELECTOR_SCORE_PROMPT,
    parse_scored_thinker, thinker_payload, selector_payload, route_scores,
    summarize, render_markdown, run_case,
)


@pytest.fixture
def catalog_fixture(tmp_path):
    url = 'sqlite:///' + str(tmp_path / 'scores.sqlite')
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        assert seed(session) == 6
        assert publish_v2(session) == 6
        assert publish_v3(session) == 6
    settings = Settings(
        _env_file=None,
        openai_api_key='test-key',
        openai_model_id='gpt-6-luna',
        localizer_database_url=url,
    )
    original = load_catalog(settings)
    catalog = expand_to_20(original)
    yield settings, catalog, engine
    engine.dispose()


def senses(source1=65, source2=65, ctx1=95, ctx2=95):
    return [
        {
            'meaning': 'A passenger road vehicle has stopped operating.',
            'term_senses': [{
                'span': 'bus', 'sense': 'Passenger road vehicle',
            }],
            'source_score': source1,
            'source_evidence': 'The sentence could describe a failed road vehicle.',
            'context_scores': {'road_transport': ctx1},
            'proposed_contexts': [],
        },
        {
            'meaning': 'A data transfer interconnect has stopped working.',
            'term_senses': [{
                'span': 'bus', 'sense': 'Computer data pathway',
            }],
            'source_score': source2,
            'source_evidence': 'The sentence could describe a failed communication bus.',
            'context_scores': {'electronics': ctx2},
            'proposed_contexts': [],
        },
    ]


def test_holdout_freeze_and_no_answers_in_model_inputs(catalog_fixture):
    _, catalog, _ = catalog_fixture
    assert len(FRESH_CASES) == 37
    assert len(USER_CASES) == 7
    assert len(AUTHOR_CASES) == 30
    assert len(REGRESSION_CASES) == 10
    assert len(catalog) == 20
    assert FRESH_CASES[0].source == 'Drop the base'
    assert [x.source for x in USER_CASES] == [
        'Drop the base', 'Blockbuster', 'Moon shine',
        'Honey badger', 'Heaving down', 'pitching', 'go to a bear',
    ]
    assert USER_CASES[-1].review_status == 'qualitative'
    assert USER_CASES[4].site_context_id == 'maritime'
    assert USER_CASES[5].site_context_id == 'maritime'
    assert all(x.site_context_id in catalog or
               x.site_context_id in EXTERNAL_SITE_CONTEXTS
               for x in FRESH_CASES + REGRESSION_CASES)
    assert len({x.case_id for x in FRESH_CASES + REGRESSION_CASES}) == 47
    for case in FRESH_CASES:
        prompt = thinker_payload(case.source, catalog)
        body = json.loads(prompt.partition('Input JSON: ')[2])
        assert body['english_source'] == case.source
        assert len(body['available_contexts']) == 20
        assert 'site_context_id' not in body
        assert 'review_target' not in body
        assert case.review_target not in prompt
    assert '0-29' in THINKER_SCORE_PROMPT
    assert '55-74' in THINKER_SCORE_PROMPT
    assert '90-100' in THINKER_SCORE_PROMPT
    assert 'scores' in SELECTOR_SCORE_PROMPT
    assert THRESHOLDS.strong_source == 90


def test_score_schema_rejects_unknown_context_range_and_missing_span(catalog_fixture):
    _, catalog, _ = catalog_fixture
    source = 'The bus failed.'
    parsed = parse_scored_thinker(json.dumps({'senses': senses()}), source, catalog)
    assert len(parsed.senses) == 2
    assert parsed.senses[1].context_scores['electronics'] == 95

    changed = senses()
    changed[0]['source_score'] = 101
    with pytest.raises(ValueError):
        parse_scored_thinker(json.dumps({'senses': changed}), source, catalog)

    changed = senses()
    changed[0]['source_score'] = True
    with pytest.raises(ValueError):
        parse_scored_thinker(json.dumps({'senses': changed}), source, catalog)

    changed = senses()
    changed[1]['context_scores'] = {'nonexistent': 90}
    with pytest.raises(ValueError, match='invented'):
        parse_scored_thinker(json.dumps({'senses': changed}), source, catalog)

    changed = senses()
    changed[1]['term_senses'][0]['span'] = 'computer'
    with pytest.raises(ValueError, match='not in original'):
        parse_scored_thinker(json.dumps({'senses': changed}), source, catalog)

    changed = senses()
    changed[1]['proposed_contexts'] = [{
        'name': 'Computer components',
        'description': 'Computer component interconnect systems.',
        'reason': 'Not listed despite established electronics profile.',
    }]
    with pytest.raises(ValueError, match='sufficient published context'):
        parse_scored_thinker(json.dumps({'senses': changed}), source, catalog)

    changed = senses()
    changed[1]['context_scores'] = {}
    changed[1]['proposed_contexts'] = [{
        'name': 'Marine engineering',
        'description': 'Ships and marine engineering systems.',
        'reason': 'The list of contexts does not include maritime operations.',
    }]
    assert parse_scored_thinker(json.dumps({'senses': changed}),
                                source, catalog).senses[1].proposed_contexts


def test_numeric_router_branches_are_independent_of_text(catalog_fixture):
    _, catalog, _ = catalog_fixture
    for first, second in ((60, 60), (65, 65)):
        decision = route_scores(senses(first, second), 'electronics', catalog)
        assert decision == {
            'call_selector': False, 'choice': 2,
            'reason': 'clear_context_score_lead',
        }

    assert route_scores(senses(92, 60), 'electronics', catalog)['reason'] == \
        'decisive_source_context_conflict'

    # An unusually strong source clue can override an irrelevant site.
    distinctive = senses(97, 45)
    assert route_scores(distinctive, 'electronics', catalog)['choice'] == 1
    assert route_scores(distinctive, 'electronics', catalog)['reason'] == \
        'high_distinctive_source_margin'

    assert route_scores(senses(45, 40), 'electronics', catalog)['reason'] == \
        'all_source_scores_low'

    a = senses(60, 63)
    a[0]['context_scores']['electronics'] = 90
    assert route_scores(a, 'electronics', catalog)['reason'] == \
        'context_score_tie'

    a = senses(88, 68)
    assert route_scores(a, 'electronics', catalog)['reason'] == \
        'competing_source_evidence'

    b = senses(70, 70, 95, 80)
    assert route_scores(b, 'road_transport', catalog)['choice'] == 1
    assert route_scores(b, 'maritime', catalog)['reason'] == \
        'website_context_outside_catalog'


def test_unknown_site_is_visible_only_to_selector(catalog_fixture):
    _, catalog, _ = catalog_fixture
    site = 'maritime'
    source = 'Heaving down'
    t_data = json.loads(thinker_payload(source, catalog).partition('Input JSON: ')[2])
    assert 'maritime' not in json.dumps(t_data)
    sel = json.loads(
        selector_payload(source, site, senses(), catalog).partition('Input JSON: ')[2]
    )
    assert sel['website_context_id'] == site
    assert sel['website_context_description']['outside_catalog']
    assert sel['website_context_description']['title'] == 'Maritime operations'
    assert len(sel['available_contexts']) == 20
    assert 'review_target_not_sent' not in sel
    assert 'careening' not in json.dumps(sel).lower()


class Response:
    def __init__(self, result):
        self.result = result

    def raise_for_status(self):
        return None

    def json(self):
        return {
            'model': 'gpt-6-luna',
            'choices': [{
                'finish_reason': 'stop',
                'message': {'content': self.result},
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
        return Response(self.replies.pop(0))


def test_live_mock_auto_one_model_call(catalog_fixture):
    settings, catalog, _ = catalog_fixture
    sample = next(c for c in FRESH_CASES if c.case_id == 'a_current_increased')
    scores = [
        {
            'meaning': 'An electrical current increased in magnitude.',
            'term_senses': [{
                'span': 'current', 'sense': 'Electric charge flowing per unit time',
            }],
            'source_score': 95,
            'source_evidence': 'Current can refer to current in an electric circuit.',
            'context_scores': {'electronics': 98},
            'proposed_contexts': [],
        },
        {
            'meaning': 'A water current increased in force.',
            'term_senses': [{
                'span': 'current', 'sense': 'Water flow',
            }],
            'source_score': 61,
            'source_evidence': 'A river or ocean current may become stronger.',
            'context_scores': {'land_management': 75},
            'proposed_contexts': [],
        },
    ]
    client = MockClient([json.dumps({'senses': scores})])
    row = asyncio.run(run_case(client, settings, sample, 1, catalog))
    assert len(client.requests) == 1
    assert row['route']['reason'] == 'high_distinctive_source_margin'
    assert row['final_status'] == 'selected'
    assert row['selected_index'] == 1
    assert row['selector_calls'] == 0
    assert summarize([row])['actual_model_calls'] == 1
    assert sample.review_target not in json.dumps(client.requests)


def test_live_mock_conditional_selector_and_human_only_report(catalog_fixture):
    settings, catalog, _ = catalog_fixture
    sample = next(c for c in FRESH_CASES if c.case_id == 'u_heaving_down')
    expected = [
        {
            'meaning': 'A vessel is deliberately inclined to expose part of its hull.',
            'term_senses': [{
                'span': 'Heaving down',
                'sense': 'A maritime ship-maintenance operation',
            }],
            'source_score': 86,
            'source_evidence': 'The phrase can be used in traditional seafaring.',
            'context_scores': {},
            'proposed_contexts': [{
                'name': 'Maritime ship maintenance',
                'description': 'Ship hull care, repairs and seafaring practices.',
                'reason': 'No catalog entry specializes in marine vessels.',
            }],
        },
        {
            'meaning': 'Something is being pulled downward with force.',
            'term_senses': [{
                'span': 'Heaving down',
                'sense': 'Effortful downward motion',
            }],
            'source_score': 52,
            'source_evidence': 'The ordinary verb could describe heavy exertion.',
            'context_scores': {},
            'proposed_contexts': [],
        },
    ]
    selection = {
        'status': 'selected',
        'choice': 1,
        'basis': 'new_context_match',
        'reason': 'The maritime context supports the ship-maintenance reading.',
        'context_mappings': [{
            'proposed_name': 'Maritime ship maintenance',
            'nearest_context_ids': [],
            'reason': 'Existing catalog omits specialized ship maintenance.',
        }],
    }
    client = MockClient([
        json.dumps({'senses': expected}), json.dumps(selection),
    ])
    row = asyncio.run(run_case(client, settings, sample, 1, catalog))
    assert len(client.requests) == 2
    assert row['route']['reason'] == 'website_context_outside_catalog'
    assert row['selected_index'] == 1
    assert row['selector_calls'] == 1
    assert summarize([row])['new_context_proposals'] == 1
    report = render_markdown([row], catalog)
    assert 'Reference for HUMAN review ONLY' in report
    assert sample.review_target in report
    assert sample.review_target not in json.dumps(client.requests)
    assert 'maritime' not in client.requests[0]['messages'][1]['content'].lower()
    assert 'maritime' in client.requests[1]['messages'][1]['content'].lower()

    # The simulated user request does not modify a DB row.
    with Session(catalog_fixture[2]) as session:
        assert session.scalar(select(func.count()).select_from(Context)) == 6
        assert session.scalar(select(func.count()).select_from(ContextVersion)) == 18


def test_retry_diagnostics_and_error_preservation(catalog_fixture):
    settings, catalog, _ = catalog_fixture
    sample = next(c for c in FRESH_CASES if c.case_id == 'a_charge_measured')
    client = MockClient(['this is not JSON', 'this is not JSON'])
    row = asyncio.run(run_case(client, settings, sample, 1, catalog))
    assert len(client.requests) == 2
    assert row['final_status'] == 'error'
    assert row['error'].startswith('thinker:')
    assert row['thinker_calls'] == 2
    assert row['selected_index'] is None
    assert summarize([row])['errors'] == 1
