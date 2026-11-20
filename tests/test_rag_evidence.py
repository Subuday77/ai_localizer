"""Offline contract and safety tests for the opt-in multilingual RAG prototype."""

import asyncio
import json
from pathlib import Path

import pytest

from app.config import Settings
from scripts.benchmark_score_cases import FRESH_CASES
from scripts.benchmark_scored_router import (
    thinker_payload, selector_payload, route_scores, run_case,
)
from scripts.rag_evidence import (
    SOURCE_LABEL, SEED_FILE, SmallConceptIndex, compact_for_model,
    evidence_gate, fetch_wiktionary_senses, get_online_query_terms,
    matched_term, search_with_optional_online,
    english_lemma_candidate, needs_english_lemma_retry,
)


@pytest.fixture(scope='module')
def index():
    return SmallConceptIndex()


def test_compact_corpus_is_multilingual_not_gold(index):
    assert 100 <= len(index.records) <= 200
    assert index.file_bytes < 100_000
    assert all(x['provenance']['provider'] == 'prototype_original'
               and x['provenance']['verified'] is False
               for x in index.records)
    assert any(set(x['terms']) >= {'en', 'ru', 'de', 'fr', 'es'}
               for x in index.records)
    sources = {case.source.casefold() for case in FRESH_CASES}
    assert not any(
        x['definition_en'].casefold() in sources
        for x in index.records
    )
    assert 'review_target_not_sent' not in SEED_FILE.read_text(
        encoding='utf-8'
    )


def test_multilingual_concepts_and_non_russian_assumption(index):
    for text, lang, cid in [
        ('The bus failed.', 'en', 'electronics_bus'),
        ('The bus failed.', 'en', 'road_bus'),
        ('pitching', 'en', 'maritime_pitching'),
        ('pitching', 'en', 'sports_pitching'),
        ('Heaving down', 'en', 'maritime_heaving'),
        ('Honey badger', 'en', 'wildlife_honey_badger'),
        ('Stampfen', 'de', 'maritime_pitching'),
        ('tangage', 'fr', 'maritime_pitching'),
        ('cabeceo', 'es', 'maritime_pitching'),
        ('килевая качка', 'ru', 'maritime_pitching'),
        ('Datenbus', 'de', 'electronics_bus'),
        ('bus de datos', 'es', 'electronics_bus'),
    ]:
        assert cid in {
            hit['concept_id'] for hit in index.search(
                text, source_language=lang,
            )
        }, (lang, text, cid)
    assert 'maritime_pitching' not in {
        hit['concept_id'] for hit in index.search(
            'pitching', source_language='de',
        )
    }


def test_exact_word_boundary_does_not_rewrite_spelling(index):
    assert matched_term('The bus failed', 'bus')
    assert not matched_term('The business failed', 'bus')
    assert not matched_term('Drop the base', 'bass')
    assert not matched_term('Moon shine', 'moonshine')
    assert matched_term('l’hôtel de train', 'hôtel')
    hits = index.search('Drop the base')
    assert 'music_bass' not in {x['concept_id'] for x in hits}
    assert 'construction_base' in {x['concept_id'] for x in hits}
    assert 'military_base' in {x['concept_id'] for x in hits}
    assert index.search('Moon shine')
    assert not index.search('moonshine', limit=8)


def test_index_constraints_and_prompt_secrecy(index):
    with pytest.raises(ValueError):
        index.search('bus', limit=10000)
    with pytest.raises(ValueError):
        index.search('bus', source_language='../../etc')
    with pytest.raises(ValueError):
        SmallConceptIndex(SEED_FILE, max_file_bytes=100)

    evidence = index.search('Heaving down')
    catalog = {'music': {'title': 'Music'}, 'military': {'title': 'Military'}}
    thinker_data = json.loads(
        thinker_payload('Heaving down', catalog, evidence)
        .partition('Input JSON: ')[2]
    )
    assert 'website_context_id' not in thinker_data
    assert 'review_target' not in thinker_data
    assert thinker_data['english_source'] == 'Heaving down'
    assert thinker_data['unverified_lexical_evidence']
    assert all('terms' not in e for e in
               thinker_data['unverified_lexical_evidence'])
    selector_data = json.loads(
        selector_payload('Heaving down', 'maritime', [], catalog, evidence)
        .partition('Input JSON: ')[2]
    )
    assert selector_data['website_context_id'] == 'maritime'
    assert len(selector_data['unverified_lexical_evidence']) > 0
    assert 'review_target_not_sent' not in json.dumps(selector_data)
    assert compact_for_model(evidence)[0]['provenance']['verified'] is False


def test_gate_never_autoselects_and_requires_actual_lexical_ambiguity(index):
    catalog = {'music': {}, 'office_administration': {}}
    two = [
        {
            'source_score': 68,
            'context_scores': {'music': 95},
            'meaning': 'A musical note persisted for four rhythmic beats.',
        },
        {
            'source_score': 62,
            'context_scores': {'office_administration': 94},
            'meaning': 'An office memo remained in place.',
        },
    ]
    numeric = route_scores(two, 'music', catalog)
    assert numeric['call_selector'] is False
    hits = index.search('The note was held for four beats.')
    assert {'music_note', 'office_note'} <= {x['concept_id'] for x in hits}
    result = evidence_gate(two, numeric, hits, 'music')
    assert result['call_selector'] is True
    assert result['choice'] is None
    assert result['reason'] == 'rag_ambiguous_term_requires_selector'
    # An unambiguous, distinctive sentence still wins on text evidence.
    two[0]['source_score'] = 95
    numeric = route_scores(two, 'music', catalog)
    assert evidence_gate(two, numeric, hits, 'music') == numeric
    assert evidence_gate(two, numeric, [], 'music') == numeric


def test_online_candidate_queries_are_bounded_and_without_full_sentence(index):
    hits = index.search('The note was held for four beats.')
    query = get_online_query_terms('The note was held for four beats.', hits)
    assert len(query) <= 2
    assert 'The note was held for four beats.' not in query
    with pytest.raises(ValueError):
        get_online_query_terms('something', [], max_terms=200)


class FakeResponse:
    status_code = 200

    def __init__(self, document):
        self.document = document

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        yield json.dumps(self.document).encode('utf-8')


class FakeHTTPClient:
    def __init__(self, document):
        self.document = document
        self.requests = []

    def stream(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return FakeResponse(self.document)


def online_fixture():
    return {
        'word': 'pitch',
        'entries': [{
            'partOfSpeech': 'noun',
            'senses': [{
                'definition': 'A property of a sound related to frequency.',
                'tags': ['music'],
            }],
        }],
        'source': {
            'url': 'https://en.wiktionary.org/wiki/pitch',
            'license': {
                'name': 'CC BY-SA 4.0',
                'url': 'https://creativecommons.org/licenses/by-sa/4.0/',
            },
        },
    }


def test_remote_dictionary_is_opt_in_bounded_attributed():
    client = FakeHTTPClient(online_fixture())
    result = asyncio.run(fetch_wiktionary_senses(client, 'pitch'))
    assert len(client.requests) == 1
    assert client.requests[0][0] == 'GET'
    assert client.requests[0][1].startswith('https://freedictionaryapi.com/api/v1/')
    assert result[0]['domain_id'] == 'unclassified'
    assert result[0]['provenance']['license'] == 'CC BY-SA 4.0'
    assert result[0]['provenance']['source_url'].startswith(
        'https://en.wiktionary.org/'
    )
    assert 'FreeDictionaryAPI.com' in SOURCE_LABEL
    assert result[0]['provenance']['verified'] is False


def test_remote_cache_is_in_memory_and_prevents_repeat_requests(index):
    cache = {}
    client = FakeHTTPClient(online_fixture())

    async def twice():
        a = await search_with_optional_online(
            'pitch', source_language='en', index=index, online=True,
            client=client, max_online_terms=1, online_cache=cache,
        )
        b = await search_with_optional_online(
            'pitch', source_language='en', index=index, online=True,
            client=client, max_online_terms=1, online_cache=cache,
        )
        return a, b

    first, second = asyncio.run(twice())
    assert first[1] == 1
    assert second[1] == 0
    assert len(client.requests) == 1
    assert len(first[0]) > 0
    assert len(second[0]) > 0


def test_online_oversize_is_rejected():
    client = FakeHTTPClient({
        **online_fixture(),
        'padding': 'z' * 1000,
    })
    with pytest.raises(ValueError, match='exceeded'):
        asyncio.run(fetch_wiktionary_senses(client, 'pitch', max_bytes=50))


class Response:
    def __init__(self, raw):
        self.raw = raw

    def raise_for_status(self):
        return None

    def json(self):
        return {
            'model': 'gpt-6-luna',
            'choices': [{
                'finish_reason': 'stop',
                'message': {'content': self.raw},
            }],
        }


class MockLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def post(self, url, *, json, headers):
        self.requests.append(json)
        return Response(self.responses.pop(0))


def thinker_document():
    return json.dumps({
        'senses': [
            {
                'meaning': 'A musical note was sustained for four beats.',
                'term_senses': [{
                    'span': 'note',
                    'sense': 'A pitched musical sound with rhythmic duration',
                }],
                'source_score': 68,
                'source_evidence': 'Beats is used for timed notes in music.',
                'context_scores': {'music': 95},
                'proposed_contexts': [],
            },
            {
                'meaning': 'A short written message was held for a time.',
                'term_senses': [{
                    'span': 'note', 'sense': 'A brief written message',
                }],
                'source_score': 62,
                'source_evidence': 'Note can also mean a written message.',
                'context_scores': {'office_administration': 94},
                'proposed_contexts': [],
            },
        ],
    })


def test_rag_assist_veto_invokes_selector_only_when_needed(index):
    settings = Settings(
        _env_file=None, openai_api_key='test-key',
        openai_model_id='gpt-6-luna',
    )
    case = next(c for c in FRESH_CASES
                if c.case_id == 'a_note_four_beats')
    catalog = {'music': {}, 'office_administration': {}}
    off_client = MockLLMClient([thinker_document()])
    off = asyncio.run(run_case(
        off_client, settings, case, 1, catalog, rag_mode='off',
    ))
    assert off['route']['call_selector'] is False
    assert off['selected_index'] == 1
    assert len(off_client.requests) == 1
    assert 'unverified_lexical_evidence' not in (
        off_client.requests[0]['messages'][1]['content']
    )

    selector_response = json.dumps({
        'status': 'selected', 'choice': 1,
        'basis': 'context_tiebreak',
        'reason': 'The term beats supports rhythmic duration of a note.',
        'context_mappings': [],
    })
    client = MockLLMClient([thinker_document(), selector_response])
    assisted = asyncio.run(run_case(
        client, settings, case, 1, catalog,
        rag_mode='assist', rag_index=index,
    ))
    assert len(client.requests) == 2
    assert assisted['original_route']['call_selector'] is False
    assert assisted['route']['call_selector'] is True
    assert assisted['rag_gate_escalation'] is True
    assert assisted['selected_index'] == 1
    assert assisted['rag_hits_count'] > 0
    assert 'unverified_lexical_evidence' in (
        client.requests[0]['messages'][1]['content']
    )
    assert 'unverified_lexical_evidence' in (
        client.requests[1]['messages'][1]['content']
    )
    assert case.review_target not in json.dumps(client.requests)


def test_rag_shadow_never_changes_decision_or_prompt(index):
    settings = Settings(_env_file=None, openai_api_key='test-key')
    case = next(c for c in FRESH_CASES
                if c.case_id == 'a_note_four_beats')
    catalog = {'music': {}, 'office_administration': {}}
    client = MockLLMClient([thinker_document()])
    row = asyncio.run(run_case(
        client, settings, case, 1, catalog, rag_mode='shadow',
        rag_index=index,
    ))
    assert row['final_status'] == 'selected'
    assert row['selected_index'] == 1
    assert row['rag_hits_count'] > 0
    assert not row['rag_gate_escalation']
    assert len(client.requests) == 1
    assert 'unverified_lexical_evidence' not in json.dumps(client.requests)


class EmptyResponse(FakeResponse):
    def __init__(self, status_code):
        super().__init__({})
        self.status_code = status_code


class StatusClient:
    def __init__(self, status_code):
        self.status_code = status_code

    def stream(self, *args, **kwargs):
        return EmptyResponse(self.status_code)


def test_online_not_found_and_rate_limit_are_distinguished(index):
    async def query(status):
        diagnostics = []
        async with httpx.AsyncClient() as unused:
            # The fake HTTP client returns a real response status, never
            # contacts the network, and requires no API key.
            hits, requests = await search_with_optional_online(
                'heaving down',
                source_language='en',
                index=index,
                online=True,
                client=StatusClient(status),
                diagnostics=diagnostics,
            )
        return hits, requests, diagnostics

    import httpx
    found, requests, trace = asyncio.run(query(404))
    assert requests == 1
    assert len(found) == 1  # local seed survives
    assert trace == [{
        'term': 'heaving down', 'status': 'not_found',
        'http_status': 404, 'definitions': 0,
    }]
    rate_limited, requests, trace = asyncio.run(query(429))
    assert requests == 1
    assert len(rate_limited) == 1
    assert trace[0]['status'] == 'rate_limited'


def test_online_network_errors_preserve_local_results_and_explain_error(index):
    class BrokenClient:
        def stream(self, *args, **kwargs):
            raise ValueError('simulated malformed response')

    async def query():
        diagnostics = []
        hits, requests = await search_with_optional_online(
            'heaving down', source_language='en', index=index,
            online=True, client=BrokenClient(), diagnostics=diagnostics,
        )
        return hits, requests, diagnostics
    hits, requests, diagnostics = asyncio.run(query())
    assert requests == 1
    assert len(hits) == 1
    assert diagnostics[0]['status'] == 'request_error'
    assert diagnostics[0]['error_type'] == 'ValueError'


def test_non_english_external_dictionary_evidence_retains_language():
    client = FakeHTTPClient(online_fixture())
    report = []
    results = asyncio.run(fetch_wiktionary_senses(
        client, 'Tonhöhe', source_language='de', status_report=report,
    ))
    assert report[0]['status'] == 'found'
    assert results[0]['language'] == 'de'
    assert results[0]['definition_en'] == ''
    assert results[0]['definition_source_language']
    assert compact_for_model(results)[0]['language'] == 'de'



class DictionaryByTerm:
    """Deterministic stand-in for the licensed provider; no network calls."""
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def stream(self, method, url, **kwargs):
        from urllib.parse import unquote
        word = unquote(url.rsplit('/', 1)[-1])
        self.requests.append(word)
        data = self.responses.get(word, {'word': word, 'entries': []})
        return FakeResponse(data)


def test_english_lemma_candidate_is_not_globally_applied():
    assert english_lemma_candidate('heaving down') == 'heave down'
    assert english_lemma_candidate('pitching') == 'pitch'
    assert english_lemma_candidate('running') == 'run'
    assert english_lemma_candidate('dancing') == 'dance'
    assert english_lemma_candidate('moon shine') is None
    assert english_lemma_candidate('go to a bear') is None
    assert english_lemma_candidate('something') is None
    assert needs_english_lemma_retry('no_definitions', [])
    assert not needs_english_lemma_retry('rate_limited', [])
    assert needs_english_lemma_retry('found', [{
        'definition_en': 'present participle and gerund of pitch',
    }])


def test_200_empty_entry_triggers_one_lemma_lookup_with_provenance(index):
    client = DictionaryByTerm({
        'heaving down': {'word': 'heaving down', 'entries': []},
        'heave down': {
            'word': 'heave down',
            'entries': [{
                'partOfSpeech': 'verb',
                'senses': [{
                    'definition': 'To incline a vessel so the hull is exposed for work.',
                }],
            }],
        },
    })
    async def run():
        diagnostics = []
        hits, requests = await search_with_optional_online(
            'heaving down', source_language='en', index=index,
            online=True, client=client, diagnostics=diagnostics,
        )
        return hits, requests, diagnostics

    hits, requests, diagnostics = asyncio.run(run())
    assert requests == 2
    assert client.requests == ['heaving down', 'heave down']
    assert diagnostics[0]['status'] == 'no_definitions'
    assert diagnostics[1]['status'] == 'found'
    assert diagnostics[1]['lookup_relation'] == 'heuristic_lemma'
    assert diagnostics[1]['derived_from'] == 'heaving down'
    assert len([h for h in hits if h['concept_id'].startswith('wiktionary:')]) == 1
    lemma_hit = next(h for h in hits if h['concept_id'].startswith('wiktionary:'))
    assert lemma_hit['matched_term'] == 'heave down'
    assert lemma_hit['derived_from'] == 'heaving down'
    assert lemma_hit['retrieval_score'] <= 48
    assert lemma_hit['provenance']['license'] == 'CC BY-SA 4.0'
    compact = compact_for_model([lemma_hit])
    assert compact[0]['lookup_relation'] == 'heuristic_lemma'


def test_morphological_entry_triggers_lemma_and_never_third_request(index):
    client = DictionaryByTerm({
        'pitching': {
            'word': 'pitching',
            'entries': [{'partOfSpeech': 'verb', 'senses': [{
                'definition': 'present participle and gerund of pitch',
            }]}],
        },
        'pitch': {
            'word': 'pitch',
            'entries': [{'partOfSpeech': 'noun', 'senses': [{
                'definition': 'The perceived highness or lowness of a sound.',
            }]}],
        },
    })
    async def run():
        trace = []
        rows, calls = await search_with_optional_online(
            'pitching', source_language='en', index=index,
            online=True, client=client, diagnostics=trace,
        )
        return rows, calls, trace

    hits, count, diag = asyncio.run(run())
    assert count == 2
    assert client.requests == ['pitching', 'pitch']
    assert diag[0]['status'] == 'found'
    assert diag[1]['lookup_relation'] == 'heuristic_lemma'
    assert any(h['derived_from'] == 'pitching'
               for h in hits if h.get('lookup_relation'))
    assert any('highness' in h['definition_en'] for h in hits)
    assert len(client.requests) <= 2


def test_other_languages_are_not_english_stemmed_and_are_still_queried(index):
    client = DictionaryByTerm({
        'Stampfen': {'word': 'Stampfen', 'entries': []},
    })
    async def run():
        trace = []
        rows, requests = await search_with_optional_online(
            'Stampfen', source_language='de', index=index,
            online=True, client=client, diagnostics=trace,
        )
        return rows, requests, trace

    hits, count, diag = asyncio.run(run())
    assert count == 1
    assert client.requests == ['Stampfen']
    assert diag[0]['status'] == 'no_definitions'
    assert 'maritime_pitching' in {h['concept_id'] for h in hits}
