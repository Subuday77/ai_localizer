"""Contract tests for context-blind lexical-first sense enumeration experiment."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from app.config import Settings
from scripts.inspect_english_senses import CASES, EXTENDED
from scripts.inspect_english_senses_lexical import (
    SYSTEM_PROMPT,
    analyze_one,
    parse_inventory,
    render_markdown,
    user_prompt,
)


def sample():
    return {
        'lexical_inventory': [
            {
                'span': 'bus',
                'readings': [
                    {'sense': 'Road passenger vehicle.', 'field': 'transportation'},
                    {'sense': 'Electrical distribution conductor.', 'field': 'engineering'},
                ],
            },
        ],
        'interpretations': [
            {
                'meaning': 'The passenger vehicle stopped working.',
                'kind': 'ordinary',
                'evidence': 'The sentence does not name a technical component.',
            },
            {
                'meaning': 'An electrical distribution system stopped functioning.',
                'kind': 'technical',
                'evidence': 'A bus can be a power or signal distribution system.',
            },
        ],
    }


def test_lexical_first_prompt_contains_no_hidden_category_or_known_ambiguous_terms():
    assert 'head nouns' in SYSTEM_PROMPT
    assert 'not only verbs' in SYSTEM_PROMPT
    assert 'STAGE 1' in SYSTEM_PROMPT and 'STAGE 2' in SYSTEM_PROMPT
    assert all(name not in SYSTEM_PROMPT.casefold() for name in (
        'electronics', 'railway', 'military', 'music',
    ))  # no held-out site profile
    for case_id, source, context in CASES + EXTENDED:
        prompt = user_prompt(source)
        assert prompt.endswith(json.dumps(source, ensure_ascii=False))
        assert context not in prompt.casefold()
        assert not any(key in SYSTEM_PROMPT.casefold() for key in (
            'bus failed', 'note was missing', 'conductor checked the tickets',
        ))
        assert case_id not in SYSTEM_PROMPT


def test_parse_validates_lexical_and_whole_sentence_senses():
    result = parse_inventory(json.dumps(sample()), 'The bus failed.')
    assert len(result.lexical_inventory) == 1
    assert len(result.interpretations) == 2
    assert result.lexical_inventory[0].span == 'bus'

    wrong_span = sample()
    wrong_span['lexical_inventory'][0]['span'] = 'electrical'
    with pytest.raises(ValueError, match='not part of'):
        parse_inventory(json.dumps(wrong_span), 'The bus failed.')

    duplicate_span = sample()
    duplicate_span['lexical_inventory'].append(
        duplicate_span['lexical_inventory'][0].copy()
    )
    with pytest.raises(ValidationError):
        parse_inventory(json.dumps(duplicate_span), 'The bus failed.')

    no_interpretations = sample()
    no_interpretations['interpretations'] = []
    with pytest.raises(ValidationError):
        parse_inventory(json.dumps(no_interpretations), 'The bus failed.')

    extra_field = sample()
    extra_field['unexpected_translation'] = 'Автобус вышел из строя'
    with pytest.raises(ValidationError):
        parse_inventory(json.dumps(extra_field), 'The bus failed.')


def test_live_request_contract_is_context_blind_and_persists_readable_report():
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                'model': 'gpt-6-luna',
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'content': json.dumps(sample())},
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
    result = asyncio.run(analyze_one(
        client, asyncio.Semaphore(1), settings,
        ('bus_failed', 'The bus failed.', 'electronics'),
    ))
    combined_prompt = ' '.join(m['content'] for m in client.sent['messages'])
    assert 'electronics' not in combined_prompt
    assert 'Russian' not in combined_prompt
    assert client.sent['model'] == 'gpt-6-luna'
    assert client.sent['max_completion_tokens'] >= 3072
    assert result['evaluation_context_not_sent'] == 'electronics'
    assert len(result['lexical_inventory']) == 1
    assert len(result['interpretations']) == 2
    assert result['error'] is None
    output = render_markdown([result])
    assert 'Road passenger vehicle' in output
    assert 'Electrical distribution conductor' in output
    assert 'An electrical distribution system stopped functioning' in output


def test_no_polysemy_may_return_one_interpretation():
    result = parse_inventory(json.dumps({
        'lexical_inventory': [],
        'interpretations': [{
            'meaning': 'An unambiguous instruction to save the changes.',
            'kind': 'ordinary',
            'evidence': 'The instruction contains no ambiguity in context.',
        }],
    }), 'Save changes')
    assert result.lexical_inventory == []
    assert len(result.interpretations) == 1
