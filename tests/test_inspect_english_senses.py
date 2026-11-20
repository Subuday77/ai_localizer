"""Offline contract tests for the isolated English sense-generation experiment."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from app.config import Settings
from scripts.inspect_english_senses import (
    CASES,
    EXTENDED,
    SYSTEM_PROMPT,
    analyze_one,
    parse_senses,
    render_markdown,
    user_prompt,
)


def test_test_cases_match_existing_benchmark_and_source_only_prompt():
    assert len(CASES) == 6
    assert len(EXTENDED) == 4
    assert 'The bus failed.' in [case[1] for case in CASES]
    for case_id, source, evaluation_context in CASES + EXTENDED:
        prompt = user_prompt(source)
        assert source in prompt
        assert evaluation_context not in prompt.casefold()
        assert case_id not in prompt
        assert 'target language' not in prompt.casefold()
    assert 'NOT a translator' in SYSTEM_PROMPT
    assert 'do not choose a best sense' in SYSTEM_PROMPT
    assert 'one meaning' in SYSTEM_PROMPT
    assert 'up to five' in SYSTEM_PROMPT


def test_parser_validates_distinct_structured_english_meanings():
    raw = json.dumps({
        'senses': [
            {'meaning': 'The road passenger vehicle stopped working.',
             'kind': 'ordinary',
             'evidence': 'The predicate failed does not decide which bus.'},
            {'meaning': 'An electrical distribution bus stopped operating.',
             'kind': 'technical',
             'evidence': 'The sentence has no disambiguating equipment details.'},
        ],
    })
    data = parse_senses(raw)
    assert len(data.senses) == 2
    assert data.senses[1].kind == 'technical'
    with pytest.raises(ValidationError):
        parse_senses('{"senses":[]}')
    with pytest.raises(ValidationError):
        parse_senses(json.dumps({'senses': [
            {'meaning': 'A vehicle stopped working.',
             'kind': 'ordinary', 'evidence': 'No extra evidence.'},
            {'meaning': 'A vehicle stopped working.',
             'kind': 'ordinary', 'evidence': 'No extra evidence.'},
        ]}))
    with pytest.raises(ValidationError):
        parse_senses('{"senses":[{"meaning":"A road vehicle stopped running","kind":"ordinary"}]}')


def test_analyzer_does_not_send_held_out_context_to_model():
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                'model': 'gpt-6-luna',
                'choices': [{
                    'finish_reason': 'stop',
                    'message': {'content': json.dumps({'senses': [
                        {'meaning': 'The passenger bus stopped functioning.',
                         'kind': 'ordinary',
                         'evidence': 'No uniquely identifying clue.'},
                    ]})},
                }],
            }

    class FakeClient:
        sent = None

        async def post(self, url, headers, json):
            self.sent = json
            assert 'Bearer test-only-key' == headers['Authorization']
            return FakeResponse()

    settings = Settings(
        _env_file=None,
        openai_api_key='test-only-key',
        openai_model_id='gpt-6-luna',
    )
    client = FakeClient()
    row = asyncio.run(analyze_one(
        client, asyncio.Semaphore(1), settings,
        ('bus_failed', 'The bus failed.', 'electronics'),
    ))
    sent = ' '.join(item['content'] for item in client.sent['messages'])
    assert 'electronics' not in sent
    assert 'ru' not in user_prompt('The bus failed.')
    assert row['evaluation_context_not_sent'] == 'electronics'
    assert row['actual_model'] == 'gpt-6-luna'
    assert len(row['senses']) == 1
    md = render_markdown([row])
    assert 'Held-out context: ' not in md  # Markdown uses a bold label.
    assert '**Held-out context:** electronics' in md
