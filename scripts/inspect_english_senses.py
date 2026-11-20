"""One-stage, context-BLIND English sense-generation experiment.

From the repository root (with OPENAI_API_KEY in .env):
    python -m scripts.inspect_english_senses
    python -m scripts.inspect_english_senses --extended

Runs one independent primary-model request per English sentence. No API server,
PostgreSQL, context profile, target language, or translation is involved.
Outputs human-readable Markdown and structured JSON; does not change production.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import get_settings
from app.services.openai import _load_model_json, build_request_payload


# The optional evaluation contexts below are deliberately NOT sent to the model.
CASES = (
    ('bus_failed', 'The bus failed.', 'electronics'),
    ('bus_overloaded', 'The bus was overloaded.', 'electronics'),
    ('note_missing', 'The note was missing.', 'music'),
    ('fruit_salad', 'The colonel had a fruit salad on his chest.', 'military'),
    ('tickets', 'The conductor checked the tickets.', 'music'),
    ('naked_conductor', 'Naked conductor runs under the wagon.', 'railway'),
)

EXTENDED = (
    ('orchestra_conductor',
     'The conductor raised his baton and the orchestra began to play.', 'railway'),
    ('literal_note', 'She left a handwritten note at the reception desk.', 'music'),
    ('literal_bus', 'The bus arrived at the stop and the passengers got off.', 'electronics'),
    ('railway_track', 'The track was removed.', 'railway'),
)

SYSTEM_PROMPT = (
    'You are an English-language semantic lexicographer, NOT a translator. '
    'Your job is to enumerate distinct, reasonably plausible senses of the '
    'WHOLE English source sentence. Do not choose a best sense and do not '
    'translate into any target language. '
    'Look for both everyday and established specialist/technical/idiomatic '
    'readings; relatively rare professional meanings must not be overlooked '
    'solely because the everyday meaning is more common. '
    'Treat actual words, verbs, objects, and collocations in the sentence as '
    'evidence, and do NOT assume an unseen surrounding context. '
    'If the wording strongly establishes one meaning, return only that '
    'meaning. Otherwise return all materially distinct plausible meanings, '
    'up to five; never pad with far-fetched or mere synonym variations. '
    'For each reading, provide a short English paraphrase of the complete '
    'sentence and briefly indicate what in the source makes it plausible '
    'or leaves it ambiguous. Do not add facts absent from the source. '
    'Return ONLY one JSON object with this exact schema: '
    '{"senses":[{"meaning":"English whole-sentence paraphrase",'
    '"kind":"ordinary|technical|idiomatic|other",'
    '"evidence":"English cue or lack of decisive cue"}]}. '
    'Do not include confidence ratings, final selection, explanations '
    'outside JSON, or markdown.'
)


class Sense(BaseModel):
    model_config = ConfigDict(extra='forbid')

    meaning: str = Field(min_length=5, max_length=400)
    kind: Literal['ordinary', 'technical', 'idiomatic', 'other']
    evidence: str = Field(min_length=3, max_length=300)


class SenseResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    senses: list[Sense] = Field(min_length=1, max_length=5)

    @model_validator(mode='after')
    def no_duplicate_meanings(self):
        readings = [x.meaning.strip().casefold() for x in self.senses]
        if len(readings) != len(set(readings)):
            raise ValueError('Duplicate senses in response')
        return self


def parse_senses(raw: str) -> SenseResult:
    """Validate actual model output; never silently fix or invent candidates."""
    return SenseResult.model_validate(_load_model_json(raw))


def user_prompt(sentence: str) -> str:
    """The request does not contain category, evaluation context or target language."""
    return (
        'Enumerate the distinct plausible English-language meanings of this '
        'exact sentence. Return JSON only. '
        'English source sentence: ' + json.dumps(sentence, ensure_ascii=False)
    )


async def analyze_one(client, sem, settings, case):
    case_id, source, eval_context = case
    started = time.monotonic()
    result = {
        'case_id': case_id,
        'source': source,
        'evaluation_context_not_sent': eval_context,
        'requested_model': settings.openai_model_id,
        'actual_model': None,
        'senses': [],
        'http_status': None,
        'error': None,
        'elapsed_ms': None,
    }
    async with sem:
        try:
            payload = build_request_payload(
                model=settings.openai_model_id,
                prompt=user_prompt(source),
                max_tokens=min(max(settings.openai_max_tokens, 2048),
                               settings.openai_max_tokens_cap),
                reasoning_effort=settings.openai_reasoning_effort,
                system_prompt=SYSTEM_PROMPT,
            )
            response = await client.post(
                settings.openai_base_url.rstrip('/') + '/chat/completions',
                headers={'Authorization': 'Bearer ' + settings.openai_api_key},
                json=payload,
            )
            result['http_status'] = response.status_code
            response.raise_for_status()
            body = response.json()
            result['actual_model'] = body.get('model')
            choice = body['choices'][0]
            if choice.get('finish_reason') not in (None, 'stop'):
                raise ValueError('Model did not finish the JSON response')
            if not isinstance(choice['message']['content'], str):
                raise ValueError('Response was not textual JSON')
            parsed = parse_senses(choice['message']['content'])
            result['senses'] = [sense.model_dump() for sense in parsed.senses]
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            result['error'] = (
                f'{type(exc).__name__}: {str(exc)[:300]}'
            )
    result['elapsed_ms'] = int((time.monotonic() - started) * 1000)
    print(f'{case_id}: '
          f'{len(result["senses"])} senses; '
          f'HTTP={result["http_status"]}; '
          f'model={result["actual_model"]}; '
          f'error={bool(result["error"])}')
    return result


def render_markdown(rows: list[dict]) -> str:
    lines = [
        '# Context-blind English sense generation',
        '',
        'The model was NOT given professional context, target language, or',
        'translations. Context labels below are for later human evaluation only.',
        '',
    ]
    for row in rows:
        lines.extend([
            f'## {row["case_id"]}',
            '',
            f'**English:** {row["source"]}',
            '',
            f'**Held-out context:** {row["evaluation_context_not_sent"]}',
            '',
            f'**Model:** {row["actual_model"] or row["requested_model"]}',
            '',
        ])
        if row['error']:
            lines.extend([f'**ERROR:** {row["error"]}', ''])
            continue
        for i, sense in enumerate(row['senses'], 1):
            lines.extend([
                f'{i}. **{sense["meaning"]}**',
                f'   - Kind: {sense["kind"]}',
                f'   - Source evidence: {sense["evidence"]}',
            ])
        lines.append('')
    return '\n'.join(lines)


async def run(cases, settings):
    sem = asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=settings.openai_timeout_seconds) as client:
        return await asyncio.gather(*(
            analyze_one(client, sem, settings, case) for case in cases
        ))


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--extended', action='store_true',
                     help='Also analyze four explicit disambiguation controls')
    args = arg.parse_args()
    settings = get_settings()
    if not settings.openai_api_key or settings.openai_api_key.startswith('your-'):
        arg.exit(2, 'OPENAI_API_KEY is missing in the project .env\n')
    if not settings.openai_model_id or settings.openai_model_id.startswith('your-'):
        arg.exit(2, 'OPENAI_MODEL_ID is missing\n')

    cases = CASES + (EXTENDED if args.extended else ())
    print(f'Primary model: {settings.openai_model_id}')
    print(f'English sources: {len(cases)} (independent calls, no context)')
    rows = asyncio.run(run(cases, settings))
    output = Path('localizer-benchmark-results')
    output.mkdir(parents=True, exist_ok=True)
    name = 'english_senses_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output / f'{name}.json'
    md_path = output / f'{name}.md'
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    md_path.write_text(render_markdown(rows), encoding='utf-8')
    print(f'\nSaved: {json_path}')
    print(f'Saved: {md_path}')
    if any(row['error'] for row in rows):
        arg.exit(1, 'Some model requests failed; see the saved report.\n')


if __name__ == '__main__':
    main()
