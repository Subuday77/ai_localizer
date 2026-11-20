"""Context-blind lexical inventory -> English sentence interpretations experiment.

From the repo root with the existing OpenAI credentials in .env:
    python -m scripts.inspect_english_senses_lexical
    python -m scripts.inspect_english_senses_lexical --extended

This is a SEPARATE experiment, not a translation API change. One Luna call per
sentence, no contextual category, no target language and no semantic selection.
It reports word/phrase senses before viable sentence-level interpretations.
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
from scripts.inspect_english_senses import CASES, EXTENDED


SYSTEM_PROMPT = (
    'You are an English semantic lexicographer analyzing a single isolated '
    'sentence. You are NOT a translator or context resolver. You receive NO '
    'website category, industry, target language, or surrounding text. '
    'Your task has TWO internal stages, but one structured JSON output. '
    'STAGE 1, lexical inventory: examine the main content words independently, '
    'especially head nouns, noun phrases, verbs, predicates and idiomatic '
    'expressions. Do not lock onto the first ordinary interpretation of a '
    'noun and then merely vary the verb. Actively look for established '
    'polysemy, including less frequent professional, engineering, scientific, '
    'computing, transport, artistic, financial, and idiomatic senses. '
    'Do this for EVERY potentially polysemous content word, not only verbs. '
    'Report a word or phrase under lexical_inventory only if it has at least '
    'two materially different, established meanings; list those meanings '
    'briefly with their semantic fields. Do not invent a meaning. '
    'STAGE 2, compositional meaning: combine the lexical alternatives into '
    'coherent interpretations of the WHOLE English sentence. Evaluate '
    'whether the exact verbs, objects, prepositions, collocations and '
    'relationships support each combination. Retain unusual established '
    'specialist meanings if the entire sentence can naturally express them. '
    'Discard readings that require invented events or unsupported additional '
    'words, and avoid trivial synonym variations of one interpretation. '
    'Do not decide which interpretation is intended, do not rank or prefer '
    'common meanings, and do not infer a website subject. '
    'If decisive wording supports only one sentence reading, give one. '
    'If ambiguous, give the materially distinct plausible interpretations '
    '(up to six). The lexical inventory may contain readings excluded from '
    'whole-sentence interpretations. '
    'Use short ENGLISH explanations and output ONLY this JSON object: '
    '{"lexical_inventory":[{"span":"exact English span from input",'
    '"readings":[{"sense":"concise English definition","field":"semantic field"}]}],'
    '"interpretations":[{"meaning":"English paraphrase of the entire sentence",'
    '"kind":"ordinary|technical|idiomatic|other",'
    '"evidence":"why these words support this interpretation"}]}. '
    'Use an empty lexical_inventory when no genuine lexical ambiguity '
    'exists. No translation, selection, confidence numbers, markdown, '
    'additional keys or text.'
)


class LexicalReading(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sense: str = Field(min_length=3, max_length=200)
    field: str = Field(min_length=2, max_length=120)


class LexicalAmbiguity(BaseModel):
    model_config = ConfigDict(extra='forbid')
    span: str = Field(min_length=1, max_length=100)
    readings: list[LexicalReading] = Field(min_length=2, max_length=7)


class Interpretation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    meaning: str = Field(min_length=5, max_length=500)
    kind: Literal['ordinary', 'technical', 'idiomatic', 'other']
    evidence: str = Field(min_length=3, max_length=300)


class SenseInventory(BaseModel):
    model_config = ConfigDict(extra='forbid')
    lexical_inventory: list[LexicalAmbiguity] = Field(max_length=8)
    interpretations: list[Interpretation] = Field(min_length=1, max_length=6)

    @model_validator(mode='after')
    def unique_items(self):
        spans = [x.span.casefold().strip() for x in self.lexical_inventory]
        if len(spans) != len(set(spans)):
            raise ValueError('Repeated lexical spans')
        for item in self.lexical_inventory:
            senses = [r.sense.casefold().strip() for r in item.readings]
            if len(senses) != len(set(senses)):
                raise ValueError('Repeated lexical readings')
        interpretations = [x.meaning.casefold().strip() for x in self.interpretations]
        if len(interpretations) != len(set(interpretations)):
            raise ValueError('Repeated sentence interpretations')
        return self


def parse_inventory(raw: str, source: str) -> SenseInventory:
    """Reject malformed output and lexical terms absent from the input."""
    parsed = SenseInventory.model_validate(_load_model_json(raw))
    source_folded = source.casefold()
    for item in parsed.lexical_inventory:
        if item.span.casefold() not in source_folded:
            raise ValueError('Lexical span is not part of the input sentence')
    return parsed


def user_prompt(source: str) -> str:
    """Never include the benchmark's hidden evaluation category in this text."""
    return (
        'Inventory distinct lexical senses before forming coherent whole-sentence '
        'readings. English sentence: ' + json.dumps(source, ensure_ascii=False)
    )


async def analyze_one(client, gate, settings, case):
    case_id, source, held_out_context = case
    record = {
        'case_id': case_id,
        'source': source,
        'evaluation_context_not_sent': held_out_context,
        'requested_model': settings.openai_model_id,
        'actual_model': None,
        'lexical_inventory': [],
        'interpretations': [],
        'http_status': None,
        'error': None,
        'elapsed_ms': None,
    }
    async with gate:
        started = time.monotonic()
        try:
            payload = build_request_payload(
                model=settings.openai_model_id,
                prompt=user_prompt(source),
                max_tokens=min(max(settings.openai_max_tokens, 3072),
                               settings.openai_max_tokens_cap),
                reasoning_effort=settings.openai_reasoning_effort,
                system_prompt=SYSTEM_PROMPT,
            )
            response = await client.post(
                settings.openai_base_url.rstrip('/') + '/chat/completions',
                headers={'Authorization': 'Bearer ' + settings.openai_api_key},
                json=payload,
            )
            record['http_status'] = response.status_code
            response.raise_for_status()
            body = response.json()
            record['actual_model'] = body.get('model')
            choice = body['choices'][0]
            if choice.get('finish_reason') not in (None, 'stop'):
                raise ValueError('Incomplete model response')
            raw = choice['message']['content']
            if not isinstance(raw, str):
                raise ValueError('Model response is not text')
            parsed = parse_inventory(raw, source)
            record['lexical_inventory'] = [
                item.model_dump() for item in parsed.lexical_inventory
            ]
            record['interpretations'] = [
                item.model_dump() for item in parsed.interpretations
            ]
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            record['error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        record['elapsed_ms'] = int((time.monotonic() - started) * 1000)
    print(
        f'{case_id}: lexical={len(record["lexical_inventory"])} '
        f'interpretations={len(record["interpretations"])} '
        f'HTTP={record["http_status"]} error={bool(record["error"])}',
        flush=True,
    )
    return record


def render_markdown(rows: list[dict]) -> str:
    lines = [
        '# Context-blind lexical inventory and English sentence meanings',
        '',
        'No professional website context or target language was supplied to the model.',
        'The held-out context is for human assessment ONLY.',
        '',
    ]
    for row in rows:
        lines += [
            f'## {row["case_id"]}',
            '',
            f'**English:** {row["source"]}',
            '',
            f'**Held-out context (NOT sent):** {row["evaluation_context_not_sent"]}',
            '',
            f'**Actual model:** {row["actual_model"]}',
            '',
        ]
        if row['error']:
            lines += [f'**ERROR:** {row["error"]}', '']
            continue
        lines += ['### Word and phrase meanings', '']
        if not row['lexical_inventory']:
            lines += ['No genuinely ambiguous content words identified.', '']
        for item in row['lexical_inventory']:
            lines += [f'**{item["span"]}**', '']
            for option in item['readings']:
                lines += [f'- {option["sense"]} ({option["field"]})']
            lines.append('')
        lines += ['### Plausible whole-sentence interpretations', '']
        for n, item in enumerate(row['interpretations'], 1):
            lines += [
                f'{n}. **{item["meaning"]}**',
                f'   - Kind: {item["kind"]}',
                f'   - Evidence: {item["evidence"]}',
            ]
        lines.append('')
    return '\n'.join(lines)


async def run(cases, settings):
    gate = asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=settings.openai_timeout_seconds) as client:
        return await asyncio.gather(*(
            analyze_one(client, gate, settings, case) for case in cases
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true',
                        help='Add four explicit disambiguation control sentences')
    args = parser.parse_args()
    settings = get_settings()
    if not settings.openai_api_key or settings.openai_api_key.startswith('your-'):
        parser.exit(2, 'OPENAI_API_KEY is missing in project .env\n')
    if not settings.openai_model_id or settings.openai_model_id.startswith('your-'):
        parser.exit(2, 'OPENAI_MODEL_ID is missing\n')
    cases = CASES + (EXTENDED if args.extended else ())
    print(f'Model: {settings.openai_model_id}; cases={len(cases)}; context=NONE')
    rows = asyncio.run(run(cases, settings))
    destination = Path('localizer-benchmark-results')
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = f'english_senses_lexical_{stamp}'
    json_file = destination / f'{output_name}.json'
    md_file = destination / f'{output_name}.md'
    json_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    md_file.write_text(render_markdown(rows), encoding='utf-8')
    print(f'Saved: {json_file}')
    print(f'Saved: {md_file}')
    if any(row['error'] for row in rows):
        parser.exit(1, 'One or more requests failed; inspect the saved report.\n')


if __name__ == '__main__':
    main()
