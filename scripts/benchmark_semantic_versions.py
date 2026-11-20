"""Real-model A/B benchmark for English website dictionary context v1/v2/v3.

Run from the repository root with FastAPI on localhost:8081:
    python -m scripts.benchmark_semantic_versions
    python -m scripts.benchmark_semantic_versions --extended --language de
    python -m scripts.benchmark_semantic_versions --runs 3

No external dependencies. Writes UTF-8 CSV/JSON to localizer-benchmark-results.
No semantic result is auto-marked correct: all translations need human review.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CASES = [
    ('bus_failed', 'electronics', 'The bus failed.'),
    ('bus_overloaded', 'electronics', 'The bus was overloaded.'),
    ('note_missing', 'music', 'The note was missing.'),
    ('fruit_salad', 'military', 'The colonel had a fruit salad on his chest.'),
    ('tickets', 'music', 'The conductor checked the tickets.'),
    ('naked_conductor', 'railway', 'Naked conductor runs under the wagon.'),
]
EXTENDED = [
    ('orchestra_conductor', 'railway',
     'The conductor raised his baton and the orchestra began to play.'),
    ('literal_note', 'music',
     'She left a handwritten note at the reception desk.'),
    ('literal_bus', 'electronics',
     'The bus arrived at the stop and the passengers got off.'),
    ('railway_track', 'railway', 'The track was removed.'),
]
VARIANTS = ('baseline', 'v1', 'v2', 'v3')


def request_json(url, body=None, api_key='', timeout=60):
    headers = {'Accept': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    if body is not None:
        headers['Content-Type'] = 'application/json; charset=utf-8'
    req = Request(
        url, data=json.dumps(body, ensure_ascii=False).encode('utf-8')
        if body is not None else None,
        headers=headers, method='POST' if body is not None else 'GET',
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        raw = exc.read().decode('utf-8', errors='replace')
        try:
            response = json.loads(raw)
        except ValueError:
            response = {'detail': raw[:500]}
        return exc.code, response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8081')
    parser.add_argument('--language', default='ru',
                        help='Target language code, e.g. ru, de, he')
    parser.add_argument('--runs', type=int, default=1,
                        help='Independent repeats per variant (default: 1)')
    parser.add_argument('--extended', action='store_true',
                        help='Include four extra counterexample scenarios')
    parser.add_argument('--variants', default=','.join(VARIANTS),
                        help='Comma separated: baseline,v1,v2,v3')
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        parser.error('--runs must be from 1 to 10')
    variants = tuple(x.strip() for x in args.variants.split(','))
    if not variants or any(x not in VARIANTS for x in variants):
        parser.error('--variants must only use baseline,v1,v2,v3')

    url = args.url.rstrip('/')
    api_key = os.environ.get('LOCALIZER_BENCHMARK_API_KEY', '')
    try:
        status, catalog = request_json(url + '/contexts', api_key=api_key)
    except (URLError, OSError, ValueError) as exc:
        parser.exit(1, f'Cannot reach local API: {exc}\n')
    if status != 200 or not isinstance(catalog.get('contexts'), list):
        parser.exit(1, f'Catalog unavailable: HTTP {status}\n')
    versions = {entry['id']: entry['version'] for entry in catalog['contexts']}
    expected = {case[1] for case in CASES + (EXTENDED if args.extended else [])}
    missing = [name for name in expected if versions.get(name) != 3]
    if missing:
        parser.exit(1, 'Expected published version 3: ' +
                    ', '.join(sorted(missing)) +
                    '. Run python -m scripts.publish_semantic_profiles.\n')

    cases = CASES + (EXTENDED if args.extended else [])
    total = len(cases) * len(variants) * args.runs
    print(f'Cases={len(cases)} variants={",".join(variants)} '
          f'repeats={args.runs} calls={total} target={args.language}')
    print('Source: English website UI strings. Model outputs saved to files.')
    rows = []
    for case_id, context, source in cases:
        print(f'CASE {case_id}')
        for run in range(1, args.runs + 1):
            for variant in variants:
                body = {
                    'dictionary': {'example': source},
                    'language_code': args.language,
                    'quality_mode': 'standard',
                }
                if variant != 'baseline':
                    body['context_id'] = context
                    body['context_version'] = int(variant[1:])
                start = time.monotonic()
                status_code, result, error = None, {}, ''
                try:
                    status_code, result = request_json(
                        url + '/translate', body, api_key=api_key,
                    )
                except (URLError, OSError, ValueError) as exc:
                    error = str(exc)
                if status_code != 200:
                    error = error or str(result.get('detail', 'HTTP error'))[:300]
                translation = result.get('dictionary', {}).get('example', '')
                fallback = result.get('fallback')
                actual_version = result.get('context_version')
                valid_version = (actual_version is None if variant == 'baseline'
                                 else actual_version == int(variant[1:]))
                ok = (status_code == 200 and not fallback and
                      isinstance(translation, str) and bool(translation)
                      and valid_version)
                row = {
                    'case_id': case_id, 'run': run,
                    'source': source, 'context': context,
                    'language_code': args.language, 'variant': variant,
                    'requested_version': '' if variant == 'baseline'
                    else int(variant[1:]),
                    'actual_version': actual_version,
                    'translation': translation, 'fallback': fallback,
                    'http_status': status_code, 'error': error,
                    'elapsed_ms': int((time.monotonic() - start) * 1000),
                    'status': 'OK' if ok else 'REVIEW',
                }
                rows.append(row)
                print(f'  run={run} {variant}: {row["status"]} '
                      f'HTTP={status_code} fallback={fallback} '
                      f'ms={row["elapsed_ms"]}')

    output = Path('localizer-benchmark-results')
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f'professional_v3_ab_{args.language}_{stamp}'
    csv_path, json_path = output / f'{name}.csv', output / f'{name}.json'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    print(f'Saved rows={len(rows)} to:')
    print(csv_path)
    print(json_path)
    print('HTTP success is NOT semantic correctness. Review outputs manually.')


if __name__ == '__main__':
    main()
