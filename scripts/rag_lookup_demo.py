"""Inspect multilingual concept retrieval without OpenAI credits or PostgreSQL.

Examples:
    python -m scripts.rag_lookup_demo --text "The bus failed."
    python -m scripts.rag_lookup_demo --text "pitching" --target-language fr
    python -m scripts.rag_lookup_demo --text "Stampfen" --language de --target-language es
    python -m scripts.rag_lookup_demo --text "heaving down" --online

--online sends at most TWO isolated dictionary terms to a Wiktionary-derived
dictionary API (CC BY-SA 4.0; see docs/rag_prototype.md). It never sends
the entire user's sentence, downloads a corpus, or stores the response in
a published context catalog.
"""

import argparse
import asyncio
import json

import httpx

from scripts.rag_evidence import (
    SmallConceptIndex, search_with_optional_online,
)


async def lookup(text: str, source_language: str, target_language: str | None,
                 online: bool, limit: int) -> dict:
    index = SmallConceptIndex()
    async with httpx.AsyncClient() as client:
        hits, requests = await search_with_optional_online(
            text, source_language=source_language, index=index,
            online=online, client=client, limit=limit,
        )
    by_id = {item['concept_id']: item for item in index.records}
    target = target_language.lower().split('-')[0] if target_language else None
    for hit in hits:
        entry = by_id.get(hit['concept_id'])
        hit['available_target_terms'] = (
            entry['terms'].get(target, []) if entry and target else []
        )
    return {
        'query': text,
        'source_language': source_language,
        'target_language': target_language,
        'source': 'local_seed_and_optional_live_dictionary',
        'source_concepts': len(index.records),
        'source_file_bytes': index.file_bytes,
        'network_dictionary_requests': requests,
        'hits': hits,
        'note': (
            'Lexical evidence only: NO final sense selected, NO translation. '
            'Missing target terms are allowed; no language is assumed.'
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', required=True)
    p.add_argument('--language', default='en', help='Input language (BCP-47)')
    p.add_argument('--target-language', default=None,
                   help='Optional target language for available example terms')
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--online', action='store_true',
                   help='Opt-in maximum two individual Wiktionary API lookups')
    args = p.parse_args()
    result = asyncio.run(lookup(
        args.text, args.language, args.target_language,
        args.online, args.limit,
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
