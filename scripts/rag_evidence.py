"""Small, multilingual, concept-first lexical evidence index for RAG experiments.

No database migration, crawler or large dictionary download. The bundled
160 demo concepts are independently authored and marked UNVERIFIED:
they provide lexical hypotheses, not validated dictionary authority.

Local retrieval supports arbitrary language codes when the record contains
term spellings for that language. Definitions remain in English because
concepts are independent of a target translation language.

Optional online enrichment fetches individual terms from
https://freedictionaryapi.com/api/v1/entries/{language}/{word}
and consumes only definitions, no quotations or examples. Content is
CC BY-SA 4.0 from Wiktionary: provenance, license, and source URLs must
travel with returned text; NEVER promote entries silently to the proprietary
published context catalog. This is experiment-only, opt-in.
"""

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]
SEED_FILE = ROOT / 'data' / 'rag_demo_concepts.json'
SOURCE_LABEL = 'FreeDictionaryAPI.com / Wiktionary (CC BY-SA 4.0)'
LICENSE_URL = 'https://creativecommons.org/licenses/by-sa/4.0/'
SERVICE_URL = 'https://freedictionaryapi.com/'
STOP_WORDS = frozenset({
    'the', 'a', 'an', 'of', 'is', 'was', 'are', 'on', 'in', 'to',
    'with', 'for', 'and', 'or', 'as', 'at', 'by', 'has', 'have',
    'their', 'this', 'that', 'from', 'be', 'one', 'two',
})


def normalize(text: str) -> str:
    """Unicode-preserving normalization; never collapse spaces inside words."""
    return unicodedata.normalize('NFKC', text).casefold().strip()


def matched_term(source: str, spelling: str) -> bool:
    term = normalize(spelling)
    if len(term) < 3 or len(source) > 2000 or len(term) > 120:
        return False
    # Unicode-aware token boundaries. NFKC does not turn "moon shine"
    # into "moonshine", or "base" into "bass".
    rx = r'(?<!\w)' + re.escape(term).replace(r'\ ', r'\s+') + r'(?!\w)'
    return re.search(rx, normalize(source), flags=re.UNICODE) is not None


class SmallConceptIndex:
    """Read-only concept index. No hidden target-language assumption."""

    def __init__(self, path: Path = SEED_FILE, *, max_file_bytes: int = 262144):
        path = Path(path)
        if path.stat().st_size > max_file_bytes:
            raise ValueError('RAG demo corpus exceeds the configured 256 KiB cap')
        document = json.loads(path.read_text(encoding='utf-8'))
        if document.get('schema_version') != 1:
            raise ValueError('Unsupported demo RAG lexicon schema')
        items = document.get('records')
        if not isinstance(items, list) or not 1 <= len(items) <= 200:
            raise ValueError('Demo concept count must be within 1..200')
        ids = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError('RAG concept row must be an object')
            ids.append(item['concept_id'])
            if not item['definition_en'] or not item['terms'].get('en'):
                raise ValueError('Concept must have an English description and terms')
            if not item.get('provenance'):
                raise ValueError('Missing RAG provenance')
            if any(
                not isinstance(words, list) or not words or
                any(not isinstance(t, str) or not t.strip() for t in words)
                for words in item['terms'].values()
            ):
                raise ValueError('Invalid language-specific terms')
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate concept IDs in RAG data')
        self.records = tuple(items)
        self.file_bytes = path.stat().st_size

    def search(self, source: str, *, source_language: str = 'en',
               limit: int = 12) -> list[dict]:
        if not 1 <= limit <= 24:
            raise ValueError('RAG limit must be 1..24')
        if not re.fullmatch(r'[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*',
                            source_language):
            raise ValueError('Invalid BCP-47 language tag')
        lang = source_language.lower().split('-')[0]
        hits = []
        for concept in self.records:
            terms = concept['terms'].get(lang, [])
            matching = [t for t in terms if matched_term(source, t)]
            if not matching:
                continue
            match = max(matching, key=lambda x: (len(normalize(x).split()), len(x)))
            whole = normalize(match) == normalize(source)
            # Pure lexical ranking, not a posterior probability or site score.
            rank = 100 if whole else 60 + min(len(match.split()) * 8, 24)
            hits.append({
                'concept_id': concept['concept_id'],
                'matched_term': match,
                'domain_id': concept['domain_id'],
                'definition_en': concept['definition_en'],
                'retrieval_score': rank,
                'provenance': dict(concept['provenance']),
            })
        return sorted(hits, key=lambda e: (
            -e['retrieval_score'], -len(e['matched_term']), e['concept_id'],
        ))[:limit]


def compact_for_model(hits: list[dict], *, limit: int = 12) -> list[dict]:
    """No target-language equivalences or gold labels enter model prompts."""
    result = []
    for hit in hits[:limit]:
        entry = {key: hit[key] for key in (
            'concept_id', 'matched_term', 'domain_id', 'definition_en',
            'provenance',
        )}
        if hit.get('definition_source_language'):
            entry['definition_source_language'] = hit['definition_source_language']
            entry['language'] = hit.get('language', 'unknown')
        result.append(entry)
    return result


def get_online_query_terms(source: str, local_hits: list[dict], *,
                           max_terms: int = 2) -> list[str]:
    """Bounded, privacy-conscious lexical queries; never send entire sentence."""
    if not 0 <= max_terms <= 2:
        raise ValueError('Online term-query budget must be 0..2')
    ordered = []
    for hit in local_hits:
        token = hit['matched_term']
        if token not in ordered and len(token) <= 64:
            ordered.append(token)
    if not ordered:
        # Conservative fallback to isolated, not user-specific long phrases.
        words = re.findall(r"[^\W\d_][\w'-]*", source, flags=re.UNICODE)
        ordered = sorted(
            {w for w in words if 4 <= len(w) <= 32 and w.casefold() not in STOP_WORDS},
            key=lambda w: (-len(w), w),
        )
    return ordered[:max_terms]


async def fetch_wiktionary_senses(
    client: httpx.AsyncClient, word: str, *,
    source_language: str = 'en', max_bytes: int = 131072,
    max_senses: int = 5,
) -> list[dict]:
    """One opt-in dictionary API GET, with strict size/term and schema limits.

    External sense definitions are NOT assigned a context_id: dictionary
    entries alone cannot establish the actual domain of a whole sentence.
    """
    if not 2 <= len(word) <= 64 or '\n' in word:
        raise ValueError('Invalid external lookup token')
    lang = source_language.lower().split('-')[0]
    if not re.fullmatch(r'[a-z]{2,3}', lang):
        raise ValueError('Invalid external language')
    if not 1 <= max_senses <= 8:
        raise ValueError('Sense limit must be 1..8')

    url = (
        f'https://freedictionaryapi.com/api/v1/entries/'
        f'{quote(lang, safe="")}/{quote(word, safe="")}'
    )
    # Narrow endpoint only, redirects disabled, bounded response.
    async with client.stream(
        'GET', url, params={'translations': 'false'},
        follow_redirects=False,
        timeout=8.0,
    ) as response:
        if response.status_code in (404, 429):
            return []
        response.raise_for_status()
        buf = bytearray()
        async for chunk in response.aiter_bytes():
            buf.extend(chunk)
            if len(buf) > max_bytes:
                raise ValueError('Wiktionary lookup exceeded 128 KiB cap')
    data = json.loads(buf)
    if not isinstance(data, dict) or not isinstance(data.get('entries'), list):
        return []
    provider = data.get('source') if isinstance(data.get('source'), dict) else {}
    source_url = provider.get('url')
    if not isinstance(source_url, str) or not source_url.startswith(
        ('https://en.wiktionary.org/', 'https://www.wiktionary.org/',
         'https://fr.wiktionary.org/', 'https://de.wiktionary.org/',
         'https://ru.wiktionary.org/', 'https://es.wiktionary.org/')
    ):
        source_url = 'https://en.wiktionary.org/wiki/' + quote(word, safe='')
    out = []
    for entry in data['entries'][:12]:
        if not isinstance(entry, dict):
            continue
        for sense in entry.get('senses', [])[:12]:
            if not isinstance(sense, dict):
                continue
            description = sense.get('definition')
            if not isinstance(description, str):
                continue
            description = ' '.join(description.split())[:320]
            if len(description) < 8:
                continue
            out.append({
                'concept_id': f'wiktionary:{lang}:{normalize(word)}:{len(out)+1}',
                'matched_term': word,
                'domain_id': 'unclassified',
                'definition_en': description if lang == 'en' else '',
                'definition_source_language': description if lang != 'en' else None,
                'language': lang,
                'retrieval_score': 52,
                'provenance': {
                    'provider': SOURCE_LABEL,
                    'license': 'CC BY-SA 4.0',
                    'source_url': source_url,
                    'service_url': SERVICE_URL,
                    'license_url': LICENSE_URL,
                    'verified': False,
                },
            })
            if len(out) >= max_senses:
                return out
    return out


async def search_with_optional_online(
    source: str, *, source_language: str, index: SmallConceptIndex,
    online: bool = False, client: httpx.AsyncClient | None = None,
    online_cache: dict | None = None,
    max_online_terms: int = 2, limit: int = 12,
) -> tuple[list[dict], int]:
    hits = index.search(source, source_language=source_language, limit=limit)
    count = 0
    if not online:
        return hits, count
    if client is None:
        raise ValueError('HTTP client required for online lookups')
    if online_cache is None:
        online_cache = {}
    for word in get_online_query_terms(source, hits, max_terms=max_online_terms):
        key = (source_language.split('-')[0].lower(), normalize(word))
        if key not in online_cache:
            try:
                online_cache[key] = await fetch_wiktionary_senses(
                    client, word, source_language=source_language,
                )
            except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                # Optional evidence must never break translation/benchmark.
                online_cache[key] = []
            count += 1
        hits.extend(online_cache[key])
    # Prefer exact/multiword curated records, then external definitions.
    hits.sort(key=lambda e: (
        -e['retrieval_score'], -len(e['matched_term']), e['concept_id'],
    ))
    return hits[:limit], count


def evidence_gate(senses: list[dict], decision: dict, hits: list[dict],
                  site_context_id: str) -> dict:
    """A conservative RAG veto only; never force an automatic sense selection.

    A matched ambiguous *term* alone cannot disprove a sense. If a term has
    multiple different corpus domains and numeric source dominance is weak,
    ask Selector to reconcile textual evidence with site context.
    Source-strong (<90 cutoff) readings and single-domain hits keep the
    original numeric routing decision.
    """
    if decision.get('call_selector') or not hits:
        return decision
    chosen = senses[decision['choice'] - 1]
    if chosen['source_score'] >= 90:
        return decision
    by_term = {}
    for hit in hits:
        by_term.setdefault(normalize(hit['matched_term']), set()).add(
            hit['domain_id']
        )
    ambiguous = any(len(domains) >= 2 for domains in by_term.values())
    if ambiguous and any(
        hit['domain_id'] == site_context_id for hit in hits
    ):
        return {
            'call_selector': True, 'choice': None,
            'reason': 'rag_ambiguous_term_requires_selector',
        }
    return decision
