# AI Localizer 2.0: multilingual RAG prototype

Experimental only in feature/localizer-2-contexts-foundation-20261114 (draft PR #15).
Not enabled on public translation routes, not a production migration, not deployed.

## Concept-first, language-neutral model

Bundled data: data/rag_demo_concepts.json. There are 160 English descriptions of
domain-specific concepts, plus term variants in five languages (English,
German, French, Spanish, Russian) where known. The concept, its professional
domain and term spellings are independent of the target language. Missing
target-language terms never block selecting or translating a concept.

All bundled definitions are freshly written DEMO data: marked
provider=prototype_original and verified=false. They are not authoritative
Wiktionary or IATE excerpts, are not a licensed professional dictionary,
and are not the gold targets from the benchmark.

The read-only index is capped at 256 KiB and 200 entries. It uses Unicode
word/phrase boundaries and exact spelling: base does not become bass and
moon shine does not become moonshine. It returns up to twelve candidates
and never selects a final sense by itself.

## Zero-cost inspection (no API keys needed)

Run from repository root:

    git pull --ff-only
    python -m scripts.rag_lookup_demo --text "pitching" --target-language fr
    python -m scripts.rag_lookup_demo --text "Stampfen" --language de --target-language es
    python -m scripts.rag_lookup_demo --text "The bus failed."
    python -m pytest -q tests/test_rag_evidence.py

The language flag specifies SOURCE text language. The optional target language
only displays curated example equivalents already present; it never requests
a translation. Other BCP-47 languages can be added to the terms map.

## Scored model benchmark (OpenAI charges apply)

As before, needs local OPENAI_API_KEY and LOCALIZER_DATABASE_URL. Six
published v3 contexts are read from PostgreSQL; fourteen experimental
catalog entries are in-memory. Maritime, wildlife and astronomy exist
only as test-site fixtures. No PostgreSQL writes.

    python -m scripts.benchmark_scored_router --rag-mode off --case u_pitching
    python -m scripts.benchmark_scored_router --rag-mode shadow --case u_pitching
    python -m scripts.benchmark_scored_router --rag-mode assist --case u_pitching
    python -m scripts.benchmark_scored_router --compare-rag --case a_note_four_beats
    python -m scripts.benchmark_scored_router --rag-mode assist

off is the UNCHANGED default behavior; shadow logs lexical evidence only;
assist provides evidence to both Thinker and conditional Selector. The Thinker
still does NOT receive the actual website domain or benchmark gold labels.
Numeric Python thresholds remain unchanged, but assist mode may VETO a
low-confidence numeric auto-selection on an ambiguous, domain-relevant word,
routing it to the Selector. RAG never forces a positive automatic selection.

compare-rag runs off and assist for the same case and writes both results
with a mode label. These calls can cost more. Model variability means the
two-arm result is not a controlled proof of RAG's effect. The original 37
cases have been viewed during development, so they are now DEVELOPMENT cases;
use a genuinely new holdout set for strong quality claims.

Reports (JSON + Markdown) are checkpointed after every case/mode in
localizer-benchmark-results/. Statistics include selected/unresolved/errors,
Python and Selector decisions, RAG hits, gate escalations, external lookup
count, model calls, and elapsed time. A successful selection is NOT a
verified correct interpretation; review the full meaning against gold.

## Opt-in online source: individual dictionary terms, not dumps

    python -m scripts.rag_lookup_demo --text "pitching" --online
    python -m scripts.benchmark_scored_router --rag-mode assist --rag-online --case u_pitching

The API at https://freedictionaryapi.com/ is backed by Wiktionary and
documents individual-term lookups in many languages. The prototype sends
at most two short dictionary TERM queries per source string, never the entire
source sentence; limits each response to 128 KiB and returns at most five
definitions per query. For English only, a 200 response with no definitions,
or an inflected-form definition like "present participle and gerund of pitch",
can trigger a SECOND query with a conservative, unverified lemma guess:
"heaving down" -> "heave down", "pitching" -> "pitch". This consumes
the existing two-query budget, never adds a third. Both the original and
guessed forms remain visible, marked lookup_relation=heuristic_lemma.
Other source languages are left untouched rather than applying English
suffix rules to German, French, Spanish or Russian.
An in-memory cache avoids duplicate lookups WITHIN ONE RUN;
it does not persist across runs or save external data in PostgreSQL.
Network/parse errors become an empty optional evidence set rather than
failing the localizer. Do not opt into external requests for confidential
customer text without separate privacy authorization.

Dictionary content is CC BY-SA 4.0. Every returned record preserves
source URL, license, provider and service URL for user-visible attribution.
Examples and third-party quotations are deliberately omitted. The provider
requests attribution of FreeDictionaryAPI.com and the original Wiktionary page.
If later publishing/redistributing this material or adaptations, satisfy
attribution/share-alike and other license obligations before deployment.

Official sources:
- https://freedictionaryapi.com/
- https://en.wiktionary.org/wiki/Wiktionary:Copyrights
- https://creativecommons.org/licenses/by-sa/4.0/

## Caveats and future work

This is lexical RAG, not pgvector, a production terminology database, full
translation, or a general semantic search engine. Exact matching will miss
some inflections/idioms; evidence can be incomplete or wrong. No corpus is
bulk downloaded. A future PostgreSQL store should be independently versioned,
track per-record provenance and licensing, and not alter the fixed context
catalog until manually approved and benchmarked.

A 200 response with no usable dictionary senses is logged as
no_definitions, which is not equivalent to a network error or proof that
the expression does not exist. The live API can return definitions for
individual verbs but not specialized multiword entries, so a lemma
fallback is only an additional lexical clue, not validation of an idiom.
