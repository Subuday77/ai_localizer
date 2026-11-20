# Localizer 2.0 — v3 semantic disambiguation experiment

## Boundary

The product still localizes **English source UI dictionaries** for websites to
the requested target language, optionally using one published professional
context. No source language autodetection, crawling, URL mapping, or new
translation endpoint. The existing v1 and v2 profiles remain addressable.

## Hypothesis

The v2 professional-role prompt regressed on a strong counterexample:
"The conductor checked the tickets." became "Дирижёр проверил билеты." in
the Music context. It also failed to interpret "bus" as an electrical bus and
"note" as a musical note in short, ambiguous strings.

v3 does not add word-level mappings. It adds one versioned profile field:
`interpretation_method="evidence_before_domain"`. In a single LLM request
and the same JSON-array response contract, the model is asked to apply:

1. **Explicit English evidence first.** Identify if the full source string
   strongly rules in a particular sense through its action, objects,
   collocations, and semantic relationships, even when that sense conflicts
   with the website domain.
2. **Professional domain as default when still ambiguous.** An action or
   adjective compatible with multiple senses is *not* decisive evidence for
   the everyday sense. For such labels, use the selected site's profession
   to pick the relevant sense.
3. **Target-language rendering.** Translate the selected sense into natural
   terminology of the requested target language. Keep the original UI
   dictionary structure, placeholders, and no-explanation JSON format.

An optional independent review prompt also follows the v3 decision policy.
No extra model request, word-specific rule, or rule-based classifier is added.

## Versioning and deployment

The v3 publisher copies all six **published v2** profiles and language-pair
rules into immutable version-3 snapshots. All v1/v2 profile data are untouched,
and parent version 3 is pinned for children. It refuses to overwrite an
existing different v3 or publish over unexpected catalog versions.

For the **local PyCharm dev environment**:

```powershell
git pull --ff-only
python -m scripts.publish_semantic_profiles
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8081
```

No new Alembic migration is needed for v3 (the domain-profile column was added
in 0002 and is already present in a working v2 installation). A fresh database
requires `alembic upgrade head`, `python -m scripts.seed_contexts`, then
`python -m scripts.publish_professional_profiles` **before** v3 publication.
Do not run the v2 publisher once v3 is active. This feature is NOT deployed
to the VPS, Render, or the legacy master branch.

## Live A/B test (requires local API and configured OpenAI key)

```powershell
python -m scripts.benchmark_semantic_versions
```

The script checks that v3 is published, then calls `baseline`, `v1`,
`v2`, and `v3` for the **same six English source strings** and target
`ru` (24 live translation requests). Results go to a JSON file and a
UTF-8-BOM CSV in `localizer-benchmark-results`.

Optional extended negative controls include a clearly musical conductor
despite Railway context, a clearly handwritten note despite Music context,
an explicit passenger bus despite Electronics context, and a Railway
track-removal action:

```powershell
python -m scripts.benchmark_semantic_versions --extended
python -m scripts.benchmark_semantic_versions --extended --language de
python -m scripts.benchmark_semantic_versions --extended --language he
python -m scripts.benchmark_semantic_versions --runs 3
```

Output requires **human semantic review**. HTTP 200 and fallback=false
do not establish correctness. One run is exploratory, not a statistical
evaluation; repeated attempts assess variance.

## Decision gate

Treat v3 as an experiment rather than an automatic upgrade in quality. At a
minimum, it should disambiguate at least some previously failed short
professional labels and must **not** introduce the Music "ticket conductor"
regression or break other explicit counterexamples.

If the same short ambiguities persist, or if explicit counterexamples fail,
do **not** make v4 by tuning synonyms in this prompt. Switch the design:
consider a separately validated *semantic disambiguation stage* operating
on coherent UI groups, or a structured website-domain description with
page/section-level localization context. Evaluate additional API cost and
error modes before implementation.

Keep v1/v2/v3 pinned for reproducible comparison.
