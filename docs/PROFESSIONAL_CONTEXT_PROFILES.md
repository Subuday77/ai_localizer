# Professional specialization profiles — Localizer 2.0

## Product boundary

AI Localizer localizes an **English website UI dictionary** into a requested
target language. It is not a free-form multilingual translator and does not
infer source languages, crawl websites, or classify a context automatically.
The existing `POST /translate` request schema remains unchanged.

## Why a professional profile instead of more word substitutions

An ambiguous English UI label often has insufficient information on its own.
A client can explicitly select a versioned professional `context_id`. The
translator should interpret ambiguous labels through the professional domain
of that site **by default**, while preserving an unmistakably different
meaning when the English string itself makes one clear.

Every published v2 profile contains four **language-independent** fields:

- `specialization`: what kind of professional website translator the model is.
- `scope`: relevant subject matter and concepts, not fixed translations.
- `audience`: intended website readership.
- `conventions`: genre-specific translation priorities.

The translation prompt uses the most specific profile as its primary role,
inheriting broader parent expertise. It explicitly instructs the model to
write in the **requested target language**, not Russian by default. Optional
terminology rules and glossary entries remain available and are filtered by
English → target-language pair; they never override unambiguous meaning.
Placeholders, ordering, JSON output and other legacy translation invariants
are unchanged.

The `context_versions.domain_profile` column is a JSON/JSONB snapshot.
This profile data is version-pinned; the mutable catalog description is *not*
used to construct a model prompt. A NULL profile means the legacy v1 behavior.
Parent contexts are pinned to exact version numbers, e.g.:

- `technology@v1 → electronics@v1`: original prompts, preserved.
- `technology@v2 → electronics@v2`: new inherited expert profiles.

No new "bus → шина" or "note → нота" rules are introduced. Existing v1
language-specific hints remain present in v2 so that an A/B comparison
isolates the new domain assignment.

## Upgrade an already running local PostgreSQL installation (without Docker)

On the existing `feature/localizer-2-contexts-foundation-20261114` branch,
with the local Python venv activated and the correct `.env` configured:

```powershell
git pull --ff-only
# Stop the local FastAPI process before upgrading the schema.
alembic upgrade head
python -m scripts.seed_contexts
python -m scripts.publish_professional_profiles
```

Expected for an installation already containing all six v1 contexts:
`Starter contexts created: 0` and
`Professional context profiles activated: 6`.

The publisher is transactional and idempotent (repeated run reports 0).
It refuses to overwrite a different existing v2 snapshot or replace unknown
published versions. Existing v1 context versions remain directly addressable.

Start the API as before, e.g.:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8081
```

`GET /contexts` should now return the six contexts with
`"version": 2`. It never invokes OpenAI.

## A/B test on actual GPT-6 Luna (not yet performed by this change)

Run the same English website string against:

1. `context_id="electronics", context_version=1`: legacy hint approach.
2. `context_id="electronics", context_version=2`: professional role.
3. No `context_id`: unconstrained baseline.

Suggested ambiguous source string: `The bus failed.`.
Test different **target** languages with `language_code` values
`ru`, `de`, `he` (source stays English). Compare semantic suitability,
not exact wording. Likewise compare `The note was missing.` for `music`
and the military `fruit salad` example for `military`.

Include explicit counterexamples that must NOT be reinterpreted as
domain jargon (e.g. `The conductor checked the tickets.` in `music`).
A model can still make mistakes; a green HTTP response is not proof of
translation quality. No real-model A/B results are claimed here.

## Safety and rollout

This feature is committed only to the isolated Localizer 2.0 PR branch.
Nothing is deployed to VPS, Render, or the legacy master branch.
The opt-in `quality_mode=verified` and its independent model policy are
not changed; its review prompt now includes the same version-pinned
professional profile chain when present.
