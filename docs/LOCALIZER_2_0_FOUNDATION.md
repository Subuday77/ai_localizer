# Localizer 2.0 — implementation status (14 November 2026)

This branch is **not a production release**. It is based on `feature/vps-public-demo-20261114`, preserving the working Render Static Site → VPS public demo, the protected `/translate` API, and current auth/fallback behavior. Do not merge it into legacy `master` or deploy it directly over the VPS service until the remaining gates are met.

## Delivered in this branch

- PostgreSQL context storage with Alembic migration `0001_contexts`, optional database settings, and separate `compose.2.0.yaml` overlay. Current `compose.yaml` does not enable PostgreSQL, so existing demo deployments remain unchanged.
- `contexts` and `context_versions` tables. Versions carry a **pinned parent context + parent version**, rule/glossary JSON, style data, publication state and public visibility of the root context.
- Read-only `GET /contexts` with **published public** context metadata; not dependent on AI calls.
- Additive `POST /translate` fields `context_id` and `context_version`. Both are optional; requests without them take the legacy translation path. Unknown/unpublished IDs and versions return HTTP 422; unconfigured DB yields 503 for context requests. The resolved ID/version are included in the result.
- Contextual disambiguation and terminology guidance passed **only** with an explicitly selected context. Relevant rules are filtered to the source→target language pair. A context must never force a mechanical substitution against the meaning of the sentence.
- Five seed contexts: `technology`, `electronics`, `railway`, `music`, `software`. These are examples, **not** the final 50–100-context library.
- `quality_mode` accepts `standard` and `verified`. The latter is now implemented as an **opt-in, internal-only** quality workflow: independent controller model, validated structured issues, four-level severity scoring, bounded single repair, and independent recheck. Without explicitly configured model, scores and verified-only API key grants, it fails closed; it never silently downgrades to `standard`.

## How to initialize the experimental DB (not the production deployment yet)

1. Prepare a unique strong password in the server-side `.env` as `LOCALIZER_POSTGRES_PASSWORD`; do not commit secrets.
2. Review the overlay, VPS free disk space, backup destination, and permissions. There is no independently verified off-VPS PostgreSQL restore procedure yet.
3. On an isolated environment only, start the dedicated DB using the overlay:

   ```bash
   docker compose -f compose.yaml -f compose.2.0.yaml up -d db
   ```

4. Apply migrations and seed from a temporary API container that uses the same overlay:

   ```bash
   docker compose -f compose.yaml -f compose.2.0.yaml run --rm --build api alembic upgrade head
   docker compose -f compose.yaml -f compose.2.0.yaml run --rm api python -m scripts.seed_contexts
   ```

5. Only after database/backup and service validation, start the API with this overlay:

   ```bash
   docker compose -f compose.yaml -f compose.2.0.yaml up -d --build api
   ```

**Important:** The overlay currently does not modify Caddy. `GET /contexts` will stay inaccessible from the public Internet until a deliberately scoped Caddy rule is added. The primary `POST /translate` remains blocked from the public Internet by Caddy; do not expose it anonymously. Confirm Quiz internal API usage before changing that behavior.

## Internal Verified QA — configuration and behavior

The feature does not pick a paid controller model or invent severity weights. The operator must explicitly provide:

```dotenv
LOCALIZER_QA_VERIFIED_ENABLED=true
LOCALIZER_QA_CONTROLLER_MODEL_ID=<independent-model-id>
LOCALIZER_QA_VERIFIED_API_KEY_HASHES=["<sha256-of-internal-verified-token>"]
LOCALIZER_QA_PENALTY_WEIGHTS={"critical":<int>,"high":<int>,"medium":<int>,"low":<int>}
LOCALIZER_QA_REPAIR_THRESHOLD=<positive-int>
```

The values above are **placeholders** and should not be applied without calibration. An internal bearer key must belong to both the ordinary API allowlist and the separate Verified allowlist; setting a browser-side flag is insufficient. The Controller uses a dedicated auditor system instruction, its own model ID and no controller fallback to a Translator model. Its JSON report identifies the source-array index, issue category, severity, evidence and contextual rule. The orchestrator deduplicates issues (keeping the most serious duplicate), calculates scores and enforces Critical/High hard-stops and the configured Medium threshold. Low issues are report-only. Verified issues are never logged as raw text.

**Provisional HTTP contract, subject to final API approval:** a Verified PASS returns the translated dictionary and `quality_report.status=pass`; repeated QA failure returns HTTP 422 with `detail.status=verification_failed` and report but **no translation dictionary**. Unavailable controller/translator returns HTTP 503. Regular Standard mode is unaffected. There are no more than two logical Translator passes and two Controller reviews. Real AI QA has not yet been executed.

## Still required for the agreed 2.0 scope

- Choose and verify the Controller model B and calibrate scoring weights/threshold from a real evaluation corpus. Current implementation enforces **B is different from BOTH configured Translator IDs**, including fallback, but no production IDs or weights have been selected.
- Agree on model B, numerical penalty weights/threshold and failure-response policy; none are secretly chosen in this branch.
- Server-side client identities, API key ownership, per-client entitlements, usage recording/quotas, and isolation of private contexts. In the current implementation, a **separate SHA-256 verified-key allowlist** restricts the internal pilot; full tenant authorization is not yet built. Existing ordinary bearer hashes remain unchanged.
- 50–100 curated, versioned context entries (incrementally), regression corpus across languages and context counterexamples, tested immutable snapshot publishing tools.
- PostgreSQL restore/backup testing, migration rollback, security review, real translation/controller E2E and performance evaluation. CI now runs PostgreSQL 17 migration + seed + pinned-context checks, in addition to offline tests.
- Independent off-VPS backups and a tested restore path **before** storing meaningful customer data.
- Safe promotion from VPS feature branch to a stable release branch; no change to production is performed by this PR.

## Test strategy

The offline suite tests the shipped migration (against SQLite), seed idempotence, public catalog filtering, parent pinning, language-pair filtering, explicit errors and backwards compatibility. QA tests cover distinct Translator/Controller IDs, four-severity scoring and dedup, low-only PASS, repair/recheck, repeated FAIL, verified-only key grants and refusal to return unverified candidates. An additional GitHub Actions job exercises the migrations and seed against real PostgreSQL 17. Real OpenAI context accuracy and costs remain unverified.
