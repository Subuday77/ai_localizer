# Localizer 2.0 — implementation status (14 November 2026)

This branch is **not a production release**. It is based on `feature/vps-public-demo-20261114`, preserving the working Render Static Site → VPS public demo, the protected `/translate` API, and current auth/fallback behavior. Do not merge it into legacy `master` or deploy it directly over the VPS service until the remaining gates are met.

## Delivered in this branch

- PostgreSQL context storage with Alembic migration `0001_contexts`, optional database settings, and separate `compose.2.0.yaml` overlay. Current `compose.yaml` does not enable PostgreSQL, so existing demo deployments remain unchanged.
- `contexts` and `context_versions` tables. Versions carry a **pinned parent context + parent version**, rule/glossary JSON, style data, publication state and public visibility of the root context.
- Read-only `GET /contexts` with **published public** context metadata; not dependent on AI calls.
- Additive `POST /translate` fields `context_id` and `context_version`. Both are optional; requests without them take the legacy translation path. Unknown/unpublished IDs and versions return HTTP 422; unconfigured DB yields 503 for context requests. The resolved ID/version are included in the result.
- Contextual disambiguation and terminology guidance passed **only** with an explicitly selected context. Relevant rules are filtered to the source→target language pair. A context must never force a mechanical substitution against the meaning of the sentence.
- Five seed contexts: `technology`, `electronics`, `railway`, `music`, `software`. These are examples, **not** the final 50–100-context library.
- `quality_mode` accepts `standard` and `verified`. The latter explicitly returns HTTP 501 on this foundation branch; never silently downgrades a request to `standard`.

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

## Still required for the agreed 2.0 scope

- Real independent Quality Controller on a **different model ID from BOTH configured Translator models**, including fallback; structured reports, severity/penalty scoring and one bounded repair + recheck.
- Agree on model B, numerical penalty weights/threshold and failure-response policy; none are secretly chosen in this branch.
- Server-side client identities, API key ownership, entitlements, usage recording/quotas, and isolation of private contexts. Existing SHA-256 bearer hashes remain unchanged for the current deployment.
- 50–100 curated, versioned context entries (incrementally), regression corpus across languages and context counterexamples, tested immutable snapshot publishing tools.
- Live PostgreSQL CI/integration testing, migration rollback/restore, access control and performance evaluation.
- Independent off-VPS backups and a tested restore path **before** storing meaningful customer data.
- Safe promotion from VPS feature branch to a stable release branch; no change to production is performed by this PR.

## Test strategy

The new offline suite tests the shipped migration (against SQLite), seed idempotence, public catalog filtering, parent version pinning, language-pair prompt filtering, explicit 422/503 errors, and backward compatibility. PostgreSQL-specific behavior and real OpenAI context quality must be tested separately before release.
