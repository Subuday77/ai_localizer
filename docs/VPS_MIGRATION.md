# AI Localizer 1.x — standalone VPS migration runbook

Status: **preparation only**. The Render API is still the production target for the Render-hosted Angular frontend. Do not cut over merely because this file exists.

## Scope and isolation

- Backend repository: `Subuday77/ai_localizer` (Render currently deploys `master`).
- Frontend repository: `Subuday77/ai_localizer_front` (`main`), Render static site `https://ai-localizer.onrender.com`.
- Old API: `https://ai-localizer-api.onrender.com`.
- Target: a **new**, independent Compose project `ai-localizer`; do not edit `/srv/ai-quiz-game`, its Compose stack, database, networks, Caddy configuration, or volumes as part of the private staging setup.
- The current Localizer has no database. No PostgreSQL migration is required.
- Public API hostname **TBD**. Do not use the Quiz domain by assumption.

## Prerequisites (verify on VPS, do not assume)

1. SSH as `deploy` with key authentication; do not open root/password login.
2. Confirm Docker/Compose, available RAM/disk and that host loopback port `18080` is free; confirm firewall/reverse proxy state.
3. Confirm exact code revision and deployment plan. Prefer a reviewed merge/release tag for production.
4. Prepare a **separate** OpenAI credential if desired, with an appropriately limited project budget. Never place it in Git, logs or chat.
5. Before any public cutover: configure abuse protection / per-client rate limiting (including `POST /translate` and `POST /demo`). CORS is not access control.

Read-only VPS checks:

```bash
sudo docker compose version
free -h
df -h
sudo ss -ltnp | grep ':18080 ' || true
sudo ufw status verbose
```

## Private staging installation

The commands below are for a fresh installation, **not** for a destructive replacement of an existing directory. Keep Quiz running.

```bash
sudo mkdir -p /srv/ai-localizer
sudo chown deploy:deploy /srv/ai-localizer
cd /srv/ai-localizer
git clone https://github.com/Subuday77/ai_localizer.git source
cd source
git switch feature/vps-migration-isolated-20261114
cp .env.example .env
chmod 600 .env
```

Edit only `/srv/ai-localizer/source/.env` locally. Minimum settings:

```dotenv
OPENAI_API_KEY=<SET_ON_VPS_ONLY>
OPENAI_MODEL_ID=gpt-6-luna
OPENAI_FALLBACK_MODEL_ID=gpt-5.4-mini
CORS_ORIGINS=["https://ai-localizer.onrender.com"]
```

Keep other values from `.env.example` unless verification proves they need adjustment. Do not display the .env contents in shared terminal output. The `HOST` / `PORT` entries in .env are not Uvicorn's container listen settings; the Dockerfile explicitly binds to `0.0.0.0:8080` inside the container.

Validate and start **only Localizer**:

```bash
cd /srv/ai-localizer/source
sudo docker compose config --quiet
sudo docker compose build
sudo docker compose up -d
sudo docker compose ps
sudo docker compose logs --tail=80 api
```

The container is accessible through `127.0.0.1:18080` on the VPS, not publicly. No host ports 8000 or 5432 are used.

## No-cost smoke tests (inside VPS)

```bash
curl -fsS http://127.0.0.1:18080/health
curl -fsS http://127.0.0.1:18080/languages
curl -fsS -X POST http://127.0.0.1:18080/translate \
  -H 'Content-Type: application/json' \
  --data '{"dictionary":{"greeting":"Hello {name}"},"language_code":"en"}'
```

Expected: `/health` returns `{"status":"ok"}`; `/languages` returns a language mapping; the English pass-through returns the original dictionary and `fallback=false` without an OpenAI call.

Then make **one** small request to a non-English target (e.g. Russian) and check that `fallback=false`, the JSON structure is preserved and the live OpenAI key works. This incurs a model call. Compare with the old Render API before cutover. Also check `POST /analytics/pageview` returns 204, `/docs` loads, and the container remains healthy after a restart.

## Public HTTPS cutover (requires separate approval and verified host configuration)

1. Choose/verify a dedicated public DNS hostname for Localizer API; point it to the VPS and confirm resolution.
2. Integrate a dedicated HTTPS virtual host with the **existing** reverse-proxy setup without replacing Quiz routes. For Caddy, **only if it is the confirmed ingress**, the route would conceptually be:

```caddyfile
api-localizer.example.com {
    reverse_proxy 127.0.0.1:18080
}
```

This is an example domain, **not** an approved or configured address. Confirm TLS issuance, timeouts and request size limits (500 dictionary entries are allowed by the current schema). Apply effective rate limiting before opening the expensive endpoints to the Internet. Do not expose port 18080 publicly.

3. From a browser, check HTTPS `GET /health`, `GET /languages`, preflight OPTIONS from `https://ai-localizer.onrender.com`, and one real translation through the new API. Verify no mixed-content/CORS failure.
4. Only then update `ai_localizer_front/src/app/config/api.config.ts` to the actual new HTTPS API origin and deploy the Render static frontend.
5. Perform an end-to-end translation from the unchanged public frontend. Test `/analytics/pageview` separately.
6. Keep the Render backend online as a rollback target until the replacement is stable. Only later disable old API deployment and update monitoring/alerts pointing at Render.

## Rollback

If the migrated API fails after switching the frontend, **first restore the frontend's previous API URL** (`https://ai-localizer-api.onrender.com`) and redeploy the static site. Leave the Render API running until that rollback is verified. Investigate the VPS container separately; do not change Quiz containers or volumes.

Avoid `docker compose down -v`, `git reset --hard`, global Docker prune commands and blanket firewall changes.

## Release gates

- [ ] Verified target VPS capacity and current Docker/network/reverse proxy configuration
- [ ] Private staging container healthy, correct models/settings, no credentials in image or logs
- [ ] Offline tests, Docker build and no-cost smoke pass
- [ ] One real OpenAI translation and fallback behavior validated
- [ ] Public HTTPS hostname verified, plus appropriate API abuse protection
- [ ] CORS preflight/production frontend E2E pass
- [ ] Rollback to Render tested
- [ ] Only then retire Render backend

## After migration

Localizer 2.0 is a **separate change**: `GET /contexts` returns predefined versioned translation contexts, and `POST /translate` accepts an optional `context_id`. No website crawling, URL mapping, auto-classification or additional analyzer model.
