# VPS Localizer 1.x — temporary authenticated API staging

Status: **code prepared in draft PR #13, not yet deployed**. VPS currently serves HTTPS `GET /health` only, with all other routes blocked by Caddy. The old Render API and frontend remain active.

## Current confirmed infrastructure (14 Nov 2026)

- VPS public IP: `202.61.249.2`
- Public hostname: `v2202611457040557919.quicksrv.de`
- Caddy 2.6.2, Let's Encrypt HTTPS validated from browser and VPS
- API: `/srv/ai-localizer/source`, Compose `ai-localizer`, private `127.0.0.1:18080`
- Public Caddy `GET /health` works; `/translate` currently returns 403
- Quiz remains in its own Compose project and is not part of this migration
- Do not assume Localizer has been cut over just because HTTPS is working

## Temporary API key mode for server-to-server clients

The VPS Compose file forces `LOCALIZER_REQUIRE_API_KEY=true`. The API process refuses to start without at least one SHA-256 hash in `LOCALIZER_API_KEY_HASHES`, a JSON list of lowercase 64-character hashes.

- Paid endpoints `POST /translate` and `POST /demo` require `Authorization: Bearer <KEY>`.
- `GET /health` and `GET /languages` are not authenticated inside the API process.
- Do not log or commit raw keys. Do not put a server-to-server key in Angular/JavaScript shipped to browsers.
- CORS does **not** authenticate callers.
- API keys are temporary 1.x infrastructure, not the Localizer 2.0 user/plan/subscription database.

### Generate a private staging key on VPS (one time)

With `deploy` logged into the VPS:

```bash
umask 077
mkdir -p /srv/ai-localizer/credentials
python3 - <<'PY'
from pathlib import Path
import hashlib
import secrets

key = 'loc_sk_' + secrets.token_urlsafe(32)
path = Path('/srv/ai-localizer/credentials/internal-test.key')
if path.exists():
    raise SystemExit('Refusing to replace existing key')
path.write_text(key + '\n', encoding='utf-8')
path.chmod(0o600)
print('API_KEY_SHA256=' + hashlib.sha256(key.encode()).hexdigest())
PY
```

This prints only the hash, not the secret. The raw key is in `/srv/ai-localizer/credentials/internal-test.key` (mode 0600, outside the Git repository).

Edit `/srv/ai-localizer/source/.env` using `nano`, **without pasting the file into chat**:

```dotenv
LOCALIZER_API_KEY_HASHES=["<API_KEY_SHA256 printed above>"]
```

The flag `LOCALIZER_REQUIRE_API_KEY` is forced to true by `compose.yaml`, regardless of any local .env value. Leave the existing OpenAI credentials unchanged.

### Update only the Localizer container

After reviewed code has been merged or the exact approved branch revision has been verified:

```bash
cd /srv/ai-localizer/source
git status --short
git fetch origin
git pull --ff-only origin feature/vps-migration-isolated-20261114
sudo docker compose config --quiet
sudo docker compose build
sudo docker compose up -d
sudo docker compose ps
curl -fsS http://127.0.0.1:18080/health
```

These commands do not deploy or reconfigure Quiz, but rebuild/replace the Localizer container. Do **not** run these before adding key hash to .env; authentication now fails closed on startup.

### Private authentication smoke

```bash
# Missing key => 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:18080/translate \
  -H 'Content-Type: application/json' \
  --data '{"dictionary":{"hello":"Hello"},"language_code":"en"}'

# Correct key => 200, no paid model call (English pass-through)
TOKEN="$(cat /srv/ai-localizer/credentials/internal-test.key)"
curl -s -i -X POST http://127.0.0.1:18080/translate \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"dictionary":{"hello":"Hello"},"language_code":"en"}'
unset TOKEN
```

Caddy should *still* return 403 on public `/translate` throughout these tests. Do not allow this route through Caddy until API authentication has passed, request-size/rate/spend controls are decided and the service clients are ready.

## Public Angular frontend and migration cutover

Angular static code at `https://ai-localizer.onrender.com` currently calls `https://ai-localizer-api.onrender.com`; do not put any persistent secret in it. This private-key mode is for trusted backends such as Quiz and Mira, not anonymous website users. The public frontend needs a separate abuse-controlled, non-secret browser access flow (or later Localizer 2.0 user sign-in) before it can switch to the VPS API. Until then, keep the old Render API and leave public translation blocked in Caddy.

## Rollback

Keep Render API intact. If the new container fails, inspect only `sudo docker compose logs --tail=100 api` from `/srv/ai-localizer/source`. Do not modify Quiz services or delete Docker volumes. For a code rollback, use a known prior release image/commit plus its compatible deployment config, after identifying the exact version; do not indiscriminately reset or prune.
