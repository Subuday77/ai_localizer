# Render-hosted public demo → VPS Localizer

Status: PREPARED, NOT ACTIVATED. Do not disable the old Render API until
the real public frontend is verified on VPS.

## Target architecture
- Angular remains at https://ai-localizer.onrender.com (Render static).
- Caddy on existing VPS hostname v2202611457040557919.quicksrv.de forwards ONLY:
  GET /languages; POST and OPTIONS /public-demo/translate; existing GET /health.
- Do not publicly forward /translate, /demo, /docs or other API routes.
- Endpoint accepts target language ONLY and translates one fixed English UI
  dictionary defined in app/api/public_demo.py. Never browser-provided text.
- CORS_ORIGINS must include https://ai-localizer.onrender.com. CORS is not
  access control: enforced quotas and narrow route allowlist are critical.
- Default limits: 20 global requests per UTC day, 3 per IP per rolling hour,
  and two concurrent translations. Persistent SQLite counter stored on
  the localizer_demo_quota Docker volume, maintained across API recreations.
- Count each accepted call even if it falls back or is an English pass-through.
- The request quota is NOT an exact dollar cap. Model retries may incur costs.
  Set provider spend limits and monitor costs independently.
- Neither a paid Localizer key nor an OpenAI key may be embedded in Angular.

## Deployment preparation
Inspect the real Caddy configuration, DNS, TLS and existing Quiz routes FIRST.
Never replace Caddy configuration blindly or expose 18080 to the Internet.
On VPS in /srv/ai-localizer/source deploy branch
feature/vps-public-demo-20261114, build only Localizer API and test.
New public route is OFF by default.

To enable, edit VPS Localizer .env (never show secrets):
LOCALIZER_DEMO_ENABLED=true
LOCALIZER_DEMO_DAILY_LIMIT=20
LOCALIZER_DEMO_PER_IP_HOUR_LIMIT=3
CORS_ORIGINS=["https://ai-localizer.onrender.com"]

Validate Compose and restart only Localizer API; confirm healthy.
Make sure the image's UID 10001 can write its persistent SQLite quota volume.
If quota storage fails, API must fail closed with HTTP 503.

## Required verification
1. On VPS, POST /public-demo/translate JSON {"language_code":"en"}
   must return 200, full demo UI dictionary and fallback=false.
2. POST with arbitrary "dictionary" field must return 422.
3. POST /translate without a secret must remain 401 on localhost.
4. Validate Caddy configuration before reloading. Explicitly allow only
   GET /languages, OPTIONS and POST /public-demo/translate, GET /health;
   limit public request body to around 4 KiB.
5. Public POST /translate must continue to return 403.
6. CORS preflight with Origin: https://ai-localizer.onrender.com must allow
   the demo endpoint; public /languages must return 200.
7. ONLY THEN merge frontend branch feature/vps-api-public-demo-20261114
   into main, which Render auto-deploys.
8. Verify the actual static site translation, RTL and custom language.
9. Retire Render API only after the website is working and rollback is known.

## Rollback
Restore frontend production API URL to
https://ai-localizer-api.onrender.com and redeploy Render (leave old API
available during testing). Disable LOCALIZER_DEMO_ENABLED on VPS and remove
only the additional Caddy public-demo routing. Never reset Quiz DB, VPS
secrets, private Localizer network or the demo quota volume.
