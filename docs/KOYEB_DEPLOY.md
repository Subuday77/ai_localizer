# Koyeb always-on deployment — AI Localizer API

This runbook migrates **only the FastAPI backend** from Render to Koyeb.
The Angular frontend stays on Render Static Site until the new API is proven healthy.
Do **not** disable the old Render API before the cutover.

## Initial Koyeb service

1. Sign in to [Koyeb](https://app.koyeb.com/) and connect your GitHub account.
   Grant Koyeb access to the `Subuday77/ai_localizer` repository.
2. Choose **Create Web Service** → **GitHub** → `Subuday77/ai_localizer`.
3. For the first smoke test choose branch `feature/koyeb-always-on-deploy-20261111`.
   Select **Buildpack**; `requirements.txt` is already in the repository root,
   and the `Procfile` starts Uvicorn with Koyeb's `PORT`.
   No Dockerfile is required.
4. Set region **Frankfurt**. Select **paid Eco Micro** (0.25 vCPU, 512 MB);
   do **not** select the Free instance, which automatically sleeps when idle.
5. Set **one running instance** (minimum 1, maximum 1). Keep **scale-to-zero
   disabled**; start with automatic GitHub deploys disabled until verification.
6. Expose an HTTP port (recommended `8000`, route `/` to that port).
   Koyeb sets `PORT` automatically from the exposed port. Use an HTTP
   health check at `/health`, not `/`.
7. Add environment values (secrets should be stored as **Koyeb Secrets**):

   | Variable | Value |
   | --- | --- |
   | `OPENAI_API_KEY` | **Koyeb Secret** containing the current OpenAI API key |
   | `OPENAI_MODEL_ID` | `gpt-6-luna` |
   | `OPENAI_FALLBACK_MODEL_ID` | `gpt-5.4-mini` |
   | `OPENAI_REASONING_EFFORT` | `none` |
   | `CORS_ORIGINS` | `["https://ai-localizer.onrender.com"]` |

   Keep existing defaults for the other `OPENAI_*` settings initially.
   Do **not** copy Render dashboard credentials, the `.env` file, or the
   OpenAI key into GitHub. The public URL is provided by Koyeb after deployment.

## Smoke test **before** cutover

- Koyeb deployment status is **Healthy** (and remains healthy without traffic).
- Open `https://<actual-koyeb-domain>/health` and expect HTTP 200.
- Open `https://<actual-koyeb-domain>/docs` to confirm the API is loaded.
- POST `/translate` with a single short German/Russian translation;
  verify HTTP 200, `fallback: false`, translated output and successful model logs.
  This probe makes a small billable OpenAI API call.
- Test at least one ~30-key batch in Hebrew using the existing integration;
  check placeholders and translation completeness.
- Test browser CORS from `https://ai-localizer.onrender.com`, and review Koyeb
  logs for request failures or memory restarts.

## Cutover — only after verification

1. Change `API_BASE_URL` in the **separate** `Subuday77/ai_localizer_front`
   repository (`src/app/config/api.config.ts`) to the **verified actual** Koyeb
   HTTPS origin. Deploy the Render Static Site and verify /languages and /translate.
2. For AI Quiz Game, update `AI_LOCALIZER_BASE_URL` in its backend environment
   to the **same verified** Koyeb HTTPS origin (without a `/translate` suffix).
   Do not change any question-generation OpenAI settings.
3. Monitor both consumers, especially cold-start latency, 5xx/429 responses,
   placeholder validation and failed translations.
4. When both clients have used Koyeb successfully, suspend the old Render
   `ai-localizer-api` web service. Keep its config available for a rollback.
   The Render static frontend does **not** need to move.

## Rollback

- Restore the previous Render API origin
  `https://ai-localizer-api.onrender.com` in `ai_localizer_front` and in the
  Quiz backend's `AI_LOCALIZER_BASE_URL`.
- Resume Render `ai-localizer-api` if it was suspended.
- Never delete the Render service or revoke its API key before an end-to-end
  Koyeb verification.

## Cost and availability

Koyeb Eco Micro lists approximately $2.68/month at continuous operation,
excluding any other billed services or usage. The actual Koyeb checkout screen
is authoritative. OpenAI usage is billed separately.

The Koyeb **Free** instance also sleeps; using it would recreate the problem.
