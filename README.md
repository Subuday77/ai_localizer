# AI Localizer

FastAPI backend for UI dictionary localization using OpenAI Chat Completions. No database required.

## Structure

- app/main.py: FastAPI setup
- app/api/routes.py: documented translation, health and demo endpoints
- app/config.py: environment-based OpenAI model, HTTP and retry settings
- app/services/openai.py: OpenAI API, JSON parsing and validation, language recognition, adaptive retries and model fallback
- app/constants.py: language names and prompts
- app/schemas.py: request and response models
- tests/: offline tests with mocked model responses

## Render Environment Variables

Set OPENAI_API_KEY to your API key in the Render service's Environment tab. These additional variables have defaults and can be overridden:

| Key | Default |
| --- | --- |
| OPENAI_MODEL_ID | gpt-6-luna |
| OPENAI_FALLBACK_MODEL_ID | gpt-5.4-mini |
| OPENAI_BASE_URL | https://api.openai.com/v1 |
| OPENAI_REASONING_EFFORT | none |
| OPENAI_MAX_TOKENS | 2048 |
| OPENAI_MAX_TOKENS_CAP | 16384 |
| OPENAI_RETRIES | 5 |
| OPENAI_QUALITY_FAILURES_BEFORE_FALLBACK | 2 |
| OPENAI_TIMEOUT_SECONDS | 45 |
| OPENAI_RETRY_MAX_DELAY_SECONDS | 12 |
| OPENAI_RETRY_JITTER_SECONDS | 0.3 |
| OPENAI_RAW_RESPONSE_LOG_CHARS | 4000 |

Legacy NVIDIA_* environment variables are ignored after migration; remove them when convenient. Never commit a live API key.
OpenAI API usage is billed separately from ChatGPT subscriptions. Make sure the API project has access to the chosen models and billing/credits are set up.

## Run locally

1. Install dependencies: python -m pip install -r requirements.txt
2. Copy .env.example to .env and set OPENAI_API_KEY.
3. Start: uvicorn app.main:app --reload --port 8080
4. Open: http://localhost:8080/docs

Endpoints: GET /health, GET /languages, POST /translate and POST /demo?language_code=de.

## Behavior

- Predefined language codes are trusted. English passes through unchanged.
- A manually entered language_name is recognized in a separate OpenAI request before translation; the recognized canonical name and language code are passed to the translator.
- Translation output must be valid JSON with the same number and order of strings, and must preserve placeholders such as {name}.
- If a response is malformed or truncated, the backend retries with an adjusted prompt/output budget. Only checkable translation quality issues trigger automated fallback; semantic mistranslations that still pass structural validation cannot be detected reliably.
- After two consecutive invalid model outputs (malformed JSON, wrong count, missing placeholders, unchanged translations), the backup model is used. Otherwise, after three failed attempts, retries also use the backup model. A 404 or 410 response from the primary model switches immediately to backup.
- Transient errors (such as 429 and 5xx) are retried with bounded exponential backoff. Non-retryable configuration/auth errors are not retried.
- Repetition loops trigger adaptive splitting of problematic translation batches.
- If models remain unavailable, the API returns the original English dictionary with fallback=true. Unrecognized custom languages return fallback=false and an explanation.
- Successful OpenAI operations log at INFO after JSON parsing and validation, once per validated operation. The line includes the operation (`translation` or `language recognition`), requested model ID (`model`), API-reported model ID when supplied (`response_model`), attempt number and total (`attempt`), and whether the backup model was used (`fallback`).
- Example: `OpenAI translation successful: model=gpt-6-luna response_model=gpt-6-luna attempt=1/6 fallback=false`.
- A successful HTTP response with invalid JSON/translation is never logged as a successful translation. API keys, authorization headers, source texts and successful translated texts are not included in the success log. Existing error diagnostics may log truncated malformed model responses.

## Tests

Run offline tests with: python -m pytest -q
Tests mock AI calls and consume no API credits.
