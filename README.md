# AI Localizer

Current backend version: **0.3.0**.

FastAPI backend for translating an English UI dictionary with NVIDIA NIM. No database required.

## Structure

- `app/main.py` — FastAPI creation and router registration only
- `app/api/routes.py` — documented HTTP endpoints
- `app/config.py` — environment settings, timeouts, retries and NVIDIA behavior
- `app/constants.py` — languages, mock strings and prompts
- `app/schemas.py` — Pydantic contracts
- `app/services/nvidia.py` — NVIDIA client, language recognition, JSON validation, diagnostic logging and retry logic
- `app/logging_setup.py` — console and file logging
- `tests/` — offline tests

## Run

```bash
python -m pip install -r requirements.txt
cp .env.example .env
# edit .env to provide NVIDIA_API_KEY and NVIDIA_MODEL_ID
uvicorn app.main:app --reload --port 8080
```

Open `http://localhost:8080/docs` or POST to `/demo?language_code=de`.

`POST /translate` accepts `dictionary` and `language_code` or `language_name`.
A custom `language_name` takes precedence over `language_code`.

Predefined dropdown languages are trusted. A manually entered `language_name` is
validated by the same NVIDIA request that performs the translation. Minor spelling
mistakes are accepted when the intended real language is unambiguous, so a value
such as `Deusch` may be normalized to `Deutsch`.

The response contains `language_recognized`:

- `true` — the target language is known or the custom name was recognized;
- `false` — a custom language name was not recognized;
- `null` — the custom language could not be checked because the NVIDIA request failed.

An unrecognized custom language returns the original English dictionary with
`fallback=false` and `error="Language not recognized"`. This is treated as user
input validation, not as a provider failure.

On NVIDIA failure the original English dictionary is returned with `fallback=true`.
At least five retries occur for transient errors; non-retryable auth/config errors fail immediately.

## NVIDIA response handling

- Reasoning/thinking is disabled by default with `NVIDIA_ENABLE_THINKING=false`. The request sends `chat_template_kwargs.enable_thinking=false` to models that support this NVIDIA chat-template option.
- If NVIDIA returns `finish_reason == "length"`, the request is retried with a larger `max_tokens` value, up to `NVIDIA_MAX_TOKENS_CAP`.
- Predefined languages use the compact JSON-array translation response.
- Manually entered languages use one combined JSON object containing language-recognition status, a normalized language name, and translations. No separate validation request is made.
- The base translation prompts explicitly require valid JSON and require double quotes inside translated text to be escaped as `\"`.
- If a model returns invalid JSON, its raw `message.content` is written to the application log before retrying. The diagnostic text is limited by `NVIDIA_RAW_RESPONSE_LOG_CHARS` (default: 4000 characters).
- After the first JSON parse failure, subsequent attempts add a stricter correction instruction. This stricter instruction remains active even if an intermediate retry fails with a transient HTTP error such as `503`.
- API keys and authorization headers are never written to the log.
- Translation arrays must contain exactly the same number of string items as the source values, in the same order.
- Placeholders such as `{name}` must survive translation unchanged.

Application logs are written to `logs/backend.log` and to stderr.
