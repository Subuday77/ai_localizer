# AI Localizer

Current backend version: **0.2.2**.

FastAPI backend for translating an English UI dictionary with NVIDIA NIM. No database required.

## Structure

- `app/main.py` — FastAPI creation and router registration only
- `app/api/routes.py` — documented HTTP endpoints
- `app/config.py` — environment settings, timeouts, retries and NVIDIA behavior
- `app/constants.py` — languages, mock strings and prompts
- `app/schemas.py` — Pydantic contracts
- `app/services/nvidia.py` — NVIDIA client, JSON validation, diagnostic logging and retry logic
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
On NVIDIA failure the original English dictionary is returned with `fallback=true`.
At least five retries occur for transient errors; non-retryable auth/config errors fail immediately.

## NVIDIA response handling

- Reasoning/thinking is disabled by default with `NVIDIA_ENABLE_THINKING=false`. The request sends `chat_template_kwargs.enable_thinking=false` to models that support this NVIDIA chat-template option.
- If NVIDIA returns `finish_reason == "length"`, the request is retried with a larger `max_tokens` value, up to `NVIDIA_MAX_TOKENS_CAP`.
- The base translation prompt explicitly requires valid JSON and requires double quotes inside translated text to be escaped as `\"`.
- If a model returns invalid JSON, its raw `message.content` is written to the application log before retrying. The diagnostic text is limited by `NVIDIA_RAW_RESPONSE_LOG_CHARS` (default: 4000 characters).
- After the first JSON parse failure, subsequent attempts add a stricter correction instruction: the JSON must be valid and all internal double quotes must be properly escaped. This stricter instruction remains active even if an intermediate retry fails with a transient HTTP error such as `503`.
- API keys and authorization headers are never written to the log.
- Model output must be a JSON array with exactly the same number of string items as the source values, in the same order.
- Placeholders such as `{name}` must survive translation unchanged.

Application logs are written to `logs/backend.log` and to stderr.
