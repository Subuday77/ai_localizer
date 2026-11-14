# AI Localizer 1.x: standalone API image. Not shared with AI Quiz Game.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LOCALIZER_LOG_TO_FILE=false

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --system --uid 10001 --create-home localizer \
    && install -d -o 10001 -g 10001 /var/lib/localizer-demo

COPY --chown=10001:10001 app/ ./app/

USER 10001:10001
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
