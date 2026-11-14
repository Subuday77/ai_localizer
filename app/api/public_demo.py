"""Cost-bounded anonymous demo for the Render-hosted AI Localizer frontend.

Unlike /translate, this endpoint NEVER accepts user-provided source text.
Requests are counted in a persistent SQLite volume before any AI work starts.
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import logging
import sqlite3
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import Settings, get_settings
from ..constants import LANGUAGES
from ..schemas import TranslateRequest, TranslateResponse
from .routes import translate

router = APIRouter()
logger = logging.getLogger("localizer")

# Keep in sync with ai_localizer_front/src/app/constants/ui-strings.ts.
# The public endpoint always uses this fixed dictionary, never browser text.
DEMO_ENGLISH_UI = {
    "brand": "AI Localizer",
    "welcome": "Welcome",
    "hero_title": "Make this page speak your language.",
    "explanation": "Choose a language below and let AI localize the interface in seconds. English always remains available as a safe fallback.",
    "language_label": "Choose a language",
    "language_placeholder": "Select a language",
    "custom_language_label": "Can't find your language in the list? Enter its native name here. For example, {russian} or {german}.",
    "custom_language_placeholder": "Type the native language name",
    "localize_button": "Localize this page",
    "spinner_text": "Localizing with AI…",
    "feedback_title": "How does it look?",
    "like_button": "I like it",
    "love_button": "I really like it",
    "dislike_button": "I don't like it",
    "thank_you": "Thank you!",
    "big_thanks": "Thank you very much!",
    "missing_language_title": "Choose a language",
    "missing_language_text": "Select a language or enter its native name first.",
    "language_not_recognized_title": "Language not recognized",
    "language_not_recognized_text": "We couldn't identify that language. Please enter the name of a real language, preferably in that language itself.",
    "localization_error_title": "Localization unavailable",
    "localization_error_text": "We couldn't translate the page. The English version is still available.",
    "server_error_title": "Server error",
    "server_error_text": "We couldn't reach the localization service. Please try again.",
    "language_list_error_title": "Language list unavailable",
    "language_list_error_text": "The server language list could not be loaded. You can still type a language name below.",
}

_active_demo = asyncio.Semaphore(2)


class PublicDemoRequest(BaseModel):
    """Only the target language may be supplied by an anonymous visitor."""

    model_config = ConfigDict(extra="forbid")
    language_code: str | None = Field(default=None, min_length=2, max_length=20)
    language_name: str | None = Field(default=None, min_length=1, max_length=60)

    @model_validator(mode="after")
    def validate_target(self):
        code = (self.language_code or "").strip().lower()
        name = (self.language_name or "").strip()
        if not code and not name:
            raise ValueError("Specify a target language")
        if code and code not in LANGUAGES:
            raise ValueError("Unsupported demo language_code")
        if name and (
            len(name) > 60 or
            not any(ch.isalpha() for ch in name) or
            not all(ch.isalpha() or ch in " -'’" for ch in name)
        ):
            raise ValueError("Invalid demo language name")
        self.language_code = code or None
        self.language_name = name or None
        return self


def _claim_quota(settings: Settings, ip_address: str) -> None:
    """Atomically reserve a request slot; fail closed on storage errors.

    SQLite state is stored on a persistent Docker volume, so an application
    restart must not replenish the global daily allowance.
    """
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = int(time.time())
    fingerprint = hashlib.sha256(ip_address.encode("utf-8")).hexdigest()
    conn = None
    try:
        conn = sqlite3.connect(settings.localizer_demo_db_path, timeout=5, isolation_level=None)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS demo_requests ("
            "id INTEGER PRIMARY KEY, day TEXT NOT NULL, "
            "ip_hash TEXT NOT NULL, requested_at INTEGER NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS demo_requests_day ON demo_requests(day)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS demo_requests_ip_time "
            "ON demo_requests(ip_hash, requested_at)"
        )
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM demo_requests WHERE requested_at < ?", (now - 172800,))
        global_count = conn.execute(
            "SELECT COUNT(*) FROM demo_requests WHERE day = ?", (day,)
        ).fetchone()[0]
        hourly_count = conn.execute(
            "SELECT COUNT(*) FROM demo_requests "
            "WHERE ip_hash = ? AND requested_at >= ?",
            (fingerprint, now - 3600),
        ).fetchone()[0]
        if global_count >= settings.localizer_demo_daily_limit:
            conn.rollback()
            raise HTTPException(
                status_code=429,
                detail="Demo daily limit reached. Please try tomorrow (UTC).",
                headers={"Retry-After": "3600"},
            )
        if hourly_count >= settings.localizer_demo_per_ip_hour_limit:
            conn.rollback()
            raise HTTPException(
                status_code=429,
                detail="Demo hourly limit reached. Please try later.",
                headers={"Retry-After": "3600"},
            )
        conn.execute(
            "INSERT INTO demo_requests (day, ip_hash, requested_at) VALUES (?, ?, ?)",
            (day, fingerprint, now),
        )
        conn.commit()
        logger.info("Public demo request reserved: day=%s used=%d", day, global_count + 1)
    except HTTPException:
        raise
    except sqlite3.Error:
        logger.exception("Public demo quota storage unavailable")
        raise HTTPException(status_code=503, detail="Demo temporarily unavailable") from None
    finally:
        if conn is not None:
            conn.close()


@router.post("/public-demo/translate", response_model=TranslateResponse, tags=["Public Demo"])
async def public_demo_translate(
    payload: PublicDemoRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> TranslateResponse:
    """Translate a fixed page only, with quotas, without exposing an API key."""
    if not settings.localizer_demo_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    if not settings.localizer_require_api_key:
        raise HTTPException(status_code=503, detail="Demo configuration unavailable")

    try:
        await asyncio.wait_for(_active_demo.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=429,
            detail="Demo is busy. Please try again later.",
            headers={"Retry-After": "30"},
        ) from None
    try:
        _claim_quota(settings, request.client.host if request.client else "unknown")
        return await translate(
            TranslateRequest(
                dictionary=DEMO_ENGLISH_UI,
                language_code=payload.language_code,
                language_name=payload.language_name,
            ),
            settings,
        )
    finally:
        _active_demo.release()
