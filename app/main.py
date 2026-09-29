"""Minimal ASGI application entry point. Run: uvicorn app.main:app --reload --port 8080."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.analytics import router as analytics_router
from .api.routes import router
from .config import get_settings
from .logging_setup import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(title='AI Localizer', version='0.3.0')

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=['GET', 'POST', 'OPTIONS'],
        allow_headers=['*'],
    )


app.include_router(router)
app.include_router(analytics_router)
