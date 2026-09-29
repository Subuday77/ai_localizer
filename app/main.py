"""Minimal ASGI application entry point. Run: uvicorn app.main:app --reload --port 8080."""
from fastapi import FastAPI
from .api.routes import router
from .logging_setup import configure_logging

configure_logging()
app = FastAPI(title='AI Localizer', version='0.3.0')
app.include_router(router)
