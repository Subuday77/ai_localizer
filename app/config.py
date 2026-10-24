"""Runtime configuration loaded from environment variables and the local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """OpenAI credentials and configurable networking/retry limits."""

    model_config = SettingsConfigDict(
        env_file=ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    openai_api_key: str = ''
    openai_model_id: str = 'gpt-6-luna'
    openai_fallback_model_id: str = 'gpt-5.4-mini'
    openai_base_url: str = 'https://api.openai.com/v1'
    openai_max_tokens: int = Field(default=2048, ge=128)
    openai_max_tokens_cap: int = Field(default=16384, ge=128)
    openai_retries: int = Field(default=5, ge=5)
    openai_quality_failures_before_fallback: int = Field(default=2, ge=1, le=6)
    openai_timeout_seconds: float = Field(default=45, gt=0)
    openai_retry_max_delay_seconds: float = Field(default=12, gt=0)
    openai_retry_jitter_seconds: float = Field(default=0.3, ge=0)
    openai_reasoning_effort: Literal['none', 'low', 'medium', 'high', 'xhigh', 'max'] = 'none'
    openai_raw_response_log_chars: int = Field(default=4000, ge=200, le=50000)
    cors_origins: list[str] = Field(default_factory=list)
    host: str = '127.0.0.1'
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    """Load and cache validated application settings.

    :return: Cached application settings populated from defaults and environment values.
    """
    return Settings()
