"""Runtime configuration loaded from environment variables and the local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """NVIDIA credentials and configurable networking/retry limits."""

    model_config = SettingsConfigDict(
        env_file=ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    nvidia_api_key: str = ''
    nvidia_model_id: str = ''
    nvidia_fallback_model_id: str = ''
    nvidia_base_url: str = 'https://integrate.api.nvidia.com/v1'
    nvidia_max_tokens: int = Field(default=2048, ge=128)
    nvidia_max_tokens_cap: int = Field(default=16384, ge=128)
    nvidia_retries: int = Field(default=5, ge=5)
    nvidia_timeout_seconds: float = Field(default=45, gt=0)
    nvidia_retry_max_delay_seconds: float = Field(default=12, gt=0)
    nvidia_retry_jitter_seconds: float = Field(default=0.3, ge=0)
    nvidia_enable_thinking: bool = False
    nvidia_raw_response_log_chars: int = Field(default=4000, ge=200, le=50000)
    host: str = '127.0.0.1'
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    """Load and cache validated application settings.

    :return: Cached application settings populated from defaults and environment values.
    """
    return Settings()
