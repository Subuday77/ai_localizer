"""Pydantic request and response contracts exposed by the API."""

from pydantic import BaseModel, Field, model_validator


class PageViewRequest(BaseModel):
    """Minimal anonymous page-view event sent by the frontend."""

    path: str = Field(min_length=1, max_length=500)
    referrer: str | None = Field(default=None, max_length=500)


class TranslateRequest(BaseModel):
    """Legacy dictionary contract plus optional pinned Localizer 2.0 context."""

    dictionary: dict[str, str] = Field(min_length=1, max_length=500)
    language_code: str | None = Field(default=None, min_length=2, max_length=20)
    language_name: str | None = Field(default=None, min_length=1, max_length=100)
    context_id: str | None = Field(
        default=None, min_length=2, max_length=80,
        pattern=r'^[a-z][a-z0-9_-]*$',
    )
    context_version: int | None = Field(default=None, ge=1)

    @model_validator(mode='after')
    def check_language(self):
        if not self.language_code and not self.language_name:
            raise ValueError('Specify language_code or language_name')
        if self.context_version is not None and self.context_id is None:
            raise ValueError('context_version requires context_id')
        return self


class TranslateResponse(BaseModel):
    """Translated strings plus optional actual context-version metadata."""

    dictionary: dict[str, str]
    language: str
    language_recognized: bool | None = True
    fallback: bool
    error: str | None = None
    context_id: str | None = None
    context_version: int | None = None
