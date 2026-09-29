"""Pydantic request and response contracts exposed by the API."""

from pydantic import BaseModel, Field, model_validator


class TranslateRequest(BaseModel):
    """English source strings and a target language code and/or native name."""

    dictionary: dict[str, str] = Field(min_length=1, max_length=500)
    language_code: str | None = Field(default=None, min_length=2, max_length=20)
    language_name: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode='after')
    def check_language(self):
        """Ensure that the request contains a target-language identifier.

        :return: The validated request model instance.
        :raises ValueError: If neither ``language_code`` nor ``language_name`` is supplied.
        """
        if not self.language_code and not self.language_name:
            raise ValueError('Specify language_code or language_name')
        return self


class TranslateResponse(BaseModel):
    """Translated or original strings, with explicit fallback status."""

    dictionary: dict[str, str]
    language: str
    fallback: bool
    error: str | None = None
