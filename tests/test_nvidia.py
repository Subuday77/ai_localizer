"""Tests for NVIDIA response validation."""

import pytest

from app.services.nvidia import (
    UnchangedTranslationError,
    _is_unsupported_translation,
    build_translation_prompt,
    parse_translation,
)


def test_parse_translation_rejects_fully_unchanged_output() -> None:
    """Reject a response when every translated string equals its source string.

    :return: None.
    :raises AssertionError: If unchanged source strings are incorrectly accepted.
    """
    originals = ['Welcome', 'Home', 'Contact']

    with pytest.raises(UnchangedTranslationError, match='original source strings unchanged'):
        parse_translation('["Welcome", "Home", "Contact"]', originals)


def test_parse_translation_allows_partially_unchanged_output() -> None:
    """Allow legitimate translations where some individual strings remain unchanged.

    :return: None.
    :raises AssertionError: If a partially translated response is incorrectly rejected.
    """
    originals = ['Welcome', 'Home', 'Contact']

    result = parse_translation('["Bienvenido", "Home", "Contacto"]', originals)

    assert result == ['Bienvenido', 'Home', 'Contacto']


def test_translation_prompt_requires_exact_target_language() -> None:
    """Require the model to avoid substituting a different language.

    :return: None.
    :raises AssertionError: If the strict target-language fallback instruction disappears.
    """
    prompt = build_translation_prompt(['Welcome', 'Home'], 'Chukchi (language code: ckt)')

    assert 'strictly into Chukchi (language code: ckt)' in prompt
    assert 'do not substitute Russian, English' in prompt
    assert 'return the original input array unchanged' in prompt


def test_unsupported_translation_requires_repeated_clean_declines() -> None:
    """Classify unsupported languages only after repeated unchanged translation responses.

    :return: None.
    :raises AssertionError: If technical or malformed failures are misclassified as unsupported.
    """
    assert _is_unsupported_translation('translation', 2, 0) is True
    assert _is_unsupported_translation('translation', 5, 0) is True
    assert _is_unsupported_translation('translation', 1, 0) is False
    assert _is_unsupported_translation('translation', 5, 1) is False
    assert _is_unsupported_translation('language recognition', 5, 0) is False
