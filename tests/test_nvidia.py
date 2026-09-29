"""Tests for NVIDIA response validation."""

import pytest

from app.services.nvidia import build_translation_prompt, parse_translation


def test_parse_translation_rejects_fully_unchanged_output() -> None:
    """Reject a response when every translated string equals its source string.

    :return: None.
    :raises AssertionError: If unchanged source strings are incorrectly accepted.
    """
    originals = ['Welcome', 'Home', 'Contact']

    with pytest.raises(ValueError, match='original source strings unchanged'):
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
