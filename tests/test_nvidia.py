"""Tests for NVIDIA response validation."""

import pytest

from app.services.nvidia import parse_translation


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
