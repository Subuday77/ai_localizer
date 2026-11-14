"""Temporary server-to-server API authentication for VPS Localizer 1.x."""

import hashlib
import hmac

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings


def require_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Authorize paid API calls before translation/model work starts.

    The browser-based frontend must never embed a long-lived API key.
    Only key hashes are stored in VPS environment settings.
    """
    if not settings.localizer_require_api_key:
        return

    authorization = request.headers.get('authorization', '')
    scheme, separator, token = authorization.partition(' ')
    if (
        scheme.lower() != 'bearer'
        or not separator
        or not token
        or len(token) > 256
        or token.strip() != token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid API credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    candidate_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    if not any(
        hmac.compare_digest(candidate_hash, expected)
        for expected in settings.localizer_api_key_hashes
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid API credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )
