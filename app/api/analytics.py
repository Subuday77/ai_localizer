"""Anonymous visit analytics endpoints."""

import logging

from fastapi import APIRouter, status

from ..schemas import PageViewRequest

router = APIRouter()
logger = logging.getLogger('localizer')


@router.post('/analytics/pageview', status_code=status.HTTP_204_NO_CONTENT, tags=['Analytics'])
def pageview(request: PageViewRequest) -> None:
    """Record one anonymous frontend page view in the application log.

    Only the page path and referrer origin supplied by the frontend are logged.
    No cookie, IP address, fingerprint, or persistent visitor identifier is stored
    by the application.

    :param request: Anonymous page path and optional referrer origin.
    :return: ``None``; successful tracking responds with HTTP 204.
    """
    logger.info(
        'PAGE_VIEW path=%r referrer=%r',
        request.path,
        request.referrer or 'direct',
    )
