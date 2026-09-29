"""Application logging to stderr and logs/backend.log."""

import logging

from .config import ROOT


def configure_logging() -> None:
    """Configure the application logger without duplicating handlers.

    :return: ``None``. The function configures the shared ``localizer`` logger in place.
    """
    logger = logging.getLogger('localizer')
    if logger.handlers:
        return

    path = ROOT / 'logs'
    path.mkdir(exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handlers = (
        logging.StreamHandler(),
        logging.FileHandler(path / 'backend.log', encoding='utf-8'),
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
