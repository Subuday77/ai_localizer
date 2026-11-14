"""Application logging to stderr; optional local file logging for non-container runs."""

import logging
import os

from .config import ROOT


def configure_logging() -> None:
    """Configure the application logger without duplicating handlers.

    By default, local and Render deployments retain logs/backend.log plus stderr.
    Containers set LOCALIZER_LOG_TO_FILE=false so logs go to Docker stdout/stderr,
    where rotation is managed by the container runtime, and no writable app
    directory is required.
    """
    logger = logging.getLogger('localizer')
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    write_file = os.getenv('LOCALIZER_LOG_TO_FILE', 'true').strip().lower() not in {
        '0', 'false', 'no', 'off',
    }
    if write_file:
        path = ROOT / 'logs'
        path.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(path / 'backend.log', encoding='utf-8'))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
