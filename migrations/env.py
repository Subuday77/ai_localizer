"""Alembic migration entry point. DB credentials only from server-side settings."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.config import get_settings
from app.contexts.models import Base

config = context.config
if config.config_file_name is not None:
    # Preserve application loggers when migrations run inside a test/process.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def migration_engine():
    settings = get_settings()
    if not settings.localizer_database_url:
        raise RuntimeError('LOCALIZER_DATABASE_URL must be set before applying migrations')
    kwargs = {'password': settings.localizer_database_password} if settings.localizer_database_password else {}
    return create_engine(settings.localizer_database_url, poolclass=pool.NullPool, connect_args=kwargs)


def run_migrations_online():
    engine = migration_engine()
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    raise RuntimeError('Offline migrations are disabled; use online migration with DATABASE_URL')
else:
    run_migrations_online()
