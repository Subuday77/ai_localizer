"""Idempotent seed of starter context versions; never modify published versions.

Usage (after alembic upgrade head): python -m scripts.seed_contexts
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import get_settings
from app.contexts.models import Context, ContextVersion
from app.contexts.repository import engine_for_url

STARTER_CONTEXTS = [
    {
        'id': 'technology', 'title': 'Technology',
        'description': 'General technology and engineering terminology.',
        'parent': None, 'style': {'tone': 'precise and professional'}, 'rules': [],
    },
    {
        'id': 'electronics', 'title': 'Electronics',
        'description': 'Circuits, electrical materials and electronic components.',
        'parent': 'technology', 'style': {},
        'rules': [{
            'id': 'electrical-conductor', 'source_language': 'en', 'target_language': 'ru',
            'term': 'conductor', 'preferred': 'проводник',
            'when': 'The source is about electrical conduction or conducting materials.',
            'note': 'Do not change an explicitly human train conductor into an electrical conductor.',
        }],
    },
    {
        'id': 'railway', 'title': 'Railway',
        'description': 'Passenger railway operations and transport staff.',
        'parent': None, 'style': {'tone': 'clear and informative'},
        'rules': [{
            'id': 'railway-conductor', 'source_language': 'en', 'target_language': 'ru',
            'term': 'conductor', 'preferred': 'кондуктор',
            'when': 'The source refers to the railway employee who attends passengers or tickets.',
            'note': 'Avoid this meaning when the source is clearly electrical.',
        }],
    },
    {
        'id': 'music', 'title': 'Music',
        'description': 'Orchestras, music performance and conducting.',
        'parent': None, 'style': {},
        'rules': [{
            'id': 'music-conductor', 'source_language': 'en', 'target_language': 'ru',
            'term': 'conductor', 'preferred': 'дирижёр',
            'when': 'The source refers to a person directing an orchestra.',
        }],
    },
    {
        'id': 'software', 'title': 'Software',
        'description': 'Operating systems, applications and software interfaces.',
        'parent': 'technology', 'style': {},
        'rules': [{
            'id': 'device-driver', 'source_language': 'en', 'target_language': 'ru',
            'term': 'driver', 'preferred': 'драйвер',
            'when': 'The source refers to software controlling a device.',
        }],
    },
]


def seed(session: Session) -> int:
    """Create missing first versions only, leaving all existing records unchanged."""
    added = 0
    for item in STARTER_CONTEXTS:
        if session.get(Context, item['id']) is None:
            session.add(Context(
                id=item['id'], title=item['title'], description=item['description'],
                is_public=True, published_version=1,
                created_at=datetime.now(timezone.utc),
            ))
            added += 1
    session.flush()
    for item in STARTER_CONTEXTS:
        if session.get(ContextVersion, (item['id'], 1)) is None:
            session.add(ContextVersion(
                context_id=item['id'], version=1, status='published',
                parent_context_id=item['parent'],
                parent_version=1 if item['parent'] is not None else None,
                rules=item['rules'], glossary=[], style=item['style'],
                created_at=datetime.now(timezone.utc),
            ))
    return added


def main():
    settings = get_settings()
    if not settings.localizer_database_url:
        raise RuntimeError('LOCALIZER_DATABASE_URL must be configured')
    engine = engine_for_url(settings.localizer_database_url, settings.localizer_database_password)
    with Session(engine) as session, session.begin():
        added = seed(session)
    print(f'Starter contexts created: {added}')


if __name__ == '__main__':
    main()
