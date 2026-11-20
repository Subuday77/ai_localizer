"""Publish v3 semantic-disambiguation profiles without modifying v1/v2 snapshots.

Usage:
    python -m scripts.publish_semantic_profiles

Requires the six existing v2 profiles from scripts.publish_professional_profiles.
Only changes the catalog's published pointer; all old versions remain pinned.
"""

from copy import deepcopy

from sqlalchemy.orm import Session

from app.config import get_settings
from app.contexts.models import Context, ContextVersion
from app.contexts.repository import DomainProfile, engine_for_url
from scripts.publish_professional_profiles import PROFESSIONAL_PROFILES


def publish(session: Session) -> int:
    """Atomically activate immutable v3 profiles for all six starter contexts."""
    activated = 0
    for context_id, _ in PROFESSIONAL_PROFILES:  # parent-first order
        context = session.get(Context, context_id)
        v2 = session.get(ContextVersion, (context_id, 2))
        if context is None or v2 is None or v2.status != 'published':
            raise ValueError(f'Published v2 context is required: {context_id}')
        if context.published_version not in (2, 3):
            raise ValueError(f'Refusing unexpected published version for {context_id}')

        # The ONLY change to the profile is a generic, language-neutral
        # interpretation procedure. No new per-word glossary or mapping.
        new_profile = deepcopy(v2.domain_profile)
        if new_profile is None:
            raise ValueError(f'Expected an expert v2 domain profile: {context_id}')
        if 'interpretation_method' in new_profile:
            raise ValueError(f'Existing v2 profile unexpectedly has a method: {context_id}')
        new_profile['interpretation_method'] = 'evidence_before_domain'
        new_profile = DomainProfile.model_validate(new_profile).model_dump(exclude_none=True)

        parent_id = v2.parent_context_id
        parent_version = 3 if parent_id is not None else None
        existing = session.get(ContextVersion, (context_id, 3))
        if existing is None:
            session.add(ContextVersion(
                context_id=context_id,
                version=3,
                status='published',
                parent_context_id=parent_id,
                parent_version=parent_version,
                rules=deepcopy(v2.rules),
                glossary=deepcopy(v2.glossary),
                style=deepcopy(v2.style),
                domain_profile=new_profile,
            ))
            session.flush()  # parent v3 foreign key must exist before child v3
        elif (
            existing.status != 'published'
            or existing.parent_context_id != parent_id
            or existing.parent_version != parent_version
            or existing.rules != v2.rules
            or existing.glossary != v2.glossary
            or existing.style != v2.style
            or existing.domain_profile != new_profile
        ):
            raise ValueError(f'Refusing to overwrite a different v3 snapshot: {context_id}')

        if context.published_version == 2:
            context.published_version = 3
            activated += 1
    return activated


def main():
    settings = get_settings()
    if not settings.localizer_database_url:
        raise RuntimeError('LOCALIZER_DATABASE_URL must be configured')
    engine = engine_for_url(
        settings.localizer_database_url, settings.localizer_database_password,
    )
    with Session(engine) as session, session.begin():
        activated = publish(session)
    print(f'Semantic context profiles activated: {activated}')


if __name__ == '__main__':
    main()
