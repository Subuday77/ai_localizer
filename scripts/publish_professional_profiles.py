"""Publish language-neutral expert profiles as immutable v2 context snapshots.

Run after 'alembic upgrade head' and 'python -m scripts.seed_contexts':
    python -m scripts.publish_professional_profiles

Existing v1 snapshots are never changed. Publish v2 only if the catalog still
points at v1 or the same v2 is already published. Transaction is atomic.
"""

from copy import deepcopy

from sqlalchemy.orm import Session

from app.config import get_settings
from app.contexts.models import Context, ContextVersion
from app.contexts.repository import DomainProfile, engine_for_url


# Parent first: parent v2 must exist before a child can pin to it.
PROFESSIONAL_PROFILES = [
    (
        'technology',
        {
            'specialization': 'technology and engineering website localization',
            'scope': 'General engineering, technical systems, hardware, devices, '
                     'measurements, industrial processes, and product documentation.',
            'audience': 'Technical professionals and users of technology products.',
            'conventions': 'Prefer established technical terminology and concise '
                           'website-interface language in the target locale.',
        },
    ),
    (
        'electronics',
        {
            'specialization': 'electronics and electrical-engineering website localization',
            'scope': 'Circuit design, electrical installations, conductors, insulation, '
                     'electronic components, signal paths, interconnects, power and '
                     'control systems, instrumentation, and equipment diagnostics.',
            'audience': 'Electrical engineers, electronics technicians, and industrial users.',
            'conventions': 'For ambiguous standalone labels, prefer established '
                           'electrical-engineering meanings over everyday meanings. '
                           'Preserve the plain-language meaning when the source '
                           'clearly describes a different subject.',
        },
    ),
    (
        'software',
        {
            'specialization': 'software products and computer-systems website localization',
            'scope': 'Applications, operating systems, configuration, user interfaces, '
                     'device software, networking, security, deployment, and diagnostics.',
            'audience': 'Software users, developers, and IT administrators.',
            'conventions': 'Use idiomatic platform terminology and short, actionable '
                           'interface labels without adding unstated functionality.',
        },
    ),
    (
        'railway',
        {
            'specialization': 'rail transport and passenger railway website localization',
            'scope': 'Railway operations, rolling stock, infrastructure, track maintenance, '
                     'station equipment, signaling, passenger services, and ticketing.',
            'audience': 'Passengers, railway operators, infrastructure engineers, and staff.',
            'conventions': 'Prefer natural railway and transport terminology in the '
                           'target locale, while respecting explicit evidence that '
                           'a sentence concerns another field.',
        },
    ),
    (
        'music',
        {
            'specialization': 'music theory and professional music website localization',
            'scope': 'Composition, musical notation, scores, orchestral performance, '
                     'rehearsal, ensembles, instruments, recording, and production.',
            'audience': 'Musicians, composers, educators, orchestral staff, and listeners.',
            'conventions': 'For ambiguous short labels, assume established musical '
                           'terminology when it fits; translate clearly non-musical '
                           'phrases according to their actual meaning.',
        },
    ),
    (
        'military',
        {
            'specialization': 'military service and defence-sector website localization',
            'scope': 'Military organizations, ranks, uniforms, insignia, decorations, '
                     'operations, equipment, personnel, and professional service jargon.',
            'audience': 'Military personnel, defence professionals, and historians.',
            'conventions': 'Interpret domain-specific idioms in their military sense '
                           'where justified, without changing an explicit literal '
                           'description into military terminology.',
        },
    ),
]


def publish(session: Session) -> int:
    """Create missing v2 snapshots and advance each catalog pointer exactly once."""
    activated = 0
    for context_id, raw_profile in PROFESSIONAL_PROFILES:
        profile = DomainProfile.model_validate(raw_profile).model_dump(exclude_none=True)
        context = session.get(Context, context_id)
        original = session.get(ContextVersion, (context_id, 1))
        if context is None or original is None or original.status != 'published':
            raise ValueError(f'Expected an existing published v1 context: {context_id}')
        if context.published_version not in (1, 2):
            raise ValueError(f'Refusing to replace an unknown published version of {context_id}')
        pinned_parent = original.parent_context_id
        pinned_version = 2 if pinned_parent is not None else None
        current = session.get(ContextVersion, (context_id, 2))
        if current is None:
            session.add(ContextVersion(
                context_id=context_id,
                version=2,
                status='published',
                parent_context_id=pinned_parent,
                parent_version=pinned_version,
                rules=deepcopy(original.rules),
                glossary=deepcopy(original.glossary),
                style=deepcopy(original.style),
                domain_profile=profile,
            ))
            # Ensure self-referential parent FK is already present before children.
            session.flush()
        elif (
            current.status != 'published'
            or current.domain_profile != profile
            or current.parent_context_id != pinned_parent
            or current.parent_version != pinned_version
            or current.rules != original.rules
            or current.glossary != original.glossary
            or current.style != original.style
        ):
            raise ValueError(f'Refusing to overwrite an existing v2 snapshot: {context_id}')
        if context.published_version == 1:
            context.published_version = 2
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
    print(f'Professional context profiles activated: {activated}')


if __name__ == '__main__':
    main()
