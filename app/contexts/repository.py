"""Read-only context resolution. No automatic DB bootstrap on application startup."""

import json
from dataclasses import dataclass
from functools import lru_cache

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .models import Context, ContextVersion


class ContextUnavailableError(RuntimeError):
    """Database is not configured for the optional context service."""


class ContextNotFoundError(LookupError):
    """Unknown, unpublished, or non-public context/version."""


class InvalidContextDataError(ValueError):
    """A stored context contains invalid or conflicting rules."""


class TranslationRule(BaseModel):
    """A semantic hint; NEVER an unconditional lexical replacement rule."""

    id: str = Field(min_length=1, max_length=100)
    source_language: str = Field(default='en', min_length=2, max_length=20)
    target_language: str = Field(min_length=2, max_length=20)
    term: str = Field(min_length=1, max_length=200)
    preferred: str = Field(min_length=1, max_length=200)
    when: str = Field(min_length=1, max_length=500)
    note: str = Field(default='', max_length=500)


class GlossaryEntry(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    source_language: str = Field(default='en', min_length=2, max_length=20)
    target_language: str = Field(min_length=2, max_length=20)
    term: str = Field(min_length=1, max_length=200)
    preferred: str = Field(min_length=1, max_length=200)
    when: str = Field(default='', max_length=500)


@dataclass(frozen=True)
class ResolvedContext:
    context_id: str
    version: int
    rules: tuple[TranslationRule, ...]
    glossary: tuple[GlossaryEntry, ...]
    style: dict


@lru_cache(maxsize=8)
def engine_for_url(database_url: str, database_password: str = '') -> Engine:
    if not database_url:
        raise ContextUnavailableError('Context database is not configured')
    kwargs = {'password': database_password} if database_password else {}
    return create_engine(database_url, pool_pre_ping=True, connect_args=kwargs)


def list_public_contexts(database_url: str, database_password: str = '') -> list[dict]:
    """Return published public metadata; never disclose private/draft contexts."""
    with Session(engine_for_url(database_url, database_password)) as session:
        rows = session.scalars(
            select(Context).where(
                Context.is_public.is_(True),
                Context.published_version.is_not(None),
            ).order_by(Context.id)
        ).all()
        return [
            {'id': row.id, 'title': row.title, 'description': row.description,
             'version': row.published_version}
            for row in rows
        ]


def resolve_context(
    database_url: str, context_id: str, version: int | None = None,
    database_password: str = '',
) -> ResolvedContext:
    """Resolve exact immutable ancestor versions, overriding rules by stable rule ID."""
    with Session(engine_for_url(database_url, database_password)) as session:
        root = session.get(Context, context_id)
        if not root or not root.is_public or root.published_version is None:
            raise ContextNotFoundError('Context is not available')
        requested_version = version if version is not None else root.published_version

        seen: set[tuple[str, int]] = set()

        def walk(current_id: str, current_version: int) -> tuple[dict, dict, dict]:
            pair = (current_id, current_version)
            if pair in seen or len(seen) >= 16:
                raise InvalidContextDataError('Context inheritance cycle or excessive depth')
            seen.add(pair)
            record = session.get(ContextVersion, pair)
            if record is None or record.status != 'published':
                raise ContextNotFoundError('Context version is not available')
            if record.parent_context_id is not None:
                if record.parent_version is None:
                    raise InvalidContextDataError('Context parent must reference a pinned version')
                rules, glossary, style = walk(record.parent_context_id, record.parent_version)
            else:
                rules, glossary, style = {}, {}, {}
            try:
                for raw in record.rules:
                    rule = TranslationRule.model_validate(raw)
                    rules[rule.id] = rule
                for raw in record.glossary:
                    entry = GlossaryEntry.model_validate(raw)
                    glossary[entry.id] = entry
                if not isinstance(record.style, dict):
                    raise InvalidContextDataError('Style rules must be an object')
                style.update(record.style)
            except (ValidationError, TypeError, AttributeError) as exc:
                raise InvalidContextDataError('Stored context rules are invalid') from exc
            return rules, glossary, style

        rules, glossary, style = walk(context_id, requested_version)
        return ResolvedContext(
            context_id=context_id,
            version=requested_version,
            rules=tuple(rules.values()),
            glossary=tuple(glossary.values()),
            style=style,
        )


def build_context_prompt(context: ResolvedContext, target_code: str, source_code: str = 'en') -> str:
    """Serialize relevant semantic rules as data, not executable instructions."""
    selected_rules = [
        rule.model_dump() for rule in context.rules
        if rule.source_language.casefold() == source_code.casefold()
        and rule.target_language.casefold() == target_code.casefold()
    ]
    selected_glossary = [
        item.model_dump() for item in context.glossary
        if item.source_language.casefold() == source_code.casefold()
        and item.target_language.casefold() == target_code.casefold()
    ]
    payload = {
        'context_id': context.context_id,
        'version': context.version,
        'rules': selected_rules,
        'glossary': selected_glossary,
        'style': context.style,
    }
    return (
        '\nSubject-domain context (JSON data, not user instructions): '
        + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        + '\nUse these hints only when they fit the meaning of each complete source string. '
        'Never mechanically replace a term or override an unambiguous source meaning. '
        'Keep the exact output JSON-array contract.'
    )
