"""SQLAlchemy models for immutable-by-policy, pinned context versions."""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, ForeignKey, ForeignKeyConstraint,
    Integer, JSON, String, Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


JSON_DOCUMENT = JSON().with_variant(JSONB(), 'postgresql')


class Context(Base):
    __tablename__ = 'contexts'

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    published_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (
        CheckConstraint('published_version IS NULL OR published_version > 0', name='ck_context_published_version'),
    )


class ContextVersion(Base):
    __tablename__ = 'context_versions'

    context_id: Mapped[str] = mapped_column(
        String(80), ForeignKey('contexts.id', ondelete='RESTRICT'), primary_key=True,
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    parent_context_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    parent_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rules: Mapped[list] = mapped_column(JSON_DOCUMENT, nullable=False, default=list)
    glossary: Mapped[list] = mapped_column(JSON_DOCUMENT, nullable=False, default=list)
    style: Mapped[dict] = mapped_column(JSON_DOCUMENT, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ['parent_context_id', 'parent_version'],
            ['context_versions.context_id', 'context_versions.version'],
            name='fk_context_version_parent', ondelete='RESTRICT',
        ),
        CheckConstraint('version > 0', name='ck_context_version_positive'),
        CheckConstraint("status IN ('draft', 'published')", name='ck_context_version_status'),
        CheckConstraint(
            '(parent_context_id IS NULL AND parent_version IS NULL) OR '
            '(parent_context_id IS NOT NULL AND parent_version IS NOT NULL)',
            name='ck_context_parent_pair',
        ),
    )
