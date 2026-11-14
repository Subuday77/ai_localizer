"""Create published context catalog and immutable-version snapshots.

Revision ID: 0001_contexts
Revises:
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '0001_contexts'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    json_type = sa.JSON().with_variant(JSONB(), 'postgresql')
    op.create_table(
        'contexts',
        sa.Column('id', sa.String(80), primary_key=True),
        sa.Column('title', sa.String(160), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('is_public', sa.Boolean(), nullable=False),
        sa.Column('published_version', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('published_version IS NULL OR published_version > 0', name='ck_context_published_version'),
    )
    op.create_table(
        'context_versions',
        sa.Column('context_id', sa.String(80), sa.ForeignKey('contexts.id', ondelete='RESTRICT'), primary_key=True),
        sa.Column('version', sa.Integer(), primary_key=True),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('parent_context_id', sa.String(80), nullable=True),
        sa.Column('parent_version', sa.Integer(), nullable=True),
        sa.Column('rules', json_type, nullable=False),
        sa.Column('glossary', json_type, nullable=False),
        sa.Column('style', json_type, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['parent_context_id', 'parent_version'],
            ['context_versions.context_id', 'context_versions.version'],
            name='fk_context_version_parent', ondelete='RESTRICT',
        ),
        sa.CheckConstraint('version > 0', name='ck_context_version_positive'),
        sa.CheckConstraint("status IN ('draft', 'published')", name='ck_context_version_status'),
        sa.CheckConstraint(
            '(parent_context_id IS NULL AND parent_version IS NULL) OR '
            '(parent_context_id IS NOT NULL AND parent_version IS NOT NULL)',
            name='ck_context_parent_pair',
        ),
    )


def downgrade():
    op.drop_table('context_versions')
    op.drop_table('contexts')
