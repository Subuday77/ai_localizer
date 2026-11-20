"""Add version-pinned, language-independent professional specialization profiles.

Revision ID: 0002_domain_profiles
Revises: 0001_contexts
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '0002_domain_profiles'
down_revision = '0001_contexts'
branch_labels = None
depends_on = None


def upgrade():
    json_type = sa.JSON().with_variant(JSONB(), 'postgresql')
    # Nullable for historical v1 snapshots: no silent reinterpretation of past versions.
    op.add_column('context_versions', sa.Column('domain_profile', json_type, nullable=True))


def downgrade():
    op.drop_column('context_versions', 'domain_profile')
