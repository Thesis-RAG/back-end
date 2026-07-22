"""remove global policy rules

Revision ID: 006_remove_global_rules
Revises: 005_message_attached_file_name
Create Date: 2026-07-22
"""
from alembic import op
import sqlalchemy as sa

revision = "006_remove_global_rules"
down_revision = "005_message_attached_file_name"
branch_labels = None
depends_on = None


def upgrade():
    # Global rules are no longer part of the policy model. Remove legacy rows
    # before enforcing that every rule belongs to a concrete domain.
    op.execute(sa.text("DELETE FROM domain_rules WHERE domain_id IS NULL"))
    op.alter_column(
        "domain_rules",
        "domain_id",
        existing_type=sa.String(36),
        nullable=False,
    )


def downgrade():
    # Downgrade restores schema compatibility but cannot restore deleted rules.
    op.alter_column(
        "domain_rules",
        "domain_id",
        existing_type=sa.String(36),
        nullable=True,
    )
