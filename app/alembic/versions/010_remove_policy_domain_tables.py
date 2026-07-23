"""Retire the legacy domain/rule policy tables.

Entity actions are now owned by document versions. The old tables are no
longer read or written by the application, so this migration removes them in
both local and production databases after migration 009 has landed.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "010_remove_policy_domain_tables"
down_revision = "009_document_entity_policies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = set(inspector.get_table_names())
    # Children first because the legacy schema has foreign keys to
    # policy_domains. These are exact legacy table names, not broad patterns.
    for table_name in ("domain_rules", "domain_entity_types", "policy_domains"):
        if table_name in existing:
            op.drop_table(table_name)


def downgrade() -> None:
    # The legacy schema was intentionally retired and is not recreated during
    # rollback. Restore from a database backup if a historical deployment
    # needs those tables again.
    pass
