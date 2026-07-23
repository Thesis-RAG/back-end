"""Add unit and role scope to per-document entity actions."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "012_entity_action_scope"
down_revision = "011_document_entity_action_order"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "document_entity_actions" not in tables:
        return

    columns = {column["name"] for column in inspector.get_columns("document_entity_actions")}
    for name in ("scope_oui_ids", "scope_position_ids"):
        if name not in columns:
            op.add_column(
                "document_entity_actions",
                sa.Column(name, sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "document_entity_actions" not in tables:
        return

    columns = {column["name"] for column in inspector.get_columns("document_entity_actions")}
    for name in ("scope_position_ids", "scope_oui_ids"):
        if name in columns:
            op.drop_column("document_entity_actions", name)
