"""Persist the user-defined order of entity actions."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "011_document_entity_action_order"
down_revision = "010_remove_policy_domain_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "document_entity_actions" not in tables:
        return

    columns = {column["name"] for column in inspector.get_columns("document_entity_actions")}
    if "sort_order" in columns:
        return

    op.add_column(
        "document_entity_actions",
        sa.Column("sort_order", sa.Integer, nullable=False, server_default="0"),
    )

    actions = sa.table(
        "document_entity_actions",
        sa.column("id", sa.String(36)),
        sa.column("document_version_id", sa.String(36)),
        sa.column("entity_type", sa.String(128)),
        sa.column("sort_order", sa.Integer),
    )
    rows = bind.execute(
        sa.select(actions.c.id, actions.c.document_version_id, actions.c.entity_type)
        .order_by(actions.c.document_version_id, actions.c.entity_type, actions.c.id)
    ).all()
    positions: dict[str, int] = {}
    for row in rows:
        position = positions.get(row.document_version_id, 0)
        bind.execute(
            actions.update()
            .where(actions.c.id == row.id)
            .values(sort_order=position)
        )
        positions[row.document_version_id] = position + 1

    op.create_index(
        "ix_document_entity_actions_sort_order",
        "document_entity_actions",
        ["document_version_id", "sort_order"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "document_entity_actions" not in tables:
        return
    columns = {column["name"] for column in inspector.get_columns("document_entity_actions")}
    if "sort_order" not in columns:
        return
    indexes = {index["name"] for index in inspector.get_indexes("document_entity_actions")}
    if "ix_document_entity_actions_sort_order" in indexes:
        op.drop_index(
            "ix_document_entity_actions_sort_order",
            table_name="document_entity_actions",
        )
    op.drop_column("document_entity_actions", "sort_order")
