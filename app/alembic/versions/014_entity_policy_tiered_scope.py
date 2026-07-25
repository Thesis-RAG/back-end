"""Split EntityPolicyRule into independent block/require/immune scopes.

block/require now each carry their own (oui_ids, position_ids) scope instead
of sharing one action+scope pair. A scope list containing the sentinel
"__ALL__" matches every user (equivalent to the old "empty scope = global"
behaviour); an empty list now means "no rule for anyone" and falls through
to the full-fallback, which is why old empty-scope rows must be migrated to
the explicit sentinel to preserve their current behaviour.
"""
from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa


revision = "014_entity_policy_tiered_scope"
down_revision = "013_global_policy_rules"
branch_labels = None
depends_on = None

ALL_SENTINEL = "__ALL__"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("entity_policy_rules")}

    new_columns = (
        ("block_scope_oui_ids", sa.Column("block_scope_oui_ids", sa.JSON(), nullable=True)),
        ("block_scope_position_ids", sa.Column("block_scope_position_ids", sa.JSON(), nullable=True)),
        ("require_scope_oui_ids", sa.Column("require_scope_oui_ids", sa.JSON(), nullable=True)),
        ("require_scope_position_ids", sa.Column("require_scope_position_ids", sa.JSON(), nullable=True)),
        ("immune_scope_oui_ids", sa.Column("immune_scope_oui_ids", sa.JSON(), nullable=True)),
        ("immune_scope_position_ids", sa.Column("immune_scope_position_ids", sa.JSON(), nullable=True)),
        ("mask_style", sa.Column("mask_style", sa.String(16), nullable=False, server_default="full")),
        ("mask_position", sa.Column("mask_position", sa.String(16), nullable=True)),
    )
    for name, column in new_columns:
        if name not in columns:
            op.add_column("entity_policy_rules", column)

    # Backfill defaults for JSON columns (server_default can't express "[]" portably).
    rules_table = sa.table(
        "entity_policy_rules",
        sa.column("id", sa.String),
        sa.column("action", sa.String),
        sa.column("scope_oui_ids", sa.JSON),
        sa.column("scope_position_ids", sa.JSON),
        sa.column("block_scope_oui_ids", sa.JSON),
        sa.column("block_scope_position_ids", sa.JSON),
        sa.column("require_scope_oui_ids", sa.JSON),
        sa.column("require_scope_position_ids", sa.JSON),
        sa.column("immune_scope_oui_ids", sa.JSON),
        sa.column("immune_scope_position_ids", sa.JSON),
    )
    rows = bind.execute(
        sa.text(
            "SELECT id, action, scope_oui_ids, scope_position_ids FROM entity_policy_rules"
        )
    ).fetchall()
    for row_id, action, scope_oui_ids, scope_position_ids in rows:
        oui_ids = scope_oui_ids if isinstance(scope_oui_ids, list) else json.loads(scope_oui_ids or "[]")
        position_ids = scope_position_ids if isinstance(scope_position_ids, list) else json.loads(scope_position_ids or "[]")
        is_global = not oui_ids and not position_ids

        block_oui, block_pos = [], []
        require_oui, require_pos = [], []
        if action == "block":
            block_oui = [ALL_SENTINEL] if is_global else oui_ids
            block_pos = [] if is_global else position_ids
        elif action == "mask":
            require_oui = [ALL_SENTINEL] if is_global else oui_ids
            require_pos = [] if is_global else position_ids
        # action == "full" -> leave both tiers empty, resolves to full fallback.

        bind.execute(
            sa.update(rules_table)
            .where(rules_table.c.id == row_id)
            .values(
                block_scope_oui_ids=block_oui,
                block_scope_position_ids=block_pos,
                require_scope_oui_ids=require_oui,
                require_scope_position_ids=require_pos,
                immune_scope_oui_ids=[],
                immune_scope_position_ids=[],
            )
        )

    # All existing rows were just backfilled above, so it is safe to enforce
    # NOT NULL without a MySQL server-side expression default (JSON_ARRAY()
    # defaults require MySQL 8.0.13+; the ORM's default=list covers new rows).
    for name in (
        "block_scope_oui_ids", "block_scope_position_ids",
        "require_scope_oui_ids", "require_scope_position_ids",
        "immune_scope_oui_ids", "immune_scope_position_ids",
    ):
        op.alter_column("entity_policy_rules", name, existing_type=sa.JSON(), nullable=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("entity_policy_rules")}
    for name in (
        "mask_position", "mask_style",
        "immune_scope_position_ids", "immune_scope_oui_ids",
        "require_scope_position_ids", "require_scope_oui_ids",
        "block_scope_position_ids", "block_scope_oui_ids",
    ):
        if name in columns:
            op.drop_column("entity_policy_rules", name)
