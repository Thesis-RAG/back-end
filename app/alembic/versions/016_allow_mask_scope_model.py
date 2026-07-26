"""Collapse block/require/immune tiers into allow_scope + mask_scope, fallback=block.

New resolve order: allow_scope (always full, checked first) > mask_scope
(mask + per-message appeal) > block (pure fallback — no longer a configurable
scope; whoever matches neither allow nor mask is hard-blocked, no appeal).

allow_scope inherits the old immune_scope's role unchanged (same "always
full, checked first" semantics). mask_scope inherits require_scope
unchanged. Old block_scope is dropped outright: a rule that used to be
block_scope=ALL with nothing else needs no explicit scope at all now, since
blocking everyone is simply what the new fallback does by default.

The one case needing a compensating write: a rule where block_scope,
require_scope AND immune_scope were all empty used to fall through to the
old fallback of "full" for everyone. Under the new fallback of "block", that
same empty rule would silently start blocking everyone instead — so those
rows get allow_scope_oui_ids=["__ALL__"] written in explicitly to preserve
their previous behaviour.
"""
from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa


revision = "016_allow_mask_scope"
down_revision = "015_message_entity_access_requests"
branch_labels = None
depends_on = None

ALL_SENTINEL = "__ALL__"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("entity_policy_rules")}

    new_columns = (
        ("allow_scope_oui_ids", sa.Column("allow_scope_oui_ids", sa.JSON(), nullable=True)),
        ("allow_scope_position_ids", sa.Column("allow_scope_position_ids", sa.JSON(), nullable=True)),
        ("mask_scope_oui_ids", sa.Column("mask_scope_oui_ids", sa.JSON(), nullable=True)),
        ("mask_scope_position_ids", sa.Column("mask_scope_position_ids", sa.JSON(), nullable=True)),
    )
    for name, column in new_columns:
        if name not in columns:
            op.add_column("entity_policy_rules", column)

    def _load(value) -> list:
        if isinstance(value, list):
            return value
        try:
            return json.loads(value or "[]")
        except (TypeError, ValueError):
            return []

    rows = bind.execute(sa.text(
        "SELECT id, block_scope_oui_ids, block_scope_position_ids, "
        "require_scope_oui_ids, require_scope_position_ids, "
        "immune_scope_oui_ids, immune_scope_position_ids FROM entity_policy_rules"
    )).fetchall()

    rules_table = sa.table(
        "entity_policy_rules",
        sa.column("id", sa.String),
        sa.column("allow_scope_oui_ids", sa.JSON),
        sa.column("allow_scope_position_ids", sa.JSON),
        sa.column("mask_scope_oui_ids", sa.JSON),
        sa.column("mask_scope_position_ids", sa.JSON),
    )
    for row_id, block_oui, block_pos, require_oui, require_pos, immune_oui, immune_pos in rows:
        block_oui, block_pos = _load(block_oui), _load(block_pos)
        require_oui, require_pos = _load(require_oui), _load(require_pos)
        allow_oui, allow_pos = _load(immune_oui), _load(immune_pos)

        if not any((block_oui, block_pos, require_oui, require_pos, allow_oui, allow_pos)):
            allow_oui = [ALL_SENTINEL]

        bind.execute(
            sa.update(rules_table)
            .where(rules_table.c.id == row_id)
            .values(
                allow_scope_oui_ids=allow_oui,
                allow_scope_position_ids=allow_pos,
                mask_scope_oui_ids=require_oui,
                mask_scope_position_ids=require_pos,
            )
        )

    for name in ("allow_scope_oui_ids", "allow_scope_position_ids", "mask_scope_oui_ids", "mask_scope_position_ids"):
        op.alter_column("entity_policy_rules", name, existing_type=sa.JSON(), nullable=False)

    for name in (
        "block_scope_oui_ids", "block_scope_position_ids",
        "require_scope_oui_ids", "require_scope_position_ids",
        "immune_scope_oui_ids", "immune_scope_position_ids",
    ):
        if name in columns:
            op.drop_column("entity_policy_rules", name)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("entity_policy_rules")}

    for name, column in (
        ("block_scope_oui_ids", sa.Column("block_scope_oui_ids", sa.JSON(), nullable=True)),
        ("block_scope_position_ids", sa.Column("block_scope_position_ids", sa.JSON(), nullable=True)),
        ("require_scope_oui_ids", sa.Column("require_scope_oui_ids", sa.JSON(), nullable=True)),
        ("require_scope_position_ids", sa.Column("require_scope_position_ids", sa.JSON(), nullable=True)),
        ("immune_scope_oui_ids", sa.Column("immune_scope_oui_ids", sa.JSON(), nullable=True)),
        ("immune_scope_position_ids", sa.Column("immune_scope_position_ids", sa.JSON(), nullable=True)),
    ):
        if name not in columns:
            op.add_column("entity_policy_rules", column)

    bind.execute(sa.text(
        "UPDATE entity_policy_rules SET "
        "immune_scope_oui_ids = allow_scope_oui_ids, immune_scope_position_ids = allow_scope_position_ids, "
        "require_scope_oui_ids = mask_scope_oui_ids, require_scope_position_ids = mask_scope_position_ids, "
        "block_scope_oui_ids = JSON_ARRAY(), block_scope_position_ids = JSON_ARRAY()"
    ))

    for name in (
        "block_scope_oui_ids", "block_scope_position_ids",
        "require_scope_oui_ids", "require_scope_position_ids",
        "immune_scope_oui_ids", "immune_scope_position_ids",
    ):
        op.alter_column("entity_policy_rules", name, existing_type=sa.JSON(), nullable=False)

    for name in ("allow_scope_oui_ids", "allow_scope_position_ids", "mask_scope_oui_ids", "mask_scope_position_ids"):
        op.drop_column("entity_policy_rules", name)
