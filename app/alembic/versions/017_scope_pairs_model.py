"""Replace flat (oui_ids, position_ids) rectangles with explicit scope pairs.

The old model AND'd two independent lists together — matching the full
cartesian product of every selected unit against every selected position.
That made it impossible to express "HCM's Trưởng chi nhánh + Hà Nội's Phó
chi nhánh" without also matching "HCM's Phó chi nhánh" and "Hà Nội's Trưởng
chi nhánh". The new model stores an explicit list of (oui_id, position_id)
pairs per scope, where either side of a pair may be null to mean "any" on
that dimension — a pair fully replaces one old (oui_ids × position_ids)
rectangle, so existing rows are migrated as the cartesian product of their
old lists (which is exactly the set of pairs they used to match).
"""
from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa


revision = "017_scope_pairs_model"
down_revision = "016_allow_mask_scope"
branch_labels = None
depends_on = None

ALL_SENTINEL = "__ALL__"


def _load(value) -> list:
    if isinstance(value, list):
        return value
    try:
        return json.loads(value or "[]")
    except (TypeError, ValueError):
        return []


def _rectangle_to_pairs(oui_ids: list, position_ids: list) -> list[dict]:
    if ALL_SENTINEL in oui_ids or ALL_SENTINEL in position_ids:
        return [{"oui_id": ALL_SENTINEL, "position_id": None}]
    if not oui_ids and not position_ids:
        return []
    if oui_ids and position_ids:
        return [{"oui_id": o, "position_id": p} for o in oui_ids for p in position_ids]
    if oui_ids:
        return [{"oui_id": o, "position_id": None} for o in oui_ids]
    return [{"oui_id": None, "position_id": p} for p in position_ids]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("entity_policy_rules")}

    if "allow_scope_pairs" not in columns:
        op.add_column("entity_policy_rules", sa.Column("allow_scope_pairs", sa.JSON(), nullable=True))
    if "mask_scope_pairs" not in columns:
        op.add_column("entity_policy_rules", sa.Column("mask_scope_pairs", sa.JSON(), nullable=True))

    rows = bind.execute(sa.text(
        "SELECT id, allow_scope_oui_ids, allow_scope_position_ids, "
        "mask_scope_oui_ids, mask_scope_position_ids FROM entity_policy_rules"
    )).fetchall()

    rules_table = sa.table(
        "entity_policy_rules",
        sa.column("id", sa.String),
        sa.column("allow_scope_pairs", sa.JSON),
        sa.column("mask_scope_pairs", sa.JSON),
    )
    for row_id, allow_oui, allow_pos, mask_oui, mask_pos in rows:
        bind.execute(
            sa.update(rules_table)
            .where(rules_table.c.id == row_id)
            .values(
                allow_scope_pairs=_rectangle_to_pairs(_load(allow_oui), _load(allow_pos)),
                mask_scope_pairs=_rectangle_to_pairs(_load(mask_oui), _load(mask_pos)),
            )
        )

    op.alter_column("entity_policy_rules", "allow_scope_pairs", existing_type=sa.JSON(), nullable=False)
    op.alter_column("entity_policy_rules", "mask_scope_pairs", existing_type=sa.JSON(), nullable=False)

    for name in ("allow_scope_oui_ids", "allow_scope_position_ids", "mask_scope_oui_ids", "mask_scope_position_ids"):
        if name in columns:
            op.drop_column("entity_policy_rules", name)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("entity_policy_rules")}

    for name, column in (
        ("allow_scope_oui_ids", sa.Column("allow_scope_oui_ids", sa.JSON(), nullable=True)),
        ("allow_scope_position_ids", sa.Column("allow_scope_position_ids", sa.JSON(), nullable=True)),
        ("mask_scope_oui_ids", sa.Column("mask_scope_oui_ids", sa.JSON(), nullable=True)),
        ("mask_scope_position_ids", sa.Column("mask_scope_position_ids", sa.JSON(), nullable=True)),
    ):
        if name not in columns:
            op.add_column("entity_policy_rules", column)

    rows = bind.execute(sa.text("SELECT id, allow_scope_pairs, mask_scope_pairs FROM entity_policy_rules")).fetchall()
    rules_table = sa.table(
        "entity_policy_rules",
        sa.column("id", sa.String),
        sa.column("allow_scope_oui_ids", sa.JSON),
        sa.column("allow_scope_position_ids", sa.JSON),
        sa.column("mask_scope_oui_ids", sa.JSON),
        sa.column("mask_scope_position_ids", sa.JSON),
    )
    for row_id, allow_pairs, mask_pairs in rows:
        allow_pairs, mask_pairs = _load(allow_pairs), _load(mask_pairs)
        allow_oui = sorted({p.get("oui_id") for p in allow_pairs if p.get("oui_id")})
        allow_pos = sorted({p.get("position_id") for p in allow_pairs if p.get("position_id")})
        mask_oui = sorted({p.get("oui_id") for p in mask_pairs if p.get("oui_id")})
        mask_pos = sorted({p.get("position_id") for p in mask_pairs if p.get("position_id")})
        bind.execute(
            sa.update(rules_table)
            .where(rules_table.c.id == row_id)
            .values(
                allow_scope_oui_ids=allow_oui, allow_scope_position_ids=allow_pos,
                mask_scope_oui_ids=mask_oui, mask_scope_position_ids=mask_pos,
            )
        )

    for name in ("allow_scope_oui_ids", "allow_scope_position_ids", "mask_scope_oui_ids", "mask_scope_position_ids"):
        op.alter_column("entity_policy_rules", name, existing_type=sa.JSON(), nullable=False)

    op.drop_column("entity_policy_rules", "allow_scope_pairs")
    op.drop_column("entity_policy_rules", "mask_scope_pairs")
