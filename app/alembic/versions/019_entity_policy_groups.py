"""Entity policy groups — a managed, creatable list of group names (Policy
page's "Nhóm" field), instead of inferring the group list from whatever
strings happen to already be in use on entity_policy_rules.group_name.
Also renames the existing hard-coded English group names to Vietnamese.

group_name on entity_policy_rules stays a plain string column, matched
against entity_policy_groups.name — not a foreign key, so a group rename
or delete never invalidates existing rule rows.
"""
from __future__ import annotations

import uuid

from alembic import op
import sqlalchemy as sa


revision = "019_entity_policy_groups"
down_revision = "018_entity_sensitivity_level"
branch_labels = None
depends_on = None

DEFAULT_POLICY_PROFILE = "enterprise_secure"

# old English group_name -> new Vietnamese name
GROUP_RENAME: dict[str, str] = {
    "security": "Bảo mật",
    "hr_financial": "Nhân sự & Tài chính",
    "financial": "Tài chính",
    "customer": "Khách hàng",
    "strategy": "Chiến lược",
    "legal": "Pháp lý",
    "pii": "Thông tin cá nhân",
    "general": "Chung",
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "entity_policy_groups" not in inspector.get_table_names():
        op.create_table(
            "entity_policy_groups",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("policy_profile", sa.String(length=64), nullable=False, server_default=DEFAULT_POLICY_PROFILE),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("policy_profile", "name", name="uq_entity_policy_group_profile_name"),
        )
        op.create_index(
            "ix_entity_policy_groups_policy_profile", "entity_policy_groups", ["policy_profile"],
        )

    groups_table = sa.table(
        "entity_policy_groups",
        sa.column("id", sa.String),
        sa.column("policy_profile", sa.String),
        sa.column("name", sa.String),
    )
    rules_table = sa.table(
        "entity_policy_rules",
        sa.column("id", sa.String),
        sa.column("group_name", sa.String),
    )

    # Rename existing rule rows' group_name to Vietnamese first, so the
    # group rows created below match what rules actually use.
    for old_name, new_name in GROUP_RENAME.items():
        bind.execute(
            sa.update(rules_table).where(rules_table.c.group_name == old_name).values(group_name=new_name)
        )

    existing = {
        row[0] for row in bind.execute(
            sa.text("SELECT name FROM entity_policy_groups WHERE policy_profile = :p"),
            {"p": DEFAULT_POLICY_PROFILE},
        ).fetchall()
    }
    distinct_group_names = {
        row[0] for row in bind.execute(sa.text("SELECT DISTINCT group_name FROM entity_policy_rules")).fetchall()
        if row[0]
    }
    for name in sorted(set(GROUP_RENAME.values()) | distinct_group_names):
        if name in existing:
            continue
        bind.execute(
            sa.insert(groups_table).values(id=str(uuid.uuid4()), policy_profile=DEFAULT_POLICY_PROFILE, name=name)
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    rules_table = sa.table(
        "entity_policy_rules",
        sa.column("id", sa.String),
        sa.column("group_name", sa.String),
    )
    for old_name, new_name in GROUP_RENAME.items():
        bind.execute(
            sa.update(rules_table).where(rules_table.c.group_name == new_name).values(group_name=old_name)
        )

    if "entity_policy_groups" in inspector.get_table_names():
        op.drop_table("entity_policy_groups")
