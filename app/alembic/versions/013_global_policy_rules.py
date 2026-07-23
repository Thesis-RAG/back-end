"""Add the administrator-managed global entity policy catalogue."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "013_global_policy_rules"
down_revision = "012_entity_action_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "entity_policy_rules" not in tables:
        op.create_table(
            "entity_policy_rules",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("policy_profile", sa.String(64), nullable=False, server_default="enterprise_secure"),
            sa.Column("entity_key", sa.String(128), nullable=False),
            sa.Column("display_name", sa.String(255), nullable=False),
            sa.Column("group_name", sa.String(128), nullable=False, server_default="general"),
            sa.Column("detection_source", sa.String(16), nullable=False, server_default="gliner"),
            sa.Column("action", sa.String(16), nullable=False, server_default="full"),
            sa.Column("enabled", sa.Boolean, nullable=False, server_default="1"),
            sa.Column("scope_oui_ids", sa.JSON(), nullable=False),
            sa.Column("scope_position_ids", sa.JSON(), nullable=False),
            sa.Column("priority", sa.Integer, nullable=False, server_default="100"),
            sa.Column("metadata_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("policy_profile", "entity_key", name="uq_entity_policy_rule_profile_key"),
        )
        op.create_index("ix_entity_policy_rules_profile", "entity_policy_rules", ["policy_profile"])
        op.create_index("ix_entity_policy_rules_entity_key", "entity_policy_rules", ["entity_key"])
        op.create_index("ix_entity_policy_rules_enabled", "entity_policy_rules", ["enabled"])
        op.create_index("ix_entity_policy_rules_priority", "entity_policy_rules", ["priority"])

    version_columns = {column["name"] for column in inspector.get_columns("document_versions")}
    for name, column in (
        ("policy_profile", sa.Column("policy_profile", sa.String(64), nullable=False, server_default="enterprise_secure")),
        ("policy_version", sa.Column("policy_version", sa.String(32), nullable=False, server_default="policy-v1")),
        ("resolved_rules_json", sa.Column("resolved_rules_json", sa.JSON(), nullable=True)),
        ("confirmed_labels_json", sa.Column("confirmed_labels_json", sa.JSON(), nullable=True)),
    ):
        if name not in version_columns:
            op.add_column("document_versions", column)

    snapshot_columns = {column["name"] for column in inspector.get_columns("document_policy_snapshots")}
    for name, column in (
        ("policy_profile", sa.Column("policy_profile", sa.String(64), nullable=False, server_default="enterprise_secure")),
        ("resolved_rules_json", sa.Column("resolved_rules_json", sa.JSON(), nullable=True)),
        ("confirmed_labels_json", sa.Column("confirmed_labels_json", sa.JSON(), nullable=True)),
    ):
        if name not in snapshot_columns:
            op.add_column("document_policy_snapshots", column)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "document_policy_snapshots" in tables:
        columns = {column["name"] for column in inspector.get_columns("document_policy_snapshots")}
        for name in ("confirmed_labels_json", "resolved_rules_json", "policy_profile"):
            if name in columns:
                op.drop_column("document_policy_snapshots", name)
    if "document_versions" in tables:
        columns = {column["name"] for column in inspector.get_columns("document_versions")}
        for name in ("confirmed_labels_json", "resolved_rules_json", "policy_version", "policy_profile"):
            if name in columns:
                op.drop_column("document_versions", name)
    if "entity_policy_rules" in tables:
        op.drop_table("entity_policy_rules")
