"""scope policy domains to concrete organization-unit instances

Revision ID: 008_policy_domain_oui_scope
Revises: 007_message_applied_rules
Create Date: 2026-07-22
"""
from alembic import op
import sqlalchemy as sa


revision = "008_policy_domain_oui_scope"
down_revision = "007_message_applied_rules"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("policy_domains")}
    if "org_unit_instance_id" not in columns:
        op.add_column(
            "policy_domains",
            sa.Column(
                "org_unit_instance_id",
                sa.String(36),
                sa.ForeignKey("org_unit_instances.id"),
                nullable=True,
            ),
        )

    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("policy_domains")}
    if "ix_policy_domains_org_unit_instance_id" not in indexes:
        op.create_index(
            "ix_policy_domains_org_unit_instance_id",
            "policy_domains",
            ["org_unit_instance_id"],
            unique=True,
        )


def downgrade():
    bind = op.get_bind()
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("policy_domains")}
    if "ix_policy_domains_org_unit_instance_id" in indexes:
        op.drop_index("ix_policy_domains_org_unit_instance_id", table_name="policy_domains")

    columns = {column["name"] for column in sa.inspect(bind).get_columns("policy_domains")}
    if "org_unit_instance_id" in columns:
        op.drop_column("policy_domains", "org_unit_instance_id")
