"""add per-document entity actions and entity access requests

Revision ID: 009_document_entity_policies
Revises: 008_policy_domain_oui_scope
"""
from alembic import op
import sqlalchemy as sa


revision = "009_document_entity_policies"
down_revision = "008_policy_domain_oui_scope"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "document_entity_actions" not in tables:
        op.create_table(
            "document_entity_actions",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("document_version_id", sa.String(36),
                      sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False),
            sa.Column("entity_type", sa.String(128), nullable=False),
            sa.Column("label", sa.String(255), nullable=True),
            sa.Column("action", sa.String(16), nullable=False, server_default="full"),
            sa.Column("source", sa.String(16), nullable=False, server_default="gliner"),
            sa.Column("enabled", sa.Boolean, nullable=False, server_default="1"),
            sa.Column("detection_count", sa.Integer, nullable=False, server_default="0"),
            sa.Column("metadata_json", sa.JSON, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("document_version_id", "entity_type", name="uq_document_version_entity_action"),
        )
        op.create_index("ix_document_entity_actions_version", "document_entity_actions", ["document_version_id"])

    if "document_entity_access_requests" not in tables:
        op.create_table(
            "document_entity_access_requests",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("document_version_id", sa.String(36), sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("entity_types_json", sa.JSON, nullable=False),
            sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
            sa.Column("expires_at", sa.DateTime, nullable=True),
            sa.Column("admin_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("admin_note", sa.Text, nullable=True),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
            sa.Column("resolved_at", sa.DateTime, nullable=True),
        )
        op.create_index("ix_entity_access_requests_doc_version", "document_entity_access_requests", ["document_id", "document_version_id"])
        op.create_index("ix_entity_access_requests_user", "document_entity_access_requests", ["user_id"])

    columns = {c["name"] for c in inspector.get_columns("document_versions")}
    if "entity_detection_json" not in columns:
        op.add_column("document_versions", sa.Column("entity_detection_json", sa.JSON, nullable=True))


def downgrade():
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "document_entity_access_requests" in tables:
        op.drop_index("ix_entity_access_requests_user", table_name="document_entity_access_requests")
        op.drop_index("ix_entity_access_requests_doc_version", table_name="document_entity_access_requests")
        op.drop_table("document_entity_access_requests")
    if "document_entity_actions" in tables:
        op.drop_index("ix_document_entity_actions_version", table_name="document_entity_actions")
        op.drop_table("document_entity_actions")
    columns = {c["name"] for c in sa.inspect(bind).get_columns("document_versions")}
    if "entity_detection_json" in columns:
        op.drop_column("document_versions", "entity_detection_json")
