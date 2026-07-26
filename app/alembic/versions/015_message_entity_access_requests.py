"""Per-message entity access requests for the 'require' policy tier."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "015_message_entity_access_req"
down_revision = "014_entity_policy_tiered_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    message_columns = {column["name"] for column in inspector.get_columns("messages")}
    if "entity_masks_json" not in message_columns:
        op.add_column("messages", sa.Column("entity_masks_json", sa.JSON(), nullable=True))

    if "message_entity_access_requests" not in tables:
        op.create_table(
            "message_entity_access_requests",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "message_id", sa.String(36),
                sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column(
                "user_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("entity_types_json", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
            sa.Column(
                "admin_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
            ),
            sa.Column("admin_note", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("resolved_at", sa.DateTime(), nullable=True),
        )
        op.create_index(
            "ix_message_entity_access_requests_message_id",
            "message_entity_access_requests", ["message_id"],
        )
        op.create_index(
            "ix_message_entity_access_requests_user_id",
            "message_entity_access_requests", ["user_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "message_entity_access_requests" in tables:
        op.drop_table("message_entity_access_requests")
    message_columns = {column["name"] for column in inspector.get_columns("messages")}
    if "entity_masks_json" in message_columns:
        op.drop_column("messages", "entity_masks_json")
