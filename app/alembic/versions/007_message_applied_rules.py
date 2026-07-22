"""store policy rules applied to assistant messages

Revision ID: 007_message_applied_rules
Revises: 006_remove_global_rules
Create Date: 2026-07-22
"""
from alembic import op
import sqlalchemy as sa


revision = "007_message_applied_rules"
down_revision = "006_remove_global_rules"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("messages")}
    if "applied_rules_json" not in columns:
        op.add_column("messages", sa.Column("applied_rules_json", sa.JSON(), nullable=True))


def downgrade():
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("messages")}
    if "applied_rules_json" in columns:
        op.drop_column("messages", "applied_rules_json")
