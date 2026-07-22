"""add attached_file_name to messages

Revision ID: 005_message_attached_file_name
Revises: 004_document_access_requests
Create Date: 2026-07-17
"""
from alembic import op
import sqlalchemy as sa

revision = '005_message_attached_file_name'
down_revision = '004_document_access_requests'
branch_labels = None
depends_on = None


def upgrade():
    # The initial migration creates tables from the current SQLAlchemy
    # metadata. On a fresh database, this column may therefore already exist
    # before revision 005 is reached. Keep the migration safe for both fresh
    # and upgraded installations.
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("messages")}
    if "attached_file_name" not in columns:
        op.add_column("messages", sa.Column("attached_file_name", sa.String(512), nullable=True))


def downgrade():
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("messages")}
    if "attached_file_name" in columns:
        op.drop_column("messages", "attached_file_name")
