"""Remove the query_scope_mode system setting.

The full_db/branch_only toggle it controlled has been removed: retrieval now
always scopes to the same document set the Documents tab shows (plus
globally public chunks) — see app/services/document_service.py
visible_document_ids and app/services/retrieval_service.py _retrieve_main.

Revision ID: 022_drop_query_scope_mode_setting
Revises: 021_document_version_signature
Create Date: 2026-08-02
"""
from alembic import op

revision = "022_drop_query_scope_mode_setting"
down_revision = "021_document_version_signature"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM system_settings WHERE `key` = 'query_scope_mode'")


def downgrade() -> None:
    op.execute(
        "INSERT IGNORE INTO system_settings (`key`, `value`, updated_at) "
        "VALUES ('query_scope_mode', 'full_db', NOW())"
    )
