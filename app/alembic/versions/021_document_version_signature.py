"""Digital signature columns for document_versions.

Each version gets its own signature over the aggregate hash of its chunks,
computed once ingest finishes (chunk_hash values only exist after chunking).
Signed with an Ed25519 private key kept outside the database (see
app/services/document_signing_service.py) so a DB-only compromise cannot
silently tamper with already-ingested content without invalidating the
signature.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "021_document_version_signature"
down_revision = "020_trace_llm_prompt_text"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # This migration was partially applied in production before Alembic
    # could advance its version row. Keep it restart-safe so an interrupted
    # deploy can finish rather than failing on duplicate columns.
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("document_versions")}
    columns = (
        ("content_hash", sa.String(128)),
        ("content_signature", sa.Text()),
        ("content_signature_key_id", sa.String(64)),
        ("content_signed_at", sa.DateTime()),
    )
    for name, column_type in columns:
        if name not in existing:
            op.add_column(
                "document_versions",
                sa.Column(name, column_type, nullable=True),
            )


def downgrade() -> None:
    op.drop_column("document_versions", "content_signed_at")
    op.drop_column("document_versions", "content_signature_key_id")
    op.drop_column("document_versions", "content_signature")
    op.drop_column("document_versions", "content_hash")
