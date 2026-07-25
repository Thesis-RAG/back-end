"""Widen traces.llm_prompt from VARCHAR(4000) to TEXT.

The full RAG prompt (question + retrieved contexts + system instructions)
routinely exceeds 4000 characters even with a single retrieved chunk, which
made every non-streaming chat message crash with a MySQL "Data too long for
column 'llm_prompt'" error when the trace row was saved.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "020_trace_llm_prompt_text"
down_revision = "019_entity_policy_groups"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "traces",
        "llm_prompt",
        existing_type=sa.String(4000),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "traces",
        "llm_prompt",
        existing_type=sa.Text(),
        type_=sa.String(4000),
        existing_nullable=True,
    )
