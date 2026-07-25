"""Add a per-entity-type sensitivity level (1-5), used to compute
chunk_sensitivity from detected entities instead of an LLM guess ±1 around
the document level.

entity_policy_rules.sensitivity is the admin-configured level shown on the
Policy page. document_entity_actions.sensitivity is a frozen snapshot of
that value copied at document-version-policy-apply time, so later edits to
a rule don't retroactively change already-ingested documents (only a
resync of chunk_sensitivity for a doc-level sensitivity change re-reads
this snapshot — it never re-detects entities).
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "018_entity_sensitivity_level"
down_revision = "017_scope_pairs_model"
branch_labels = None
depends_on = None

# entity_key -> default sensitivity level (1-5), seeded once for existing
# rows; editable per-rule afterwards via the Policy page.
DEFAULT_SENSITIVITY: dict[str, int] = {
    "credential": 5, "password": 5, "api_key": 5, "token": 5, "merger": 5,
    "salary": 4, "bonus": 4, "income": 4, "payroll": 4, "social_insurance": 4,
    "national_id": 4, "dob": 4, "bank_account": 4, "tax_id": 4,
    "financial_data": 4, "contract": 4, "contract_number": 4, "strategy": 4,
    "business_plan": 4, "roadmap": 4,
    "money": 3, "percentage": 3, "customer": 3, "customer_id": 3,
    "customer_email": 3, "customer_phone": 3, "person_name": 3, "address": 3,
    "phone": 3, "email": 3, "project": 3, "project_code": 3,
    "organization": 1, "date": 1, "date_generic": 1, "location": 1,
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    rule_columns = {c["name"] for c in inspector.get_columns("entity_policy_rules")}
    if "sensitivity" not in rule_columns:
        op.add_column(
            "entity_policy_rules",
            sa.Column("sensitivity", sa.Integer(), nullable=True),
        )

    rules_table = sa.table(
        "entity_policy_rules",
        sa.column("id", sa.String),
        sa.column("entity_key", sa.String),
        sa.column("sensitivity", sa.Integer),
    )
    rows = bind.execute(sa.text("SELECT id, entity_key FROM entity_policy_rules")).fetchall()
    for row_id, entity_key in rows:
        level = DEFAULT_SENSITIVITY.get(entity_key, 2)
        bind.execute(
            sa.update(rules_table).where(rules_table.c.id == row_id).values(sensitivity=level)
        )

    op.alter_column(
        "entity_policy_rules", "sensitivity",
        existing_type=sa.Integer(), nullable=False, server_default="2",
    )

    action_columns = {c["name"] for c in inspector.get_columns("document_entity_actions")}
    if "sensitivity" not in action_columns:
        op.add_column(
            "document_entity_actions",
            sa.Column("sensitivity", sa.Integer(), nullable=False, server_default="1"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    action_columns = {c["name"] for c in inspector.get_columns("document_entity_actions")}
    if "sensitivity" in action_columns:
        op.drop_column("document_entity_actions", "sensitivity")

    rule_columns = {c["name"] for c in inspector.get_columns("entity_policy_rules")}
    if "sensitivity" in rule_columns:
        op.drop_column("entity_policy_rules", "sensitivity")
