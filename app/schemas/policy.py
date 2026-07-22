from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Entity Types ──────────────────────────────────────────────────────────────

class EntityTypeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    domain_id: str
    entity_type: str
    label_vi: Optional[str] = None
    is_system_suggested: bool
    is_active: bool
    created_at: datetime
    boolean_labels: list[str] = []   # Computed from ENTITY_FLAG_MAP, not stored in DB.


class EntityTypeCreate(BaseModel):
    entity_type: str = Field(..., min_length=1, max_length=128)
    label_vi: Optional[str] = Field(None, max_length=255)
    boolean_labels: list[str] = []   # Populated when returned from suggestion/list endpoints.


class EntityTypeBulkCreate(BaseModel):
    entity_types: list[EntityTypeCreate]


# ── Rules ─────────────────────────────────────────────────────────────────────

class RuleConditions(BaseModel):
    min_sensitivity: Optional[str] = None           # Public|Internal|Confidential|Restricted|TopSecret.
    applicable_roles: list[str] = []                # Exemption: This role → rule does not apply.
    blocked_roles: list[str] = []                   # Coercion: This role → Rule automatically applied.
    cross_dept_only: bool = False                   # Activated when the document is at a higher organizational level than the user.
    applicable_intents: list[str] = []              # Leave blank = all intents.
    min_user_level: Optional[int] = None
    # Field/entity scope.  Empty means the rule keeps the legacy chunk-wide behaviour.
    target_entity_types: list[str] = []              # e.g. money, email, person_name.
    target_flags: list[str] = []                    # e.g. has_financial, has_pii.
    # Concrete organisation-unit-instance IDs.  Empty means all departments.
    applicable_oui_ids: list[str] = []


class RuleContract(BaseModel):
    # Top-level violation.
    violation_action: str = "conditional"    # Block | Watermark | Allow | Conditional.

    # Use only when violation_action = "conditional".
    max_detail: str = "generalize"           # Redact | Anonymize | Generalize | Summarize.
    numeric_granularity: str = "aggregated"  # Hidden | Aggregated | Range_only | Exact.


def derive_action(violation_action: str, max_detail: str = "") -> str:
    """Map violation_action (+ max_detail khi conditional) → internal action."""
    v = violation_action.lower()
    if v == "block":
        return "DENY"
    if v == "watermark":
        return "ALLOW_WITH_WATERMARK"
    if v == "allow":
        return "ALLOW"
    if v == "conditional":
        d = max_detail.lower()
        if d == "redact":
            return "REDACT"
        if d == "anonymize":
            return "ANONYMIZE"
        if d == "summarize":
            return "SUMMARIZE"
        return "GENERALIZE"   # Default conditional = generalize.
    # Backward-compat: old violation_action values.
    _legacy = {"mask": "REDACT", "generalize": "GENERALIZE"}
    return _legacy.get(v, "ALLOW")


class DomainRuleCreate(BaseModel):
    rule_code: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=255)
    priority: int = Field(default=50, ge=0, le=100)
    mandatory: bool = False
    audit_log: bool = True
    conditions: RuleConditions = Field(default_factory=RuleConditions)
    contract: RuleContract = Field(default_factory=RuleContract)


class DomainRuleUpdate(BaseModel):
    name: Optional[str] = None
    priority: Optional[int] = Field(None, ge=0, le=100)
    mandatory: Optional[bool] = None
    is_active: Optional[bool] = None
    audit_log: Optional[bool] = None
    conditions: Optional[RuleConditions] = None
    contract: Optional[RuleContract] = None


class DomainRuleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    domain_id: str
    rule_code: str
    name: str
    action: str           # Derived from violation_action, used internally.
    priority: int
    mandatory: bool
    is_active: bool
    audit_log: bool
    conditions_json: dict
    contract_json: dict
    created_at: datetime
    updated_at: datetime


class InstallRuleTemplatesRequest(BaseModel):
    template_codes: list[str] = []  # Empty = install the full recommended bundle.
    domain_id: str


# ── Domains ───────────────────────────────────────────────────────────────────

class PolicyDomainCreate(BaseModel):
    code: str = Field(..., min_length=1, max_length=64, pattern=r'^[A-Z0-9\-]+$')
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    base_sensitivity: int = Field(default=2, ge=1, le=5)


class PolicyDomainUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    base_sensitivity: Optional[int] = Field(None, ge=1, le=5)
    is_active: Optional[bool] = None


class PolicyDomainRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    code: str
    name: str
    description: Optional[str]
    base_sensitivity: int
    is_active: bool
    entity_types: list[EntityTypeRead] = []
    rules: list[DomainRuleRead] = []
    created_at: datetime
    updated_at: datetime


class PolicyDomainSummary(BaseModel):
    """Compact view for list endpoint."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    code: str
    name: str
    description: Optional[str]
    base_sensitivity: int
    is_active: bool
    entity_type_count: int = 0
    rule_count: int = 0
    created_at: datetime


# ── Entity Suggestion ─────────────────────────────────────────────────────────

class SuggestEntitiesRequest(BaseModel):
    name: str
    description: Optional[str] = None


class SuggestEntitiesResponse(BaseModel):
    entity_types: list[EntityTypeCreate]


# ── Policy Contract (runtime, not stored in DB) ───────────────────────────────

class PolicyContractRead(BaseModel):
    contract_id: str
    chunk_id: str
    generated_at: str

    domains: list[dict]
    effective_sensitivity: str
    pii_detected: bool

    decision: Literal["ALLOW", "DENY", "REDACT", "ANONYMIZE", "GENERALIZE", "SUMMARIZE", "ALLOW_WITH_WATERMARK", "FIELD_SCOPED"]
    max_detail: str
    numeric_granularity: str
    violation_action: str

    applied_rules: list[dict]
    needs_human_review: bool
    intent_class: str
