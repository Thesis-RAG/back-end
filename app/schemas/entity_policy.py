from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


EntityAction = Literal["block", "full", "mask"]
DetectionSource = Literal["gliner", "regex", "manual"]
MaskStyle = Literal["full", "partial"]
MaskPosition = Literal["head", "center", "tail"]


class ScopePair(BaseModel):
    # oui_id/position_id null means "any" on that dimension. A pair fully
    # replaces the old (oui_ids × position_ids) rectangle — each pair is one
    # explicit (unit, role) combination, not cross-multiplied against others.
    oui_id: Optional[str] = None
    position_id: Optional[str] = None


class PolicyRuleInput(BaseModel):
    entity_key: str = Field(..., min_length=1, max_length=128)
    display_name: str = Field(..., min_length=1, max_length=255)
    group_name: str = Field(default="general", min_length=1, max_length=128)
    detection_source: DetectionSource = "manual"
    # Legacy single-tier field, kept for the ingest-time snapshot copy.
    # Live chat/file resolution uses the tiered scopes below.
    action: EntityAction = "full"
    # Absolute sensitivity level (1-5, same scale as Document.sensitivity)
    # this entity type carries — feeds chunk_sensitivity computation at
    # ingest/resync time (see entity_extractor.compute_chunk_sensitivity).
    sensitivity: int = Field(default=2, ge=1, le=5)
    enabled: bool = True
    scope_oui_ids: list[str] = Field(default_factory=list)
    scope_position_ids: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=0, le=100000)
    metadata_json: dict = Field(default_factory=dict)

    # allow: always full, checked first. mask: mask + per-message appeal,
    # checked second. Anyone matching neither is hard-blocked (fallback,
    # not a configurable scope) — enforced by resolve_tier(), not here.
    allow_scope_pairs: list[ScopePair] = Field(default_factory=list)
    mask_scope_pairs: list[ScopePair] = Field(default_factory=list)

    mask_style: MaskStyle = "full"
    mask_position: Optional[MaskPosition] = None


class TranslateEntityKeyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=255)


class TranslateEntityKeyResponse(BaseModel):
    entity_key: str


class PolicyGroupRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    policy_profile: str
    name: str
    created_at: datetime
    updated_at: datetime


class PolicyGroupCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)


# Overwrites (not merges) allow_scope_pairs/mask_scope_pairs on every rule
# currently in the target group — see PolicyRuleService.apply_group_scope.
class ApplyGroupScopeRequest(BaseModel):
    allow_scope_pairs: list[ScopePair] = Field(default_factory=list)
    mask_scope_pairs: list[ScopePair] = Field(default_factory=list)


class PolicyRuleRead(PolicyRuleInput):
    model_config = ConfigDict(from_attributes=True)

    id: str
    policy_profile: str
    created_at: datetime
    updated_at: datetime

    # Live result of policy_rule_service.compute_graph_tiers() — the same
    # allow>mask>block fan-out the ontology graph is built from (see
    # ontology_sync_service.py). Attached onto the ORM object by the
    # endpoints below before serialization; NOT derived from `action`
    # above, which is a separate legacy single-tier field only used for
    # the ingest-time snapshot.
    resolved_allow: Optional[bool] = None
    resolved_mask: Optional[bool] = None
    resolved_block: Optional[bool] = None


class PolicyPreviewRule(BaseModel):
    entity_key: str
    display_name: str
    action: EntityAction
    detection_count: int = 0


class PolicyPreviewSummary(BaseModel):
    block: int = 0
    mask: int = 0
    full: int = 0


class EntityActionInput(BaseModel):
    entity_type: str = Field(..., min_length=1, max_length=128)
    label: Optional[str] = Field(None, max_length=255)
    action: EntityAction = "full"
    source: Literal["gliner", "regex", "manual"] = "gliner"
    enabled: bool = True
    scope_oui_ids: list[str] = Field(default_factory=list)
    scope_position_ids: list[str] = Field(default_factory=list)


class EntityActionRead(EntityActionInput):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_version_id: str
    detection_count: int = 0
    sort_order: int = 0
    metadata_json: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class EntityPreviewEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: float = 0.0
    source: str = "gliner"
    flags: list[str] = Field(default_factory=list)


class EntityPreviewResponse(BaseModel):
    file_name: str
    text_preview: str
    text_truncated: bool
    entities: list[EntityPreviewEntity] = Field(default_factory=list)
    entity_types: list[str] = Field(default_factory=list)
    confirmed_labels: list[str] = Field(default_factory=list)
    policy_profile: str = "enterprise_secure"
    policy_version: str = "policy-v1"
    applied_rules: list[PolicyPreviewRule] = Field(default_factory=list)
    action_summary: PolicyPreviewSummary = Field(default_factory=PolicyPreviewSummary)


class EntityConfigurationRead(BaseModel):
    document_id: str
    document_title: str
    document_version_id: str
    version_no: int
    file_name: str
    entity_detection_json: dict = Field(default_factory=dict)
    actions: list[EntityActionRead] = Field(default_factory=list)


class EntityConfigurationUpdate(BaseModel):
    actions: list[EntityActionInput] = Field(default_factory=list)


class RuntimeEntityContract(BaseModel):
    contract_id: str
    document_id: str
    document_version_id: str
    chunk_id: str
    query_entities: list[str] = Field(default_factory=list)
    detected_entities: list[str] = Field(default_factory=list)
    matched_entities: list[dict] = Field(default_factory=list)
    decision: str = "full"
    requires_access_request: bool = False
