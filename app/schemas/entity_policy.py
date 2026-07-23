from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


EntityAction = Literal["block", "full", "mask"]
DetectionSource = Literal["gliner", "regex", "manual"]


class PolicyRuleInput(BaseModel):
    entity_key: str = Field(..., min_length=1, max_length=128)
    display_name: str = Field(..., min_length=1, max_length=255)
    group_name: str = Field(default="general", min_length=1, max_length=128)
    detection_source: DetectionSource = "manual"
    action: EntityAction = "full"
    enabled: bool = True
    scope_oui_ids: list[str] = Field(default_factory=list)
    scope_position_ids: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=0, le=100000)
    metadata_json: dict = Field(default_factory=dict)


class PolicyRuleRead(PolicyRuleInput):
    model_config = ConfigDict(from_attributes=True)

    id: str
    policy_profile: str
    created_at: datetime
    updated_at: datetime


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
