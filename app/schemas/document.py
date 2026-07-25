from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from app.schemas.job import JobRead

try:
    import app.schemas.job as _job
    JobRead = _job.JobRead
except Exception:
    JobRead = None


# Chunking config
class ChunkingConfig(BaseModel):
    """
    The chunking parameter is sent when uploading the version.
    By default, an internal (legacy) chunker is used.
    When mode = hierarchical/hybrid, Docling is used.
    """
    mode: Literal["legacy", "hierarchical", "hybrid", "llm_structured"] = "llm_structured"
    max_tokens: int = Field(default=1500, ge=64, le=4096)
    overlap_tokens: int = Field(default=80, ge=0, le=512)
    ocr: bool = False

    def to_json(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_json(cls, data: dict | None) -> "ChunkingConfig":
        if not data:
            return cls()
        return cls(**data)


# Document CRUD 
class DocumentCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    oui_ids: list[str] = []           # Multi OUI.
    sensitivity: int = 2              # 1-5.
    document_type: str = "general"
    data_type: str = "text"
    tags: Optional[list[str]] = None


class DocumentUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    oui_ids: Optional[list[str]] = None
    sensitivity: Optional[int] = None
    document_type: Optional[str] = None
    tags: Optional[list[str]] = None


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    description: Optional[str] = None
    oui_ids: list[str] = []        # ← Computed from doc.ouis
    owner_user_id: str
    owner_name: Optional[str] = None
    document_type: str
    sensitivity: int               # 1-5.
    data_type: str
    tags: list[str] = []
    status: str
    current_version_id: Optional[str] = None
    file_name: Optional[str] = None    # ← From current_version, for file-extension/icon display only.
    version_count: int = 0
    created_at: datetime
    updated_at: datetime

    @classmethod
    def model_validate(cls, obj, **kwargs):
        data = super().model_validate(obj, **kwargs)
        if hasattr(obj, "ouis"):
            data.oui_ids = [o.id for o in (obj.ouis or [])]
        if hasattr(obj, "owner") and obj.owner:
            data.owner_name = obj.owner.name
        if hasattr(obj, "current_version") and obj.current_version:
            data.file_name = obj.current_version.file_name
        return data


class DocumentVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    version_id: str = Field(alias="id")
    version_no: int
    file_name: str
    mime_type: str
    checksum: str
    source_object_id: str
    normalized_object_id: Optional[str] = None
    ingest_status: str
    parse_status: str
    chunk_status: str
    embed_status: str
    error_message: Optional[str] = None
    rule_version: str
    policy_profile: str = "enterprise_secure"
    policy_version: str = "policy-v1"
    resolved_rules_json: Optional[list] = None
    confirmed_labels_json: Optional[list[str]] = None
    chunk_config_json: Optional[dict] = None
    entity_detection_json: Optional[dict] = None
    created_at: datetime
    updated_at: datetime


class UploadVersionResponse(BaseModel):
    document: DocumentRead
    version: DocumentVersionRead
    job: "JobRead"
    queued: bool


class PolicySnapshotRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_version_id: str
    policy_version: str
    policy_profile: str = "enterprise_secure"
    resolved_rules_json: Optional[list] = None
    confirmed_labels_json: Optional[list[str]] = None
    contract_json: dict
