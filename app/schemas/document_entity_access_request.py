from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


def _utc(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() + "Z" if value else None


class EntityAccessRequestCreate(BaseModel):
    document_id: str
    document_version_id: str
    entity_types: list[str] = Field(..., min_length=1)


class EntityAccessRequestRead(BaseModel):
    id: str
    request_kind: str = "entity"
    document_id: str
    document_version_id: str
    document_title: Optional[str] = None
    user_id: str
    requester_name: Optional[str] = None
    requester_email: Optional[str] = None
    entity_types: list[str] = Field(default_factory=list)
    status: str
    expires_at: Optional[datetime] = None
    admin_id: Optional[str] = None
    admin_note: Optional[str] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None

    @field_serializer("expires_at", "created_at", "resolved_at")
    def _serialize_dt(self, value):
        return _utc(value)
