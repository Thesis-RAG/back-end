from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_serializer


def _utc(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() + "Z" if value else None


class MessageEntityAccessRequestCreate(BaseModel):
    message_id: str
    entity_types: list[str] = Field(..., min_length=1)


class MessageEntityAccessRequestRead(BaseModel):
    id: str
    message_id: str
    conversation_id: Optional[str] = None
    message_excerpt: Optional[str] = None
    user_id: str
    requester_name: Optional[str] = None
    requester_email: Optional[str] = None
    entity_types: list[str] = Field(default_factory=list)
    status: str
    admin_id: Optional[str] = None
    admin_note: Optional[str] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None

    @field_serializer("created_at", "resolved_at")
    def _serialize_dt(self, value):
        return _utc(value)


class MessageAccessRequestResolve(BaseModel):
    admin_note: Optional[str] = None
