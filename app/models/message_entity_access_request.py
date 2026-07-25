from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text

from app.db.base import Base
from app.utils.ids import new_uuid


class MessageEntityAccessRequest(Base):
    """Per-message request to reveal a 'require' entity's real value.

    Unlike DocumentEntityAccessRequest (permanent, document-scoped grant),
    approval here only reveals the entity inside this one message — it never
    persists into Message.content and never applies to any other message,
    even a repeat of the same question.
    """

    __tablename__ = "message_entity_access_requests"

    id = Column(String(36), primary_key=True, default=new_uuid)
    message_id = Column(
        String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_types_json = Column(JSON, nullable=False, default=list)
    status = Column(String(16), nullable=False, default="pending")  # pending | approved | rejected
    admin_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    admin_note = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
