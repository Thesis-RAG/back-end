from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text

from app.db.base import Base
from app.utils.ids import new_uuid


class DocumentEntityAccessRequest(Base):
    """Least-privilege request to reveal blocked entities in a document version."""

    __tablename__ = "document_entity_access_requests"

    id = Column(String(36), primary_key=True, default=new_uuid)
    document_id = Column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_version_id = Column(
        String(36), ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_types_json = Column(JSON, nullable=False, default=list)
    status = Column(String(16), nullable=False, default="pending")
    expires_at = Column(DateTime, nullable=True)
    admin_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    admin_note = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
