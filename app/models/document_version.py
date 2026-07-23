from sqlalchemy import Column, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin
from app.utils.ids import new_uuid


class DocumentVersion(Base, TimestampMixin):
    __tablename__ = "document_versions"

    id = Column(String(36), primary_key=True, default=new_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    version_no = Column(Integer, nullable=False)

    file_name = Column(String(255), nullable=False)
    mime_type = Column(String(128), nullable=False)
    checksum = Column(String(128), nullable=False, index=True)

    source_object_id = Column(String(36), ForeignKey("storage_objects.id"), nullable=False, index=True)
    normalized_object_id = Column(String(36), ForeignKey("storage_objects.id"), nullable=True, index=True)

    ingest_status = Column(String(32), nullable=False, default="queued")
    parse_status = Column(String(32), nullable=False, default="pending")
    chunk_status = Column(String(32), nullable=False, default="pending")
    embed_status = Column(String(32), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    rule_version = Column(String(32), nullable=False, default="v1")
    policy_profile = Column(String(64), nullable=False, default="enterprise_secure")
    policy_version = Column(String(32), nullable=False, default="policy-v1")
    resolved_rules_json = Column(JSON, nullable=True)
    confirmed_labels_json = Column(JSON, nullable=True)

    chunk_config_json = Column(JSON, nullable=True)
    # Full-document entity detection snapshot produced during ingest.
    entity_detection_json = Column(JSON, nullable=True)

    document = relationship("Document", back_populates="versions", foreign_keys=[document_id])
    source_object = relationship("StorageObject", foreign_keys=[source_object_id])
    normalized_object = relationship("StorageObject", foreign_keys=[normalized_object_id])

    chunks = relationship(
        "DocumentChunk",
        back_populates="version",
        cascade="all, delete-orphan",
        foreign_keys="DocumentChunk.document_version_id",
    )
    entity_actions = relationship(
        "DocumentEntityAction",
        back_populates="version",
        cascade="all, delete-orphan",
        foreign_keys="DocumentEntityAction.document_version_id",
        order_by="DocumentEntityAction.sort_order",
    )
