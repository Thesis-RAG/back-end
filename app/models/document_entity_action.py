from sqlalchemy import Boolean, Column, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin
from app.utils.ids import new_uuid


class DocumentEntityAction(Base, TimestampMixin):
    """Action configured for one detected entity type in one document version."""

    __tablename__ = "document_entity_actions"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "entity_type",
            name="uq_document_version_entity_action",
        ),
    )

    id = Column(String(36), primary_key=True, default=new_uuid)
    document_version_id = Column(
        String(36),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_type = Column(String(128), nullable=False)
    label = Column(String(255), nullable=True)
    action = Column(String(16), nullable=False, default="full")  # block | full | mask
    # Snapshot of the entity type's sensitivity level (1-5) at the time the
    # global policy was copied onto this version. See
    # entity_extractor.compute_chunk_sensitivity.
    sensitivity = Column(Integer, nullable=False, default=1)
    source = Column(String(16), nullable=False, default="gliner")
    enabled = Column(Boolean, nullable=False, default=True)
    detection_count = Column(Integer, nullable=False, default=0)
    sort_order = Column(Integer, nullable=False, default=0, index=True)
    scope_oui_ids = Column(JSON, nullable=False, default=list)
    scope_position_ids = Column(JSON, nullable=False, default=list)
    metadata_json = Column(JSON, nullable=False, default=dict)

    version = relationship("DocumentVersion", back_populates="entity_actions")
