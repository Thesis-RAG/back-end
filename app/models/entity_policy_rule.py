from sqlalchemy import Boolean, Column, Integer, JSON, String, UniqueConstraint

from app.db.base import Base, TimestampMixin
from app.utils.ids import new_uuid


class EntityPolicyRule(Base, TimestampMixin):
    """An administrator-managed rule in the active enterprise policy profile."""

    __tablename__ = "entity_policy_rules"
    __table_args__ = (
        UniqueConstraint(
            "policy_profile",
            "entity_key",
            name="uq_entity_policy_rule_profile_key",
        ),
    )

    id = Column(String(36), primary_key=True, default=new_uuid)
    policy_profile = Column(String(64), nullable=False, default="enterprise_secure", index=True)
    entity_key = Column(String(128), nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    group_name = Column(String(128), nullable=False, default="general")
    detection_source = Column(String(16), nullable=False, default="gliner")
    action = Column(String(16), nullable=False, default="full")
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    scope_oui_ids = Column(JSON, nullable=False, default=list)
    scope_position_ids = Column(JSON, nullable=False, default=list)
    priority = Column(Integer, nullable=False, default=100, index=True)
    metadata_json = Column(JSON, nullable=False, default=dict)
