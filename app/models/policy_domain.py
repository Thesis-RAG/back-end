from sqlalchemy import Boolean, Column, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin
from app.utils.ids import new_uuid


class PolicyDomain(Base, TimestampMixin):
    # Define a policy domain, which groups related access control rules and entity types.
    __tablename__ = "policy_domains"

    id          = Column(String(36), primary_key=True, default=new_uuid)
    code        = Column(String(64), nullable=False, unique=True, index=True)  # e.g. HR-01
    name        = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    base_sensitivity = Column(Integer, nullable=False, default=2)  # 1-5
    is_active   = Column(Boolean, nullable=False, default=True)

    entity_types = relationship(
        "DomainEntityType",
        back_populates="domain",
        cascade="all, delete-orphan",
    )
    rules = relationship(
        "DomainRule",
        back_populates="domain",
        cascade="all, delete-orphan",
        foreign_keys="DomainRule.domain_id",
    )


class DomainEntityType(Base, TimestampMixin):
    # Entity types detected within the domain (GLiNER labels).
    __tablename__ = "domain_entity_types"
    __table_args__ = (
        UniqueConstraint("domain_id", "entity_type", name="uq_domain_entity"),
    )

    id             = Column(String(36), primary_key=True, default=new_uuid)
    domain_id      = Column(String(36), ForeignKey("policy_domains.id"), nullable=False, index=True)
    entity_type    = Column(String(128), nullable=False)          # e.g. "salary_amount"
    label_vi       = Column(String(255), nullable=True)           # e.g. "Mức lương"
    boolean_labels = Column(JSON, nullable=False, default=list)   # e.g. ["has_pii", "has_financial"]
    is_system_suggested = Column(Boolean, nullable=False, default=False)
    is_active      = Column(Boolean, nullable=False, default=True)

    domain = relationship("PolicyDomain", back_populates="entity_types")


class DomainRule(Base, TimestampMixin):
    # Access control rules always belong to a policy domain.
    __tablename__ = "domain_rules"

    id        = Column(String(36), primary_key=True, default=new_uuid)
    domain_id = Column(String(36), ForeignKey("policy_domains.id"), nullable=False, index=True)
    rule_code = Column(String(64), nullable=False, unique=True, index=True)   # e.g. HR-01-R001
    name      = Column(String(255), nullable=False)
    action    = Column(String(32), nullable=False)   # ALLOW | DENY | REDACT | ALLOW_WITH_WATERMARK
    priority  = Column(Integer, nullable=False, default=50)  # 0-100, cao hơn = ưu tiên hơn
    mandatory = Column(Boolean, nullable=False, default=False)
    risk_level = Column(String(32), nullable=False, default="low")  # low|medium|high|very_high
    is_active = Column(Boolean, nullable=False, default=True)
    audit_log = Column(Boolean, nullable=False, default=True)

    # Conditions for activating the rule
    conditions_json = Column(JSON, nullable=False, default=dict)
    # {
    #   "min_sensitivity": "Confidential",     # None = không giới hạn
    #   "applicable_roles": [],                # [] = tất cả roles
    #   "blocked_roles": [],
    #   "cross_dept_only": false,
    #   "require_pii_detected": false,
    #   "applicable_intents": [],              # [] = tất cả
    #   "min_user_level": null,
    #   "require_intent_risk": null            # normal|cross_dept|bulk_extraction|suspicious
    # }

    # Conditions of the policy-contract when the rule is activated
    contract_json = Column(JSON, nullable=False, default=dict)
    # {
    #   "max_detail": "department",            # company|department|project|individual
    #   "numeric_granularity": "aggregated",   # hidden|aggregated|exact
    #   "violation_action": "redact"           # mask|generalize|deny|regenerate
    # }

    domain = relationship("PolicyDomain", back_populates="rules", foreign_keys=[domain_id])

