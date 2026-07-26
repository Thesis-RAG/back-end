from sqlalchemy import Column, String, UniqueConstraint

from app.db.base import Base, TimestampMixin
from app.utils.ids import new_uuid


class EntityPolicyGroup(Base, TimestampMixin):
    """A named, managed group entity policy rules can belong to (rule.group_name
    matches this row's name — kept as a plain string on the rule rather than a
    foreign key so existing rules/data never go invalid if a group is renamed
    or removed; this table exists to give the admin UI a real, creatable list
    instead of inferring "groups" from whatever strings happen to already be
    in use)."""

    __tablename__ = "entity_policy_groups"
    __table_args__ = (
        UniqueConstraint("policy_profile", "name", name="uq_entity_policy_group_profile_name"),
    )

    id = Column(String(36), primary_key=True, default=new_uuid)
    policy_profile = Column(String(64), nullable=False, default="enterprise_secure", index=True)
    name = Column(String(128), nullable=False)
