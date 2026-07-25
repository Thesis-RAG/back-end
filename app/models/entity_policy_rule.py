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
    group_name = Column(String(128), nullable=False, default="Chung")
    detection_source = Column(String(16), nullable=False, default="gliner")
    # Legacy single-tier action, kept for the ingest-time snapshot copy
    # (document_service._apply_policy_snapshot) and API backward compatibility.
    # Live chat/file resolution uses the block/require/immune scopes below.
    action = Column(String(16), nullable=False, default="full")
    # Absolute sensitivity level (1-5, same scale as Document.sensitivity)
    # carried by this entity type. Used to compute chunk_sensitivity at
    # ingest/resync time — see entity_extractor.compute_chunk_sensitivity.
    sensitivity = Column(Integer, nullable=False, default=2)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    scope_oui_ids = Column(JSON, nullable=False, default=list)
    scope_position_ids = Column(JSON, nullable=False, default=list)
    priority = Column(Integer, nullable=False, default=100, index=True)
    metadata_json = Column(JSON, nullable=False, default=dict)

    # Each scope is a list of explicit (oui_id, position_id) pairs — not two
    # independent lists AND'd together — so "HCM's Trưởng chi nhánh" and
    # "Hà Nội's Phó chi nhánh" can be selected without also matching "HCM's
    # Phó chi nhánh" (the cartesian-product bug a flat pair of lists has).
    # A pair's oui_id/position_id is null to mean "any" on that dimension
    # (eg. {oui_id: X, position_id: null} = "whole unit X, any position").
    # Always full, checked first — the only tier an admin explicitly grants.
    allow_scope_pairs = Column(JSON, nullable=False, default=list)
    # Span-level mask + per-message access request appeal, checked second.
    mask_scope_pairs = Column(JSON, nullable=False, default=list)
    # No block_scope column: block is the pure fallback (not user-scoped) —
    # anyone matching neither allow_scope nor mask_scope is hard-blocked,
    # no appeal, by construction of resolve_tier().

    mask_style = Column(String(16), nullable=False, default="full")      # full | partial
    mask_position = Column(String(16), nullable=True)                    # head | center | tail
