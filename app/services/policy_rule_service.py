"""Global policy rule management and immutable policy snapshot helpers."""
from __future__ import annotations

import re
from typing import Iterable

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.entity_policy_rule import EntityPolicyRule
from app.repositories.system_setting_repository import system_setting_repository


DEFAULT_POLICY_PROFILE = "enterprise_secure"
POLICY_VERSION_KEY = "policy.version"

# Marker stored inside a scope list to mean "every user", distinct from an
# empty list which means "no rule configured for anyone" on that tier. A
# user matching neither allow_scope nor mask_scope fails closed to block.
ALL_SENTINEL = "__ALL__"


def normalize_entity_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower()).strip("_")[:128]


# Return True if any of the user's OUI/Position assignments matches one of
# the scope's explicit (oui_id, position_id) pairs. A null side of a pair
# means "any" on that dimension. Pairs are OR'd together; within one pair,
# the two dimensions are AND'd — this is what lets "HCM's Trưởng chi nhánh"
# and "Hà Nội's Phó chi nhánh" be selected independently, unlike the old
# two-flat-lists model which always matched their full cartesian product.
def _scope_matches(pairs: list[dict] | None, user) -> bool:
    pairs = pairs or []
    if any(pair.get("oui_id") == ALL_SENTINEL or pair.get("position_id") == ALL_SENTINEL for pair in pairs):
        return True
    if not pairs:
        return False
    for assignment in getattr(user, "oui_positions", []) or []:
        oui_id = str(getattr(assignment, "oui_id", "") or "")
        position_id = str(getattr(assignment, "position_id", "") or "")
        for pair in pairs:
            pair_oui = pair.get("oui_id")
            pair_pos = pair.get("position_id")
            if pair_oui and str(pair_oui) != oui_id:
                continue
            if pair_pos and str(pair_pos) != position_id:
                continue
            return True
    return False


# Resolve the effective tier for one rule against one user.
# Priority: allow > mask > block. block is the pure fallback — anyone
# matching neither allow_scope nor mask_scope is hard-blocked, no appeal.
def resolve_tier(rule: EntityPolicyRule, user) -> dict:
    if not rule.enabled:
        return {"tier": "block"}
    if _scope_matches(rule.allow_scope_pairs, user):
        return {"tier": "full"}
    if _scope_matches(rule.mask_scope_pairs, user):
        return {
            "tier": "require",
            "mask_style": rule.mask_style or "full",
            "mask_position": rule.mask_position,
        }
    return {"tier": "block"}


# Seed data is intentionally data, not the runtime source of truth. Once seeded,
# EntityPolicyRule is the only active label/rule catalogue used by detection.
DEFAULT_RULES: tuple[dict, ...] = (
    {"entity_key": "credential", "display_name": "Thông tin xác thực", "group_name": "Bảo mật", "detection_source": "regex", "action": "block", "sensitivity": 5, "metadata_json": {"boolean_labels": ["has_credential"]}},
    {"entity_key": "password", "display_name": "Mật khẩu", "group_name": "Bảo mật", "detection_source": "gliner", "action": "block", "sensitivity": 5, "metadata_json": {"boolean_labels": ["has_credential"]}},
    {"entity_key": "api_key", "display_name": "API key", "group_name": "Bảo mật", "detection_source": "regex", "action": "block", "sensitivity": 5, "metadata_json": {"boolean_labels": ["has_credential"]}},
    {"entity_key": "token", "display_name": "Token", "group_name": "Bảo mật", "detection_source": "regex", "action": "block", "sensitivity": 5, "metadata_json": {"boolean_labels": ["has_credential"]}},
    {"entity_key": "salary", "display_name": "Lương", "group_name": "Nhân sự & Tài chính", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_financial", "has_hr"]}},
    {"entity_key": "bonus", "display_name": "Tiền thưởng", "group_name": "Nhân sự & Tài chính", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_financial", "has_hr"]}},
    {"entity_key": "income", "display_name": "Thu nhập", "group_name": "Nhân sự & Tài chính", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_financial", "has_hr"]}},
    {"entity_key": "payroll", "display_name": "Bảng lương", "group_name": "Nhân sự & Tài chính", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_financial", "has_hr"]}},
    {"entity_key": "bank_account", "display_name": "Tài khoản ngân hàng", "group_name": "Tài chính", "detection_source": "regex", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_pii", "has_financial"]}},
    {"entity_key": "tax_id", "display_name": "Mã số thuế", "group_name": "Tài chính", "detection_source": "regex", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_pii", "has_financial"]}},
    {"entity_key": "financial_data", "display_name": "Dữ liệu tài chính", "group_name": "Tài chính", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_financial"]}},
    {"entity_key": "money", "display_name": "Số tiền", "group_name": "Tài chính", "detection_source": "regex", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_financial"]}},
    {"entity_key": "percentage", "display_name": "Tỷ lệ", "group_name": "Tài chính", "detection_source": "regex", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_financial"]}},
    {"entity_key": "customer", "display_name": "Khách hàng", "group_name": "Khách hàng", "detection_source": "gliner", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "customer_id", "display_name": "Mã khách hàng", "group_name": "Khách hàng", "detection_source": "gliner", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "customer_email", "display_name": "Email khách hàng", "group_name": "Khách hàng", "detection_source": "regex", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "customer_phone", "display_name": "Điện thoại khách hàng", "group_name": "Khách hàng", "detection_source": "regex", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "project", "display_name": "Dự án", "group_name": "Chiến lược", "detection_source": "gliner", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "project_code", "display_name": "Mã dự án", "group_name": "Chiến lược", "detection_source": "gliner", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "roadmap", "display_name": "Lộ trình", "group_name": "Chiến lược", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "strategy", "display_name": "Chiến lược", "group_name": "Chiến lược", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "business_plan", "display_name": "Kế hoạch kinh doanh", "group_name": "Chiến lược", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "merger", "display_name": "Sáp nhập", "group_name": "Chiến lược", "detection_source": "gliner", "action": "block", "sensitivity": 5, "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "contract", "display_name": "Hợp đồng", "group_name": "Pháp lý", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_legal"]}},
    {"entity_key": "contract_number", "display_name": "Số hợp đồng", "group_name": "Pháp lý", "detection_source": "gliner", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_legal"]}},
    {"entity_key": "person_name", "display_name": "Họ tên", "group_name": "Thông tin cá nhân", "detection_source": "gliner", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "national_id", "display_name": "Số định danh", "group_name": "Thông tin cá nhân", "detection_source": "regex", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_pii", "has_hr"]}},
    {"entity_key": "dob", "display_name": "Ngày sinh", "group_name": "Thông tin cá nhân", "detection_source": "regex", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_pii", "has_hr"]}},
    {"entity_key": "social_insurance", "display_name": "Bảo hiểm xã hội", "group_name": "Nhân sự & Tài chính", "detection_source": "regex", "action": "mask", "sensitivity": 4, "metadata_json": {"boolean_labels": ["has_pii", "has_hr"]}},
    {"entity_key": "address", "display_name": "Địa chỉ", "group_name": "Thông tin cá nhân", "detection_source": "gliner", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "phone", "display_name": "Số điện thoại", "group_name": "Thông tin cá nhân", "detection_source": "regex", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "email", "display_name": "Email", "group_name": "Thông tin cá nhân", "detection_source": "regex", "action": "mask", "sensitivity": 3, "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "organization", "display_name": "Tổ chức", "group_name": "Chung", "detection_source": "gliner", "action": "full", "sensitivity": 1},
    {"entity_key": "date", "display_name": "Ngày tháng", "group_name": "Chung", "detection_source": "gliner", "action": "full", "sensitivity": 1},
    {"entity_key": "date_generic", "display_name": "Ngày tháng dạng số", "group_name": "Chung", "detection_source": "regex", "action": "full", "sensitivity": 1},
    {"entity_key": "location", "display_name": "Địa điểm", "group_name": "Chung", "detection_source": "gliner", "action": "full", "sensitivity": 1},
)


def _clean_scope(values: Iterable[str] | None) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in (values or []) if str(value).strip()))


def _clean_pairs(values: Iterable[dict] | None) -> list[dict]:
    seen: set[tuple[str | None, str | None]] = set()
    result: list[dict] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        oui_id = str(item.get("oui_id")).strip() if item.get("oui_id") else None
        position_id = str(item.get("position_id")).strip() if item.get("position_id") else None
        if oui_id is None and position_id is None:
            continue
        key = (oui_id, position_id)
        if key in seen:
            continue
        seen.add(key)
        result.append({"oui_id": oui_id, "position_id": position_id})
    return result


class PolicyRuleService:
    def current_version(self, db: Session) -> str:
        return str(system_setting_repository.get(db, POLICY_VERSION_KEY) or settings.default_policy_version)

    def bump_version(self, db: Session) -> str:
        current = self.current_version(db)
        match = re.match(r"^(.*?)(\d+)$", current)
        version = f"{match.group(1)}{int(match.group(2)) + 1}" if match else f"{current}-2"
        system_setting_repository.set(db, POLICY_VERSION_KEY, version)
        return version

    def list_rules(self, db: Session, profile: str = DEFAULT_POLICY_PROFILE) -> list[EntityPolicyRule]:
        return (
            db.query(EntityPolicyRule)
            .filter(EntityPolicyRule.policy_profile == profile)
            .order_by(EntityPolicyRule.priority.asc(), EntityPolicyRule.entity_key.asc())
            .all()
        )

    def active_rules(self, db: Session, profile: str = DEFAULT_POLICY_PROFILE) -> list[EntityPolicyRule]:
        return [rule for rule in self.list_rules(db, profile) if rule.enabled]

    # Batch-lookup active rules by entity_key, for live resolution against a
    # set of entity types detected in a chunk/document (not per-document).
    def rules_by_key(
        self, db: Session, entity_keys: Iterable[str], profile: str = DEFAULT_POLICY_PROFILE
    ) -> dict[str, EntityPolicyRule]:
        keys = {normalize_entity_key(key) for key in entity_keys if key}
        if not keys:
            return {}
        rows = (
            db.query(EntityPolicyRule)
            .filter(
                EntityPolicyRule.policy_profile == profile,
                EntityPolicyRule.entity_key.in_(keys),
                EntityPolicyRule.enabled.is_(True),
            )
            .all()
        )
        return {row.entity_key: row for row in rows}

    @staticmethod
    def serialize_rule(rule: EntityPolicyRule) -> dict:
        return {
            "id": rule.id,
            "policy_profile": rule.policy_profile,
            "entity_key": rule.entity_key,
            "display_name": rule.display_name,
            "group_name": rule.group_name,
            "detection_source": rule.detection_source,
            "action": rule.action,
            "sensitivity": int(rule.sensitivity or 1),
            "enabled": bool(rule.enabled),
            "scope_oui_ids": list(rule.scope_oui_ids or []),
            "scope_position_ids": list(rule.scope_position_ids or []),
            "priority": int(rule.priority or 0),
            "metadata_json": dict(rule.metadata_json or {}),
        }

    def snapshot(self, db: Session, profile: str = DEFAULT_POLICY_PROFILE) -> dict:
        rules = self.active_rules(db, profile)
        return {
            "policy_profile": profile,
            "policy_version": self.current_version(db),
            "resolved_rules": [self.serialize_rule(rule) for rule in rules],
            "confirmed_labels": [],
        }

    def seed_defaults(self, db: Session) -> None:
        from app.models.entity_policy_group import EntityPolicyGroup

        existing_groups = {
            row.name for row in db.query(EntityPolicyGroup)
            .filter(EntityPolicyGroup.policy_profile == DEFAULT_POLICY_PROFILE).all()
        }
        for name in sorted({item["group_name"] for item in DEFAULT_RULES}):
            if name not in existing_groups:
                db.add(EntityPolicyGroup(policy_profile=DEFAULT_POLICY_PROFILE, name=name))
                existing_groups.add(name)

        for index, item in enumerate(DEFAULT_RULES):
            entity_key = normalize_entity_key(item["entity_key"])
            rule = (
                db.query(EntityPolicyRule)
                .filter(
                    EntityPolicyRule.policy_profile == DEFAULT_POLICY_PROFILE,
                    EntityPolicyRule.entity_key == entity_key,
                )
                .first()
            )
            if rule is None:
                # Seed data still authors a single legacy `action`; translate
                # it into allow_scope/mask_scope so a fresh install behaves
                # the same as a migrated one. action=="block" needs neither
                # scope set — blocking everyone is just the fallback.
                # action=="mask" masks everyone (mask_scope=ALL).
                # action=="full" must explicitly allow everyone, since the
                # fallback is now block, not full.
                action = item["action"]
                db.add(EntityPolicyRule(
                    policy_profile=DEFAULT_POLICY_PROFILE,
                    entity_key=entity_key,
                    display_name=item["display_name"],
                    group_name=item["group_name"],
                    detection_source=item["detection_source"],
                    action=action,
                    sensitivity=int(item.get("sensitivity") or 2),
                    priority=index * 10,
                    metadata_json=item.get("metadata_json") or {},
                    allow_scope_pairs=[{"oui_id": ALL_SENTINEL, "position_id": None}] if action == "full" else [],
                    mask_scope_pairs=[{"oui_id": ALL_SENTINEL, "position_id": None}] if action == "mask" else [],
                ))
        if system_setting_repository.get(db, POLICY_VERSION_KEY) is None:
            system_setting_repository.set(db, POLICY_VERSION_KEY, settings.default_policy_version)
        db.flush()

    def create(self, db: Session, data: dict) -> EntityPolicyRule:
        rule = EntityPolicyRule(
            policy_profile=data.get("policy_profile") or DEFAULT_POLICY_PROFILE,
            entity_key=normalize_entity_key(data.get("entity_key") or ""),
            display_name=str(data.get("display_name") or data.get("entity_key") or "").strip(),
            group_name=str(data.get("group_name") or "Chung").strip(),
            detection_source=str(data.get("detection_source") or "manual").lower(),
            action=str(data.get("action") or "full").lower(),
            sensitivity=int(data.get("sensitivity") or 2),
            enabled=bool(data.get("enabled", True)),
            scope_oui_ids=_clean_scope(data.get("scope_oui_ids")),
            scope_position_ids=_clean_scope(data.get("scope_position_ids")),
            priority=int(data.get("priority") or 100),
            metadata_json=dict(data.get("metadata_json") or {}),
            allow_scope_pairs=_clean_pairs(data.get("allow_scope_pairs")),
            mask_scope_pairs=_clean_pairs(data.get("mask_scope_pairs")),
            mask_style=str(data.get("mask_style") or "full").lower(),
            mask_position=(str(data["mask_position"]).lower() if data.get("mask_position") else None),
        )
        db.add(rule)
        db.flush()
        return rule

    # ---- groups ------------------------------------------------------
    # group_name on EntityPolicyRule is a plain string, not a foreign key
    # (see EntityPolicyGroup's docstring) — these just manage the list of
    # "known" names the Policy page's group picker offers, plus a bulk
    # scope-overwrite action across every rule currently using a name.

    def list_groups(self, db: Session, profile: str = DEFAULT_POLICY_PROFILE):
        from app.models.entity_policy_group import EntityPolicyGroup
        return (
            db.query(EntityPolicyGroup)
            .filter(EntityPolicyGroup.policy_profile == profile)
            .order_by(EntityPolicyGroup.name.asc())
            .all()
        )

    def create_group(self, db: Session, name: str, profile: str = DEFAULT_POLICY_PROFILE):
        from app.models.entity_policy_group import EntityPolicyGroup
        name = name.strip()
        if not name:
            raise ValueError("Tên nhóm không được để trống")
        existing = (
            db.query(EntityPolicyGroup)
            .filter(EntityPolicyGroup.policy_profile == profile, EntityPolicyGroup.name == name)
            .first()
        )
        if existing:
            raise ValueError(f"Nhóm '{name}' đã tồn tại")
        group = EntityPolicyGroup(policy_profile=profile, name=name)
        db.add(group)
        db.flush()
        return group

    def delete_group(self, db: Session, group_id: str, profile: str = DEFAULT_POLICY_PROFILE) -> None:
        from app.models.entity_policy_group import EntityPolicyGroup
        group = (
            db.query(EntityPolicyGroup)
            .filter(EntityPolicyGroup.id == group_id, EntityPolicyGroup.policy_profile == profile)
            .first()
        )
        if not group:
            raise ValueError("Không tìm thấy nhóm")
        db.delete(group)
        db.flush()

    # Overwrite allow_scope_pairs/mask_scope_pairs on every rule currently
    # in `group_name`, in one shot — "đè lên phần scope settings chi tiết
    # trong mỗi thực thể" per the admin's own request, not a merge. Returns
    # the affected rows so the caller can re-push each one to the ontology
    # graph (this service has no Neo4j dependency of its own).
    def apply_group_scope(
        self, db: Session, group_name: str, allow_scope_pairs: list[dict], mask_scope_pairs: list[dict],
        profile: str = DEFAULT_POLICY_PROFILE,
    ) -> list[EntityPolicyRule]:
        allow_pairs = _clean_pairs(allow_scope_pairs)
        mask_pairs = _clean_pairs(mask_scope_pairs)
        validate_tiered_scopes({"allow_scope_pairs": allow_pairs, "mask_scope_pairs": mask_pairs})

        rules = (
            db.query(EntityPolicyRule)
            .filter(EntityPolicyRule.policy_profile == profile, EntityPolicyRule.group_name == group_name)
            .all()
        )
        for rule in rules:
            rule.allow_scope_pairs = allow_pairs
            rule.mask_scope_pairs = mask_pairs
        if rules:
            self.bump_version(db)
        db.flush()
        return rules


# A pair missing one side (oui_id or position_id) reads as "any" on that
# dimension — deliberately, for _scope_matches(). But it can't be told apart
# from an independent, broader pair once mirrored onto the ontology graph as
# separate edges (see OrgScopePicker.tsx), so new/edited pairs must always
# name both sides. The __ALL__ sentinel is a different mechanism entirely
# (the "Tất cả" toggle, bypassing OrgScopePicker) and stays exempt.
def _require_both_sides(pairs: list[dict], field_label: str) -> None:
    for pair in pairs:
        if pair.get("oui_id") == ALL_SENTINEL:
            continue
        if not pair.get("oui_id") or not pair.get("position_id"):
            raise ValueError(
                f"{field_label}: mỗi phạm vi phải chọn cả đơn vị lẫn vị trí, "
                "không được để trống 1 bên"
            )


# Raise ValueError if allow_scope=ALL while mask_scope still has a scope
# configured — allow is resolved before mask, so allow=ALL would make any
# mask_scope permanently unreachable (dead configuration).
def validate_tiered_scopes(data: dict) -> None:
    allow_pairs = _clean_pairs(data.get("allow_scope_pairs"))
    mask_pairs = _clean_pairs(data.get("mask_scope_pairs"))

    _require_both_sides(allow_pairs, "Cho phép")
    _require_both_sides(mask_pairs, "Che")

    allow_is_all = any(p.get("oui_id") == ALL_SENTINEL for p in allow_pairs)
    if allow_is_all and mask_pairs:
        raise ValueError(
            "Cho phép đang áp dụng cho Tất cả — Che sẽ không bao giờ được resolve tới, "
            "hãy xoá mask_scope trước khi đặt allow = Tất cả"
        )


# ---- ontology graph tier fan-out ------------------------------------------
# Decides which of Allow/Mask/Block policy nodes the ontology graph sync
# should materialize for a rule, and — for Allow/Mask — which scope pairs
# each one APPLIES_TO. Kept next to resolve_tier() since both encode the
# same allow > mask > block precedence, just for different consumers (one
# resolves a single user, this one decides which graph nodes to draw).

def _all_valid_scope_combos(db: Session) -> set[tuple[str, str]]:
    """Every (oui_instance_id, position_id) pair that could structurally
    exist — a Position belongs to an OU type, so it pairs with every
    instance of that type, regardless of whether anyone is assigned yet."""
    from app.models.org_unit_instance import OrgUnitInstance
    from app.models.position import Position

    positions_by_ou: dict[str, list[str]] = {}
    for position_id, ou_id in db.query(Position.id, Position.ou_id).all():
        positions_by_ou.setdefault(ou_id, []).append(position_id)

    combos: set[tuple[str, str]] = set()
    for oui_id, ou_id in db.query(OrgUnitInstance.id, OrgUnitInstance.ou_id).all():
        for position_id in positions_by_ou.get(ou_id, []):
            combos.add((oui_id, position_id))
    return combos


def _covered_scope_combos(pairs: list[dict], all_combos: set[tuple[str, str]]) -> set[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for pair in pairs or []:
        oui_id, position_id = pair.get("oui_id"), pair.get("position_id")
        if oui_id == ALL_SENTINEL or position_id == ALL_SENTINEL:
            return set(all_combos)
        if oui_id and position_id:
            if (oui_id, position_id) in all_combos:
                covered.add((oui_id, position_id))
        elif oui_id:
            covered |= {combo for combo in all_combos if combo[0] == oui_id}
        elif position_id:
            covered |= {combo for combo in all_combos if combo[1] == position_id}
    return covered


def _covers_full_scope(db: Session, allow_pairs: list[dict], mask_pairs: list[dict]) -> bool:
    all_combos = _all_valid_scope_combos(db)
    if not all_combos:
        return True
    covered = _covered_scope_combos(allow_pairs, all_combos) | _covered_scope_combos(mask_pairs, all_combos)
    return covered >= all_combos


def compute_graph_tiers(db: Session, rule: EntityPolicyRule) -> dict:
    allow_pairs = list(rule.allow_scope_pairs or [])
    mask_pairs = list(rule.mask_scope_pairs or [])
    allow_is_all = any(pair.get("oui_id") == ALL_SENTINEL for pair in allow_pairs)

    if allow_is_all:
        return {
            "allow": {"needed": True, "all": True, "pairs": []},
            "mask": {"needed": False, "pairs": []},
            "block": {"needed": False},
        }

    needs_allow = bool(allow_pairs)
    needs_mask = bool(mask_pairs)
    # Pure fallback (neither tier configured) is always block. Otherwise
    # block only survives if allow+mask together don't cover every real
    # (unit, position) combination — matches the fallback semantics of
    # resolve_tier(): anyone in neither scope is hard-blocked.
    needs_block = True if not (needs_allow or needs_mask) else not _covers_full_scope(db, allow_pairs, mask_pairs)

    return {
        "allow": {"needed": needs_allow, "all": False, "pairs": allow_pairs},
        "mask": {"needed": needs_mask, "pairs": mask_pairs},
        "block": {"needed": needs_block},
    }


policy_rule_service = PolicyRuleService()
