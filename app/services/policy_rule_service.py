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


def normalize_entity_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower()).strip("_")[:128]


# Seed data is intentionally data, not the runtime source of truth. Once seeded,
# EntityPolicyRule is the only active label/rule catalogue used by detection.
DEFAULT_RULES: tuple[dict, ...] = (
    {"entity_key": "credential", "display_name": "Thông tin xác thực", "group_name": "security", "detection_source": "regex", "action": "block", "metadata_json": {"boolean_labels": ["has_credential"]}},
    {"entity_key": "password", "display_name": "Mật khẩu", "group_name": "security", "detection_source": "gliner", "action": "block", "metadata_json": {"boolean_labels": ["has_credential"]}},
    {"entity_key": "api_key", "display_name": "API key", "group_name": "security", "detection_source": "regex", "action": "block", "metadata_json": {"boolean_labels": ["has_credential"]}},
    {"entity_key": "token", "display_name": "Token", "group_name": "security", "detection_source": "regex", "action": "block", "metadata_json": {"boolean_labels": ["has_credential"]}},
    {"entity_key": "salary", "display_name": "Lương", "group_name": "hr_financial", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_financial", "has_hr"]}},
    {"entity_key": "bonus", "display_name": "Tiền thưởng", "group_name": "hr_financial", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_financial", "has_hr"]}},
    {"entity_key": "income", "display_name": "Thu nhập", "group_name": "hr_financial", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_financial", "has_hr"]}},
    {"entity_key": "payroll", "display_name": "Bảng lương", "group_name": "hr_financial", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_financial", "has_hr"]}},
    {"entity_key": "bank_account", "display_name": "Tài khoản ngân hàng", "group_name": "financial", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii", "has_financial"]}},
    {"entity_key": "tax_id", "display_name": "Mã số thuế", "group_name": "financial", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii", "has_financial"]}},
    {"entity_key": "financial_data", "display_name": "Dữ liệu tài chính", "group_name": "financial", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_financial"]}},
    {"entity_key": "money", "display_name": "Số tiền", "group_name": "financial", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_financial"]}},
    {"entity_key": "percentage", "display_name": "Tỷ lệ", "group_name": "financial", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_financial"]}},
    {"entity_key": "customer", "display_name": "Khách hàng", "group_name": "customer", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "customer_id", "display_name": "Mã khách hàng", "group_name": "customer", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "customer_email", "display_name": "Email khách hàng", "group_name": "customer", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "customer_phone", "display_name": "Điện thoại khách hàng", "group_name": "customer", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "project", "display_name": "Dự án", "group_name": "strategy", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "project_code", "display_name": "Mã dự án", "group_name": "strategy", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "roadmap", "display_name": "Lộ trình", "group_name": "strategy", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "strategy", "display_name": "Chiến lược", "group_name": "strategy", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "business_plan", "display_name": "Kế hoạch kinh doanh", "group_name": "strategy", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "merger", "display_name": "Sáp nhập", "group_name": "strategy", "detection_source": "gliner", "action": "block", "metadata_json": {"boolean_labels": ["has_strategic"]}},
    {"entity_key": "contract", "display_name": "Hợp đồng", "group_name": "legal", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_legal"]}},
    {"entity_key": "contract_number", "display_name": "Số hợp đồng", "group_name": "legal", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_legal"]}},
    {"entity_key": "person_name", "display_name": "Họ tên", "group_name": "pii", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "national_id", "display_name": "Số định danh", "group_name": "pii", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii", "has_hr"]}},
    {"entity_key": "dob", "display_name": "Ngày sinh", "group_name": "pii", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii", "has_hr"]}},
    {"entity_key": "social_insurance", "display_name": "Bảo hiểm xã hội", "group_name": "hr_financial", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii", "has_hr"]}},
    {"entity_key": "address", "display_name": "Địa chỉ", "group_name": "pii", "detection_source": "gliner", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "phone", "display_name": "Số điện thoại", "group_name": "pii", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "email", "display_name": "Email", "group_name": "pii", "detection_source": "regex", "action": "mask", "metadata_json": {"boolean_labels": ["has_pii"]}},
    {"entity_key": "organization", "display_name": "Tổ chức", "group_name": "general", "detection_source": "gliner", "action": "full"},
    {"entity_key": "date", "display_name": "Ngày tháng", "group_name": "general", "detection_source": "gliner", "action": "full"},
    {"entity_key": "date_generic", "display_name": "Ngày tháng dạng số", "group_name": "general", "detection_source": "regex", "action": "full"},
    {"entity_key": "location", "display_name": "Địa điểm", "group_name": "general", "detection_source": "gliner", "action": "full"},
)


def _clean_scope(values: Iterable[str] | None) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in (values or []) if str(value).strip()))


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
                db.add(EntityPolicyRule(
                    policy_profile=DEFAULT_POLICY_PROFILE,
                    entity_key=entity_key,
                    display_name=item["display_name"],
                    group_name=item["group_name"],
                    detection_source=item["detection_source"],
                    action=item["action"],
                    priority=index * 10,
                    metadata_json=item.get("metadata_json") or {},
                ))
        if system_setting_repository.get(db, POLICY_VERSION_KEY) is None:
            system_setting_repository.set(db, POLICY_VERSION_KEY, settings.default_policy_version)
        db.flush()

    def create(self, db: Session, data: dict) -> EntityPolicyRule:
        rule = EntityPolicyRule(
            policy_profile=data.get("policy_profile") or DEFAULT_POLICY_PROFILE,
            entity_key=normalize_entity_key(data.get("entity_key") or ""),
            display_name=str(data.get("display_name") or data.get("entity_key") or "").strip(),
            group_name=str(data.get("group_name") or "general").strip(),
            detection_source=str(data.get("detection_source") or "manual").lower(),
            action=str(data.get("action") or "full").lower(),
            enabled=bool(data.get("enabled", True)),
            scope_oui_ids=_clean_scope(data.get("scope_oui_ids")),
            scope_position_ids=_clean_scope(data.get("scope_position_ids")),
            priority=int(data.get("priority") or 100),
            metadata_json=dict(data.get("metadata_json") or {}),
        )
        db.add(rule)
        db.flush()
        return rule


policy_rule_service = PolicyRuleService()
