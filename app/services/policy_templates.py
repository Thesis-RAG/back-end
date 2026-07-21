"""Built-in policy bundles for a first-run enterprise setup.

Templates are intentionally data-only. Installing one creates an ordinary
DomainRule, so administrators can edit, disable or delete it exactly like a
manually-created rule.
"""
from __future__ import annotations

from app.schemas.policy import DomainRuleCreate, RuleConditions, RuleContract


BUILT_IN_RULE_TEMPLATES: list[dict] = [
    {
        "template_code": "HR-SALARY-DENY",
        "name": "Chặn thông tin lương",
        "description": "Ẩn số tiền lương, thưởng và dữ liệu tài chính nhạy cảm theo từng trường.",
        "rule": DomainRuleCreate(
            rule_code="HR-SALARY-DENY",
            name="Chặn thông tin lương",
            priority=95,
            mandatory=True,
            conditions=RuleConditions(
                target_entity_types=["money", "salary", "salary_amount"],
                target_flags=["has_financial", "has_hr"],
            ),
            contract=RuleContract(violation_action="block", max_detail="redact", numeric_granularity="hidden"),
        ),
    },
    {
        "template_code": "HR-PERSONAL-CONDITIONAL",
        "name": "Thông tin cá nhân có điều kiện",
        "description": "Cho phép câu trả lời nhưng khái quát hoặc ẩn từng trường cá nhân.",
        "rule": DomainRuleCreate(
            rule_code="HR-PERSONAL-CONDITIONAL",
            name="Thông tin cá nhân có điều kiện",
            priority=80,
            conditions=RuleConditions(
                target_entity_types=["person_name", "name", "email", "phone", "address", "dob", "national_id"],
                target_flags=["has_pii"],
            ),
            contract=RuleContract(violation_action="conditional", max_detail="generalize", numeric_granularity="hidden"),
        ),
    },
    {
        "template_code": "GLOBAL-CREDENTIAL-DENY",
        "name": "Chặn thông tin xác thực",
        "description": "Không đưa mật khẩu, token, API key hoặc OTP vào ngữ cảnh trả lời.",
        "rule": DomainRuleCreate(
            rule_code="GLOBAL-CREDENTIAL-DENY",
            name="Chặn thông tin xác thực",
            priority=100,
            mandatory=True,
            conditions=RuleConditions(target_flags=["has_credential"]),
            contract=RuleContract(violation_action="block", max_detail="redact", numeric_granularity="hidden"),
        ),
    },
    {
        "template_code": "CROSS-DEPARTMENT-CONDITIONAL",
        "name": "Khái quát dữ liệu khác phòng ban",
        "description": "Giảm mức chi tiết khi người dùng truy cập tài liệu thuộc nhánh tổ chức khác.",
        "rule": DomainRuleCreate(
            rule_code="CROSS-DEPARTMENT-CONDITIONAL",
            name="Khái quát dữ liệu khác phòng ban",
            priority=60,
            conditions=RuleConditions(cross_dept_only=True),
            contract=RuleContract(violation_action="conditional", max_detail="generalize", numeric_granularity="aggregated"),
        ),
    },
]


def list_policy_templates() -> list[dict]:
    return [
        {
            "template_code": item["template_code"],
            "name": item["name"],
            "description": item["description"],
            "rule": item["rule"].model_dump(),
        }
        for item in BUILT_IN_RULE_TEMPLATES
    ]


def get_policy_template(code: str) -> dict | None:
    normalized = code.strip().upper()
    return next((item for item in BUILT_IN_RULE_TEMPLATES if item["template_code"] == normalized), None)
