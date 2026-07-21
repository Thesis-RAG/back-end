"""Enterprise document-governance rule templates.

The templates are deliberately limited to controls that the current policy
engine can enforce: sensitivity, roles, departments, intents, entity types
and boolean data classifications.  Installing a template creates an ordinary
editable DomainRule; it is never a hidden or immutable policy.
"""
from __future__ import annotations

from app.schemas.policy import DomainRuleCreate, RuleConditions, RuleContract


def _template(
    code: str,
    name: str,
    description: str,
    *,
    category: str,
    department: str,
    document_types: list[str],
    priority: int,
    conditions: RuleConditions,
    contract: RuleContract,
    recommended: bool = False,
) -> dict:
    return {
        "template_code": code,
        "name": name,
        "description": description,
        "category": category,
        "department": department,
        "document_types": document_types,
        "recommended": recommended,
        "rule": DomainRuleCreate(
            rule_code=code,
            name=name,
            priority=priority,
            mandatory=recommended and priority >= 95,
            conditions=conditions,
            contract=contract,
        ),
    }


BUILT_IN_RULE_TEMPLATES: list[dict] = [
    # Core DMS baseline: safe defaults for a medium/large enterprise.
    _template(
        "DMS-PII-CONDITIONAL", "Thông tin cá nhân có điều kiện",
        "Ẩn danh hoặc khái quát tên, email, điện thoại, địa chỉ và mã định danh theo từng trường.",
        category="Privacy", department="Company",
        document_types=["Employee Records", "Customers", "Contracts"], priority=90,
        conditions=RuleConditions(
            target_entity_types=["person_name", "name", "email", "phone", "address", "dob", "national_id"],
            target_flags=["has_pii"],
        ),
        contract=RuleContract(violation_action="conditional", max_detail="anonymize", numeric_granularity="hidden"),
        recommended=True,
    ),
    _template(
        "DMS-SALARY-BLOCK", "Chặn lương và dữ liệu thu nhập",
        "Không trả số lương, thưởng, phụ cấp hoặc giá trị thu nhập riêng lẻ cho người không đủ quyền.",
        category="Privacy", department="HR",
        document_types=["Employee Records", "Payroll", "Performance Reviews"], priority=100,
        conditions=RuleConditions(
            target_entity_types=["money", "salary", "salary_amount"],
            target_flags=["has_financial", "has_hr"],
        ),
        contract=RuleContract(violation_action="block", max_detail="redact", numeric_granularity="hidden"),
        recommended=True,
    ),
    _template(
        "DMS-CREDENTIAL-BLOCK", "Chặn thông tin xác thực",
        "Chặn mật khẩu, API key, token, secret và OTP trước khi nội dung đi vào ngữ cảnh LLM.",
        category="Security", department="IT",
        document_types=["Source Code Docs", "API Documents", "Deployment Guides"], priority=100,
        conditions=RuleConditions(target_entity_types=["credential", "secret", "api_key"], target_flags=["has_credential"]),
        contract=RuleContract(violation_action="block", max_detail="redact", numeric_granularity="hidden"),
        recommended=True,
    ),
    _template(
        "DMS-RESTRICTED-CONDITIONAL", "Tài liệu hạn chế - khái quát",
        "Giảm mức chi tiết với tài liệu Restricted nếu người dùng chưa có clearance cấp 4.",
        category="Classification", department="Company",
        document_types=["Policies", "Contracts", "Financial Reports"], priority=85,
        conditions=RuleConditions(min_sensitivity="Restricted", min_user_level=4),
        contract=RuleContract(violation_action="conditional", max_detail="generalize", numeric_granularity="aggregated"),
        recommended=True,
    ),
    _template(
        "DMS-TOPSECRET-BLOCK", "Tài liệu tuyệt mật - chặn",
        "Chặn toàn bộ tài liệu TopSecret với người không có clearance cấp 5.",
        category="Classification", department="Company",
        document_types=["Board Minutes", "Strategic Plans", "R&D Designs"], priority=100,
        conditions=RuleConditions(min_sensitivity="TopSecret", min_user_level=5),
        contract=RuleContract(violation_action="block", max_detail="redact", numeric_granularity="hidden"),
        recommended=True,
    ),
    _template(
        "DMS-CROSS-DEPARTMENT", "Khác phòng ban - khái quát",
        "Giảm chi tiết khi tài liệu thuộc nhánh tổ chức khác với người truy vấn.",
        category="Access Control", department="Company",
        document_types=["All document types"], priority=70,
        conditions=RuleConditions(cross_dept_only=True),
        contract=RuleContract(violation_action="conditional", max_detail="generalize", numeric_granularity="aggregated"),
        recommended=True,
    ),
    _template(
        "DMS-RESTRICTED-WATERMARK", "Tài liệu hạn chế - watermark",
        "Cho phép xem tài liệu Restricted đúng clearance nhưng ghi nhận watermark/audit.",
        category="Audit", department="Company",
        document_types=["Policies", "Contracts", "Reports"], priority=65,
        conditions=RuleConditions(min_sensitivity="Restricted", min_user_level=4),
        contract=RuleContract(violation_action="watermark", max_detail="generalize", numeric_granularity="aggregated"),
        recommended=True,
    ),
    _template(
        "DMS-EXPORT-REDACTION", "Xuất dữ liệu - che trường nhạy cảm",
        "Khi export, che PII và số liệu tài chính thay vì đưa dữ liệu gốc ra ngoài.",
        category="Export Control", department="Company",
        document_types=["Reports", "Customer Lists", "Financial Reports"], priority=90,
        conditions=RuleConditions(applicable_intents=["export"], target_flags=["has_pii", "has_financial"]),
        contract=RuleContract(violation_action="conditional", max_detail="redact", numeric_granularity="aggregated"),
        recommended=True,
    ),

    # Department-specific optional bundles.
    _template(
        "HR-EMPLOYEE-RECORDS", "Hồ sơ nhân viên - HR only",
        "Khái quát hồ sơ nhân viên, hợp đồng lao động và đánh giá năng lực ngoài HR/lãnh đạo.",
        category="Department", department="HR",
        document_types=["Employee Records", "Employment Contracts", "Performance Reviews"], priority=82,
        conditions=RuleConditions(target_flags=["has_pii", "has_hr"]),
        contract=RuleContract(violation_action="conditional", max_detail="anonymize", numeric_granularity="hidden"),
    ),
    _template(
        "FINANCE-REPORTS", "Báo cáo tài chính - số liệu tổng hợp",
        "Không hiển thị số liệu tài chính chi tiết; chỉ cho phép tổng hợp hoặc khoảng giá trị.",
        category="Department", department="Finance",
        document_types=["Invoices", "Payments", "Financial Reports", "Accounting Books"], priority=84,
        conditions=RuleConditions(target_flags=["has_financial"]),
        contract=RuleContract(violation_action="conditional", max_detail="generalize", numeric_granularity="range_only"),
    ),
    _template(
        "LEGAL-CONTRACTS", "Hợp đồng pháp lý - kiểm soát chi tiết",
        "Khái quát điều khoản nhạy cảm trong hợp đồng, hồ sơ pháp lý và giấy phép.",
        category="Department", department="Legal",
        document_types=["Contracts", "Legal Files", "Licenses"], priority=78,
        conditions=RuleConditions(target_flags=["has_legal", "has_pii"]),
        contract=RuleContract(violation_action="conditional", max_detail="summarize", numeric_granularity="hidden"),
    ),
    _template(
        "SALES-CUSTOMER-PII", "Khách hàng - bảo vệ PII",
        "Che thông tin liên hệ và định danh khách hàng khi truy vấn ngoài phạm vi phụ trách.",
        category="Department", department="Sales",
        document_types=["Customers", "Quotations", "Sales Contracts"], priority=80,
        conditions=RuleConditions(cross_dept_only=True, target_flags=["has_pii"]),
        contract=RuleContract(violation_action="conditional", max_detail="anonymize", numeric_granularity="hidden"),
    ),
    _template(
        "IT-ARCHITECTURE-CONDITIONAL", "Kiến trúc và triển khai IT",
        "Khái quát kiến trúc, source-code documentation và hướng dẫn triển khai với người ngoài IT.",
        category="Department", department="IT",
        document_types=["Architecture", "Source Code Docs", "Deployment Guides"], priority=76,
        conditions=RuleConditions(target_flags=["has_credential", "has_strategic"]),
        contract=RuleContract(violation_action="conditional", max_detail="summarize", numeric_granularity="hidden"),
    ),
    _template(
        "RND-STRATEGIC-DESIGN", "R&D - bảo vệ thiết kế và bí mật kinh doanh",
        "Khái quát thiết kế, prototype, bản vẽ và nội dung nghiên cứu chưa công bố.",
        category="Department", department="R&D",
        document_types=["Product Designs", "Research", "Blueprints", "Prototype"], priority=88,
        conditions=RuleConditions(target_flags=["has_strategic"]),
        contract=RuleContract(violation_action="conditional", max_detail="generalize", numeric_granularity="hidden"),
    ),
    _template(
        "PMO-PROJECT-LIFECYCLE", "PMO - tài liệu theo vòng đời dự án",
        "Khái quát kế hoạch, ngân sách, risk register và báo cáo dự án ngoài nhóm dự án.",
        category="Department", department="PMO",
        document_types=["Charter", "Project Plan", "WBS", "Risk Register", "Final Report"], priority=72,
        conditions=RuleConditions(cross_dept_only=True, target_flags=["has_financial", "has_strategic"]),
        contract=RuleContract(violation_action="conditional", max_detail="generalize", numeric_granularity="aggregated"),
    ),
    _template(
        "PROCUREMENT-SUPPLIER", "Mua sắm - nhà cung cấp",
        "Che giá, tài khoản và PII trong đơn mua hàng, hồ sơ nhà cung cấp và hợp đồng mua.",
        category="Department", department="Procurement",
        document_types=["Purchase Orders", "Supplier Lists", "Purchase Contracts"], priority=74,
        conditions=RuleConditions(target_flags=["has_financial", "has_pii"]),
        contract=RuleContract(violation_action="conditional", max_detail="redact", numeric_granularity="range_only"),
    ),
    _template(
        "QA-AUDIT-TRAIL", "QA/QC - bảo toàn hồ sơ kiểm tra",
        "Tóm tắt biên bản kiểm tra, CAPA và audit report nhưng không làm lộ chi tiết nhạy cảm.",
        category="Department", department="QA/QC",
        document_types=["Quality Standards", "Inspection Records", "CAPA", "Audit"], priority=68,
        conditions=RuleConditions(applicable_intents=["summarize"], target_flags=["has_legal", "has_strategic"]),
        contract=RuleContract(violation_action="conditional", max_detail="summarize", numeric_granularity="aggregated"),
    ),
]


def list_policy_templates() -> list[dict]:
    return [
        {
            "template_code": item["template_code"],
            "name": item["name"],
            "description": item["description"],
            "category": item["category"],
            "department": item["department"],
            "document_types": item["document_types"],
            "recommended": item["recommended"],
            "rule": item["rule"].model_dump(),
        }
        for item in BUILT_IN_RULE_TEMPLATES
    ]


def get_policy_template(code: str) -> dict | None:
    normalized = code.strip().upper()
    return next((item for item in BUILT_IN_RULE_TEMPLATES if item["template_code"] == normalized), None)
