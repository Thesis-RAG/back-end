from app.services.chat_service import ChatService
from app.services.policy_agent.rule_selector import RuleSelector, ScoredRule


def make_rule(
    code: str,
    action: str,
    priority: int,
    *,
    target_entity_types: list[str] | None = None,
    target_flags: list[str] | None = None,
    max_detail: str = "generalize",
) -> ScoredRule:
    return ScoredRule(
        rule_id=code,
        rule_code=code,
        name=code,
        action=action.upper(),
        priority=priority,
        mandatory=False,
        risk_level="low",
        domain_code="HR-01",
        score=1.0,
        contract={
            "violation_action": action,
            "max_detail": max_detail,
            "numeric_granularity": "aggregated",
        },
        target_entity_types=target_entity_types or [],
        target_flags=target_flags or [],
    )


def test_field_block_does_not_block_the_whole_chunk():
    resolution = RuleSelector().resolve_conflicts([
        make_rule(
            "HR-SALARY-BLOCK",
            "block",
            80,
            target_entity_types=["money", "salary_amount"],
        ),
        make_rule(
            "HR-PERSON-ALLOW",
            "allow",
            20,
            target_entity_types=["person_name"],
        ),
    ])

    assert resolution["final_action"] == "field_scoped"
    assert {
        rule["rule_code"] for rule in resolution["contract"]["field_rules"]
    } == {"HR-SALARY-BLOCK", "HR-PERSON-ALLOW"}


def test_explicit_whole_chunk_block_still_blocks_everything():
    resolution = RuleSelector().resolve_conflicts([
        make_rule("HR-WHOLE-CHUNK-BLOCK", "block", 10),
        make_rule(
            "HR-SALARY-BLOCK",
            "block",
            90,
            target_entity_types=["money"],
        ),
    ])

    assert resolution["final_action"] == "block"
    assert resolution["reason"] == "block_overrides"


def test_field_allow_keeps_person_name_while_salary_is_masked():
    text = "Nguyen Van A nhan luong 30 trieu"
    person_start = text.index("Nguyen")
    person_end = person_start + len("Nguyen Van A")
    money_start = text.index("30 trieu")
    money_end = money_start + len("30 trieu")

    resolution = RuleSelector().resolve_conflicts([
        make_rule(
            "HR-SALARY-BLOCK",
            "block",
            80,
            target_entity_types=["money"],
        ),
        make_rule(
            "HR-PERSON-ALLOW",
            "allow",
            20,
            target_entity_types=["person_name"],
        ),
    ])

    chunk = {
        "chunk_id": "chunk-1",
        "document_text": text,
        "_policy_contract": resolution["contract"],
        "_policy_entities": [
            {"start": person_start, "end": person_end, "label": "person_name"},
            {"start": money_start, "end": money_end, "label": "money"},
        ],
        "_needs_field_policy": True,
    }

    result = ChatService.__new__(ChatService)._apply_field_scoped_policy(chunk)

    assert "Nguyen Van A" in result["document_text"]
    assert "30 trieu" not in result["document_text"]
    assert "giá trị đã được ẩn theo chính sách" in result["document_text"]
    assert "[" not in result["document_text"]
    assert "_policy_entities" not in result
    assert "_needs_field_policy" not in result


def test_field_allow_alone_is_a_noop():
    text = "Nguyen Van A"
    chunk = {
        "document_text": text,
        "_policy_contract": {
            "field_rules": [
                {
                    "action": "allow",
                    "priority": 50,
                    "target_entity_types": ["person_name"],
                    "target_flags": [],
                    "contract": {"violation_action": "allow", "max_detail": "generalize"},
                }
            ]
        },
        "_policy_entities": [{"start": 0, "end": len(text), "label": "person_name"}],
        "_needs_field_policy": True,
    }

    result = ChatService.__new__(ChatService)._apply_field_scoped_policy(chunk)

    assert result["document_text"] == text
    assert "_policy_entities" not in result
    assert "_needs_field_policy" not in result
