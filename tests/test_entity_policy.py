from types import SimpleNamespace

from app.services.entity_extractor import adjust_chunk_sensitivity, compute_chunk_sensitivity
from app.services.chat_service import _question_only_history
from app.services.entity_policy_service import (
    EntityPolicyService,
    _replace_entity_spans,
    normalize_actions,
)


def test_actions_are_per_entity_and_invalid_values_default_to_full():
    actions = normalize_actions([
        {"entity_type": "salary amount", "action": "block"},
        {"entity_type": "salary amount", "action": "mask"},
        {"entity_type": "person_name", "action": "unknown"},
    ])

    assert actions == [
        {
            "entity_type": "salary_amount",
            "label": "salary_amount",
            "action": "block",
            "source": "manual",
            "enabled": True,
            "detection_count": 0,
            "scope_oui_ids": [],
            "scope_position_ids": [],
            "metadata_json": {},
        },
        {
            "entity_type": "person_name",
            "label": "person_name",
            "action": "full",
            "source": "manual",
            "enabled": True,
            "detection_count": 0,
            "scope_oui_ids": [],
            "scope_position_ids": [],
            "metadata_json": {},
        },
    ]


def test_blocking_one_entity_masks_only_its_span():
    text = "Nguyen Van A nhan luong 30 trieu"
    result = _replace_entity_spans(
        text,
        [
            {"label": "person_name", "start": 0, "end": 12},
            {"label": "money", "start": 24, "end": 32},
        ],
        {"money": "[ENTITY_ACCESS_REQUIRED]"},
    )

    assert "Nguyen Van A" in result
    assert "30 trieu" not in result
    assert "[ENTITY_ACCESS_REQUIRED]" in result


def test_entity_action_scope_matches_unit_and_role_assignment():
    service = EntityPolicyService()
    action = SimpleNamespace(
        scope_oui_ids=["oui-hr"],
        scope_position_ids=["position-manager"],
    )
    matching_user = SimpleNamespace(oui_positions=[SimpleNamespace(
        oui_id="oui-hr",
        position_id="position-manager",
    )])
    different_role = SimpleNamespace(oui_positions=[SimpleNamespace(
        oui_id="oui-hr",
        position_id="position-staff",
    )])

    assert service.action_applies_to_user(action, matching_user)
    assert not service.action_applies_to_user(action, different_role)


def test_sensitivity_only_lowers_for_gliner_common_public_entities():
    date_entity = [{"label": "date", "source": "gliner", "start": 0, "end": 10}]

    assert adjust_chunk_sensitivity(3, 3, date_entity, set()) == 2
    assert compute_chunk_sensitivity(3, {"has_pii": False, "has_financial": False}) == 3
    assert adjust_chunk_sensitivity(3, 3, [], set()) == 3
    assert adjust_chunk_sensitivity(
        3, 3, [{"label": "organization", "source": "gliner"}], set()
    ) == 3


def test_sensitivity_raise_is_preserved_and_sensitive_mix_is_not_lowered():
    date_entity = [{"label": "date", "source": "gliner"}]
    mixed_entities = date_entity + [{"label": "salary", "source": "gliner"}]

    assert adjust_chunk_sensitivity(3, 4, date_entity, set()) == 4
    assert adjust_chunk_sensitivity(
        3, 3, mixed_entities, {"has_financial", "has_hr"}
    ) == 3


def test_saving_entity_actions_preserves_ingest_detection_count(monkeypatch):
    service = EntityPolicyService()
    previous = SimpleNamespace(
        entity_type="salary",
        detection_count=7,
        metadata_json={"boolean_labels": ["has_financial"]},
    )
    captured = {}

    monkeypatch.setattr(
        "app.services.entity_policy_service.document_entity_repository.list_actions",
        lambda db, version_id: [previous],
    )
    monkeypatch.setattr(
        "app.services.entity_policy_service.document_entity_repository.replace_actions",
        lambda db, version_id, actions: captured.setdefault("actions", actions),
    )
    service.replace_actions(
        object(),
        "version-1",
        [{"entity_type": "salary", "action": "mask"}],
    )

    assert captured["actions"][0]["action"] == "mask"
    assert captured["actions"][0]["detection_count"] == 7
    assert captured["actions"][0]["metadata_json"] == {"boolean_labels": ["has_financial"]}


def test_query_rewrite_history_keeps_questions_only():
    assert _question_only_history([
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "response must not be rewritten"},
        {"role": "system", "content": "summary"},
    ]) == [{"role": "user", "content": "previous question"}]


def test_secret_clearance_bypasses_entity_actions_without_running_detection():
    service = EntityPolicyService()
    high_clearance_user = SimpleNamespace(
        oui_positions=[SimpleNamespace(position=SimpleNamespace(clearance=4))]
    )
    chunks = [{
        "document_text": "salary 100",
        "metadata": {"document_id": "doc-1", "document_version_id": "ver-1"},
    }]

    assert service.bypasses_entity_actions(high_clearance_user)
    processed, contracts = service.apply_to_retrieved(
        object(), high_clearance_user, "salary", chunks
    )

    assert processed is chunks
    assert contracts == []
    assert service.bypasses_entity_actions(SimpleNamespace(max_clearance=5))
    assert not service.bypasses_entity_actions(SimpleNamespace(max_clearance=3))
