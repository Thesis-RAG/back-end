from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models.entity_policy_rule import EntityPolicyRule
from app.services import entity_extractor
from app.services.policy_rule_service import policy_rule_service


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_default_rules_are_seeded_and_new_labels_are_active():
    engine = make_db()
    with Session(engine) as db:
        policy_rule_service.seed_defaults(db)
        db.commit()
        keys = {rule.entity_key for rule in policy_rule_service.active_rules(db)}
        assert {"credential", "salary", "bonus", "customer", "project_code"}.issubset(keys)

        db.add(EntityPolicyRule(
            entity_key="custom_label",
            display_name="Custom label",
            group_name="custom",
            detection_source="manual",
            action="mask",
            scope_oui_ids=[],
            scope_position_ids=[],
            metadata_json={},
        ))
        db.commit()
        entity_extractor.invalidate_label_cache()
        labels, _ = entity_extractor._refresh_cache(db)
        assert "custom_label" in labels


def test_chunk_detection_only_uses_confirmed_labels(monkeypatch):
    engine = make_db()
    monkeypatch.setattr(entity_extractor, "_get_gliner", lambda: None)
    with Session(engine) as db:
        policy_rule_service.seed_defaults(db)
        db.commit()
        entity_extractor.invalidate_label_cache()
        details = entity_extractor.extract_realtime_batch_detailed(
            ["Contact a@example.com and salary 100"],
            db=db,
            labels=["email"],
        )
        assert {item["label"] for item in details[0]["entities"]} == {"email"}
        assert details[0]["entity_types"] == {"email"}


def test_resolved_policy_snapshot_does_not_change_after_rule_edit():
    engine = make_db()
    with Session(engine) as db:
        policy_rule_service.seed_defaults(db)
        db.commit()
        old_snapshot = policy_rule_service.snapshot(db)
        salary = db.query(EntityPolicyRule).filter(EntityPolicyRule.entity_key == "salary").one()
        salary.action = "block"
        db.commit()
        new_snapshot = policy_rule_service.snapshot(db)

        old_salary = next(item for item in old_snapshot["resolved_rules"] if item["entity_key"] == "salary")
        new_salary = next(item for item in new_snapshot["resolved_rules"] if item["entity_key"] == "salary")
        assert old_salary["action"] == "mask"
        assert new_salary["action"] == "block"
