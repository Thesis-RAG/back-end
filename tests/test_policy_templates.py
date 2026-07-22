import app.services.policy_templates as policy_templates


def test_rule_template_scope_is_generated_by_llm(monkeypatch):
    policy_templates._SCOPE_CACHE.clear()

    monkeypatch.setattr(policy_templates.llm_service, "is_configured", lambda: True)

    def fake_generate(*args, **kwargs):
        return (
            '{"scopes":{"DMS-SALARY-BLOCK":{'
            '"target_entity_types":["money"],'
            '"target_flags":["has_financial"]}}}',
            None,
            "test",
        )

    monkeypatch.setattr(policy_templates.llm_service, "generate", fake_generate)

    template = policy_templates.get_policy_template("DMS-SALARY-BLOCK")

    assert template is not None
    assert template["rule"].conditions.target_entity_types == ["money"]
    assert template["rule"].conditions.target_flags == ["has_financial"]


def test_invalid_llm_scope_keeps_template_fallback(monkeypatch):
    policy_templates._SCOPE_CACHE.clear()

    monkeypatch.setattr(policy_templates.llm_service, "is_configured", lambda: True)
    monkeypatch.setattr(
        policy_templates.llm_service,
        "generate",
        lambda *args, **kwargs: ("not-json", None, "test"),
    )

    template = policy_templates.get_policy_template("DMS-SALARY-BLOCK")

    assert template is not None
    assert "money" in template["rule"].conditions.target_entity_types
    assert "has_financial" in template["rule"].conditions.target_flags
