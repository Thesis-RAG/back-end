from types import SimpleNamespace

from app.services.policy_rule_service import resolve_tier


def make_rule(**overrides):
    defaults = dict(
        enabled=True,
        allow_scope_pairs=[],
        mask_scope_pairs=[],
        mask_style="full",
        mask_position=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_user(oui_id, position_id=None):
    assignment = SimpleNamespace(oui_id=oui_id, position_id=position_id)
    return SimpleNamespace(oui_positions=[assignment])


def test_exact_oui_match_still_works_without_graph_expansion():
    rule = make_rule(mask_scope_pairs=[{"oui_id": "hcm-branch", "position_id": None}])
    user = make_user("hcm-branch")
    assert resolve_tier(rule, user)["tier"] == "require"


def test_no_match_and_no_expansion_falls_back_to_block():
    rule = make_rule(mask_scope_pairs=[{"oui_id": "hcm-branch", "position_id": None}])
    user = make_user("head-office")
    assert resolve_tier(rule, user)["tier"] == "block"


def test_ancestor_user_inherits_child_ous_scope_via_graph_expansion():
    # head-office manager, no exact assignment to hcm-branch — only the
    # graph-computed descendant set (as if hcm-branch were beneath them)
    # should let them inherit the mask tier, mirroring FGA's ancestor_viewer.
    rule = make_rule(mask_scope_pairs=[{"oui_id": "hcm-branch", "position_id": None}])
    user = make_user("head-office")
    expanded = {"head-office", "hcm-branch"}
    assert resolve_tier(rule, user, expanded)["tier"] == "require"


def test_ancestor_inherits_even_when_pair_pins_a_specific_position():
    # A pair naming Marketing dept + "Trưởng phòng ban" still lets the
    # branch manager above Marketing inherit it — they're not a peer within
    # Marketing being loosely matched, they're a manager above it, so the
    # pair's position pin (which only picks a role *within* that OUI) has
    # no bearing on them. Mirrors FGA's ancestor_viewer, which doesn't care
    # which position at the descendant OUI a document belongs to either.
    rule = make_rule(mask_scope_pairs=[{"oui_id": "marketing-dept", "position_id": "truong-phong"}])
    user = make_user("da-nang-branch", position_id="truong-chi-nhanh")
    expanded = {"da-nang-branch", "marketing-dept"}
    assert resolve_tier(rule, user, expanded)["tier"] == "require"


def test_direct_position_pin_still_exact_matches_within_its_own_oui():
    # Sanity check the untouched direct-match path: someone actually
    # assigned to marketing-dept still needs the exact position to match.
    rule = make_rule(mask_scope_pairs=[{"oui_id": "marketing-dept", "position_id": "truong-phong"}])
    wrong_position = make_user("marketing-dept", position_id="nhan-vien")
    assert resolve_tier(rule, wrong_position)["tier"] == "block"
    right_position = make_user("marketing-dept", position_id="truong-phong")
    assert resolve_tier(rule, right_position)["tier"] == "require"


def test_allow_still_takes_priority_over_mask_with_graph_expansion():
    rule = make_rule(
        allow_scope_pairs=[{"oui_id": "hr-dept", "position_id": None}],
        mask_scope_pairs=[{"oui_id": "__ALL__", "position_id": None}],
    )
    user = make_user("head-office")
    expanded = {"head-office", "hr-dept"}
    assert resolve_tier(rule, user, expanded)["tier"] == "full"
