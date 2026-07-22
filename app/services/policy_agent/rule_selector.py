"""
rule_selector.py
================
Loads rules from DB (domain_rules table), scores them, applies deny-overrides conflict resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.policy_agent.risk_analyzer import sensitivity_rank

# Restrictiveness order: higher = more restrictive
_MAX_DETAIL_ORDER = {"redact": 4, "anonymize": 3, "generalize": 2, "summarize": 1}
_NUMERIC_ORDER    = {"hidden": 4, "aggregated": 3, "range_only": 2, "exact": 1}


def _infer_legacy_targets(rule: dict) -> tuple[list[str], list[str]]:
    """Infer field scope for older rules created before target fields existed."""
    explicit_types = [str(v).lower() for v in (rule.get("target_entity_types") or [])]
    explicit_flags = [str(v).lower() for v in (rule.get("target_flags") or [])]
    if explicit_types or explicit_flags:
        return explicit_types, explicit_flags
    text = f"{rule.get('rule_code', '')} {rule.get('name', '')}".lower()
    if any(word in text for word in ("salary", "lương", "luong", "payroll", "wage", "compensation", "thưởng", "thuong")):
        return ["money", "salary", "salary_amount"], ["has_financial", "has_hr"]
    if any(word in text for word in ("personal", "cá nhân", "ca nhan", "pii", "contact", "thông tin cá nhân", "thong tin ca nhan")):
        return ["person_name", "name", "email", "phone", "address", "dob", "national_id"], ["has_pii"]
    return [], []


@dataclass
class ScoredRule:
    rule_id:     str
    rule_code:   str
    name:        str
    action:      str          # ALLOW | DENY | REDACT | ALLOW_WITH_WATERMARK
    priority:    int
    mandatory:   bool
    risk_level:  str
    domain_code: str
    score:       float
    reasons:     list[str] = field(default_factory=list)
    contract:    dict       = field(default_factory=dict)
    target_entity_types: list[str] = field(default_factory=list)
    target_flags: list[str] = field(default_factory=list)


@dataclass
class SelectionContext:
    domain_codes:              list[str]
    effective_sensitivity:     str
    pii_detected:              bool
    user_role:                 str
    user_level:                int
    user_department:           str
    chunk_department:          str
    is_owner:                  bool
    is_direct_manager_or_hr:   bool
    subject_is_direct_report:  bool
    is_subject:                bool
    has_assigned_customer:     bool
    intent_class:              str
    intent_risk_signal:        str
    # Org hierarchy: level = depth from root (0=company, 1=branch, 2=group, ...)
    user_ou_levels:            list[int] = None   # all positions held by the user
    chunk_ou_level:            int = None         # org-hierarchy level of the document
    is_before_publish_date:    bool = False
    is_after_publish_date:     bool = False
    # Effective clearance derived from the user position closest to the chunk (1–5).
    effective_clearance:       int = 1
    detected_entity_types:     set[str] = field(default_factory=set)
    detected_flags:            set[str] = field(default_factory=set)
    user_oui_ids:              set[str] = field(default_factory=set)
    chunk_oui_ids:             set[str] = field(default_factory=set)


class RuleSelector:
    SCORE_THRESHOLD = 0.4
    MAX_RULES_WARN  = 8

    # Score all db_rules against ctx and return selected rules, review flag, and candidate count.
    # db_rules: list[DomainRule] ORM objects loaded from DB.
    def select(self, ctx: SelectionContext, db_rules: list) -> dict:
        candidates: list[ScoredRule] = []

        for db_rule in db_rules:
            # Map ORM → rule dict for scoring
            rule_dict = {
                "rule_id":    db_rule.id,
                "rule_code":  db_rule.rule_code,
                "name":       db_rule.name,
                "action":     db_rule.action,
                "priority":   db_rule.priority,
                "mandatory":  db_rule.mandatory,
                "risk_level": db_rule.risk_level,
                "contract":   db_rule.contract_json or {},
                **(db_rule.conditions_json or {}),
            }
            domain_code = ctx.domain_codes[0] if ctx.domain_codes else "UNCLASSIFIED"

            scored = self._score_rule(rule_dict, domain_code, ctx)
            if scored:
                candidates.append(scored)

        selected = [
            c for c in candidates
            if c.score >= self.SCORE_THRESHOLD or c.mandatory
        ]
        # ensure all mandatory rules included
        mandatory_ids = {c.rule_id for c in candidates if c.mandatory}
        existing_ids  = {c.rule_id for c in selected}
        for c in candidates:
            if c.rule_id in mandatory_ids and c.rule_id not in existing_ids:
                selected.append(c)
                existing_ids.add(c.rule_id)

        if not selected:
            # fallback: default-allow contract
            fallback = ScoredRule(
                rule_id="FALLBACK", rule_code="FALLBACK", name="Default Allow",
                action="ALLOW", priority=0, mandatory=False, risk_level="low",
                domain_code=ctx.domain_codes[0] if ctx.domain_codes else "UNCLASSIFIED",
                score=1.0, reasons=["no_rule_match"],
                contract={
                    "violation_action": "allow",
                    "max_detail": "summarize",
                    "numeric_granularity": "exact",
                },
            )
            selected = [fallback]

        return {
            "selected_rules":    selected,
            "needs_human_review": len(selected) > self.MAX_RULES_WARN,
            "candidate_count":   len(candidates),
        }

    # Resolve whole-chunk rules normally, but keep entity-targeted rules
    # independent so a salary deny does not discard allowed personal fields.
    def resolve_conflicts(self, selected_rules: list[ScoredRule]) -> dict:
        if not selected_rules:
            return {
                "final_action": "block",
                "winning_rule": None,
                "contract":     {},
                "reason":       "no_rule_default_deny",
            }

        def _action(r: ScoredRule) -> str:
            action = str(r.contract.get("violation_action") or r.action or "allow").lower()
            return {"deny": "block", "redact": "conditional", "anonymize": "conditional", "generalize": "conditional", "allow_with_watermark": "watermark"}.get(action, action)

        def top(rules: list[ScoredRule]) -> ScoredRule:
            return max(rules, key=lambda r: r.priority)

        scoped_rules = [r for r in selected_rules if r.target_entity_types or r.target_flags]
        whole_chunk_rules = [r for r in selected_rules if r not in scoped_rules]

        def resolve_whole_chunk(rules: list[ScoredRule]) -> dict:
            block_rules       = [r for r in rules if _action(r) == "block"]
            conditional_rules = [r for r in rules if _action(r) == "conditional"]
            watermark_rules   = [r for r in rules if _action(r) == "watermark"]
            allow_rules       = [r for r in rules if _action(r) == "allow"]
            if block_rules:
                return {"final_action": "block", "winning_rule": top(block_rules),
                        "contract": {"violation_action": "block"}, "reason": "block_overrides"}
            if conditional_rules:
                return {"final_action": "conditional", "winning_rule": top(conditional_rules),
                        "contract": self._merge_conditional(conditional_rules), "reason": "conditional_merged"}
            if watermark_rules:
                return {"final_action": "watermark", "winning_rule": top(watermark_rules),
                        "contract": {"violation_action": "watermark"}, "reason": "watermark_only"}
            if allow_rules:
                return {"final_action": "allow", "winning_rule": top(allow_rules),
                        "contract": {"violation_action": "allow"}, "reason": "allow_only"}
            return {"final_action": "allow", "winning_rule": None, "contract": {}, "reason": "no_whole_chunk_rule"}

        whole_resolution = resolve_whole_chunk(whole_chunk_rules)

        # A field-scoped rule must never turn into a whole-chunk decision.
        # Only an explicitly unscoped BLOCK can discard the complete chunk.
        # For every other whole-chunk result, keep resolving targeted rules
        # independently so (for example) a salary deny does not hide an
        # otherwise allowed person_name entity in the same chunk.
        if scoped_rules and whole_resolution["final_action"] != "block":
            field_rules = []
            for rule in sorted(scoped_rules, key=lambda r: r.priority, reverse=True):
                field_rules.append({
                    "scope": "field",
                    "rule_code": rule.rule_code,
                    "name": rule.name,
                    "action": _action(rule),
                    "priority": rule.priority,
                    "target_entity_types": rule.target_entity_types,
                    "target_flags": rule.target_flags,
                    "contract": rule.contract,
                })
            return {
                "final_action": "field_scoped",
                "winning_rule": top(scoped_rules),
                "contract": {
                    "violation_action": "field_scoped",
                    "field_rules": field_rules,
                    "whole_chunk_action": whole_resolution["final_action"],
                    "whole_chunk_contract": whole_resolution.get("contract") or {},
                },
                "reason": "field_scoped_conflicts_resolved_independently",
            }
        if whole_chunk_rules:
            return whole_resolution
        return {"final_action": "allow", "winning_rule": None, "contract": {}, "reason": "no_scoped_target"}

    # Merge multiple conditional rules into one contract: most restrictive per field wins.
    def _merge_conditional(self, rules: list[ScoredRule]) -> dict:
        best_detail  = max(rules, key=lambda r: _MAX_DETAIL_ORDER.get(r.contract.get("max_detail", ""), 0))
        best_numeric = max(rules, key=lambda r: _NUMERIC_ORDER.get(r.contract.get("numeric_granularity", ""), 0))
        return {
            "violation_action":    "conditional",
            "max_detail":          best_detail.contract.get("max_detail", "generalize"),
            "numeric_granularity": best_numeric.contract.get("numeric_granularity", "aggregated"),
        }

    # ── Internal scoring ──────────────────────────────────────────────────────

    # Score a single rule against the selection context; return None if the rule does not apply.
    def _score_rule(
        self, rule: dict, domain_code: str, ctx: SelectionContext
    ) -> ScoredRule | None:
        reasons: list[str] = []
        target_entity_types, target_flags = _infer_legacy_targets(rule)

        # ── 1. Exemption roles — if the user holds one of these, the rule does not apply.
        exempt_roles = rule.get("applicable_roles") or []
        if exempt_roles and ctx.user_role in exempt_roles:
            return None

        # ── 2. Blocked roles — if the user holds one of these, the rule is force-applied.
        blocked_roles = rule.get("blocked_roles") or []
        if blocked_roles and ctx.user_role in blocked_roles:
            score = 0.95
            if rule.get("mandatory"):
                score = 1.0
                reasons.append("mandatory")
            reasons.append("role_force_applied")
            return ScoredRule(
                rule_id    = rule["rule_id"],
                rule_code  = rule.get("rule_code", rule["rule_id"]),
                name       = rule.get("name", ""),
                action     = rule["action"],
                priority   = rule.get("priority", 50),
                mandatory  = bool(rule.get("mandatory")),
                risk_level = rule.get("risk_level", "low"),
                domain_code= domain_code,
                score      = round(score, 3),
                reasons    = reasons,
                contract   = rule.get("contract") or {},
                target_entity_types = target_entity_types,
                target_flags = target_flags,
            )

        # ── 3. Sensitivity check ───────────────────────────────────────────────
        min_sens = rule.get("min_sensitivity")
        if min_sens and sensitivity_rank(ctx.effective_sensitivity) < sensitivity_rank(min_sens):
            return None
        reasons.append("sensitivity_ok")

        # ── 4. Clearance check ────────────────────────────────────────────────
        # min_user_level = minimum clearance required to VIEW the content (exempts from the rule).
        # If the user meets the clearance requirement, the rule does not apply.
        # If not, the rule fires (block/redact/...).
        if rule.get("min_user_level") and ctx.effective_clearance >= rule["min_user_level"]:
            return None
        intents = rule.get("applicable_intents") or []
        if intents and ctx.intent_class not in intents:
            return None

        applicable_oui_ids = {str(v) for v in (rule.get("applicable_oui_ids") or [])}
        if applicable_oui_ids and not applicable_oui_ids.intersection(ctx.user_oui_ids | ctx.chunk_oui_ids):
            return None

        detected_types = {str(v).lower() for v in ctx.detected_entity_types}
        detected_flags = {str(v).lower() for v in ctx.detected_flags}
        if target_entity_types and not detected_types.intersection(target_entity_types):
            return None
        if target_flags and not detected_flags.intersection(target_flags):
            return None
        if target_entity_types or target_flags:
            reasons.append("field_target_match")

        # ── 5. Cross-department / org-hierarchy check ──────────────────────────
        if rule.get("cross_dept_only"):
            user_levels = ctx.user_ou_levels or []
            chunk_level = ctx.chunk_ou_level
            if chunk_level is None or not user_levels:
                # No org-hierarchy data available — fall back to department string comparison.
                if ctx.user_department and ctx.chunk_department:
                    if ctx.user_department.lower() == ctx.chunk_department.lower():
                        return None  # Same department — do not trigger the rule.
            else:
                # Trigger when the document is at a higher level (smaller index) than all user positions.
                # Do not trigger if the user holds any position at or above the document level.
                has_adequate_level = any(ul <= chunk_level for ul in user_levels)
                if has_adequate_level:
                    return None

        reasons.append("conditions_ok")

        # ── 7. Score (rule with no role condition → neutral) ──────────────────
        score = 0.6 if not blocked_roles and not exempt_roles else 0.8
        reasons.append("role_neutral")
        if rule.get("mandatory"):
            score += 0.2
            reasons.append("mandatory")
        score = min(1.0, score)

        return ScoredRule(
            rule_id    = rule["rule_id"],
            rule_code  = rule.get("rule_code", rule["rule_id"]),
            name       = rule.get("name", ""),
            action     = rule["action"],
            priority   = rule.get("priority", 50),
            mandatory  = bool(rule.get("mandatory")),
            risk_level = rule.get("risk_level", "low"),
            domain_code= domain_code,
            score      = round(score, 3),
            reasons    = reasons,
            contract   = rule.get("contract") or {},
            target_entity_types = target_entity_types,
            target_flags = target_flags,
        )
