"""
policy_contract_agent.py
========================
Orchestrates the full policy-contract generation pipeline:
  1. Domain Classification (LLM + DB domains)
  2. Sensitivity Scoring (declared + domain base + PII)
  3. Intent Risk Analysis
  4. Rule Selection (from DB)
  5. Conflict Resolution (deny-overrides)
  6. Policy-Contract Generation (4-component output)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from app.services.policy_agent.domain_classifier import DomainClassifier
from app.services.policy_agent.risk_analyzer import IntentRiskAnalyzer, SensitivityScorer
from app.services.policy_agent.rule_selector import RuleSelector, SelectionContext

logger = logging.getLogger(__name__)


# Return the set of all ancestor OUI IDs (including the node itself) via BFS.
def _get_oui_ancestors(db, oui_id: str) -> set[str]:
    from app.models.org_unit_instance import OrgUnitInstance
    visited: set[str] = set()
    queue = [oui_id]
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        oui = db.get(OrgUnitInstance, cur)
        if oui:
            for parent in oui.parents:
                if parent.id not in visited:
                    queue.append(parent.id)
    return visited


# Compute the effective clearance of a user relative to a chunk's OUI nodes.
# Exact match → use that position's clearance.
# Document above user (chunk is ancestor of user OUI) → take the lowest clearance among positions below the chunk.
# Document below user (user position is ancestor of chunk OUI) → take the highest clearance among those positions.
# No relationship → return 1 (lowest clearance, triggers rules).
def _compute_effective_clearance(
    user_positions: list[dict],   # [{oui_id, clearance}]
    chunk_oui_ids: set[str],      # OUI IDs of the document containing the chunk
    db,
) -> int:
    if not user_positions or not chunk_oui_ids:
        return max((p["clearance"] for p in user_positions), default=1)

    # Case 1: exact match
    for oui_id in chunk_oui_ids:
        for pos in user_positions:
            if pos["oui_id"] == oui_id:
                return pos["clearance"]

    # Build ancestor sets for all chunk OUI nodes.
    chunk_ancestor_sets: dict[str, set[str]] = {}
    for oui_id in chunk_oui_ids:
        chunk_ancestor_sets[oui_id] = _get_oui_ancestors(db, oui_id)

    # Case 2: document above user — chunk_oui is an ancestor of the user's OUI.
    positions_under_chunk: list[dict] = []
    for pos in user_positions:
        pos_ancestors = _get_oui_ancestors(db, pos["oui_id"])
        for oui_id in chunk_oui_ids:
            if oui_id in pos_ancestors and oui_id != pos["oui_id"]:
                positions_under_chunk.append(pos)
                break
    if positions_under_chunk:
        # Multiple branches — take the lowest clearance (most restrictive).
        return min(p["clearance"] for p in positions_under_chunk)

    # Case 3: document below user — user position is an ancestor of the chunk OUI.
    all_chunk_ancestors = set().union(*chunk_ancestor_sets.values())
    positions_above_chunk = [
        pos for pos in user_positions
        if pos["oui_id"] in all_chunk_ancestors
    ]
    if positions_above_chunk:
        return max(p["clearance"] for p in positions_above_chunk)

    return 1  # No relationship — return the lowest clearance.


_INT_TO_SENSITIVITY = {1: "Public", 2: "Internal", 3: "Confidential", 4: "Restricted", 5: "TopSecret"}

# Default policy-contract when no domain/rule is configured
_DEFAULT_CONTRACT = {
    "violation_action":    "allow",
    "max_detail":          "summarize",
    "numeric_granularity": "exact",
}


class PolicyContractAgent:

    # Initialise sub-components for the policy-contract pipeline.
    def __init__(self):
        self.domain_classifier  = DomainClassifier()
        self.sensitivity_scorer = SensitivityScorer()
        self.intent_analyzer    = IntentRiskAnalyzer()
        self.rule_selector      = RuleSelector()

    # LLM-based rule relevance filter — keeps only rules actually applicable to the query.
    # request-scoped cache ensures the same (rule set, query) pair calls LLM only once.
    def _filter_rules_by_relevance(self, rules, raw_query: str, *, cache: dict | None = None):
        if not rules:
            return rules

        cache_key = (frozenset(r.rule_code for r in rules), raw_query[:120])
        if cache is not None and cache_key in cache:
            relevant_codes = cache[cache_key]
            print(f"[POLICY] 5.5. cache hit → {relevant_codes}")
            filtered = [r for r in rules if r.rule_code in relevant_codes]
            mandatory = [r for r in rules if r.mandatory and r not in filtered]
            return filtered + mandatory or rules

        from app.services.llm_service import llm_service
        if not llm_service.is_configured():
            return rules

        rule_lines = "\n".join(f"- {r.rule_code}: {r.name}" for r in rules)
        system = (
            "Bạn là bộ lọc quy tắc chính sách doanh nghiệp. "
            "Chỉ giữ lại các rule_code thực sự liên quan đến loại thông tin mà câu hỏi đang yêu cầu. "
            "Trả về JSON duy nhất: {\"relevant\": [\"<rule_code>\", ...]}. Không giải thích."
        )
        prompt = (
            f"Câu hỏi: \"{raw_query}\"\n\n"
            f"Danh sách rule:\n{rule_lines}\n\n"
            "Rule nào thực sự liên quan đến nội dung câu hỏi trên?"
        )
        try:
            import re as _re, json as _json
            text, _, _ = llm_service.generate(
                prompt=prompt, system=system, max_tokens=64, temperature=0.0,
            )
            m = _re.search(r'\{[\s\S]*\}', text)
            if m:
                relevant_codes = set(_json.loads(m.group()).get("relevant", []))
                if cache is not None:
                    cache[cache_key] = relevant_codes
                print(f"[POLICY] 5.5. LLM → {relevant_codes}")
                filtered = [r for r in rules if r.rule_code in relevant_codes]
                mandatory = [r for r in rules if r.mandatory and r not in filtered]
                return filtered + mandatory or rules
        except Exception as exc:
            logger.warning("Rule relevance filter failed: %s", exc)
        return rules

    # Run the full pipeline and return a policy-contract dict for the chunk.
    # Output keys: contract_id, chunk_id, generated_at, domains, effective_sensitivity,
    # pii_detected, decision, max_detail, numeric_granularity,
    # violation_action, applied_rules, needs_human_review, intent_class, intent_risk.
    def generate_contract(
        self,
        *,
        chunk_id:             str,
        chunk_text:           str,
        chunk_metadata:       dict,
        declared_sensitivity: int,      # 1-5 from chunk_sensitivity at ingest
        user_role:            str,
        user_level:           int,
        user_department:      str,
        user_id:              str,
        intent_class:         str,
        raw_query:            str,
        is_off_hours:             bool = False,
        user_positions:           list[dict] | None = None,  # [{oui_id, clearance}]
        detected_entity_types:    set[str] | None = None,    # pre-computed by batch extractor
        detected_flags:           set[str] | None = None,
        detected_entities:        list[dict] | None = None,
        rule_filter_cache:        dict | None = None,        # request-scoped LLM relevance cache
        db=None,
    ) -> dict:

        cid = chunk_id[:8]
        print(f"\n[POLICY] ── chunk={cid} ──────────────────────────────────")

        # ── 1. Realtime Entity Extraction ──────────────────────────────────
        # Skip if already pre-computed by the batch extractor in chat_service.
        if detected_entity_types is None or detected_flags is None:
            try:
                from app.services.entity_extractor import run_pipeline
                extracted = run_pipeline(chunk_text, db=db)
                detected_entity_types = set(extracted.get("entity_types") or [])
                detected_flags = {k for k, value in (extracted.get("labels") or {}).items() if value}
                detected_entities = extracted.get("entities") or []
            except Exception as exc:
                logger.warning("Realtime entity extraction failed chunk=%s: %s", chunk_id, exc)
                detected_entity_types = set()
                detected_flags = set()
                detected_entities = []
        detected_entity_types = detected_entity_types or set()
        detected_flags = detected_flags or set()
        print(f"[POLICY] 1. entities detected: {sorted(detected_entity_types) or '(none)'}")

        # ── 2. Domain Classification ───────────────────────────────────────
        chunk_dept = chunk_metadata.get("section_heading", "")

        try:
            domain_predictions = self.domain_classifier.classify(
                chunk_text,
                db=db,
                detected_entity_types=detected_entity_types,
            )
            domain_codes = [p.domain_code for p in domain_predictions]
            domains_info = [{"code": p.domain_code, "confidence": p.confidence} for p in domain_predictions]
        except Exception as exc:
            logger.warning("Domain classification failed for chunk %s: %s", chunk_id, exc)
            domain_codes = ["GEN-00"]
            domains_info = [{"code": "GEN-00", "confidence": 1.0}]
        print(f"[POLICY] 2. domains: {[(p['code'], p['confidence']) for p in domains_info]}")

        # ── 3. Sensitivity Scoring ─────────────────────────────────────────
        declared_label = _INT_TO_SENSITIVITY.get(declared_sensitivity, "Internal")
        pii_detected   = "has_pii" in detected_flags

        # Get domain base sensitivity from DB
        domain_base_sensitivity = 2  # default: Internal
        if db is not None:
            try:
                from app.repositories.policy_repository import policy_repository
                primary_domain = policy_repository.get_domain_by_code(db, domain_codes[0])
                if primary_domain:
                    domain_base_sensitivity = primary_domain.base_sensitivity
            except Exception:
                pass

        sensitivity_result = self.sensitivity_scorer.score(
            declared_sensitivity=declared_label,
            domain_base_sensitivity=domain_base_sensitivity,
            pii_detected=pii_detected,
        )

        # ── 4. Intent Risk Analysis ────────────────────────────────────────
        intent_result = self.intent_analyzer.analyze(
            intent_class=intent_class,
            raw_query=raw_query,
            user_department=user_department,
            chunk_department=chunk_dept,
            user_level=user_level,
            is_off_hours=is_off_hours,
        )

        # ── 5. Rule Selection ─────────────────────────────────────────────
        # Compute effective clearance from the user position closest to the chunk OUI.
        chunk_oui_ids: set[str] = set(
            filter(None, (chunk_metadata.get("oui_id") or "").split(","))
        )
        if db is not None and user_positions and chunk_oui_ids:
            eff_clearance = _compute_effective_clearance(user_positions, chunk_oui_ids, db)
        else:
            eff_clearance = user_level  # Fallback when OUI data is unavailable.

        ctx = SelectionContext(
            domain_codes=domain_codes,
            effective_sensitivity=sensitivity_result.effective_sensitivity,
            pii_detected=pii_detected,
            user_role=user_role,
            user_level=user_level,
            user_department=user_department,
            chunk_department=chunk_dept,
            is_owner=False,
            is_direct_manager_or_hr=False,
            subject_is_direct_report=False,
            is_subject=False,
            has_assigned_customer=False,
            intent_class=intent_class,
            intent_risk_signal=intent_result.risk_signal,
            effective_clearance=eff_clearance,
            detected_entity_types=set(detected_entity_types),
            detected_flags=set(detected_flags),
            user_oui_ids={str(p.get("oui_id")) for p in (user_positions or []) if p.get("oui_id")},
            chunk_oui_ids={str(v) for v in chunk_oui_ids},
        )

        db_rules: list = []
        if db is not None:
            try:
                from app.repositories.policy_repository import policy_repository
                db_rules = policy_repository.get_rules_for_domains(db, domain_codes)
            except Exception as exc:
                logger.warning("Failed to load rules from DB: %s", exc)

        selection = self.rule_selector.select(ctx, db_rules)
        selected_rules = selection["selected_rules"]
        print(f"[POLICY] 5. rules selected ({len(selected_rules)}): {[(r.rule_code, r.contract.get('violation_action','?')) for r in selected_rules]}")

        # ── 5.5. LLM Relevance Filter ──────────────────────────────────────
        if len(selected_rules) > 1:
            selected_rules = self._filter_rules_by_relevance(
                selected_rules, raw_query, cache=rule_filter_cache,
            )
            print(f"[POLICY] 5.5. after filter ({len(selected_rules)}): {[r.rule_code for r in selected_rules]}")

        # ── 6. Conflict Resolution ─────────────────────────────────────────
        resolution = self.rule_selector.resolve_conflicts(selected_rules)
        print(f"[POLICY] 6. decision={resolution['final_action']} contract={resolution.get('contract')} reason={resolution['reason']}")

        # ── 7. Build Policy-Contract ───────────────────────────────────────
        # For conditional: resolution["contract"] is the merged contract across all matched rules.
        contract_terms = resolution.get("contract") or _DEFAULT_CONTRACT

        return {
            "contract_id":         f"PC-{uuid.uuid4().hex[:12]}",
            "chunk_id":            chunk_id,
            "generated_at":        datetime.utcnow().isoformat(),
            # Domain
            "domains":             domains_info,
            # Sensitivity
            "effective_sensitivity": sensitivity_result.effective_sensitivity,
            "pii_detected":          pii_detected,
            # Decision
            "decision":            resolution["final_action"],
            "decision_reason":     resolution["reason"],
            # 4-component policy-contract (merged when multiple conditional rules apply)
            "max_detail":          contract_terms.get("max_detail", "generalize"),
            "numeric_granularity": contract_terms.get("numeric_granularity", "aggregated"),
            "violation_action":    contract_terms.get("violation_action", "allow"),
            "contract":             contract_terms,
            "field_rules":          contract_terms.get("field_rules", []),
            # Applied rules summary
            "applied_rules": [
                {
                    "rule_id":    r.rule_id,
                    "rule_code":  r.rule_code,
                    "domain":     r.domain_code,
                    "name":       r.name,
                    "action":     r.contract.get("violation_action", r.action),
                    "scope":      "field" if (r.target_entity_types or r.target_flags) else "chunk",
                    "priority":   r.priority,
                    "score":      r.score,
                    "reasons":    r.reasons,
                    "target_entity_types": r.target_entity_types,
                    "target_flags": r.target_flags,
                }
                for r in selected_rules
            ],
            "needs_human_review": selection["needs_human_review"],
            "intent_class":       intent_class,
            "intent_risk": {
                "signal":    intent_result.risk_signal,
                "score":     intent_result.risk_score,
                "reasoning": intent_result.reasoning,
            },
        }


# Module-level singleton; imported by chat_service for per-chunk policy evaluation.
policy_contract_agent = PolicyContractAgent()
