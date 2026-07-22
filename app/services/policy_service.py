"""
Business logic for domain/entity/rule management.
Includes LLM-based entity type suggestion when a domain is created.
"""
from __future__ import annotations

import json
import logging
import re

from sqlalchemy.orm import Session

from app.models.policy_domain import DomainEntityType, DomainRule, PolicyDomain
from app.repositories.policy_repository import policy_repository
from app.schemas.policy import (
    DomainRuleCreate,
    DomainRuleUpdate,
    derive_action,
    EntityTypeCreate,
    PolicyDomainCreate,
    PolicyDomainUpdate,
)
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

# Fixed set of boolean flags. LLM must classify each entity type into one or more of these.
BOOLEAN_FLAGS = [
    "has_pii",        # personal identifiable information
    "has_financial",  # financial / quantitative business data
    "has_credential", # authentication secrets (passwords, tokens, keys)
    "has_legal",      # legal / regulatory / contractual content
    "has_strategic",  # strategic / competitive plans
    "has_hr",         # HR-specific sensitive data (salary, employment, medical)
]

_SUGGEST_SYSTEM = (
    "You are a data governance expert. "
    "Given a business domain name and description, generate entity types that commonly appear "
    "in documents of this domain. Entity type names must be in snake_case English. "
    "For each entity type, classify it into one or more of these boolean flags:\n"
    "  has_pii        — personal identifiable information (names, IDs, contacts, addresses)\n"
    "  has_financial  — financial or quantitative business data (money, revenue, %, investments)\n"
    "  has_credential — authentication secrets (passwords, tokens, API keys, OTPs)\n"
    "  has_legal      — legal/regulatory/contractual content (clauses, law refs, contracts)\n"
    "  has_strategic  — strategic or competitive information (plans, M&A, roadmaps, launches)\n"
    "  has_hr         — HR-specific sensitive data (salary, employment records, medical, BHXH)\n"
    "Respond with JSON only:\n"
    "{\"entity_types\": [{\"entity_type\": \"<name>\", \"label_vi\": \"<Vietnamese label>\", "
    "\"boolean_labels\": [\"<flag>\", ...]}, ...]}\n"
    "Generate up to 10 entity types most relevant to this domain — fewer is fine if the domain "
    "does not have that many distinct entity types worth tracking. "
    "Use [] for boolean_labels if the entity type is not sensitive. No explanation, only JSON."
)


class PolicyService:

    # ── Domains ───────────────────────────────────────────────────────────────

    # Return all domains, optionally filtering to active-only.
    def list_domains(self, db: Session, *, active_only: bool = False) -> list[PolicyDomain]:
        return policy_repository.list_domains(db, active_only=active_only)

    # Return a domain by ID, raising ValueError if not found.
    def get_domain(self, db: Session, domain_id: str) -> PolicyDomain:
        domain = policy_repository.get_domain(db, domain_id)
        if not domain:
            raise ValueError(f"Domain {domain_id} not found")
        return domain

    # Create a domain and optionally auto-suggest entity types via LLM.
    def create_domain(
        self, db: Session, data: PolicyDomainCreate, *, auto_suggest: bool = True
    ) -> PolicyDomain:
        if policy_repository.get_domain_by_code(db, data.code):
            raise ValueError(f"Domain code '{data.code}' already exists")

        domain = policy_repository.create_domain(
            db,
            code=data.code,
            name=data.name,
            description=data.description,
            base_sensitivity=data.base_sensitivity,
        )

        if auto_suggest:
            try:
                suggested = self._suggest_entity_types(data.name, data.description)
                for et in suggested:
                    existing = policy_repository.get_entity_type_by_name(db, domain.id, et.entity_type)
                    if not existing:
                        policy_repository.create_entity_type(
                            db,
                            domain_id=domain.id,
                            entity_type=et.entity_type,
                            label_vi=et.label_vi,
                            boolean_labels=et.boolean_labels,
                            is_system_suggested=True,
                        )
            except Exception as exc:
                logger.warning("Entity type suggestion failed for domain %s: %s", data.code, exc)

        db.commit()
        db.refresh(domain)
        return domain

    # Update mutable fields of an existing domain.
    def update_domain(self, db: Session, domain_id: str, data: PolicyDomainUpdate) -> PolicyDomain:
        domain = self.get_domain(db, domain_id)
        updates = data.model_dump(exclude_none=True)
        policy_repository.update_domain(db, domain, **updates)
        db.commit()
        db.refresh(domain)
        return domain

    # Delete a domain permanently.
    def delete_domain(self, db: Session, domain_id: str) -> None:
        domain = self.get_domain(db, domain_id)
        policy_repository.delete_domain(db, domain)
        db.commit()

    # ── Entity Types ──────────────────────────────────────────────────────────

    # Public wrapper: call the LLM to suggest entity types for a given domain name/description.
    def suggest_entity_types(self, name: str, description: str | None) -> list[EntityTypeCreate]:
        return self._suggest_entity_types(name, description)

    # Call the LLM and parse the JSON response into EntityTypeCreate objects.
    def _suggest_entity_types(
        self, name: str, description: str | None
    ) -> list[EntityTypeCreate]:
        prompt = (
            f"Business domain: {name}\n"
            f"Description: {description or 'N/A'}\n\n"
            "Generate up to 10 entity types most commonly found in documents of this domain — "
            "use fewer if the domain is narrow. "
            "For each, classify it into the appropriate boolean flag(s) from the list above."
        )
        try:
            text, _, _ = llm_service.generate(
                prompt=prompt,
                system=_SUGGEST_SYSTEM,
                max_tokens=512,
                temperature=0.0,
            )
            raw = _extract_json(text)
            items: list[dict] = raw.get("entity_types", [])
        except Exception as exc:
            logger.error("LLM entity suggestion error: %s", exc)
            items = []

        result = []
        seen: set[str] = set()
        for item in items:
            et = item.get("entity_type", "").strip().lower().replace(" ", "_")
            if not et or et in seen:
                continue
            seen.add(et)
            # Validate flags — only keep known flag names
            flags = [f for f in (item.get("boolean_labels") or []) if f in BOOLEAN_FLAGS]
            result.append(EntityTypeCreate(
                entity_type=et,
                label_vi=item.get("label_vi"),
                boolean_labels=flags,
            ))
        return result[:10]

    # Add a manually-defined entity type to a domain, raising ValueError on duplicate.
    def add_entity_type(
        self,
        db: Session,
        domain_id: str,
        data: EntityTypeCreate,
    ) -> DomainEntityType:
        self.get_domain(db, domain_id)
        existing = policy_repository.get_entity_type_by_name(db, domain_id, data.entity_type)
        if existing:
            raise ValueError(f"Entity type '{data.entity_type}' already exists in this domain")
        obj = policy_repository.create_entity_type(
            db,
            domain_id=domain_id,
            entity_type=data.entity_type,
            label_vi=data.label_vi,
            boolean_labels=data.boolean_labels,
            is_system_suggested=False,
        )
        db.commit()
        return obj

    # Delete an entity type from a domain, raising ValueError if not found.
    def delete_entity_type(self, db: Session, domain_id: str, entity_type_id: str) -> None:
        self.get_domain(db, domain_id)
        et = policy_repository.get_entity_type(db, entity_type_id)
        if not et or et.domain_id != domain_id:
            raise ValueError("Entity type not found in this domain")
        policy_repository.delete_entity_type(db, et)
        db.commit()

    # ── Rules ─────────────────────────────────────────────────────────────────

    # Return all rules for a specific domain.
    def list_rules(self, db: Session, domain_id: str) -> list[DomainRule]:
        return policy_repository.list_rules(db, domain_id)

    # Create a new rule, raising ValueError on duplicate rule_code.
    def create_rule(
        self, db: Session, domain_id: str, data: DomainRuleCreate
    ) -> DomainRule:
        self.get_domain(db, domain_id)
        if policy_repository.get_rule_by_code(db, data.rule_code):
            raise ValueError(f"Rule code '{data.rule_code}' already exists")
        rule = policy_repository.create_rule(
            db,
            domain_id=domain_id,
            rule_code=data.rule_code,
            name=data.name,
            action=derive_action(data.contract.violation_action, data.contract.max_detail),
            priority=data.priority,
            mandatory=data.mandatory,
            risk_level="low",
            audit_log=data.audit_log,
            conditions_json=data.conditions.model_dump(),
            contract_json=data.contract.model_dump(),
        )
        db.commit()
        return rule

    # Apply partial updates to an existing rule.
    def update_rule(self, db: Session, rule_id: str, data: DomainRuleUpdate) -> DomainRule:
        rule = policy_repository.get_rule(db, rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id} not found")
        updates: dict = {}
        if data.name is not None:
            updates["name"] = data.name
        if data.priority is not None:
            updates["priority"] = data.priority
        if data.mandatory is not None:
            updates["mandatory"] = data.mandatory
        if data.is_active is not None:
            updates["is_active"] = data.is_active
        if data.audit_log is not None:
            updates["audit_log"] = data.audit_log
        if data.conditions is not None:
            updates["conditions_json"] = data.conditions.model_dump()
        if data.contract is not None:
            updates["contract_json"] = data.contract.model_dump()
            updates["action"] = derive_action(data.contract.violation_action, data.contract.max_detail)
        policy_repository.update_rule(db, rule, **updates)
        db.commit()
        return rule

    # Delete a rule by ID, raising ValueError if not found.
    def delete_rule(self, db: Session, rule_id: str) -> None:
        rule = policy_repository.get_rule(db, rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id} not found")
        policy_repository.delete_rule(db, rule)
        db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

# Extract the first JSON object from LLM output text.
def _extract_json(text: str) -> dict:
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group())
    return {}


# Module-level singleton; imported by the policy API router and policy agent.
policy_service = PolicyService()
