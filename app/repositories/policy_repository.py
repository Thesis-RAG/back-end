"""
Repository for policy domains, entity types, and domain rules persistence and retrieval.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.policy_domain import DomainEntityType, DomainRule, PolicyDomain
from app.utils.ids import new_uuid


class PolicyRepository:

    # ── Domains ───────────────────────────────────────────────────────────────

    # Return all policy domains, optionally filtered to active ones only.
    def list_domains(self, db: Session, *, active_only: bool = False) -> list[PolicyDomain]:
        q = db.query(PolicyDomain)
        if active_only:
            q = q.filter(PolicyDomain.is_active.is_(True))
        return q.order_by(PolicyDomain.code).all()

    # Return a policy domain by ID.
    def get_domain(self, db: Session, domain_id: str) -> PolicyDomain | None:
        return db.get(PolicyDomain, domain_id)

    # Return a policy domain by its unique code.
    def get_domain_by_code(self, db: Session, code: str) -> PolicyDomain | None:
        return db.query(PolicyDomain).filter(PolicyDomain.code == code).first()

    # Create and persist a new policy domain.
    def create_domain(self, db: Session, *, code: str, name: str,
                      description: str | None, base_sensitivity: int) -> PolicyDomain:
        domain = PolicyDomain(
            id=new_uuid(),
            code=code,
            name=name,
            description=description,
            base_sensitivity=base_sensitivity,
        )
        db.add(domain)
        db.flush()
        return domain

    # Update non-None fields on an existing policy domain.
    def update_domain(self, db: Session, domain: PolicyDomain, **kwargs) -> PolicyDomain:
        for k, v in kwargs.items():
            if v is not None:
                setattr(domain, k, v)
        db.flush()
        return domain

    # Delete a policy domain.
    def delete_domain(self, db: Session, domain: PolicyDomain) -> None:
        db.delete(domain)
        db.flush()

    # ── Entity Types ──────────────────────────────────────────────────────────

    # Return all entity types for a domain ordered by entity_type name.
    def list_entity_types(self, db: Session, domain_id: str) -> list[DomainEntityType]:
        return (
            db.query(DomainEntityType)
            .filter(DomainEntityType.domain_id == domain_id)
            .order_by(DomainEntityType.entity_type)
            .all()
        )

    # Return all active entity types across all active domains (used by the entity extractor).
    def get_all_active_entity_types(self, db: Session) -> list[DomainEntityType]:
        return (
            db.query(DomainEntityType)
            .join(PolicyDomain, DomainEntityType.domain_id == PolicyDomain.id)
            .filter(
                DomainEntityType.is_active.is_(True),
                PolicyDomain.is_active.is_(True),
            )
            .all()
        )

    # Return {domain_code: {entity_type strings}} for entity overlap scoring in domain classification.
    def get_entity_types_grouped_by_domain(self, db: Session) -> dict[str, set[str]]:
        rows = (
            db.query(PolicyDomain.code, DomainEntityType.entity_type)
            .join(DomainEntityType, DomainEntityType.domain_id == PolicyDomain.id)
            .filter(
                DomainEntityType.is_active.is_(True),
                PolicyDomain.is_active.is_(True),
            )
            .all()
        )
        result: dict[str, set[str]] = {}
        for code, entity_type in rows:
            result.setdefault(code, set()).add(entity_type)
        return result

    # Create and persist a new entity type within a domain.
    def create_entity_type(
        self,
        db: Session,
        *,
        domain_id: str,
        entity_type: str,
        label_vi: str | None = None,
        boolean_labels: list | None = None,
        is_system_suggested: bool = False,
    ) -> DomainEntityType:
        obj = DomainEntityType(
            id=new_uuid(),
            domain_id=domain_id,
            entity_type=entity_type,
            label_vi=label_vi,
            boolean_labels=boolean_labels or [],
            is_system_suggested=is_system_suggested,
        )
        db.add(obj)
        db.flush()
        return obj

    # Delete an entity type.
    def delete_entity_type(self, db: Session, entity_type: DomainEntityType) -> None:
        db.delete(entity_type)
        db.flush()

    # Return an entity type by ID.
    def get_entity_type(self, db: Session, entity_type_id: str) -> DomainEntityType | None:
        return db.get(DomainEntityType, entity_type_id)

    # Return an entity type by domain ID and type name.
    def get_entity_type_by_name(
        self, db: Session, domain_id: str, entity_type: str
    ) -> DomainEntityType | None:
        return (
            db.query(DomainEntityType)
            .filter(
                DomainEntityType.domain_id == domain_id,
                DomainEntityType.entity_type == entity_type,
            )
            .first()
        )

    # ── Rules ─────────────────────────────────────────────────────────────────

    # Return rules for one domain, sorted by priority descending.
    def list_rules(self, db: Session, domain_id: str, *,
                   active_only: bool = False) -> list[DomainRule]:
        q = db.query(DomainRule).filter(DomainRule.domain_id == domain_id)
        if active_only:
            q = q.filter(DomainRule.is_active.is_(True))
        return q.order_by(DomainRule.priority.desc()).all()

    # Load active rules for the given domains.
    def get_rules_for_domains(self, db: Session, domain_codes: list[str]) -> list[DomainRule]:
        domain_ids = (
            db.query(PolicyDomain.id)
            .filter(PolicyDomain.code.in_(domain_codes), PolicyDomain.is_active.is_(True))
            .all()
        )
        ids = [r[0] for r in domain_ids]
        return (
            db.query(DomainRule)
            .filter(
                DomainRule.is_active.is_(True),
                DomainRule.domain_id.in_(ids),
            )
            .order_by(DomainRule.priority.desc())
            .all()
        )

    # Return a rule by ID.
    def get_rule(self, db: Session, rule_id: str) -> DomainRule | None:
        return db.get(DomainRule, rule_id)

    # Return a rule by its unique code.
    def get_rule_by_code(self, db: Session, rule_code: str) -> DomainRule | None:
        return db.query(DomainRule).filter(DomainRule.rule_code == rule_code).first()

    # Create and persist a new domain rule.
    def create_rule(
        self,
        db: Session,
        *,
        domain_id: str,
        rule_code: str,
        name: str,
        action: str,
        priority: int,
        mandatory: bool,
        risk_level: str,
        audit_log: bool,
        conditions_json: dict,
        contract_json: dict,
    ) -> DomainRule:
        rule = DomainRule(
            id=new_uuid(),
            domain_id=domain_id,
            rule_code=rule_code,
            name=name,
            action=action,
            priority=priority,
            mandatory=mandatory,
            risk_level=risk_level,
            audit_log=audit_log,
            conditions_json=conditions_json,
            contract_json=contract_json,
        )
        db.add(rule)
        db.flush()
        return rule

    # Update non-None fields on an existing rule.
    def update_rule(self, db: Session, rule: DomainRule, **kwargs) -> DomainRule:
        for k, v in kwargs.items():
            if v is not None:
                setattr(rule, k, v)
        db.flush()
        return rule

    # Delete a rule.
    def delete_rule(self, db: Session, rule: DomainRule) -> None:
        db.delete(rule)
        db.flush()


# Module-level singleton; imported by the policy API router and policy agent.
policy_repository = PolicyRepository()
