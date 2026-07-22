"""
REST API for policy domain, entity type, and rule management.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.models.user import User
from app.repositories.policy_repository import policy_repository
from app.schemas.policy import (
    DomainRuleCreate,
    DomainRuleRead,
    DomainRuleUpdate,
    EntityTypeBulkCreate,
    EntityTypeCreate,
    EntityTypeRead,
    PolicyDomainCreate,
    PolicyDomainRead,
    PolicyDomainSummary,
    PolicyDomainUpdate,
    SuggestEntitiesRequest,
    SuggestEntitiesResponse,
    InstallRuleTemplatesRequest,
)
from app.services.entity_extractor import invalidate_label_cache
from app.services.policy_agent.domain_classifier import invalidate_domain_cache
from app.services.policy_service import policy_service
from app.services.policy_templates import get_policy_templates, list_policy_templates

router = APIRouter()


@router.get("/rule-templates")
def list_rule_templates(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Return built-in rules without writing anything to the database."""
    return list_policy_templates(db)


@router.post("/rule-templates/install")
def install_rule_templates(
    payload: InstallRuleTemplatesRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Install selected built-ins as normal editable rules; duplicates are skipped."""
    codes = payload.template_codes or [item["template_code"] for item in list_policy_templates(db)]
    templates = {
        item["template_code"]: item
        for item in get_policy_templates(codes, db)
    }
    created, skipped = [], []
    for code in codes:
        template = templates.get(code.strip().upper())
        if not template:
            raise HTTPException(status_code=404, detail=f"Unknown rule template: {code}")
        rule_data = template["rule"]
        if policy_repository.get_rule_by_code(db, rule_data.rule_code):
            skipped.append(rule_data.rule_code)
            continue
        created.append(policy_service.create_rule(db, payload.domain_id, rule_data))
    return {"created": [DomainRuleRead.model_validate(rule) for rule in created], "skipped": skipped}


# ── Domains ───────────────────────────────────────────────────────────────────

# Return a summary list of all policy domains, optionally filtered to active only.
@router.get("/domains", response_model=list[PolicyDomainSummary])
def list_domains(
    active_only: bool = Query(False),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    domains = policy_service.list_domains(db, active_only=active_only)
    result = []
    for d in domains:
        result.append(PolicyDomainSummary(
            id=d.id, code=d.code, name=d.name, description=d.description,
            base_sensitivity=d.base_sensitivity, is_active=d.is_active,
            entity_type_count=len(d.entity_types),
            rule_count=len(d.rules),
            created_at=d.created_at,
        ))
    return result


# Retrieve a single policy domain by ID.
@router.get("/domains/{domain_id}", response_model=PolicyDomainRead)
def get_domain(
    domain_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        domain = policy_service.get_domain(db, domain_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return domain


# Create a domain and auto-suggest entity types via LLM, then invalidate domain cache.
@router.post("/domains", response_model=PolicyDomainRead, status_code=201)
def create_domain(
    payload: PolicyDomainCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Creates a domain and auto-suggests entity types via LLM.
    """
    try:
        domain = policy_service.create_domain(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    invalidate_domain_cache()
    return domain


# Update an existing domain and invalidate the domain classifier cache.
@router.put("/domains/{domain_id}", response_model=PolicyDomainRead)
def update_domain(
    domain_id: str,
    payload: PolicyDomainUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        domain = policy_service.update_domain(db, domain_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    invalidate_domain_cache()
    return domain


# Delete a domain and invalidate both the domain and entity label caches.
@router.delete("/domains/{domain_id}", status_code=204)
def delete_domain(
    domain_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        policy_service.delete_domain(db, domain_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    invalidate_domain_cache()
    invalidate_label_cache()


# ── Entity Type Suggestion ────────────────────────────────────────────────────

# Call the LLM to suggest entity types for a given domain name and description.
@router.post("/domains/suggest-entities", response_model=SuggestEntitiesResponse)
def suggest_entities(
    payload: SuggestEntitiesRequest,
    _: User = Depends(get_current_user),
):
    """
    Call LLM to suggest entity types for a given domain name/description.
    Does not save to DB — frontend can use this to preview before saving.
    """
    suggestions = policy_service.suggest_entity_types(payload.name, payload.description)
    return SuggestEntitiesResponse(entity_types=suggestions)


# ── Entity Types ──────────────────────────────────────────────────────────────

# List all entity types registered under a domain.
@router.get("/domains/{domain_id}/entity-types", response_model=list[EntityTypeRead])
def list_entity_types(
    domain_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        policy_service.get_domain(db, domain_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return policy_repository.list_entity_types(db, domain_id)


# Add a single entity type to a domain and invalidate the label cache.
@router.post("/domains/{domain_id}/entity-types", response_model=EntityTypeRead, status_code=201)
def add_entity_type(
    domain_id: str,
    payload: EntityTypeCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        obj = policy_service.add_entity_type(db, domain_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    invalidate_label_cache()
    return obj


# Add multiple entity types at once, skipping duplicates, then invalidate the label cache.
@router.post("/domains/{domain_id}/entity-types/bulk", response_model=list[EntityTypeRead], status_code=201)
def add_entity_types_bulk(
    domain_id: str,
    payload: EntityTypeBulkCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Add multiple entity types at once (skip duplicates)."""
    try:
        policy_service.get_domain(db, domain_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    created = []
    for et in payload.entity_types:
        try:
            obj = policy_service.add_entity_type(db, domain_id, et)
            created.append(obj)
        except ValueError:
            pass  # skip duplicates
    invalidate_label_cache()
    return created


# Delete an entity type from a domain and invalidate the label cache.
@router.delete("/domains/{domain_id}/entity-types/{entity_type_id}", status_code=204)
def delete_entity_type(
    domain_id: str,
    entity_type_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        policy_service.delete_entity_type(db, domain_id, entity_type_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    invalidate_label_cache()


# ── Rules (per domain) ────────────────────────────────────────────────────────

# List all rules associated with a specific domain.
@router.get("/domains/{domain_id}/rules", response_model=list[DomainRuleRead])
def list_domain_rules(
    domain_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return policy_service.list_rules(db, domain_id)


# Create a new rule for a specific domain.
@router.post("/domains/{domain_id}/rules", response_model=DomainRuleRead, status_code=201)
def create_domain_rule(
    domain_id: str,
    payload: DomainRuleCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return policy_service.create_rule(db, domain_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ── Global Rules (domain_id = None) ──────────────────────────────────────────

# List all global rules that apply regardless of domain.
@router.get("/global-rules", response_model=list[DomainRuleRead])
def list_global_rules(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return policy_service.list_rules(db, domain_id=None)


# Create a new global rule (not bound to any domain).
@router.post("/global-rules", response_model=DomainRuleRead, status_code=201)
def create_global_rule(
    payload: DomainRuleCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return policy_service.create_rule(db, domain_id=None, data=payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ── Rules (update / delete by rule id) ───────────────────────────────────────

# Update an existing rule by its ID.
@router.put("/rules/{rule_id}", response_model=DomainRuleRead)
def update_rule(
    rule_id: str,
    payload: DomainRuleUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        return policy_service.update_rule(db, rule_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# Delete a rule by its ID.
@router.delete("/rules/{rule_id}", status_code=204)
def delete_rule(
    rule_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    try:
        policy_service.delete_rule(db, rule_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
