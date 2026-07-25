"""Centralized global Rules API plus compatibility endpoints for version snapshots."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session, joinedload

from app.core.deps import get_current_user, get_db
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.entity_policy_rule import EntityPolicyRule
from app.models.user import User
from app.repositories.document_entity_repository import document_entity_repository
from app.schemas.entity_policy import (
    ApplyGroupScopeRequest,
    EntityActionRead,
    EntityConfigurationRead,
    EntityConfigurationUpdate,
    PolicyGroupCreate,
    PolicyGroupRead,
    PolicyRuleInput,
    PolicyRuleRead,
    TranslateEntityKeyRequest,
    TranslateEntityKeyResponse,
)
from app.services.audit_service import audit_service
from app.services.document_service import document_service
from app.services.entity_policy_service import entity_policy_service
from app.services.entity_extractor import invalidate_label_cache
from app.services.ontology_sync_service import ontology_sync_service
from app.services.policy_rule_service import (
    DEFAULT_POLICY_PROFILE,
    compute_graph_tiers,
    normalize_entity_key,
    policy_rule_service,
    validate_tiered_scopes,
)
from app.services.translate_service import translate_vi_to_en
from app.services.user_service import user_service

router = APIRouter()


def _require_entity_admin(db: Session, user: User) -> None:
    if not user_service.build_user_response(db, user).is_corp_member:
        raise HTTPException(status_code=403, detail="Corp-level required")


# Attach the live allow/mask/block fan-out (same computation the ontology
# graph is built from) as transient attributes so PolicyRuleRead's
# from_attributes pickup finds them — not persisted columns, recomputed on
# every read since it depends on current org structure, not just the rule.
def _attach_resolved_tiers(db: Session, rule: EntityPolicyRule) -> EntityPolicyRule:
    tiers = compute_graph_tiers(db, rule)
    rule.resolved_allow = bool(tiers["allow"]["needed"])
    rule.resolved_mask = bool(tiers["mask"]["needed"])
    rule.resolved_block = bool(tiers["block"]["needed"])
    return rule


@router.post("/translate-entity-key", response_model=TranslateEntityKeyResponse)
def translate_entity_key(
    payload: TranslateEntityKeyRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    translated = translate_vi_to_en(payload.text)
    return TranslateEntityKeyResponse(entity_key=normalize_entity_key(translated))


@router.get("/rules", response_model=list[PolicyRuleRead])
def list_policy_rules(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    rules = policy_rule_service.list_rules(db, DEFAULT_POLICY_PROFILE)
    return [_attach_resolved_tiers(db, rule) for rule in rules]


@router.post("/rules", response_model=PolicyRuleRead, status_code=201)
def create_policy_rule(
    payload: PolicyRuleInput,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    entity_key = normalize_entity_key(payload.entity_key)
    if not entity_key:
        raise HTTPException(status_code=422, detail="entity_key is required")
    if db.query(EntityPolicyRule).filter(
        EntityPolicyRule.policy_profile == DEFAULT_POLICY_PROFILE,
        EntityPolicyRule.entity_key == entity_key,
    ).first():
        raise HTTPException(status_code=409, detail="Entity rule already exists")
    try:
        validate_tiered_scopes(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rule = policy_rule_service.create(db, {**payload.model_dump(), "entity_key": entity_key})
    policy_version = policy_rule_service.bump_version(db)
    invalidate_label_cache()
    audit_service.log_action(
        db, trace_id=getattr(request.state, "trace_id", uuid.uuid4().hex),
        user_id=current_user.id, action="policy.rule.created",
        resource_type="entity_policy_rule", resource_id=rule.id, decision="allow",
        input_json=payload.model_dump(mode="json"),
        output_json={"entity_key": rule.entity_key, "action": rule.action, "policy_version": policy_version},
    )
    db.commit()
    db.refresh(rule)
    ontology_sync_service.sync_entity_rule(db, rule)
    return _attach_resolved_tiers(db, rule)


@router.put("/rules/{rule_id}", response_model=PolicyRuleRead)
def update_policy_rule(
    rule_id: str,
    payload: PolicyRuleInput,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    rule = db.query(EntityPolicyRule).filter(
        EntityPolicyRule.id == rule_id,
        EntityPolicyRule.policy_profile == DEFAULT_POLICY_PROFILE,
    ).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Policy rule not found")
    entity_key = normalize_entity_key(payload.entity_key)
    if not entity_key:
        raise HTTPException(status_code=422, detail="entity_key is required")
    duplicate = db.query(EntityPolicyRule).filter(
        EntityPolicyRule.policy_profile == DEFAULT_POLICY_PROFILE,
        EntityPolicyRule.entity_key == entity_key,
        EntityPolicyRule.id != rule_id,
    ).first()
    if duplicate:
        raise HTTPException(status_code=409, detail="Entity rule already exists")
    data = payload.model_dump()
    try:
        validate_tiered_scopes(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rule.entity_key = entity_key
    rule.display_name = data["display_name"]
    rule.group_name = data["group_name"]
    rule.detection_source = data["detection_source"]
    rule.action = data["action"]
    rule.sensitivity = data["sensitivity"]
    rule.enabled = data["enabled"]
    rule.scope_oui_ids = data["scope_oui_ids"]
    rule.scope_position_ids = data["scope_position_ids"]
    rule.priority = data["priority"]
    rule.metadata_json = data["metadata_json"]
    rule.allow_scope_pairs = data["allow_scope_pairs"]
    rule.mask_scope_pairs = data["mask_scope_pairs"]
    rule.mask_style = data["mask_style"]
    rule.mask_position = data["mask_position"]
    policy_version = policy_rule_service.bump_version(db)
    invalidate_label_cache()
    audit_service.log_action(
        db, trace_id=getattr(request.state, "trace_id", uuid.uuid4().hex),
        user_id=current_user.id, action="policy.rule.updated",
        resource_type="entity_policy_rule", resource_id=rule.id, decision="allow",
        input_json=payload.model_dump(mode="json"),
        output_json={"entity_key": rule.entity_key, "action": rule.action, "policy_version": policy_version},
    )
    db.commit()
    db.refresh(rule)
    ontology_sync_service.sync_entity_rule(db, rule)
    return _attach_resolved_tiers(db, rule)


@router.delete("/rules/{rule_id}")
def delete_policy_rule(
    rule_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    rule = db.query(EntityPolicyRule).filter(
        EntityPolicyRule.id == rule_id,
        EntityPolicyRule.policy_profile == DEFAULT_POLICY_PROFILE,
    ).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Policy rule not found")
    entity_key = rule.entity_key
    scope = {
        "scope_oui_ids": list(rule.scope_oui_ids or []),
        "scope_position_ids": list(rule.scope_position_ids or []),
    }
    db.delete(rule)
    policy_version = policy_rule_service.bump_version(db)
    invalidate_label_cache()
    audit_service.log_action(
        db, trace_id=getattr(request.state, "trace_id", uuid.uuid4().hex),
        user_id=current_user.id, action="policy.rule.deleted",
        resource_type="entity_policy_rule", resource_id=rule_id, decision="allow",
        output_json={"entity_key": entity_key, **scope, "policy_version": policy_version},
    )
    db.commit()
    ontology_sync_service.delete_entity_rule(rule_id)
    return {"ok": True, "policy_version": policy_version}


# ── Groups ─────────────────────────────────────────────────────────────────
# group_name on EntityPolicyRule is a plain string, not a foreign key — see
# EntityPolicyGroup's docstring. These endpoints back the Policy page's
# group picker (list/create/delete) and the "apply scope to whole group"
# bulk action.

@router.get("/groups", response_model=list[PolicyGroupRead])
def list_policy_groups(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    return policy_rule_service.list_groups(db, DEFAULT_POLICY_PROFILE)


@router.post("/groups", response_model=PolicyGroupRead, status_code=201)
def create_policy_group(
    payload: PolicyGroupCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    try:
        group = policy_rule_service.create_group(db, payload.name, DEFAULT_POLICY_PROFILE)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    audit_service.log_action(
        db, trace_id=getattr(request.state, "trace_id", uuid.uuid4().hex),
        user_id=current_user.id, action="policy.group.created",
        resource_type="entity_policy_group", resource_id=group.id, decision="allow",
        output_json={"name": group.name},
    )
    db.commit()
    db.refresh(group)
    return group


@router.delete("/groups/{group_id}")
def delete_policy_group(
    group_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    try:
        policy_rule_service.delete_group(db, group_id, DEFAULT_POLICY_PROFILE)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    audit_service.log_action(
        db, trace_id=getattr(request.state, "trace_id", uuid.uuid4().hex),
        user_id=current_user.id, action="policy.group.deleted",
        resource_type="entity_policy_group", resource_id=group_id, decision="allow",
    )
    db.commit()
    return {"ok": True}


@router.post("/groups/{group_id}/apply-scope")
def apply_policy_group_scope(
    group_id: str,
    payload: ApplyGroupScopeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    from app.models.entity_policy_group import EntityPolicyGroup

    group = db.get(EntityPolicyGroup, group_id)
    if not group or group.policy_profile != DEFAULT_POLICY_PROFILE:
        raise HTTPException(status_code=404, detail="Không tìm thấy nhóm")

    try:
        rules = policy_rule_service.apply_group_scope(
            db, group.name,
            [pair.model_dump() for pair in payload.allow_scope_pairs],
            [pair.model_dump() for pair in payload.mask_scope_pairs],
            DEFAULT_POLICY_PROFILE,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    invalidate_label_cache()
    audit_service.log_action(
        db, trace_id=getattr(request.state, "trace_id", uuid.uuid4().hex),
        user_id=current_user.id, action="policy.group.scope_applied",
        resource_type="entity_policy_group", resource_id=group_id, decision="allow",
        input_json=payload.model_dump(mode="json"),
        output_json={"group_name": group.name, "affected_rule_count": len(rules)},
    )
    db.commit()
    for rule in rules:
        db.refresh(rule)
        ontology_sync_service.sync_entity_rule(db, rule)
    return {"ok": True, "affected_rule_count": len(rules)}


@router.post("/reprocess/{version_id}", response_model=dict)
def reprocess_document_version(
    version_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    version = db.get(DocumentVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Document version not found")
    job = document_service.reprocess_policy(
        db, current_user, version.document_id, version.id,
        getattr(request.state, "trace_id", uuid.uuid4().hex),
    )
    return {"job_id": job.id, "version_id": version.id, "status": job.status}


def _configuration(version: DocumentVersion) -> EntityConfigurationRead:
    doc = version.document
    return EntityConfigurationRead(
        document_id=doc.id,
        document_title=doc.title,
        document_version_id=version.id,
        version_no=version.version_no,
        file_name=version.file_name,
        entity_detection_json=version.entity_detection_json or {},
        actions=[EntityActionRead.model_validate(row) for row in version.entity_actions],
    )


@router.get("/entity-configurations", response_model=list[EntityConfigurationRead])
def list_entity_configurations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    versions = (
        db.query(DocumentVersion)
        .options(joinedload(DocumentVersion.document), joinedload(DocumentVersion.entity_actions))
        .join(Document, Document.id == DocumentVersion.document_id)
        .order_by(Document.updated_at.desc(), DocumentVersion.version_no.desc())
        .all()
    )
    return [_configuration(version) for version in versions]


@router.get("/entity-configurations/{version_id}", response_model=EntityConfigurationRead)
def get_entity_configuration(
    version_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    version = (
        db.query(DocumentVersion)
        .options(joinedload(DocumentVersion.document), joinedload(DocumentVersion.entity_actions))
        .filter(DocumentVersion.id == version_id)
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Document version not found")
    return _configuration(version)


@router.put("/entity-configurations/{version_id}", response_model=EntityConfigurationRead)
def update_entity_configuration(
    version_id: str,
    payload: EntityConfigurationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_entity_admin(db, current_user)
    version = (
        db.query(DocumentVersion)
        .options(joinedload(DocumentVersion.document))
        .filter(DocumentVersion.id == version_id)
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Document version not found")

    entity_policy_service.replace_actions(
        db, version_id, [item.model_dump() for item in payload.actions]
    )
    db.commit()
    version = (
        db.query(DocumentVersion)
        .options(joinedload(DocumentVersion.document), joinedload(DocumentVersion.entity_actions))
        .filter(DocumentVersion.id == version_id)
        .first()
    )
    return _configuration(version)
