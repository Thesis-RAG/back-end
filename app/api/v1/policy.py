"""Per-document entity action configuration API.

Domain/rule authoring was removed.  Policy output is generated at query time
from the entity actions attached to each document version.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from app.core.deps import get_current_user, get_db
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.user import User
from app.repositories.document_entity_repository import document_entity_repository
from app.schemas.entity_policy import (
    EntityActionRead,
    EntityConfigurationRead,
    EntityConfigurationUpdate,
)
from app.services.entity_policy_service import entity_policy_service
from app.services.user_service import user_service

router = APIRouter()


def _require_entity_admin(db: Session, user: User) -> None:
    if not user_service.build_user_response(db, user).is_corp_member:
        raise HTTPException(status_code=403, detail="Corp-level required")


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
