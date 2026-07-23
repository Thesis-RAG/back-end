"""
Endpoints for document access requests on documents containing sensitive chunks.

User flow:
  POST /access-requests               — submit a new access request
  GET  /access-requests/my            — list the current user's requests

Admin flow:
  GET  /access-requests               — list all requests
  PUT  /access-requests/{id}/approve  — approve a request (optional expires_at)
  PUT  /access-requests/{id}/reject   — reject a request

Document access-status:
  GET  /documents/{doc_id}/access-status  — check has_restricted_chunks and request status
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.models.document import Document
from app.models.document_access_request import DocumentAccessRequest
from app.models.document_entity_access_request import DocumentEntityAccessRequest
from app.models.document_version import DocumentVersion
from app.models.user import User
from app.services.chroma_service import chroma_service
from app.repositories.document_access_request_repository import doc_access_request_repo
from app.schemas.document_access_request import (
    AccessRequestApprove,
    AccessRequestCreate,
    AccessRequestRead,
    AccessRequestReject,
    DocumentAccessStatus,
)
from app.services.user_service import user_service as _user_service
from app.repositories.document_entity_repository import document_entity_repository
from app.schemas.document_entity_access_request import (
    EntityAccessRequestCreate,
    EntityAccessRequestRead,
)
from app.services.entity_policy_service import entity_policy_service, normalize_entity_type

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

# Return True if the user holds a corp-level membership.
def _is_corp_member(db: Session, user: User) -> bool:
    return _user_service.build_user_response(db, user).is_corp_member


# Return the highest clearance level held by the user across all their positions.
def _get_user_clearance(user: User) -> int:
    max_c = 1
    for uop in getattr(user, "oui_positions", []):
        if uop.position and uop.position.clearance > max_c:
            max_c = uop.position.clearance
    return max_c


# Build a full AccessRequestRead response object from a DocumentAccessRequest ORM row.
def _build_read(db: Session, obj: DocumentAccessRequest) -> AccessRequestRead:
    doc       = db.get(Document, obj.document_id)
    requester = db.get(User, obj.user_id)
    name  = None
    email = None
    if requester:
        name  = getattr(requester, "name", None) or getattr(requester, "email", None)
        email = getattr(requester, "email", None)
    return AccessRequestRead(
        id                   = obj.id,
        document_id          = obj.document_id,
        document_title       = doc.title if doc else None,
        document_sensitivity = doc.sensitivity if doc else None,
        user_id              = obj.user_id,
        requester_name       = name,
        requester_email      = email,
        status               = obj.status,
        expires_at           = obj.expires_at,
        admin_id             = obj.admin_id,
        admin_note           = obj.admin_note,
        created_at           = obj.created_at,
        resolved_at          = obj.resolved_at,
    )


def _build_entity_read(db: Session, obj: DocumentEntityAccessRequest) -> EntityAccessRequestRead:
    doc = db.get(Document, obj.document_id)
    requester = db.get(User, obj.user_id)
    return EntityAccessRequestRead(
        id=obj.id,
        document_id=obj.document_id,
        document_version_id=obj.document_version_id,
        document_title=doc.title if doc else None,
        user_id=obj.user_id,
        requester_name=(getattr(requester, "name", None) or getattr(requester, "email", None)) if requester else None,
        requester_email=getattr(requester, "email", None) if requester else None,
        entity_types=list(obj.entity_types_json or []),
        status=obj.status,
        expires_at=obj.expires_at,
        admin_id=obj.admin_id,
        admin_note=obj.admin_note,
        created_at=obj.created_at,
        resolved_at=obj.resolved_at,
    )


# ── User endpoints ────────────────────────────────────────────────────────────

# Submit a new access request for a document. Rejects if a pending request already exists.
@router.post("/access-requests", response_model=AccessRequestRead, status_code=201)
def create_access_request(
    payload: AccessRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if doc_access_request_repo.has_pending(db, str(current_user.id), payload.document_id):
        raise HTTPException(status_code=409, detail="Yêu cầu đang chờ duyệt")
    obj = doc_access_request_repo.create(
        db, document_id=payload.document_id, user_id=str(current_user.id)
    )
    db.commit()
    return _build_read(db, obj)


# Return all access requests submitted by the current user.
@router.get("/access-requests/my", response_model=list[AccessRequestRead])
def my_access_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return [_build_read(db, r) for r in doc_access_request_repo.list_for_user(db, str(current_user.id))]


# Revoke an approved access request. The owner or an admin may perform this action.
@router.post("/access-requests/{request_id}/revoke", response_model=AccessRequestRead)
def revoke_access_request(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    obj = db.get(DocumentAccessRequest, request_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Không tìm thấy yêu cầu")
    is_admin = _is_corp_member(db, current_user)
    if obj.user_id != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    if obj.status != "approved":
        raise HTTPException(status_code=400, detail="Chỉ có thể hủy quyền đang được duyệt")
    now = datetime.utcnow()
    if obj.expires_at is not None and obj.expires_at <= now:
        raise HTTPException(status_code=400, detail="Quyền truy cập đã hết hạn, không cần thu hồi thủ công")
    obj.status = "revoked"
    obj.resolved_at = now
    db.commit()
    return _build_read(db, obj)


# ── Admin endpoints ───────────────────────────────────────────────────────────

# Return all access requests across all users. Requires corp-level access.
@router.get("/access-requests", response_model=list[AccessRequestRead])
def list_access_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")
    return [_build_read(db, r) for r in doc_access_request_repo.list_all(db)]


# Approve an access request, optionally setting an expiry date.
@router.put("/access-requests/{request_id}/approve", response_model=AccessRequestRead)
def approve_access_request(
    request_id: str,
    payload: AccessRequestApprove,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        obj = doc_access_request_repo.resolve(
            db, request_id,
            status     = "approved",
            admin_id   = str(current_user.id),
            admin_note = payload.admin_note,
            expires_at = payload.expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return _build_read(db, obj)


# Reject an access request with an optional admin note.
@router.put("/access-requests/{request_id}/reject", response_model=AccessRequestRead)
def reject_access_request(
    request_id: str,
    payload: AccessRequestReject,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        obj = doc_access_request_repo.resolve(
            db, request_id,
            status     = "rejected",
            admin_id   = str(current_user.id),
            admin_note = payload.admin_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    db.commit()
    return _build_read(db, obj)


# ── Document access-status ────────────────────────────────────────────────────

# Return whether a document has chunks above the user's clearance and the latest request status.
@router.get("/documents/{document_id}/access-status", response_model=DocumentAccessStatus)
def get_document_access_status(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns:
      has_restricted_chunks  — True if any chunk sensitivity exceeds the user's clearance
      access_request_status  — status of the latest request (None if none submitted)
      approved_until         — expiry of the approved grant (None if permanent or not approved)
    """
    user_clearance = _get_user_clearance(current_user)

    # Read max chunk_sensitivity from Chroma (consistent with retrieval_service blur logic).
    max_chunk_sens = chroma_service.get_max_chunk_sensitivity(document_id)
    has_restricted = max_chunk_sens > user_clearance

    latest = doc_access_request_repo.get_latest_for_user_doc(db, str(current_user.id), document_id)
    status: str | None = None
    approved_until: datetime | None = None
    if latest:
        status = latest.status
        if latest.status == "approved":
            approved_until = latest.expires_at

    return DocumentAccessStatus(
        has_restricted_chunks = has_restricted,
        access_request_status = status,
        approved_until        = approved_until,
    )


# ── Entity-level access requests ────────────────────────────────────────────

@router.post("/entity-access-requests", response_model=EntityAccessRequestRead, status_code=201)
def create_entity_access_request(
    payload: EntityAccessRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    version = db.get(DocumentVersion, payload.document_version_id)
    if not version or version.document_id != payload.document_id:
        raise HTTPException(status_code=404, detail="Document version not found")

    blocked = entity_policy_service.blocked_types_for_version(
        db, version.id, current_user
    )
    requested = {normalize_entity_type(value) for value in payload.entity_types}
    requested = requested.intersection(blocked)
    if not requested:
        raise HTTPException(status_code=400, detail="No blocked entity requested")
    if entity_policy_service.has_entity_access(
        db, current_user, payload.document_id, payload.document_version_id, requested
    ):
        raise HTTPException(status_code=409, detail="Entity access already granted")
    if document_entity_repository.has_pending(
        db, str(current_user.id), payload.document_id, payload.document_version_id, list(requested)
    ):
        raise HTTPException(status_code=409, detail="Entity access request already pending")

    obj = document_entity_repository.create_access_request(
        db,
        document_id=payload.document_id,
        version_id=payload.document_version_id,
        user_id=str(current_user.id),
        entity_types=sorted(requested),
    )
    db.commit()
    return _build_entity_read(db, obj)


@router.get("/entity-access-requests/my", response_model=list[EntityAccessRequestRead])
def my_entity_access_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return [
        _build_entity_read(db, row)
        for row in document_entity_repository.list_requests(db, user_id=str(current_user.id))
    ]


@router.get("/entity-access-requests", response_model=list[EntityAccessRequestRead])
def list_entity_access_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")
    return [_build_entity_read(db, row) for row in document_entity_repository.list_requests(db)]


@router.put("/entity-access-requests/{request_id}/approve", response_model=EntityAccessRequestRead)
def approve_entity_access_request(
    request_id: str,
    payload: AccessRequestApprove,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        obj = document_entity_repository.resolve(
            db, request_id, status="approved", admin_id=str(current_user.id),
            admin_note=payload.admin_note, expires_at=payload.expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    db.commit()
    return _build_entity_read(db, obj)


@router.put("/entity-access-requests/{request_id}/reject", response_model=EntityAccessRequestRead)
def reject_entity_access_request(
    request_id: str,
    payload: AccessRequestReject,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        obj = document_entity_repository.resolve(
            db, request_id, status="rejected", admin_id=str(current_user.id),
            admin_note=payload.admin_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    db.commit()
    return _build_entity_read(db, obj)
