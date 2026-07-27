"""
Document management endpoints: create, list, update, delete, version upload,
file streaming, review workflow (submit/approve/reject), and ingest triggering.
"""
import io
import urllib.parse

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.models.document import Document as DocumentModel
from app.models.document_version import DocumentVersion
from app.models.job import Job as JobModel
from app.models.user import User
from app.repositories.document_access_request_repository import doc_access_request_repo
from app.repositories.storage_repository import StorageRepository
from app.schemas.document import (
    ChunkingConfig,
    DocumentChunkRead,
    DocumentChunkUpdate,
    DocumentCreateRequest,
    DocumentRead,
    DocumentUpdateRequest,
    DocumentVersionRead,
    UploadVersionResponse,
)
from app.schemas.job import JobRead
from app.services.document_service import document_service
from app.services.entity_policy_service import entity_policy_service
from app.schemas.entity_policy import EntityPreviewResponse
from app.services.user_service import user_service as _user_service
from app.workers.ingest_tasks import process_ingest_job

router = APIRouter()


# Return True if the user holds a corp-level (root OUI) membership.
def _is_corp_member(db: Session, user: User) -> bool:
    return _user_service.build_user_response(db, user).is_corp_member


# Find the latest pending-approval job for a document and queue it for processing.
def _trigger_pending_job(db: Session, document_id: str) -> None:
    job = db.query(JobModel).filter(
        JobModel.document_id == document_id,
        JobModel.status == "pending_approval",
    ).order_by(JobModel.created_at.desc()).first()
    if job:
        job.status = "queued"
        db.commit()
        process_ingest_job.delay(job.id)


@router.post("/entity-preview", response_model=EntityPreviewResponse)
async def preview_file_entities(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Detect configurable entities in the full file before it is persisted."""
    raw_bytes = await file.read()
    if len(raw_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        result = entity_policy_service.preview(
            raw_bytes,
            file.filename or "upload.bin",
            file.content_type or "application/octet-stream",
            db,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot detect entities: {exc}") from exc


# Create a new document record for the current user.
@router.post("", response_model=DocumentRead)
def create_document(
    payload: DocumentCreateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = document_service.create_document(db, current_user, payload, request.state.trace_id)
    return DocumentRead.model_validate(doc)


# List all documents visible to the current user.
@router.get("", response_model=list[DocumentRead])
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    docs = document_service.list_documents(db, current_user)
    return [DocumentRead.model_validate(d) for d in docs]


# Return the count of documents currently awaiting corp-level review.
@router.get("/pending-review-count")
def pending_review_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        return {"count": 0}
    count = db.query(DocumentModel).filter(DocumentModel.status.in_(["review", "uploaded"])).count()
    return {"count": count}


@router.get("/pending-approvals", response_model=list[DocumentRead])
def pending_approvals(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return every document waiting for corp-level approval.

    This intentionally does not use the normal FGA/viewable-document query:
    reviewers need to see pending documents owned by other employees before
    deciding whether to approve them.
    """
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Corp-level required")
    docs = (
        db.query(DocumentModel)
        .filter(DocumentModel.status.in_(["review", "uploaded"]))
        .order_by(DocumentModel.updated_at.desc())
        .all()
    )
    return [DocumentRead.model_validate(doc) for doc in docs]


# Retrieve a single document by ID, enforcing access control.
@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = document_service.get_document(db, current_user, document_id)
    return DocumentRead.model_validate(doc)


# Update document metadata (title, sensitivity, OUI assignment, etc.).
@router.patch("/{document_id}", response_model=DocumentRead)
def update_document(
    document_id: str,
    payload: DocumentUpdateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = document_service.update_document(db, current_user, document_id, payload, request.state.trace_id)
    return DocumentRead.model_validate(doc)


# List all versions for a document.
@router.get("/{document_id}/versions", response_model=list[DocumentVersionRead])
def get_versions(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return [DocumentVersionRead.model_validate(v)
            for v in document_service.get_versions(db, current_user, document_id)]


# Upload a new file version (max 50 MB) and create a pending ingest job.
@router.post("/{document_id}/versions", response_model=UploadVersionResponse)
async def upload_version(
    document_id: str,
    file: UploadFile = File(...),
    request: Request = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    chunk_max_tokens: int = Form(default=512),
    chunk_overlap_tokens: int = Form(default=80),
    chunk_ocr: bool = Form(default=False),
    # Kept out of the public workflow: actions come from the active global
    # policy and are snapshotted automatically for this version.
):
    raw_bytes = await file.read()
    if len(raw_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    chunking_config = ChunkingConfig(
        mode="llm_structured", max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens, ocr=chunk_ocr,
    )
    doc, version, job, queued = document_service.create_version(
        db, current_user, document_id,
        raw_bytes=raw_bytes, filename=file.filename or "upload.bin",
        content_type=file.content_type or "application/octet-stream",
        trace_id=request.state.trace_id, chunking_config=chunking_config,
    )

    if _is_corp_member(db, current_user):
        # Corp members bypass the review step; trigger ingest immediately.
        doc.status = "approved"
        db.commit()
        _trigger_pending_job(db, document_id)

    return UploadVersionResponse(
        document=DocumentRead.model_validate(doc),
        version=DocumentVersionRead.model_validate(version),
        job=JobRead.model_validate(job),
        queued=queued,
    )


# Stream the source file for a document version, enforcing chunk-level clearance.
@router.get("/{document_id}/versions/{version_id}/file")
def view_document_file(
    document_id: str,
    version_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document_service.get_document(db, current_user, document_id)

    # Chunk-level sensitivity gate: block if any chunk > user clearance without approval
    user_clearance = max(
        (uop.position.clearance for uop in getattr(current_user, "oui_positions", []) if uop.position),
        default=1,
    )
    if not _is_corp_member(db, current_user):
        # Read from Chroma — same source as /access-status (the lock icon) so both
        # always agree. Legacy-ingested chunks lack chunk_sensitivity in MySQL.
        from app.services.chroma_service import chroma_service
        max_chunk_sens = chroma_service.get_max_chunk_sensitivity(document_id)
        if max_chunk_sens > user_clearance:
            approved_ids = doc_access_request_repo.get_active_approved_doc_ids(db, str(current_user.id))
            if document_id not in approved_ids:
                raise HTTPException(
                    status_code=403,
                    detail="Tài liệu này chứa nội dung vượt mức độ phân quyền của bạn. Vui lòng gửi yêu cầu xem tài liệu.",
                )

    version = db.query(DocumentVersion).filter(
        DocumentVersion.id == version_id,
        DocumentVersion.document_id == document_id,
    ).first()
    if not version or not version.source_object:
        raise HTTPException(status_code=404, detail="File not found")

    src_obj = version.source_object
    data = StorageRepository().get_bytes(src_obj.bucket, src_obj.object_key)
    encoded_name = urllib.parse.quote(src_obj.original_filename)

    return StreamingResponse(
        io.BytesIO(data),
        media_type=src_obj.content_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{encoded_name}",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# List chunks (heading + text) for a version — backs the chunk-metadata editor.
@router.get("/{document_id}/versions/{version_id}/chunks", response_model=list[DocumentChunkRead])
def list_document_chunks(
    document_id: str,
    version_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chunks = document_service.list_chunks(db, current_user, document_id, version_id)
    return [DocumentChunkRead.model_validate(c) for c in chunks]


# Edit a chunk's section_heading + chunk_text. Re-embeds into Chroma and
# re-signs the version's content signature so the edit is fully applied
# everywhere retrieval reads from (see document_service.update_chunk).
@router.put(
    "/{document_id}/versions/{version_id}/chunks/{chunk_id}",
    response_model=DocumentChunkRead,
)
def update_document_chunk(
    document_id: str,
    version_id: str,
    chunk_id: str,
    payload: DocumentChunkUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chunk = document_service.update_chunk(
        db, current_user, document_id, version_id, chunk_id,
        section_heading=payload.section_heading, chunk_text=payload.chunk_text,
        chunk_sensitivity=payload.chunk_sensitivity,
    )
    return DocumentChunkRead.model_validate(chunk)


# Delete a chunk — removes it from Chroma + MySQL, renumbers the remaining
# chunks' chunk_index/position_ratio, and returns the updated chunk list
# (see document_service.delete_chunk).
@router.delete(
    "/{document_id}/versions/{version_id}/chunks/{chunk_id}",
    response_model=list[DocumentChunkRead],
)
def delete_document_chunk(
    document_id: str,
    version_id: str,
    chunk_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    remaining = document_service.delete_chunk(
        db, current_user, document_id, version_id, chunk_id,
    )
    return [DocumentChunkRead.model_validate(c) for c in remaining]


# Submit a document for corp-level review, or auto-approve if caller is a corp member.
@router.post("/{document_id}/submit-review")
def submit_for_review(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only owner can submit for review")
    if doc.status != "uploaded":
        raise HTTPException(status_code=400, detail=f"Cannot submit with status '{doc.status}'")

    if _is_corp_member(db, current_user):
        doc.status = "approved"
        db.commit()
        _trigger_pending_job(db, doc.id)
    else:
        doc.status = "review"
        db.commit()

    db.refresh(doc)
    return DocumentRead.model_validate(doc)


# Approve a document and trigger its ingest pipeline. Requires corp-level access.
@router.post("/{document_id}/approve")
def approve_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Corp-level required")
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status not in {"review", "uploaded"}:
        raise HTTPException(status_code=400, detail=f"Cannot approve with status '{doc.status}'")
    doc.status = "approved"
    db.commit()
    _trigger_pending_job(db, doc.id)
    db.refresh(doc)
    return DocumentRead.model_validate(doc)


# Reject a document and cancel its pending ingest job. Requires corp-level access.
@router.post("/{document_id}/reject")
def reject_document(
    document_id: str,
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not _is_corp_member(db, current_user):
        raise HTTPException(status_code=403, detail="Corp-level required")
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status not in {"review", "uploaded"}:
        raise HTTPException(status_code=400, detail=f"Cannot reject with status '{doc.status}'")
    doc.status = "draft"
    job = db.query(JobModel).filter(
        JobModel.document_id == doc.id,
        JobModel.status == "pending_approval",
    ).order_by(JobModel.created_at.desc()).first()
    if job:
        job.status = "cancelled"
        job.error_message = "Rejected by reviewer"
    db.commit()
    db.refresh(doc)
    return DocumentRead.model_validate(doc)


# Permanently delete a document and all associated data.
@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document_service.delete_document(db, current_user, document_id, request.state.trace_id)
    return {"ok": True}


# Manually start or re-trigger the ingest pipeline for a specific document version.
@router.post("/{document_id}/ingest", response_model=JobRead)
def start_ingest(
    document_id: str,
    version_id: str | None = None,
    force_new: bool = False,
    request: Request = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = document_service.start_ingest(
        db, current_user, document_id,
        version_id=version_id, force_new=force_new,
        trace_id=request.state.trace_id,
    )
    if job.status == "queued":
        process_ingest_job.delay(job.id)
    return JobRead.model_validate(job)
