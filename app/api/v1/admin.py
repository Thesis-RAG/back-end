"""
Admin-only endpoints for system operations: rule reload, FGA sync, reindex,
job management (retry/cancel), and audit trace inspection.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.fga.adapter import fga_adapter
from app.models.document import Document as DocumentModel
from app.models.job import Job as JobModel
from app.models.trace import Trace
from app.models.user import User
from app.services.document_service import document_service
from app.services.user_service import user_service as _user_service
from app.workers.ingest_tasks import process_ingest_job

router = APIRouter()


# Raise 403 if the caller is not a corp-level member.
def require_admin(user: User, db: Session):
    user_resp = _user_service.build_user_response(db, user)
    if not user_resp.is_corp_member:
        raise HTTPException(status_code=403, detail="Corp-level admin required")


# Reload authorization rules (placeholder — returns ok immediately).
@router.post("/rules/reload")
def reload_rules(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    require_admin(current_user, db)
    return {"status": "ok", "message": "rules reloaded"}


# Re-sync FGA tuples for all approved documents. Fixes stale access grants after org changes.
@router.post("/fga/sync")
def sync_fga(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    require_admin(current_user, db)
    membership_synced = fga_adapter.sync_user_memberships(db)
    docs = db.query(DocumentModel).filter(DocumentModel.status == "approved").all()
    synced = 0
    errors = 0
    for doc in docs:
        try:
            old = fga_adapter.get_document_tuples(doc.id)
            fga_adapter.delete_document_tuples(doc.id, old)
            document_service._sync_fga(db, doc)
            synced += 1
        except Exception:
            errors += 1
    return {
        "status": "ok",
        "membership_synced": membership_synced,
        "documents_synced": synced,
        "errors": errors,
    }


# Queue a full document reindex (placeholder — returns ok immediately).
@router.post("/reindex")
def reindex(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    require_admin(current_user, db)
    return {"status": "ok", "message": "reindex queued"}


# Accept a manual metadata override request (placeholder — returns ok immediately).
@router.post("/override-metadata")
def override_metadata(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    require_admin(current_user, db)
    return {"status": "ok", "message": "override accepted"}


# Return the 200 most recent request traces ordered by creation time.
@router.get("/traces")
def list_traces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    traces = db.query(Trace).order_by(Trace.created_at.desc()).limit(200).all()
    result = []
    for t in traces:
        # Returns user_id only; the frontend resolves display names.
        result.append({
            "id": t.id,
            "trace_id": t.trace_id,
            "user_id": t.user_id,
            "user_input": t.user_input,
            "assistant_output_summary": t.assistant_output_summary,
            "retrieved_sources": t.retrieved_sources or [],
            "timings": t.timings,
            "token_usage": t.token_usage,
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        })
    return result


# Return the 200 most recent ingest jobs ordered by creation time.
@router.get("/jobs")
def list_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    jobs = db.query(JobModel).order_by(JobModel.created_at.desc()).limit(200).all()
    result = []
    for j in jobs:
        doc_title = j.document.title if j.document else None
        ver_no = j.version.version_no if j.version else None
        result.append({
            "id": j.id,
            "job_type": j.job_type,
            "status": j.status,
            "progress": j.progress,
            "document_id": j.document_id,
            "document_version_id": j.document_version_id,
            "retry_count": j.retry_count,
            "error_message": j.error_message,
            "started_at": j.started_at,
            "finished_at": j.finished_at,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "doc_title": doc_title,
            "version_no": ver_no,
        })
    return result


# Reset a failed job to queued and re-dispatch it to the Celery worker.
@router.post("/jobs/{job_id}/retry")
def retry_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.status = "queued"
    job.retry_count += 1
    job.error_message = None
    db.commit()
    process_ingest_job.delay(job.id)
    return {"status": "queued"}


# Mark a queued or running job as failed with a cancellation message.
@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("queued", "running"):
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    job.status = "failed"
    job.error_message = "Cancelled by admin"
    db.commit()
    return {"status": "cancelled"}
