from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.models.document_entity_access_request import DocumentEntityAccessRequest
from app.models.document_entity_action import DocumentEntityAction


class DocumentEntityRepository:
    def list_actions(self, db: Session, version_id: str) -> list[DocumentEntityAction]:
        return (
            db.query(DocumentEntityAction)
            .filter(DocumentEntityAction.document_version_id == version_id)
            .order_by(DocumentEntityAction.sort_order.asc(), DocumentEntityAction.entity_type.asc())
            .all()
        )

    def get_action_map(self, db: Session, version_ids: list[str]) -> dict[str, list[DocumentEntityAction]]:
        if not version_ids:
            return {}
        rows = (
            db.query(DocumentEntityAction)
            .filter(
                DocumentEntityAction.document_version_id.in_(version_ids),
                DocumentEntityAction.enabled.is_(True),
            )
            .all()
        )
        result: dict[str, list[DocumentEntityAction]] = {}
        for row in rows:
            result.setdefault(row.document_version_id, []).append(row)
        return result

    def replace_actions(self, db: Session, version_id: str, actions: list[dict]) -> list[DocumentEntityAction]:
        db.query(DocumentEntityAction).filter(
            DocumentEntityAction.document_version_id == version_id
        ).delete(synchronize_session=False)
        rows = []
        for sort_order, action in enumerate(actions):
            row = DocumentEntityAction(
                document_version_id=version_id,
                sort_order=sort_order,
                **action,
            )
            db.add(row)
            rows.append(row)
        db.flush()
        return rows

    def create_access_request(
        self, db: Session, *, document_id: str, version_id: str,
        user_id: str, entity_types: list[str],
    ) -> DocumentEntityAccessRequest:
        row = DocumentEntityAccessRequest(
            document_id=document_id,
            document_version_id=version_id,
            user_id=user_id,
            entity_types_json=entity_types,
            status="pending",
        )
        db.add(row)
        db.flush()
        return row

    def list_requests(self, db: Session, *, user_id: str | None = None) -> list[DocumentEntityAccessRequest]:
        query = db.query(DocumentEntityAccessRequest)
        if user_id is not None:
            query = query.filter(DocumentEntityAccessRequest.user_id == user_id)
        return query.order_by(DocumentEntityAccessRequest.created_at.desc()).all()

    def get(self, db: Session, request_id: str) -> DocumentEntityAccessRequest | None:
        return db.get(DocumentEntityAccessRequest, request_id)

    def has_pending(self, db: Session, user_id: str, document_id: str, version_id: str, entity_types: list[str]) -> bool:
        wanted = {str(v).lower() for v in entity_types}
        rows = (
            db.query(DocumentEntityAccessRequest)
            .filter(
                DocumentEntityAccessRequest.user_id == user_id,
                DocumentEntityAccessRequest.document_id == document_id,
                DocumentEntityAccessRequest.document_version_id == version_id,
                DocumentEntityAccessRequest.status == "pending",
            ).all()
        )
        return any(wanted.intersection({str(v).lower() for v in (row.entity_types_json or [])}) for row in rows)

    def active_granted_types(self, db: Session, user_id: str, document_id: str, version_id: str) -> set[str]:
        now = datetime.utcnow()
        rows = (
            db.query(DocumentEntityAccessRequest)
            .filter(
                DocumentEntityAccessRequest.user_id == user_id,
                DocumentEntityAccessRequest.document_id == document_id,
                DocumentEntityAccessRequest.document_version_id == version_id,
                DocumentEntityAccessRequest.status == "approved",
            ).all()
        )
        granted: set[str] = set()
        for row in rows:
            if row.expires_at is not None and row.expires_at <= now:
                row.status = "revoked"
                row.resolved_at = now
                continue
            granted.update(str(v).lower() for v in (row.entity_types_json or []))
        db.flush()
        return granted

    def resolve(self, db: Session, request_id: str, *, status: str, admin_id: str,
                admin_note: str | None = None, expires_at=None) -> DocumentEntityAccessRequest:
        row = self.get(db, request_id)
        if not row:
            raise ValueError(f"Entity access request {request_id} not found")
        row.status = status
        row.admin_id = admin_id
        row.admin_note = admin_note
        row.resolved_at = datetime.utcnow()
        row.expires_at = expires_at if status == "approved" else None
        db.flush()
        return row


document_entity_repository = DocumentEntityRepository()
