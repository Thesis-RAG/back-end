from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.models.message_entity_access_request import MessageEntityAccessRequest


class MessageEntityRepository:
    def create_access_request(
        self, db: Session, *, message_id: str, user_id: str, entity_types: list[str],
    ) -> MessageEntityAccessRequest:
        row = MessageEntityAccessRequest(
            message_id=message_id,
            user_id=user_id,
            entity_types_json=entity_types,
            status="pending",
        )
        db.add(row)
        db.flush()
        return row

    def list_requests(self, db: Session, *, user_id: str | None = None) -> list[MessageEntityAccessRequest]:
        query = db.query(MessageEntityAccessRequest)
        if user_id is not None:
            query = query.filter(MessageEntityAccessRequest.user_id == user_id)
        return query.order_by(MessageEntityAccessRequest.created_at.desc()).all()

    def get(self, db: Session, request_id: str) -> MessageEntityAccessRequest | None:
        return db.get(MessageEntityAccessRequest, request_id)

    def has_pending(self, db: Session, user_id: str, message_id: str, entity_types: list[str]) -> bool:
        wanted = {str(v).lower() for v in entity_types}
        rows = (
            db.query(MessageEntityAccessRequest)
            .filter(
                MessageEntityAccessRequest.user_id == user_id,
                MessageEntityAccessRequest.message_id == message_id,
                MessageEntityAccessRequest.status == "pending",
            ).all()
        )
        return any(wanted.intersection({str(v).lower() for v in (row.entity_types_json or [])}) for row in rows)

    # Approved entity types for this exact message + user. No expiry, no
    # cross-message reuse — a repeat question is a new message_id and needs
    # a fresh request, by design (per-query justification).
    def granted_types(self, db: Session, user_id: str, message_id: str) -> set[str]:
        rows = (
            db.query(MessageEntityAccessRequest)
            .filter(
                MessageEntityAccessRequest.user_id == user_id,
                MessageEntityAccessRequest.message_id == message_id,
                MessageEntityAccessRequest.status == "approved",
            ).all()
        )
        granted: set[str] = set()
        for row in rows:
            granted.update(str(v).lower() for v in (row.entity_types_json or []))
        return granted

    def resolve(
        self, db: Session, request_id: str, *, status: str, admin_id: str, admin_note: str | None = None,
    ) -> MessageEntityAccessRequest:
        row = self.get(db, request_id)
        if not row:
            raise ValueError(f"Message entity access request {request_id} not found")
        row.status = status
        row.admin_id = admin_id
        row.admin_note = admin_note
        row.resolved_at = datetime.utcnow()
        db.flush()
        return row


message_entity_repository = MessageEntityRepository()
