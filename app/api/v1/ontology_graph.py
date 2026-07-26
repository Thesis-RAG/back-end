"""Ontology graph endpoints — front-end/src/pages/GraphPage.tsx reads and
writes through here. Same JWT auth as the rest of the app; admin-gated the
same way policy.py is (corp-level members only), since editing this graph
can change the same entity-policy scope pairs "Cấu hình thực thể" does.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.models.user import User
from app.services import ontology_graph_service
from app.services.user_service import user_service

router = APIRouter()


def _require_admin(db: Session, user: User) -> None:
    if not user_service.build_user_response(db, user).is_corp_member:
        raise HTTPException(status_code=403, detail="Corp-level required")


class NodeCreate(BaseModel):
    name: str
    category: str
    type: Optional[str] = None
    description: Optional[str] = None


class NodeUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None


class RelationshipCreate(BaseModel):
    fromId: str
    toId: str
    relType: str


@router.get("/meta")
def get_meta(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _require_admin(db, current_user)
    return ontology_graph_service.get_meta()


@router.get("/graph")
def get_graph(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _require_admin(db, current_user)
    return ontology_graph_service.get_graph()


@router.post("/nodes", status_code=201)
def create_node(
    payload: NodeCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    _require_admin(db, current_user)
    try:
        return ontology_graph_service.create_node(payload.name, payload.category, payload.type, payload.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/nodes/{node_id}")
def update_node(
    node_id: str, payload: NodeUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    _require_admin(db, current_user)
    try:
        result = ontology_graph_service.update_node(
            db, node_id, payload.name, payload.category, payload.type, payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy node")
    return result


@router.delete("/nodes/{node_id}", status_code=204)
def delete_node(node_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _require_admin(db, current_user)
    ontology_graph_service.delete_node(node_id)


@router.post("/relationships", status_code=201)
def create_relationship(
    payload: RelationshipCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    _require_admin(db, current_user)
    try:
        return ontology_graph_service.create_relationship(db, payload.fromId, payload.toId, payload.relType)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/relationships/{rel_id}", status_code=204)
def delete_relationship(rel_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _require_admin(db, current_user)
    ontology_graph_service.delete_relationship(db, rel_id)
