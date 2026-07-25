"""
Org-unit hierarchy management endpoints: OU types, OUI instances, positions,
and user assignment/unassignment within the organization tree.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.models.user import User
from app.models.org_unit import OrgUnit
from app.models.org_unit_instance import OrgUnitInstance
from app.models.position import Position
from app.models.user_oui_position import UserOuiPosition
from app.fga.adapter import fga_adapter
from app.services.ontology_sync_service import ontology_sync_service
from app.services.oui_tree_service import oui_tree_service
from app.services.user_service import user_service as _user_service


router = APIRouter()


# Raise 403 if the caller is not a corp-level member.
def require_admin(user: User, db: Session):
    user_resp = _user_service.build_user_response(db, user)
    if not user_resp.is_corp_member:
        raise HTTPException(status_code=403, detail="Corp-level admin required")


# ══ Schemas ══════════════════════════════════════════════════════════════════

class OrgUnitCreate(BaseModel):
    name: str
    parent_id: str | None = None  # None = root corp OU

class OrgUnitInstanceCreate(BaseModel):
    name: str
    ou_id: str
    parent_oui_ids: list[str] = []   # Multi-parent

class PositionCreate(BaseModel):
    name: str
    ou_id: str
    clearance: int  # 1–5

class OrgUnitInstanceUpdate(BaseModel):
    parent_oui_ids: list[str]

class AssignUserRequest(BaseModel):
    user_id: str
    oui_id: str
    position_id: str

class UnassignUserRequest(BaseModel):
    user_id: str
    oui_id: str


# ══ OU endpoints ══════════════════════════════════════════════════════════════

# Return the full OU type tree.
@router.get("/org-units")
def list_org_units(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    units = db.query(OrgUnit).all()
    return [
        {
            "id": u.id,
            "name": u.name,
            "parent_id": u.parent_id,
        }
        for u in units
    ]


# Create a new OU type (e.g. Division, Branch, Team). Requires admin.
@router.post("/org-units")
def create_org_unit(
    payload: OrgUnitCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    if payload.parent_id:
        parent = db.get(OrgUnit, payload.parent_id)
        if not parent:
            raise HTTPException(status_code=404, detail="Parent OU not found")
    existing = db.query(OrgUnit).filter(OrgUnit.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"OU '{payload.name}' already exists")
    unit = OrgUnit(name=payload.name, parent_id=payload.parent_id)
    db.add(unit)
    db.commit()
    db.refresh(unit)
    ontology_sync_service.sync_org_unit(unit)
    return {"id": unit.id, "name": unit.name, "parent_id": unit.parent_id}


# Delete an OU type. The root corp OU cannot be deleted.
@router.delete("/org-units/{ou_id}")
def delete_org_unit(
    ou_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    unit = db.get(OrgUnit, ou_id)
    if not unit:
        raise HTTPException(status_code=404, detail="OU not found")
    if unit.parent_id is None:
        raise HTTPException(status_code=400, detail="Cannot delete root Corp. OU")
    if unit.children:
        raise HTTPException(status_code=400, detail="Cannot delete OU with child OUs")
    if unit.instances:
        raise HTTPException(status_code=400, detail="Cannot delete OU that has instances")
    db.delete(unit)
    db.commit()
    ontology_sync_service.delete_org_unit(ou_id)
    return {"status": "deleted"}


# ══ OUI endpoints ════════════════════════════════════════════════════════════

# Return all OUI instances with their parent IDs.
@router.get("/org-unit-instances")
def list_oui(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    instances = db.query(OrgUnitInstance).all()
    return [
        {
            "id": i.id,
            "name": i.name,
            "ou_id": i.ou_id,
            "parent_oui_ids": [p.id for p in i.parents],
        }
        for i in instances
    ]


# Create a new OUI instance (e.g. HR, Marketing). Requires admin.
@router.post("/org-unit-instances")
def create_oui(
    payload: OrgUnitInstanceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    ou = db.get(OrgUnit, payload.ou_id)
    if not ou:
        raise HTTPException(status_code=404, detail="OU not found")

    parents = []
    for pid in payload.parent_oui_ids:
        p = db.get(OrgUnitInstance, pid)
        if not p:
            raise HTTPException(status_code=404, detail=f"Parent OUI {pid} not found")
        parents.append(p)

    instance = OrgUnitInstance(name=payload.name, ou_id=payload.ou_id)
    instance.parents = parents
    db.add(instance)
    db.commit()
    db.refresh(instance)

    # Sync parent relationships into FGA.
    for p in parents:
        fga_adapter.link_oui_parent(instance.id, p.id)

    ontology_sync_service.sync_org_unit_instance(instance, db)

    return {
        "id": instance.id,
        "name": instance.name,
        "ou_id": instance.ou_id,
        "parent_oui_ids": [p.id for p in parents],
    }


# Update the parent_oui_ids of an existing OUI instance (replaces the full list).
@router.put("/org-unit-instances/{oui_id}")
def update_oui(
    oui_id: str,
    payload: OrgUnitInstanceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    instance = db.get(OrgUnitInstance, oui_id)
    if not instance:
        raise HTTPException(status_code=404, detail="OUI not found")

    new_parents = []
    for pid in payload.parent_oui_ids:
        p = db.get(OrgUnitInstance, pid)
        if not p:
            raise HTTPException(status_code=404, detail=f"Parent OUI {pid} not found")
        new_parents.append(p)

    old_parent_ids = {p.id for p in instance.parents}
    new_parent_ids = {p.id for p in new_parents}

    for pid in old_parent_ids - new_parent_ids:
        fga_adapter.unlink_oui_parent(oui_id, pid)
    for pid in new_parent_ids - old_parent_ids:
        fga_adapter.link_oui_parent(oui_id, pid)

    instance.parents = new_parents
    db.commit()
    db.refresh(instance)

    # Re-sync document FGA tuples so stale viewer grants are cleaned up.
    _resync_docs_for_oui(db, oui_id)
    ontology_sync_service.sync_org_unit_instance(instance, db)

    return {
        "id": instance.id,
        "name": instance.name,
        "ou_id": instance.ou_id,
        "parent_oui_ids": [p.id for p in instance.parents],
    }


# Delete an OUI instance. Cannot delete if it has children, assigned users, or documents.
@router.delete("/org-unit-instances/{oui_id}")
def delete_oui(
    oui_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    instance = db.get(OrgUnitInstance, oui_id)
    if not instance:
        raise HTTPException(status_code=404, detail="OUI not found")
    if instance.children:
        raise HTTPException(status_code=400, detail="Cannot delete OUI with children")
    if instance.user_positions:
        raise HTTPException(status_code=400, detail="Cannot delete OUI that has assigned users")
    if instance.documents:
        raise HTTPException(status_code=400, detail="Cannot delete OUI that owns documents")
    # Unlink FGA parent relations
    for p in instance.parents:
        fga_adapter.unlink_oui_parent(oui_id, p.id)
    db.delete(instance)
    db.commit()
    ontology_sync_service.delete_org_unit_instance(oui_id)
    return {"status": "deleted"}


# ══ Position endpoints ════════════════════════════════════════════════════════

# Return all positions across all OU types.
@router.get("/positions")
def list_positions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    positions = db.query(Position).all()
    return [
        {"id": p.id, "name": p.name, "ou_id": p.ou_id, "clearance": p.clearance}
        for p in positions
    ]


# Create a new position for an OU type (e.g. Dept Manager, clearance=4). Requires admin.
@router.post("/positions")
def create_position(
    payload: PositionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    if not 1 <= payload.clearance <= 5:
        raise HTTPException(status_code=400, detail="Clearance must be 1–5")
    ou = db.get(OrgUnit, payload.ou_id)
    if not ou:
        raise HTTPException(status_code=404, detail="OU not found")
    pos = Position(name=payload.name, ou_id=payload.ou_id, clearance=payload.clearance)
    db.add(pos)
    db.commit()
    db.refresh(pos)
    ontology_sync_service.sync_position(pos)
    return {"id": pos.id, "name": pos.name, "ou_id": pos.ou_id, "clearance": pos.clearance}


# Update a position's name or clearance level. Triggers a document re-sync for affected OUIs.
@router.put("/positions/{position_id}")
def update_position(
    position_id: str,
    payload: PositionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    pos = db.get(Position, position_id)
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")
    if not 1 <= payload.clearance <= 5:
        raise HTTPException(status_code=400, detail="Clearance must be 1–5")
    pos.name = payload.name
    pos.clearance = payload.clearance
    db.commit()
    db.refresh(pos)

    # Clearance changed: re-sync all documents owned by OUIs using this position.
    _resync_docs_for_position(db, position_id)
    ontology_sync_service.sync_position(pos)

    return {"id": pos.id, "name": pos.name, "clearance": pos.clearance}


# Delete a position only when it is not assigned to any user.
@router.delete("/positions/{position_id}")
def delete_position(
    position_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    pos = db.get(Position, position_id)
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")

    assignment_count = db.query(UserOuiPosition).filter(
        UserOuiPosition.position_id == position_id,
    ).count()
    if assignment_count:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Không thể xóa vị trí đang được gán cho {assignment_count} người dùng. "
                "Hãy chuyển hoặc gỡ vị trí của các người dùng trước."
            ),
        )

    db.delete(pos)
    db.commit()
    ontology_sync_service.delete_position(position_id)
    return {"status": "deleted", "position_id": position_id}


# Re-sync FGA document tuples for all OUIs that use the given position.
def _resync_docs_for_position(db: Session, position_id: str):
    from app.services.document_service import document_service

    affected_oui_ids = [
        r.oui_id for r in db.query(UserOuiPosition).filter(
            UserOuiPosition.position_id == position_id
        ).all()
    ]
    for oui_id in affected_oui_ids:
        _resync_docs_for_oui(db, oui_id)


# ══ Assign / unassign user ════════════════════════════════════════════════════

# Assign a user to an OUI with a given position. Enforces branch-conflict rules.
@router.post("/users/assign-oui")
def assign_user_to_oui(
    payload: AssignUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Assign a user to an (OUI + Position) pair.
    Conflict rule: a user cannot hold two assignments on the same OUI branch.
    """
    require_admin(current_user, db)

    # Validate
    user = db.get(User, payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    oui = db.get(OrgUnitInstance, payload.oui_id)
    if not oui:
        raise HTTPException(status_code=404, detail="OUI not found")
    pos = db.get(Position, payload.position_id)
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")
    # Position must belong to the same OU type as the OUI.
    if pos.ou_id != oui.ou_id:
        raise HTTPException(
            status_code=400,
            detail=f"Position '{pos.name}' không thuộc OU type của OUI này"
        )

    # Conflict check
    conflict = oui_tree_service.check_conflict(db, payload.user_id, payload.oui_id)
    if conflict:
        raise HTTPException(status_code=409, detail=conflict)

    # Persist the assignment.
    record = UserOuiPosition(
        user_id=payload.user_id,
        oui_id=payload.oui_id,
        position_id=payload.position_id,
    )
    db.add(record)
    db.commit()

    fga_adapter.add_oui_member(payload.user_id, payload.oui_id)
    ontology_sync_service.sync_user_assignment(user, payload.oui_id, payload.position_id)

    return {"status": "assigned", "user_id": payload.user_id, "oui": oui.name, "position": pos.name}


# Remove a user from an OUI and revoke their FGA membership tuple.
@router.post("/users/unassign-oui")
def unassign_user_from_oui(
    payload: UnassignUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    record = db.query(UserOuiPosition).filter(
        UserOuiPosition.user_id == payload.user_id,
        UserOuiPosition.oui_id == payload.oui_id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Assignment not found")

    db.delete(record)
    db.commit()

    fga_adapter.remove_oui_member(payload.user_id, payload.oui_id)
    ontology_sync_service.delete_user_assignment(payload.user_id, payload.oui_id)

    return {"status": "unassigned"}


# Change a user's position within an OUI without altering their OUI membership.
@router.put("/users/{user_id}/oui/{oui_id}/position")
def change_position(
    user_id: str,
    oui_id: str,
    payload: AssignUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_admin(current_user, db)
    record = db.query(UserOuiPosition).filter(
        UserOuiPosition.user_id == user_id,
        UserOuiPosition.oui_id == oui_id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Assignment not found")
    pos = db.get(Position, payload.position_id)
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")

    record.position_id = payload.position_id
    db.commit()

    user = db.get(User, user_id)
    if user:
        ontology_sync_service.sync_user_assignment(user, oui_id, payload.position_id)

    return {"status": "updated"}


# Re-sync FGA document tuples for all documents owned by the given OUI.
def _resync_docs_for_oui(db: Session, oui_id: str):
    from app.services.document_service import document_service

    oui = db.get(OrgUnitInstance, oui_id)
    if not oui:
        return
    docs = oui.documents  # ORM relationship: OrgUnitInstance → documents
    for doc in docs:
        old = fga_adapter.get_document_tuples(doc.id)
        fga_adapter.delete_document_tuples(doc.id, old)
        document_service._sync_fga(db, doc)
