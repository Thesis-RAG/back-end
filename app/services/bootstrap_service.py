"""Idempotent first-start bootstrap for the organization and default admin."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.org_unit import OrgUnit
from app.models.org_unit_instance import OrgUnitInstance
from app.models.position import Position
from app.models.user import User
from app.models.user_oui_position import UserOuiPosition

CORP_OU_NAME = "Công ty"
ADMIN_EMAIL = "admin@rag.com"
ADMIN_NAME = "Quản trị viên"
ADMIN_PASS = "Admin@123"

logger = logging.getLogger(__name__)


class BootstrapService:
    """Create the minimum enterprise hierarchy without creating duplicates."""

    def _get_or_create_ou(self, db: Session, name: str, parent: OrgUnit | None) -> OrgUnit:
        ou = db.query(OrgUnit).filter(OrgUnit.name == name).first()
        if ou is None:
            ou = OrgUnit(name=name, parent_id=parent.id if parent else None)
            db.add(ou)
            db.flush()
        elif parent is not None and ou.parent_id != parent.id:
            ou.parent_id = parent.id
            db.flush()
        return ou

    def _get_or_create_position(
        self,
        db: Session,
        name: str,
        ou: OrgUnit,
        clearance: int,
    ) -> Position:
        position = (
            db.query(Position)
            .filter(Position.name == name, Position.ou_id == ou.id)
            .first()
        )
        if position is None:
            position = Position(name=name, ou_id=ou.id, clearance=clearance)
            db.add(position)
            db.flush()
        elif position.clearance != clearance:
            position.clearance = clearance
            db.flush()
        return position

    def _get_or_create_oui(self, db: Session, name: str, ou: OrgUnit) -> OrgUnitInstance:
        oui = (
            db.query(OrgUnitInstance)
            .filter(OrgUnitInstance.name == name, OrgUnitInstance.ou_id == ou.id)
            .first()
        )
        if oui is None:
            oui = OrgUnitInstance(name=name, ou_id=ou.id)
            db.add(oui)
            db.flush()
        return oui

    def _ensure_user_assignment(
        self,
        db: Session,
        user: User,
        oui: OrgUnitInstance,
        position: Position,
    ) -> None:
        assignment = (
            db.query(UserOuiPosition)
            .filter(
                UserOuiPosition.user_id == user.id,
                UserOuiPosition.oui_id == oui.id,
            )
            .first()
        )
        if assignment is None:
            db.add(UserOuiPosition(
                user_id=user.id,
                oui_id=oui.id,
                position_id=position.id,
            ))
        elif assignment.position_id != position.id:
            assignment.position_id = position.id
        db.flush()

    def seed_defaults(self, db: Session) -> None:
        # Reuse the first existing root. This prevents the old minimal bootstrap
        # and the sample hierarchy bootstrap from creating two company roots.
        root = (
            db.query(OrgUnit)
            .filter(OrgUnit.parent_id.is_(None))
            .order_by(OrgUnit.created_at.asc())
            .first()
        )
        if root is None:
            root = self._get_or_create_ou(db, CORP_OU_NAME, None)

        department = self._get_or_create_ou(db, "Department", root)
        division = self._get_or_create_ou(db, "Division", root)
        branch = self._get_or_create_ou(db, "Branch", root)
        project = self._get_or_create_ou(db, "Project", department)
        self._get_or_create_ou(db, "Team", department)
        self._get_or_create_ou(db, "Group", division)
        self._get_or_create_ou(db, "Program", division)
        self._get_or_create_ou(db, "Support Unit", branch)

        admin_position = self._get_or_create_position(db, "Admin", root, 5)
        self._get_or_create_position(db, "Director", root, 4)
        self._get_or_create_position(db, "Dept Manager", department, 4)
        self._get_or_create_position(db, "Deputy Dept Manager", department, 3)
        self._get_or_create_position(db, "Employee", department, 2)
        self._get_or_create_position(db, "Project Leader", project, 3)
        self._get_or_create_position(db, "Member", project, 2)

        root_oui = (
            db.query(OrgUnitInstance)
            .filter(OrgUnitInstance.ou_id == root.id)
            .order_by(OrgUnitInstance.created_at.asc())
            .first()
        )
        if root_oui is None:
            root_oui = self._get_or_create_oui(db, root.name, root)
        child_ouis = [
            self._get_or_create_oui(db, "HR", department),
            self._get_or_create_oui(db, "Marketing", department),
            self._get_or_create_oui(db, "Finance", department),
        ]
        for child in child_ouis:
            if root_oui not in child.parents:
                child.parents.append(root_oui)
        db.flush()

        admin = db.query(User).filter(User.email == ADMIN_EMAIL).first()
        if admin is None:
            admin = User(
                email=ADMIN_EMAIL,
                name=ADMIN_NAME,
                status="active",
                password_hash=hash_password(ADMIN_PASS),
            )
            db.add(admin)
            db.flush()
        self._ensure_user_assignment(db, admin, root_oui, admin_position)

        db.commit()

        # FGA is initialized before this method during application startup.
        from app.fga.adapter import fga_adapter

        for child in child_ouis:
            fga_adapter.link_oui_parent(child.id, root_oui.id)
        self._sync_user_memberships(db)
        logger.info(
            "Bootstrap reconciled root=%s root_oui=%s child_ouis=%d",
            root.id,
            root_oui.id,
            len(child_ouis),
        )

    @staticmethod
    def _sync_user_memberships(db: Session) -> None:
        from app.fga.adapter import fga_adapter

        synced = fga_adapter.sync_user_memberships(db)
        logger.info("OpenFGA OUI memberships reconciled: %d", synced)


bootstrap_service = BootstrapService()
