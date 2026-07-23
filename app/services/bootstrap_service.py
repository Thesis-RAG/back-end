"""Idempotent first-start bootstrap for the organization and default users."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.org_unit import OrgUnit
from app.models.org_unit_instance import OrgUnitInstance
from app.models.position import Position
from app.models.user import User
from app.models.user_oui_position import UserOuiPosition
from app.services.policy_rule_service import policy_rule_service


CORP_OU_NAME = "Công ty (Company)"
ADMIN_EMAIL = "admin@rag.com"
ADMIN_NAME = "Quản trị hệ thống (System Administrator)"
ADMIN_PASS = "Admin@123"

OU_MANAGEMENT = "Ban quản lý (Management Board)"
OU_DEPARTMENT = "Phòng ban (Department)"
OU_PROJECT = "Dự án (Project)"
OU_PARTNER = "Đối tác (Partner)"
OU_BRANCH = "Chi nhánh (Branch)"
OU_REPRESENTATIVE_OFFICE = "Văn phòng đại diện (Representative Office)"

COMPANY_POSITIONS = (
    ("Tổng giám đốc (Chief Executive Officer - CEO)", 5),
    ("Phó giám đốc (Deputy Director)", 4),
    ("Trưởng phòng (Department Manager)", 4),
    ("Trưởng bộ phận (Head of Function)", 3),
    ("Chuyên viên (Specialist)", 2),
    ("Nhân viên (Staff)", 2),
    ("Thực tập sinh (Intern)", 1),
)

DEFAULT_USERS = (
    ("Nguyễn Minh Quân", "nmq@rag.com"),
    ("Trần Anh Đức", "tad@rag.com"),
    ("Lê Thu Hà", "lth@rag.com"),
    ("Phạm Quốc Bảo", "pqb@rag.com"),
    ("Nguyễn Hoàng Minh", "nhm@rag.com"),
    ("Võ Ngọc Lan", "vnl@rag.com"),
    ("Đỗ Gia Huy", "dgh@rag.com"),
)

DEFAULT_DEPARTMENTS = (
    ("Phòng Nhân sự (HR)", ("HR",)),
    ("Phòng Kinh doanh (Sales)", ("Sales",)),
    ("Phòng Marketing (Marketing)", ("Marketing",)),
    ("Phòng Hỗ trợ Công nghệ thông tin (IT Helpdesk)", ("IT Helpdesk", "IT Support")),
    ("Phòng Nghiên cứu và Phát triển (R&D)", ("R&D", "Research and Development")),
    ("Phòng Sản phẩm (Product)", ("Product",)),
    ("Phòng Thuê ngoài (Outsourcing)", ("Outsourcing",)),
)

logger = logging.getLogger(__name__)


class BootstrapService:
    """Create and reconcile the standard enterprise hierarchy without duplicates."""

    def _get_or_create_ou(
        self,
        db: Session,
        name: str,
        parent: OrgUnit | None,
        aliases: tuple[str, ...] = (),
    ) -> OrgUnit:
        ou = db.query(OrgUnit).filter(OrgUnit.name == name).first()
        if ou is None and aliases:
            ou = db.query(OrgUnit).filter(OrgUnit.name.in_(aliases)).first()

        parent_id = parent.id if parent else None
        if ou is None:
            ou = OrgUnit(name=name, parent_id=parent_id)
            db.add(ou)
            db.flush()
        else:
            if ou.name != name:
                ou.name = name
            if ou.parent_id != parent_id:
                ou.parent_id = parent_id
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

    def _get_or_create_oui(
        self,
        db: Session,
        name: str,
        ou: OrgUnit,
        aliases: tuple[str, ...] = (),
    ) -> OrgUnitInstance:
        oui = (
            db.query(OrgUnitInstance)
            .filter(
                OrgUnitInstance.name == name,
                OrgUnitInstance.ou_id == ou.id,
            )
            .first()
        )
        if oui is None and aliases:
            oui = (
                db.query(OrgUnitInstance)
                .filter(
                    OrgUnitInstance.name.in_(aliases),
                    OrgUnitInstance.ou_id == ou.id,
                )
                .first()
            )
        if oui is None:
            oui = OrgUnitInstance(name=name, ou_id=ou.id)
            db.add(oui)
            db.flush()
        elif oui.name != name:
            oui.name = name
            db.flush()
        return oui

    def _ensure_default_users(self, db: Session) -> None:
        """Create realistic users with the default password, without assignments."""
        for name, email in DEFAULT_USERS:
            user = db.query(User).filter(User.email == email).first()
            if user is None:
                db.add(User(
                    email=email,
                    name=name,
                    status="active",
                    password_hash=hash_password(ADMIN_PASS),
                ))
        db.flush()

    def _remove_empty_legacy_ou_types(self, db: Session) -> None:
        """Remove only unused OU types from the previous sample bootstrap."""
        legacy_names = ("Team", "Group", "Program", "Support Unit")
        legacy_ous = db.query(OrgUnit).filter(OrgUnit.name.in_(legacy_names)).all()
        for ou in legacy_ous:
            # Never remove a type that already has business data attached to
            # it; preserving user-created data is more important than cleanup.
            if ou.instances or ou.positions or ou.children:
                logger.warning("Keeping legacy OU type with data: %s", ou.name)
                continue
            db.delete(ou)
        db.flush()

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
        # Reuse and normalize the first existing root when upgrading from the
        # previous English/minimal bootstrap.
        root = (
            db.query(OrgUnit)
            .filter(OrgUnit.parent_id.is_(None))
            .order_by(OrgUnit.created_at.asc())
            .first()
        )
        if root is None:
            root = self._get_or_create_ou(db, CORP_OU_NAME, None)
        elif root.name != CORP_OU_NAME:
            root.name = CORP_OU_NAME
            db.flush()

        management = self._get_or_create_ou(
            db,
            OU_MANAGEMENT,
            root,
            aliases=("Division", "Management Board"),
        )
        department = self._get_or_create_ou(
            db,
            OU_DEPARTMENT,
            management,
            aliases=("Department",),
        )
        project = self._get_or_create_ou(
            db,
            OU_PROJECT,
            department,
            aliases=("Project",),
        )
        partner = self._get_or_create_ou(db, OU_PARTNER, root, aliases=("Partner",))
        branch = self._get_or_create_ou(db, OU_BRANCH, root, aliases=("Branch",))
        representative = self._get_or_create_ou(
            db,
            OU_REPRESENTATIVE_OFFICE,
            branch,
            aliases=("Representative Office",),
        )
        self._remove_empty_legacy_ou_types(db)

        # The same standardized position catalog is available for every OU
        # type that may contain employees. Assignment is still left to admin.
        for ou in (root, management, department, project, partner, branch, representative):
            for position_name, clearance in COMPANY_POSITIONS:
                self._get_or_create_position(db, position_name, ou, clearance)
        admin_position = self._get_or_create_position(
            db,
            "Quản trị viên hệ thống (System Administrator)",
            root,
            5,
        )

        root_oui = (
            db.query(OrgUnitInstance)
            .filter(OrgUnitInstance.ou_id == root.id)
            .order_by(OrgUnitInstance.created_at.asc())
            .first()
        )
        if root_oui is None:
            root_oui = self._get_or_create_oui(db, root.name, root)
        elif root_oui.name != root.name:
            root_oui.name = root.name

        management_oui = self._get_or_create_oui(
            db,
            "Ban quản lý Công ty (Management Board)",
            management,
            aliases=("Management Board",),
        )
        department_ouis = [
            self._get_or_create_oui(db, name, department, aliases=aliases)
            for name, aliases in DEFAULT_DEPARTMENTS
        ]

        # Concrete tree: Company -> Management Board -> Departments.
        old_department_parent_ids = {
            child.id: {parent.id for parent in child.parents}
            for child in department_ouis
        }
        root_oui.parents = []
        management_oui.parents = [root_oui]
        for child in department_ouis:
            child.parents = [management_oui]
        db.flush()

        # The platform admin remains company-level so the organization UI is
        # usable. The sample employee users above deliberately have no roles.
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
        self._ensure_default_users(db)
        policy_rule_service.seed_defaults(db)

        db.commit()

        from app.fga.adapter import fga_adapter

        fga_adapter.link_oui_parent(management_oui.id, root_oui.id)
        for child in department_ouis:
            for old_parent_id in old_department_parent_ids.get(child.id, set()) - {management_oui.id}:
                fga_adapter.unlink_oui_parent(child.id, old_parent_id)
            fga_adapter.link_oui_parent(child.id, management_oui.id)
        self._sync_user_memberships(db)
        logger.info(
            "Bootstrap reconciled root=%s root_oui=%s departments=%d",
            root.id,
            root_oui.id,
            len(department_ouis),
        )

    @staticmethod
    def _sync_user_memberships(db: Session) -> None:
        from app.fga.adapter import fga_adapter

        synced = fga_adapter.sync_user_memberships(db)
        logger.info("OpenFGA OUI memberships reconciled: %d", synced)


bootstrap_service = BootstrapService()
