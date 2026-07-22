"""
Service for seeding the initial organisational structure and admin user on first startup.
"""
import logging

from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.user import User
from app.models.org_unit import OrgUnit
from app.models.org_unit_instance import OrgUnitInstance
from app.models.position import Position
from app.models.user_oui_position import UserOuiPosition

CORP_OU_NAME = "Công ty"
ADMIN_EMAIL  = "admin@rag.com"
ADMIN_NAME   = "Quản trị viên"
ADMIN_PASS   = "Admin@123"

logger = logging.getLogger(__name__)


class BootstrapService:
    # Create the root OrgUnit, root OUI, Admin position, and the default admin user if none exist.
    def seed_defaults(self, db: Session):
        if db.query(User).count() > 0:
            self._sync_user_memberships(db)
            return

        # 1. Corp. OU
        corp_ou = OrgUnit(name=CORP_OU_NAME)
        db.add(corp_ou)
        db.flush()

        # 2. Corp. OUI (instance duy nhất của Corp.)
        corp_oui = OrgUnitInstance(name=CORP_OU_NAME, ou_id=corp_ou.id)
        db.add(corp_oui)
        db.flush()

        # 3. Positions cho Corp. OU
        admin_pos    = Position(name="Admin",    ou_id=corp_ou.id, clearance=5)
        db.add_all([admin_pos])
        db.flush()

        # 4. Admin user
        admin = User(
            email=ADMIN_EMAIL,
            name=ADMIN_NAME,
            status="active",
            password_hash=hash_password(ADMIN_PASS),
        )
        db.add(admin)
        db.flush()

        # 5. Assign admin → Corp. OUI + Admin position
        db.add(UserOuiPosition(
            user_id=admin.id,
            oui_id=corp_oui.id,
            position_id=admin_pos.id,
        ))

        db.commit()

        self._sync_user_memberships(db)

    @staticmethod
    def _sync_user_memberships(db: Session) -> None:
        """Repair OpenFGA membership tuples for existing DB assignments."""
        from app.fga.adapter import fga_adapter

        try:
            synced = fga_adapter.sync_user_memberships(db)
            logger.info("OpenFGA OUI memberships reconciled: %d", synced)
        except Exception:
            # Keep startup available while authorization remains fail-closed;
            # the admin FGA sync endpoint can retry once OpenFGA is available.
            logger.exception("Unable to reconcile OpenFGA OUI memberships")


# Module-level singleton; imported by the startup lifespan handler.
bootstrap_service = BootstrapService()
