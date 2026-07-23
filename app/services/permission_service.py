"""
Role- and FGA-based permission checks for document view, create, and update operations.
"""
from app.models.document import Document
from app.models.user import User
from app.fga.adapter import fga_adapter
from app.services.sensitivity_levels import MIN_SENSITIVITY


class PermissionService:
    EDIT_ROLES = {"department_manager", "director", "admin_auditor"}

    # Map a sensitivity level string to a numeric rank for comparison.
    @staticmethod
    def _rank(level: str) -> int:
        return {
            "public": 1,
            "internal": 2,
            "confidential": 3,
            "restricted": 4,
            "top_secret": 5,
        }.get(level or "internal", 0)

    # Return True if FGA grants the user view access to the document.
    def can_view_document(self, user: User, doc: Document) -> tuple[bool, str]:
        if doc.sensitivity == MIN_SENSITIVITY:
            return True, "ok"
        if not fga_adapter.can_view(user.id, doc.id):
            return False, "fga_denied"
        return True, "ok"

    # Return True if the user's role permits creating a document in the given department scope.
    def can_create_document(self, user: User, department_id: str | None) -> tuple[bool, str]:
        # Director and admin_auditor may create any document type.
        if user.role in {"director", "admin_auditor"}:
            return True, "ok"

        # Company-wide doc (no department) — restricted to managers and above.
        if department_id is None:
            if user.role == "department_manager":
                return True, "ok"
            return False, "only_managers_can_create_company_doc"

        # Dept-scoped doc — user may only upload to their own department.
        if user.role in {"department_manager", "employee"}:
            if user.department_id == department_id:
                return True, "ok"
            return False, "department_scope_denied"

        return False, "role_or_scope_denied"

    # Return True if the user may edit the document based on ownership, FGA, status, and clearance.
    def can_update_document(self, user: User, doc: Document) -> tuple[bool, str]:
        # Owner may always update their own document.
        if doc.owner_user_id == user.id:
            return True, "ok"

        if not fga_adapter.can_edit(user.id, doc.id):
            return False, "fga_denied"

        if doc.status == "archived" and user.role != "admin_auditor":
            return False, "archived_denied"

        if self._rank(user.clearance_level) < self._rank(doc.sensitivity_level):
            return False, "clearance_denied"

        if user.role in self.EDIT_ROLES:
            return True, "ok"

        return False, "role_denied"


# Module-level singleton; imported by the document service and API routers.
permission_service = PermissionService()
