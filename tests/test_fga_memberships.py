from types import SimpleNamespace

# Load the services package first, matching the application's import order and
# avoiding the package-level service re-export cycle during isolated test import.
from app.services.document_service import document_service  # noqa: F401
from app.fga import adapter as adapter_module


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeDb:
    def __init__(self, rows):
        self.rows = rows

    def query(self, _model):
        return FakeQuery(self.rows)


def test_sync_user_memberships_writes_persisted_assignments(monkeypatch):
    assignments = [
        SimpleNamespace(user_id="user-1", oui_id="oui-root"),
        SimpleNamespace(user_id="user-2", oui_id="oui-hr"),
    ]
    captured = []

    monkeypatch.setattr(
        adapter_module.fga_client,
        "write",
        lambda tuples: captured.extend(tuples),
    )

    synced = adapter_module.fga_adapter.sync_user_memberships(FakeDb(assignments))

    assert synced == 2
    assert captured == [
        {"user": "user:user-1", "relation": "member", "object": "oui:oui-root"},
        {"user": "user:user-2", "relation": "member", "object": "oui:oui-hr"},
    ]
