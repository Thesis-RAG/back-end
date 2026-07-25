"""One-off backfill: push every existing EntityPolicyRule / OrgUnit /
OrgUnitInstance / Position / UserOuiPosition row into the ontology graph.

The live hooks in app/api/v1/policy.py and app/api/v1/org_units.py only
fire on new writes going forward — rows that already existed before those
hooks were added need this to show up on the graph at all. Safe to re-run
any time (every step is an idempotent upsert, see ontology_sync_service.py's
MERGE usage); order matters here (parents before children, org structure
before entity rules that reference it via scope pairs).

Run once after seeding the static taxonomy, inside the API container:
    docker exec back-end-api-1 python -m app.scripts.seed_ontology
    docker exec back-end-api-1 python -m app.scripts.resync_ontology
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from app.db.neo4j_client import get_driver
from app.db.session import SessionLocal
from app.models.entity_policy_rule import EntityPolicyRule
from app.models.org_unit import OrgUnit
from app.models.org_unit_instance import OrgUnitInstance
from app.models.position import Position
from app.models.user import User
from app.models.user_oui_position import UserOuiPosition
from app.services.ontology_sync_service import ontology_sync_service

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def _topo_org_units(units: list[OrgUnit]) -> list[OrgUnit]:
    """Roots first, then children whose parent has already been ordered —
    org-unit sync looks up the parent node by sourceId, so a child synced
    before its parent exists would just end up IS_A'd to nothing."""
    done: set[str] = set()
    ordered: list[OrgUnit] = []
    remaining = list(units)
    while remaining:
        ready = [u for u in remaining if u.parent_id is None or u.parent_id in done]
        if not ready:
            ordered.extend(remaining)  # dangling/cyclic parent_id — bail rather than loop forever
            break
        for unit in ready:
            ordered.append(unit)
            done.add(unit.id)
        remaining = [u for u in remaining if u.id not in done]
    return ordered


# Core logic, reusable — takes an existing session, never exits the
# process. Called both by main() below (its own session, CLI use) and by
# app.main's startup hook (the request-scoped session there).
def run(db: "Session") -> None:
    org_units = _topo_org_units(db.query(OrgUnit).all())
    print(f"[1/5] {len(org_units)} org unit (loại đơn vị)...")
    for unit in org_units:
        ontology_sync_service.sync_org_unit(unit)

    instances = db.query(OrgUnitInstance).all()
    print(f"[2/5] {len(instances)} org unit instance...")
    for instance in instances:
        ontology_sync_service.sync_org_unit_instance(instance)

    positions = db.query(Position).all()
    print(f"[3/5] {len(positions)} position...")
    for position in positions:
        ontology_sync_service.sync_position(position)

    assignments = db.query(UserOuiPosition).all()
    users_by_id = {u.id: u for u in db.query(User).all()}
    print(f"[4/5] {len(assignments)} user assignment...")
    for assignment in assignments:
        user = users_by_id.get(assignment.user_id)
        if not user:
            continue
        ontology_sync_service.sync_user_assignment(user, assignment.oui_id, assignment.position_id)

    rules = db.query(EntityPolicyRule).all()
    print(f"[5/5] {len(rules)} entity policy rule...")
    for rule in rules:
        ontology_sync_service.sync_entity_rule(db, rule)

    print("Xong.")


def main() -> None:
    try:
        get_driver().verify_connectivity()
    except Exception as exc:
        print(f"Không kết nối được Neo4j ({exc}) — kiểm tra service neo4j đã chạy chưa (docker compose up -d neo4j).")
        sys.exit(1)

    db = SessionLocal()
    try:
        run(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
