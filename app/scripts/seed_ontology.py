"""Seed script — builds the STATIC class taxonomy of the Enterprise RAG
Access-Control Ontology in Neo4j. Run with:

    docker exec back-end-api-1 python -m app.scripts.seed_ontology

This does NOT seed instance data (entity rules, org units, positions, user
assignments) — that's synced live from real MySQL data by
app/services/ontology_sync_service.py on every relevant write, and
backfilled for pre-existing rows by app/scripts/resync_ontology.py (run
that *after* this one). This script only lays down the fixed class
scaffold instance nodes attach to via IS_A. Safe to re-run —
wipes and rebuilds the whole graph, including instance data, so always
follow it with resync_ontology.py.
"""
from __future__ import annotations

import uuid

from app.db.neo4j_client import get_session

# key -> generated uuid, so nodes can be referenced by short key when building edges
_ids: dict[str, str] = {}


def _uid(key: str) -> str:
    if key not in _ids:
        _ids[key] = str(uuid.uuid4())
    return _ids[key]


# 20 hard-coded document categories, chosen at upload time via a dropdown
# (front-end/src/lib/document-types.ts — keep both lists in sync by hand).
DOCUMENT_TYPES = [
    "EmployeeProfile", "Contract", "FinancialReport", "TechnicalDocument", "MeetingMinutes",
    "Email", "Specification", "Policy", "ProjectPlan", "Invoice", "PurchaseOrder", "Proposal",
    "PerformanceReview", "TrainingMaterial", "Presentation", "AuditReport", "LegalDocument",
    "Memo", "Handbook", "Other",
]

# (key, name, category, type)
NODES: list[tuple[str, str, str, str]] = [
    # --- Organization --- (OU types/instances synced live; only root here.)
    ("Enterprise", "Enterprise", "Organization", "Root"),

    # --- Employee --- (User nodes synced live; root kept for taxonomy.)
    ("Employee", "Employee", "Employee", "Root"),

    # --- Document ---
    ("Document", "Document", "Document", "Root"),
    *[(key, key, "Document", "DocumentType") for key in DOCUMENT_TYPES],

    # --- Entity taxonomy --- (leaf nodes synced live from
    # entity_policy_rules; only the 5-tier class scaffold stays here,
    # matching the 5 real sensitivity levels 1-5.)
    ("Entity", "Entity", "Entity", "Root"),
    ("PublicEntity", "PublicEntity", "Entity", "EntityClass"),
    ("InternalEntity", "InternalEntity", "Entity", "EntityClass"),
    ("ConfidentialEntity", "ConfidentialEntity", "Entity", "EntityClass"),
    ("SecretEntity", "SecretEntity", "Entity", "EntityClass"),
    ("TopSecretEntity", "TopSecretEntity", "Entity", "EntityClass"),

    # --- Policy --- (per-entity Allow/Mask nodes synced live, one pair per
    # entity_policy_rule, literally named "Allow"/"Mask". The class nodes
    # here are named "AllowPolicy"/"MaskPolicy" instead, so the two never
    # share a label. No Block class/instance nodes at all — block has no
    # scope list of its own (it's "everyone not in allow or mask"), so
    # there is nothing for a block node to ever point APPLIES_TO at; see
    # ontology_sync_service.py's sync_entity_rule for the full reasoning.)
    ("AccessPolicy", "AccessPolicy", "Policy", "Root"),
    ("AllowPolicyClass", "AllowPolicy", "Policy", "PolicyClass"),
    ("MaskPolicyClass", "MaskPolicy", "Policy", "PolicyClass"),
    ("AuditPolicy", "AuditPolicy", "Policy", "PolicyClass"),

    # --- Scope --- (per-OU-type scope classes, eg. "Phòng banScope", and
    # OUI instance/Position nodes synced live. RoleScope stays static since
    # every real Position maps to it regardless of OU type.)
    ("Scope", "Scope", "Scope", "Root"),
    ("RoleScope", "RoleScope", "Scope", "ScopeClass"),

    # --- Permission ---
    ("Permission", "Permission", "Permission", "Root"),
    ("AllowEntity", "AllowEntity", "Permission", "PermissionClass"),
    ("MaskEntity", "MaskEntity", "Permission", "PermissionClass"),

    # --- Decision outcomes ---
    ("Allow", "ALLOW", "Decision", "Decision"),
    ("Mask", "MASK", "Decision", "Decision"),
    ("Block", "BLOCK", "Decision", "Decision"),
    ("Audit", "AUDIT", "Decision", "Decision"),
]

# (fromKey, relType, toKey)
EDGES: list[tuple[str, str, str]] = [
    *[(key, "IS_A", "Document") for key in DOCUMENT_TYPES],

    ("PublicEntity", "IS_A", "Entity"),
    ("InternalEntity", "IS_A", "Entity"),
    ("ConfidentialEntity", "IS_A", "Entity"),
    ("SecretEntity", "IS_A", "Entity"),
    ("TopSecretEntity", "IS_A", "Entity"),

    ("AllowPolicyClass", "IS_A", "AccessPolicy"),
    ("MaskPolicyClass", "IS_A", "AccessPolicy"),
    ("AuditPolicy", "IS_A", "AccessPolicy"),

    ("RoleScope", "IS_A", "Scope"),

    ("AllowEntity", "IS_A", "Permission"),
    ("MaskEntity", "IS_A", "Permission"),

    # Only the 3 tiers a policy can actually result in; Audit is a
    # taxonomy leftover with no producer today.
    ("Permission", "RESULTS_IN", "Allow"),
    ("Permission", "RESULTS_IN", "Mask"),
    ("Permission", "RESULTS_IN", "Block"),
]


def main() -> None:
    with get_session() as session:
        print("[seed] Xoá dữ liệu ontology cũ (nếu có)...")
        session.run("MATCH (n:OntologyNode) DETACH DELETE n")

        print(f"[seed] Tạo {len(NODES)} node...")
        for key, name, category, type_ in NODES:
            session.run(
                """
                CREATE (n:OntologyNode {
                  id: $id, name: $name, category: $category, type: $type,
                  description: '', createdAt: datetime()
                })
                """,
                id=_uid(key), name=name, category=category, type=type_,
            )

        print(f"[seed] Tạo {len(EDGES)} quan hệ...")
        for from_key, rel_type, to_key in EDGES:
            session.run(
                f"""
                MATCH (a:OntologyNode {{id: $fromId}}), (b:OntologyNode {{id: $toId}})
                CREATE (a)-[r:{rel_type} {{id: $relId}}]->(b)
                """,
                fromId=_uid(from_key), toId=_uid(to_key), relId=str(uuid.uuid4()),
            )

        print("[seed] Hoàn tất! Ontology đã được nạp vào Neo4j.")

    # Deliberately not closing the driver here — it's a shared singleton
    # (app/db/neo4j_client.py) also used by the running API process when
    # this is called from app.main's startup hook, not just standalone.


if __name__ == "__main__":
    main()
