"""Generic ontology-graph node/edge CRUD, backing app/api/v1/ontology_graph.py
(the graph-editing UI in front-end/src/pages/GraphPage.tsx). Ported from
what used to be a separate Node/Express service's routes/graph.js.

Also does reverse sync: when an edit touches a node/edge carrying
{sourceType, sourceId} (put there by ontology_sync_service.py — a
"recognized" node, not one a user drew freehand), it mirrors the change
back into the real MySQL row, in the same process, no network hop. Node
*deletion* is deliberately NOT mirrored — the real delete endpoints
(org_units.py, policy.py) enforce guards (eg. "cannot delete OU with
instances") that a graph-side delete would bypass, so playing with the
graph can't destroy real org/policy data. Renames and entity-policy
scope-pair edits are the two things this graph safely round-trips.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.db.neo4j_client import get_session as get_neo4j_session
from app.services import ontology_sync_service as sync_engine

REL_TYPES = sync_engine.REL_TYPES  # same whitelist, single source of truth

CATEGORIES = {
    "Organization", "Employee", "Document", "Entity",
    "Policy", "Scope", "Permission", "Decision",
}


def _clean_props(props: dict) -> dict:
    result = dict(props)
    created_at = result.get("createdAt")
    if created_at is not None and hasattr(created_at, "iso_format"):
        result["createdAt"] = created_at.iso_format()
    return result


def get_meta() -> dict:
    return {"relTypes": sorted(REL_TYPES), "categories": sorted(CATEGORIES)}


def get_graph() -> dict:
    with get_neo4j_session() as session:
        result = session.run(
            """
            MATCH (n:OntologyNode)
            OPTIONAL MATCH (n)-[r]->(m:OntologyNode)
            RETURN n, r, m
            """
        )
        nodes_by_id: dict[str, dict] = {}
        edges: list[dict] = []
        for record in result:
            n, m, r = record["n"], record["m"], record["r"]
            if n is not None:
                nodes_by_id[n["id"]] = _clean_props(dict(n))
            if m is not None:
                nodes_by_id[m["id"]] = _clean_props(dict(m))
            if r is not None:
                edges.append({"id": r.get("id"), "type": r.type, "source": n["id"], "target": m["id"]})
        return {"nodes": list(nodes_by_id.values()), "edges": edges}


def create_node(name: str, category: str, type_: str | None, description: str | None) -> dict:
    if not name or not category:
        raise ValueError("name và category là bắt buộc")
    if category not in CATEGORIES:
        raise ValueError(f"category không hợp lệ. Cho phép: {', '.join(sorted(CATEGORIES))}")
    node_id = str(uuid.uuid4())
    with get_neo4j_session() as session:
        result = session.run(
            """
            CREATE (n:OntologyNode {
              id: $id, name: $name, category: $category, type: $type,
              description: $description, createdAt: datetime()
            })
            RETURN n
            """,
            id=node_id, name=name, category=category, type=type_ or category, description=description or "",
        )
        return _clean_props(dict(result.single()["n"]))


def update_node(
    db: Session, node_id: str, name: str | None, category: str | None, type_: str | None, description: str | None,
) -> dict | None:
    if category and category not in CATEGORIES:
        raise ValueError(f"category không hợp lệ. Cho phép: {', '.join(sorted(CATEGORIES))}")
    with get_neo4j_session() as session:
        result = session.run(
            """
            MATCH (n:OntologyNode {id: $id})
            SET n.name = coalesce($name, n.name),
                n.category = coalesce($category, n.category),
                n.type = coalesce($type, n.type),
                n.description = coalesce($description, n.description)
            RETURN n
            """,
            id=node_id, name=name, category=category, type=type_, description=description,
        )
        record = result.single()
        if not record:
            return None
        props = _clean_props(dict(record["n"]))

    if name and props.get("sourceType"):
        _reverse_sync_rename(db, props["sourceType"], props["sourceId"], name)
    return props


def delete_node(node_id: str) -> None:
    # Deliberately not reverse-synced — see module docstring.
    with get_neo4j_session() as session:
        session.run("MATCH (n:OntologyNode {id: $id}) DETACH DELETE n", id=node_id)


def create_relationship(db: Session, from_id: str, to_id: str, rel_type: str) -> dict:
    if not from_id or not to_id or not rel_type:
        raise ValueError("fromId, toId, relType là bắt buộc")
    if rel_type not in REL_TYPES:
        raise ValueError(f"relType không hợp lệ. Cho phép: {', '.join(sorted(REL_TYPES))}")
    rel_id = str(uuid.uuid4())
    with get_neo4j_session() as session:
        result = session.run(
            f"""
            MATCH (a:OntologyNode {{id: $fromId}}), (b:OntologyNode {{id: $toId}})
            CREATE (a)-[r:{rel_type} {{id: $id}}]->(b)
            RETURN r
            """,
            fromId=from_id, toId=to_id, id=rel_id,
        )
        if result.single() is None:
            raise LookupError("Không tìm thấy node nguồn hoặc đích")
        if rel_type == "APPLIES_TO":
            _reverse_sync_scope_edge(db, session, from_id, to_id, "add")
    return {"id": rel_id, "type": rel_type, "source": from_id, "target": to_id}


def delete_relationship(db: Session, rel_id: str) -> None:
    with get_neo4j_session() as session:
        info = session.run(
            "MATCH (a:OntologyNode)-[r {id: $id}]->(b:OntologyNode) "
            "RETURN type(r) AS relType, a.id AS fromId, b.id AS toId",
            id=rel_id,
        )
        record = info.single()
        session.run("MATCH ()-[r {id: $id}]->() DELETE r", id=rel_id)
        if record and record["relType"] == "APPLIES_TO":
            _reverse_sync_scope_edge(db, session, record["fromId"], record["toId"], "remove")


def wipe_graph() -> None:
    with get_neo4j_session() as session:
        session.run("MATCH (n:OntologyNode) DETACH DELETE n")


# ---- reverse sync: graph edit -> real MySQL write -------------------------

def _fetch_node_source(session, node_id: str) -> tuple[str, str] | None:
    result = session.run(
        "MATCH (n:OntologyNode {id:$id}) RETURN n.sourceType AS sourceType, n.sourceId AS sourceId",
        id=node_id,
    )
    record = result.single()
    if not record or not record["sourceType"]:
        return None
    return record["sourceType"], record["sourceId"]


def _reverse_sync_rename(db: Session, source_type: str, source_id: str, name: str) -> None:
    from app.models.entity_policy_rule import EntityPolicyRule
    from app.models.org_unit import OrgUnit
    from app.models.org_unit_instance import OrgUnitInstance
    from app.models.position import Position
    from app.models.user import User

    name = name.strip()
    if not name:
        return

    if source_type == "entity_policy_rule":
        row = db.get(EntityPolicyRule, source_id)
        if row:
            row.display_name = name
            db.commit()
            sync_engine.ontology_sync_service.sync_entity_rule(db, row)
    elif source_type == "org_unit":
        row = db.get(OrgUnit, source_id)
        if row:
            row.name = name
            db.commit()
            sync_engine.ontology_sync_service.sync_org_unit(row)
    elif source_type == "org_unit_instance":
        row = db.get(OrgUnitInstance, source_id)
        if row:
            row.name = name
            db.commit()
            sync_engine.ontology_sync_service.sync_org_unit_instance(row, db)
    elif source_type == "position_instance":
        row = db.get(Position, source_id.split(":", 1)[0])
        if row:
            row.name = name
            db.commit()
            sync_engine.ontology_sync_service.sync_position(row)
    elif source_type == "user_assignment":
        row = db.get(User, source_id.split(":", 1)[0])
        if row:
            row.name = name
            db.commit()
    # entity_policy_rule_tier, org_unit_scope, and static seed classes are
    # not renamed this way — their names are always derived, not editable.


# The one edge type this graph lets a user meaningfully change that maps
# onto a specific MySQL field: an APPLIES_TO edge from a per-entity
# Allow/Mask policy node onto an org-unit-instance or position-instance
# node round-trips onto one element of entity_policy_rule.allow_scope_pairs
# / .mask_scope_pairs. Anything else (block, the static "Scope" root used
# for an ALL pair, freeform edges) is intentionally left graph-only.
def _reverse_sync_scope_edge(db: Session, neo4j_session, from_id: str, to_id: str, action: str) -> None:
    from_source = _fetch_node_source(neo4j_session, from_id)
    if not from_source or from_source[0] != "entity_policy_rule_tier":
        return
    rule_id, tier = from_source[1].split(":", 1)
    if tier not in ("allow", "mask"):
        return  # block has no editable scope

    to_source = _fetch_node_source(neo4j_session, to_id)
    if not to_source:
        return
    to_type, to_source_id = to_source
    if to_type == "org_unit_instance":
        oui_id, position_id = to_source_id, None
    elif to_type == "position_instance":
        position_id, oui_id = to_source_id.split(":", 1)
    else:
        return

    from app.models.entity_policy_rule import EntityPolicyRule
    from app.services.policy_rule_service import policy_rule_service

    rule = db.get(EntityPolicyRule, rule_id)
    if not rule:
        return
    field = "allow_scope_pairs" if tier == "allow" else "mask_scope_pairs"
    pairs = [dict(p) for p in (getattr(rule, field) or [])]
    pair = {"oui_id": oui_id, "position_id": position_id}
    if action == "add":
        if pair not in pairs:
            pairs.append(pair)
    else:
        pairs = [p for p in pairs if p != pair]

    setattr(rule, field, pairs)
    policy_rule_service.bump_version(db)
    db.commit()
    db.refresh(rule)
    # Re-push so the graph gets normalized back (dedupe, GRANTS edges, etc.)
    # rather than trusting the single edit was already fully consistent.
    sync_engine.ontology_sync_service.sync_entity_rule(db, rule)
