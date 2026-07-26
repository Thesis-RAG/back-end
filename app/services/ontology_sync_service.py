"""Push policy/org changes into the ontology graph (Neo4j), keeping it a
live mirror of real data. Best-effort: a Neo4j hiccup is logged and
swallowed, never raised — the graph is a visualization layered on top of
the real system, not the source of truth, and must never be able to fail a
real policy/org/document write. Route handlers call these after their own
db.commit(), never before, so a sync failure can't roll back real data.

Ported from what used to be a separate Node/Express service's routes/
sync.js — now the same in-process Neo4j session as everything else, so no
network hop and no shared-secret auth between services. The reverse
direction (a user editing the graph UI, which should update real data) is
handled by app/services/ontology_graph_service.py, in the same process.
"""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from app.db.neo4j_client import get_session
from app.services.policy_rule_service import ALL_SENTINEL

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.models.entity_policy_rule import EntityPolicyRule
    from app.models.org_unit import OrgUnit
    from app.models.org_unit_instance import OrgUnitInstance
    from app.models.position import Position
    from app.models.user import User

logger = logging.getLogger(__name__)

# Kept in sync by hand with app/services/ontology_graph_service.py's whitelist.
REL_TYPES = {
    "BELONGS_TO", "PARTICIPATES_IN", "ATTENDS", "CONTAINS",
    "PROTECTED_BY", "APPLIES_TO", "GRANTS", "RESULTS_IN", "IS_A", "IS_A_ROLE_OF",
}

ENTITY_CLASS_BY_SENSITIVITY = {
    1: "PublicEntity", 2: "InternalEntity", 3: "ConfidentialEntity", 4: "SecretEntity", 5: "TopSecretEntity",
}
# No "block" entry — block is never materialized as a graph node/edge at
# all (see the module docstring's note in sync_entity_rule).
GRANT_PERMS = {
    "allow": ["AllowEntity"],
    "mask": ["MaskEntity"],
}
# Per-entity instance nodes are named "Allow"/"Mask"; the static class
# they IS_A into is named "AllowPolicy"/"MaskPolicy" — swapped from the
# other way around per feedback. find_by_name() only ever matches static
# nodes (sourceType IS NULL, see its query), so instance nodes sharing a
# name with a class is never actually ambiguous — safe either way round.
POLICY_LABEL = {"allow": "Allow", "mask": "Mask"}
# Class nodes' actual `name` property in Neo4j (seed_ontology.py stores
# them as ("AllowPolicyClass", "AllowPolicy", "Policy", "PolicyClass") —
# "AllowPolicyClass" is only the seed script's own internal lookup key).
POLICY_CLASS = {"allow": "AllowPolicy", "mask": "MaskPolicy"}


# ---- low-level graph helpers ---------------------------------------------

def find_by_source(session, source_type: str, source_id: str) -> str | None:
    result = session.run(
        "MATCH (n:OntologyNode {sourceType:$sourceType, sourceId:$sourceId}) RETURN n.id AS id",
        sourceType=source_type, sourceId=source_id,
    )
    record = result.single()
    return record["id"] if record else None


# Static class/taxonomy nodes (from the ontology seed) have no sourceType —
# matched by name alone, safe because the seed deliberately keeps every
# static node's name distinct from anything a sync call could produce (eg.
# per-entity policy nodes are named "AllowPolicy", the static class "Allow").
def find_by_name(session, name: str) -> str | None:
    result = session.run(
        "MATCH (n:OntologyNode {name:$name}) WHERE n.sourceType IS NULL RETURN n.id AS id LIMIT 1",
        name=name,
    )
    record = result.single()
    return record["id"] if record else None


def ensure_node(session, source_type: str, source_id: str, name: str, category: str, type_: str, extra: dict | None = None) -> str:
    new_id = str(uuid.uuid4())
    result = session.run(
        """
        MERGE (n:OntologyNode {sourceType:$sourceType, sourceId:$sourceId})
        ON CREATE SET n.id = $newId, n.createdAt = datetime()
        SET n.name = $name, n.category = $category, n.type = $type,
            n.description = coalesce(n.description, ''), n += $extra
        RETURN n.id AS id
        """,
        sourceType=source_type, sourceId=source_id, name=name, category=category,
        type=type_, newId=new_id, extra=extra or {},
    )
    return result.single()["id"]


def ensure_edge(session, from_id: str | None, to_id: str | None, rel_type: str) -> None:
    if not from_id or not to_id:
        return
    if rel_type not in REL_TYPES:
        raise ValueError(f"Unknown rel type {rel_type}")
    session.run(
        f"""
        MATCH (a:OntologyNode {{id:$fromId}}), (b:OntologyNode {{id:$toId}})
        MERGE (a)-[r:{rel_type}]->(b)
        ON CREATE SET r.id = $relId
        """,
        fromId=from_id, toId=to_id, relId=str(uuid.uuid4()),
    )


def ensure_edge_by_name(session, from_id: str | None, to_name: str, rel_type: str) -> None:
    to_id = find_by_name(session, to_name)
    if to_id:
        ensure_edge(session, from_id, to_id, rel_type)


def clear_edges(session, node_id: str | None, rel_type: str) -> None:
    if not node_id:
        return
    if rel_type not in REL_TYPES:
        raise ValueError(f"Unknown rel type {rel_type}")
    session.run(f"MATCH (n:OntologyNode {{id:$nodeId}})-[r:{rel_type}]->() DELETE r", nodeId=node_id)


def delete_node_by_source(session, source_type: str, source_id: str) -> None:
    session.run(
        "MATCH (n:OntologyNode {sourceType:$sourceType, sourceId:$sourceId}) DETACH DELETE n",
        sourceType=source_type, sourceId=source_id,
    )


def grant_from(session, target_id: str | None, tier: str) -> None:
    for perm in GRANT_PERMS.get(tier, []):
        ensure_edge_by_name(session, target_id, perm, "GRANTS")


# A scope pair always names either one or both sides (OrgScopePicker no
# longer allows a fully-empty pair; __ALL__ is handled by the caller). Both
# sides set -> the one Position-instance node already specific to that
# (position, unit) combination — no separate "unit-only" edge needed,
# IS_A_ROLE_OF already carries that. Position-only -> every instance node
# for that position (the one case with no single precise target — legacy
# data only, see policy_rule_service._require_both_sides).
def resolve_pair_targets(session, pair: dict) -> list[str]:
    oui_id = pair.get("oui_id") or None
    position_id = pair.get("position_id") or None
    # The __ALL__ sentinel ("Tất cả" toggle) can show up in *either* tier's
    # pair list — allow's is short-circuited earlier in sync_entity_rule,
    # but mask has no equivalent shortcut, so a mask-ALL pair reaches here
    # as a literal pair needing the same "applies to everyone" resolution.
    if oui_id == ALL_SENTINEL or position_id == ALL_SENTINEL:
        target = find_by_name(session, "Scope")
        return [target] if target else []
    if oui_id and position_id:
        target = find_by_source(session, "position_instance", f"{position_id}:{oui_id}")
        return [target] if target else []
    if oui_id:
        target = find_by_source(session, "org_unit_instance", oui_id)
        return [target] if target else []
    if position_id:
        result = session.run(
            "MATCH (n:OntologyNode {sourceType:'position_instance'}) WHERE n.sourceId STARTS WITH $prefix RETURN n.id AS id",
            prefix=f"{position_id}:",
        )
        return [record["id"] for record in result]
    return []


def ensure_position_instance(session, position_id: str, name: str, instance_id: str) -> str:
    source_id = f"{position_id}:{instance_id}"
    instance_node_id = find_by_source(session, "org_unit_instance", instance_id)
    # Bare position name ("Trưởng chi nhánh") is ambiguous once there's more
    # than one instance of that OU type — fold the unit's own name into the
    # label so it reads unambiguously on the graph without needing to trace
    # the IS_A_ROLE_OF edge first.
    label = name
    if instance_node_id:
        result = session.run("MATCH (n:OntologyNode {id:$id}) RETURN n.name AS name", id=instance_node_id)
        record = result.single()
        if record and record["name"]:
            label = f"{name} ({record['name']})"
    node_id = ensure_node(session, "position_instance", source_id, label, "Scope", "PositionInstance")
    clear_edges(session, node_id, "IS_A_ROLE_OF")
    clear_edges(session, node_id, "IS_A")
    if instance_node_id:
        ensure_edge(session, node_id, instance_node_id, "IS_A_ROLE_OF")
    ensure_edge_by_name(session, node_id, "RoleScope", "IS_A")
    return node_id


class OntologySyncService:
    def sync_entity_rule(self, db: "DbSession", rule: "EntityPolicyRule") -> None:
        from app.services.policy_rule_service import compute_graph_tiers

        try:
            tiers = compute_graph_tiers(db, rule)
            with get_session() as session:
                entity_id = ensure_node(
                    session, "entity_policy_rule", rule.id, rule.display_name, "Entity", "EntityInstance",
                )
                clear_edges(session, entity_id, "IS_A")
                sensitivity = int(rule.sensitivity or 1)
                ensure_edge_by_name(session, entity_id, ENTITY_CLASS_BY_SENSITIVITY.get(sensitivity, "PublicEntity"), "IS_A")

                # Block has no scope list of its own — it's "everyone not in
                # allow or mask", defined by exclusion, so there's no real
                # node it could correctly point APPLIES_TO at without
                # enumerating the whole org minus a few people. A node with
                # no outgoing edge is just clutter, so it isn't materialized
                # in the graph at all (unlike allow/mask, which always get a
                # node once needed — see the loop below).
                delete_node_by_source(session, "entity_policy_rule_tier", f"{rule.id}:block")

                for tier in ("allow", "mask"):
                    spec = tiers.get(tier) or {}
                    tier_source_id = f"{rule.id}:{tier}"
                    if not spec.get("needed"):
                        delete_node_by_source(session, "entity_policy_rule_tier", tier_source_id)
                        continue
                    policy_id = ensure_node(
                        session, "entity_policy_rule_tier", tier_source_id,
                        POLICY_LABEL[tier], "Policy", "PolicyInstance",
                    )
                    clear_edges(session, policy_id, "IS_A")
                    ensure_edge_by_name(session, policy_id, POLICY_CLASS[tier], "IS_A")
                    ensure_edge(session, entity_id, policy_id, "PROTECTED_BY")
                    clear_edges(session, policy_id, "APPLIES_TO")
                    # GRANTS lives on this per-entity Allow/Mask node, not on
                    # whatever it APPLIES_TO — a scope/position target can be
                    # shared by several unrelated entities (one Allow, another
                    # Mask), so granting from the target mixed their signals
                    # together. This node is already unambiguous (one per
                    # entity+tier), so granting from here is always correct.
                    clear_edges(session, policy_id, "GRANTS")
                    grant_from(session, policy_id, tier)

                    if tier == "allow" and spec.get("all"):
                        ensure_edge_by_name(session, policy_id, "Scope", "APPLIES_TO")
                        continue
                    for pair in spec.get("pairs") or []:
                        for target_id in resolve_pair_targets(session, pair):
                            ensure_edge(session, policy_id, target_id, "APPLIES_TO")
        except Exception:
            logger.warning("[ontology-sync] sync_entity_rule failed", exc_info=True)

    def delete_entity_rule(self, rule_id: str) -> None:
        try:
            with get_session() as session:
                for tier in ("allow", "mask"):
                    delete_node_by_source(session, "entity_policy_rule_tier", f"{rule_id}:{tier}")
                delete_node_by_source(session, "entity_policy_rule", rule_id)
        except Exception:
            logger.warning("[ontology-sync] delete_entity_rule failed", exc_info=True)

    def sync_org_unit(self, org_unit: "OrgUnit") -> None:
        try:
            with get_session() as session:
                node_id = ensure_node(session, "org_unit", org_unit.id, org_unit.name, "Organization", "OrgUnitType")
                clear_edges(session, node_id, "IS_A")
                if org_unit.parent_id:
                    parent_id = find_by_source(session, "org_unit", org_unit.parent_id)
                    if parent_id:
                        ensure_edge(session, node_id, parent_id, "IS_A")
                else:
                    ensure_edge_by_name(session, node_id, "Enterprise", "IS_A")
                # Companion Scope-class for this OU type, so its instances
                # have somewhere to IS_A into that a Policy can APPLIES_TO —
                # eg. "Dự án" gets "Dự ánScope", "Chi nhánh" gets "Chi nhánhScope".
                scope_id = ensure_node(
                    session, "org_unit_scope", f"{org_unit.id}:scope", f"{org_unit.name}Scope", "Scope", "ScopeClass",
                )
                clear_edges(session, scope_id, "IS_A")
                ensure_edge_by_name(session, scope_id, "Scope", "IS_A")
        except Exception:
            logger.warning("[ontology-sync] sync_org_unit failed", exc_info=True)

    def delete_org_unit(self, org_unit_id: str) -> None:
        try:
            with get_session() as session:
                delete_node_by_source(session, "org_unit_scope", f"{org_unit_id}:scope")
                delete_node_by_source(session, "org_unit", org_unit_id)
        except Exception:
            logger.warning("[ontology-sync] delete_org_unit failed", exc_info=True)

    def sync_org_unit_instance(self, instance: "OrgUnitInstance", db: "DbSession | None" = None) -> None:
        try:
            with get_session() as session:
                node_id = ensure_node(
                    session, "org_unit_instance", instance.id, instance.name, "Organization", "OrgUnitInstance",
                    extra={"ouId": instance.ou_id},  # read back by sync_position() to fan out per-instance Position nodes
                )
                clear_edges(session, node_id, "IS_A")
                type_node_id = find_by_source(session, "org_unit", instance.ou_id)
                if type_node_id:
                    ensure_edge(session, node_id, type_node_id, "IS_A")
                scope_node_id = find_by_source(session, "org_unit_scope", f"{instance.ou_id}:scope")
                if scope_node_id:
                    ensure_edge(session, node_id, scope_node_id, "IS_A")

                # Real OUI-instance parent tree (oui_parents table — same
                # relation FGA's ancestor_viewer walks), separate from the
                # IS_A edge above (which only points at the *type* taxonomy).
                # This is what lets entity-policy scope matching later expand
                # "my OUI" into "my OUI and everything beneath it" via a
                # graph path query, instead of only exact-matching — closing
                # the gap where a parent-OUI user can already view a child
                # OUI's document (via FGA ancestor_viewer) but every field in
                # it would otherwise hard-block since no scope pair names
                # their own OUI exactly.
                clear_edges(session, node_id, "BELONGS_TO")
                for parent in instance.parents:
                    parent_node_id = find_by_source(session, "org_unit_instance", parent.id)
                    if parent_node_id:
                        ensure_edge(session, node_id, parent_node_id, "BELONGS_TO")

            # A brand-new instance needs a position-instance node for every
            # Position that already exists for its OU type — sync_position()
            # normally does that fan-out, but only runs when a Position
            # itself changes, not when a fresh instance shows up on the
            # other side. Re-running it per existing Position here closes
            # that gap (idempotent — harmless for positions that already
            # covered this instance).
            if db is not None:
                from app.models.position import Position

                for position in db.query(Position).filter(Position.ou_id == instance.ou_id).all():
                    self.sync_position(position)
        except Exception:
            logger.warning("[ontology-sync] sync_org_unit_instance failed", exc_info=True)

    def delete_org_unit_instance(self, instance_id: str) -> None:
        try:
            with get_session() as session:
                session.run(
                    "MATCH (n:OntologyNode {sourceType:'position_instance'}) WHERE n.sourceId ENDS WITH $suffix DETACH DELETE n",
                    suffix=f":{instance_id}",
                )
                session.run(
                    "MATCH (n:OntologyNode {sourceType:'user_assignment'}) WHERE n.sourceId ENDS WITH $suffix DETACH DELETE n",
                    suffix=f":{instance_id}",
                )
                delete_node_by_source(session, "org_unit_instance", instance_id)
        except Exception:
            logger.warning("[ontology-sync] delete_org_unit_instance failed", exc_info=True)

    # A Position row is scoped to an OU *type* — fans out into one node per
    # existing OrgUnitInstance of that type.
    def sync_position(self, position: "Position") -> None:
        try:
            with get_session() as session:
                result = session.run(
                    "MATCH (n:OntologyNode {sourceType:'org_unit_instance', ouId:$ouId}) RETURN n.sourceId AS instanceId",
                    ouId=position.ou_id,
                )
                for record in result:
                    ensure_position_instance(session, position.id, position.name, record["instanceId"])
        except Exception:
            logger.warning("[ontology-sync] sync_position failed", exc_info=True)

    def delete_position(self, position_id: str) -> None:
        try:
            with get_session() as session:
                session.run(
                    "MATCH (n:OntologyNode {sourceType:'position_instance'}) WHERE n.sourceId STARTS WITH $prefix DETACH DELETE n",
                    prefix=f"{position_id}:",
                )
        except Exception:
            logger.warning("[ontology-sync] delete_position failed", exc_info=True)

    # One node per assignment (a user can hold assignments in more than one
    # OUI branch — UserOuiPosition is unique(user_id, oui_id), not unique on
    # user_id alone), so the source key is the (user, oui) pair.
    def sync_user_assignment(self, user: "User", oui_id: str, position_id: str) -> None:
        try:
            with get_session() as session:
                source_id = f"{user.id}:{oui_id}"
                node_id = ensure_node(session, "user_assignment", source_id, user.name or "?", "Employee", "Employee")
                clear_edges(session, node_id, "IS_A")
                position_instance_id = find_by_source(session, "position_instance", f"{position_id}:{oui_id}")
                if position_instance_id:
                    ensure_edge(session, node_id, position_instance_id, "IS_A")
        except Exception:
            logger.warning("[ontology-sync] sync_user_assignment failed", exc_info=True)

    def delete_user_assignment(self, user_id: str, oui_id: str) -> None:
        try:
            with get_session() as session:
                delete_node_by_source(session, "user_assignment", f"{user_id}:{oui_id}")
        except Exception:
            logger.warning("[ontology-sync] delete_user_assignment failed", exc_info=True)

    # Document-type nodes are static (seeded, 20 fixed values) — this only
    # ensures the CONTAINS edge once ingest has actually seen that entity
    # type inside a chunk belonging to a document of that type.
    def sync_document_entity(self, document_type: str, entity_rule_id: str) -> None:
        try:
            with get_session() as session:
                doc_type_id = find_by_name(session, document_type)
                entity_id = find_by_source(session, "entity_policy_rule", entity_rule_id)
                if doc_type_id and entity_id:
                    ensure_edge(session, doc_type_id, entity_id, "CONTAINS")
        except Exception:
            logger.warning("[ontology-sync] sync_document_entity failed", exc_info=True)


ontology_sync_service = OntologySyncService()
