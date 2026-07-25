"""Neo4j driver singleton for the ontology graph.

Merged in-process from what used to be a separate Node/Express service
(enterprise-ontology/backend) — same database, same node/edge shape, just
one fewer process to run and no cross-service auth needed anymore.
"""
from __future__ import annotations

from neo4j import Driver, GraphDatabase, Session

from app.core.config import settings

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


def get_session() -> Session:
    return get_driver().session()
