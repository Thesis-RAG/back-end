"""
FastAPI application entry point: middleware, CORS, startup hooks, router registration, and auth dependency.
"""
import logging
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.v1.auth import router as auth_router
from app.api.v1.documents import router as documents_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.health import router as health_router
from app.api.v1.admin import router as admin_router
from app.api.v1.chat import router as chat_router
from app.api.v1.users import router as users_router
from app.api.v1.gmail import router as gmail_router
from app.api.v1.org_units import router as org_units_router
from app.api.v1.policy import router as policy_router
from app.api.v1.settings import router as settings_router
from app.api.v1.document_access_requests import router as access_requests_router
from app.api.v1.ontology_graph import router as ontology_graph_router
from app.core.exceptions import register_exception_handlers
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.init_db import init_db
from app.db.session import SessionLocal, engine, get_db
from app.fga.client import fga_client
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse
from app.schemas.health import HealthResponse
from app.services.auth_service import auth_service
from app.services.bootstrap_service import bootstrap_service
from app.services.document_service import document_service
from app.services.job_service import job_service
from app.services.storage_service import storage_service
from app.workers.ingest_tasks import process_ingest_job


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

configure_logging()
logger = logging.getLogger(__name__)

# Module-level FastAPI application instance; imported by uvicorn as the ASGI entrypoint.
app = FastAPI(title="rag-role-enterprise-api", version="1.0.0")


# Attach a trace ID to every request and echo it in the response headers.
@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    request.state.trace_id = request.headers.get("X-Trace-Id", uuid.uuid4().hex)
    response = await call_next(request)
    response.headers["X-Trace-Id"] = request.state.trace_id
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.cors_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

# Extract the trace ID from request state, generating a fallback UUID if absent.
def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", uuid.uuid4().hex)


# Poll MySQL up to 30 times (2 s apart) until it accepts connections, then raise on timeout.
def wait_for_database():
    for _ in range(30):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("MySQL is not ready")


# Ensure OpenFGA is reachable and has a reusable store/model before seeding tuples.
def wait_for_openfga():
    for attempt in range(30):
        try:
            # Probe the service even when IDs came from the shared config file.
            fga_client.list_stores()
            store_id, model_id = fga_client.ensure_configured()
            logger.info(
                "OpenFGA ready store=%s model=%s",
                store_id,
                model_id,
            )
            return
        except Exception:
            logger.warning(
                "OpenFGA is not ready (attempt %d/30)",
                attempt + 1,
                exc_info=True,
            )
            time.sleep(2)
    raise RuntimeError("OpenFGA did not become ready")


# Seed the ontology graph's static taxonomy and backfill real data into it,
# but only if it's still empty (first run) — re-seeding on every restart
# would wipe any nodes/edges a user added directly through the graph
# editor, since seed_ontology.main() does a full DETACH DELETE first.
# Non-fatal: the ontology graph is a visualization layered on top of the
# real app (see app/services/ontology_sync_service.py's module docstring),
# not a load-bearing dependency — it must never block API startup.
def bootstrap_ontology_graph():
    from app.db.neo4j_client import get_driver, get_session

    for attempt in range(30):
        try:
            get_driver().verify_connectivity()
            break
        except Exception:
            logger.warning("Neo4j is not ready (attempt %d/30)", attempt + 1, exc_info=True)
            time.sleep(2)
    else:
        logger.warning("Neo4j did not become ready — ontology graph will stay unseeded until the next restart")
        return

    try:
        with get_session() as session:
            count = session.run("MATCH (n:OntologyNode) RETURN count(n) AS c").single()["c"]
        if count > 0:
            logger.info("Ontology graph already has %d node(s) — skipping seed/backfill", count)
            return

        logger.info("Ontology graph is empty — seeding taxonomy and backfilling real data")
        from app.scripts import resync_ontology, seed_ontology

        seed_ontology.main()
        with SessionLocal() as db:
            resync_ontology.run(db)
    except Exception:
        logger.warning("Ontology graph bootstrap failed", exc_info=True)


# Wait for MinIO before declaring the API ready.
def wait_for_storage():
    for attempt in range(30):
        try:
            storage_service.ensure_buckets()
            logger.info("MinIO buckets ready")
            return
        except Exception:
            logger.warning(
                "MinIO is not ready (attempt %d/30)",
                attempt + 1,
                exc_info=True,
            )
            time.sleep(2)
    raise RuntimeError("MinIO did not become ready")


# Wait for DB readiness, run migrations, seed defaults, and ensure MinIO buckets exist.
@app.on_event("startup")
def startup_event():
    wait_for_database()
    init_db()
    wait_for_openfga()
    with SessionLocal() as db:
        bootstrap_service.seed_defaults(db)
    wait_for_storage()
    bootstrap_ontology_graph()


# FastAPI dependency: decode the Bearer token and return the active User, raising 401 otherwise.
def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    payload = auth_service.decode_access_token(token)
    user = db.get(User, payload["sub"])
    if not user or user.status != "active":
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])
app.include_router(chat_router, prefix="", tags=["chat"])
app.include_router(users_router, prefix="", tags=["users"])
app.include_router(admin_router, prefix="/audit", tags=["audit"])
app.include_router(org_units_router, prefix="", tags=["org-units"])
app.include_router(gmail_router, prefix="", tags=["gmail"])
app.include_router(policy_router, prefix="/policy", tags=["policy"])
app.include_router(settings_router, prefix="/settings", tags=["settings"])
app.include_router(access_requests_router, prefix="", tags=["access-requests"])
app.include_router(ontology_graph_router, prefix="/ontology-graph", tags=["ontology-graph"])



