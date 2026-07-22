"""Backward-compatible manual entry point for the idempotent bootstrap."""
import sys

sys.path.insert(0, ".")

from app.db.session import SessionLocal
from app.fga.client import fga_client
from app.services.bootstrap_service import bootstrap_service


def main() -> None:
    fga_client.ensure_configured()
    with SessionLocal() as db:
        bootstrap_service.seed_defaults(db)
    print("Bootstrap completed.")


if __name__ == "__main__":
    main()
