"""
Low-level HTTP client for the OpenFGA authorization service.
Wraps the /check, /write, /read, and /list-objects REST endpoints.
"""
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# Thin HTTP wrapper around the OpenFGA API; one instance is shared application-wide.
class FGAClient:
    # Initialize a shared synchronous httpx client with a short timeout.
    def __init__(self):
        self._http = httpx.Client(timeout=5.0)

    # Build the base URL for the configured store.
    @property
    def _base(self) -> str:
        return f"{settings.openfga_url}/stores/{settings.openfga_store_id}"

    # Return True if the user has the given relation on the object; False on any error.
    def check(self, user: str, relation: str, object: str, context: dict | None = None) -> bool:
        try:
            body: dict = {"tuple_key": {"user": user, "relation": relation, "object": object}}
            if context:
                body["context"] = context
            resp = self._http.post(f"{self._base}/check", json=body)
            resp.raise_for_status()
            return resp.json().get("allowed", False)
        except Exception:
            logger.exception(
                "OpenFGA check failed: user=%s relation=%s object=%s",
                user,
                relation,
                object,
            )
            return False

    # Write tuples one-by-one; skip 400 responses (duplicate tuple).
    def write(self, tuples: list[dict]) -> None:
        if not tuples:
            return
        for t in tuples:
            try:
                self._http.post(
                    f"{self._base}/write",
                    json={"writes": {"tuple_keys": [t]}},
                ).raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    continue
                raise

    # Delete tuples one-by-one; skip 400 responses (tuple not found).
    def delete(self, tuples: list[dict]) -> None:
        if not tuples:
            return
        for t in tuples:
            # Strip condition field — delete only needs the base tuple key
            base = {k: v for k, v in t.items() if k != "condition"}
            try:
                self._http.post(
                    f"{self._base}/write",
                    json={"deletes": {"tuple_keys": [base]}},
                ).raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    continue
                raise

    # Create a new FGA store and return its ID.
    def create_store(self, name: str) -> str:
        resp = self._http.post(
            f"{settings.openfga_url}/stores",
            json={"name": name},
        )
        resp.raise_for_status()
        return resp.json()["id"]

    # Upload an authorization model and return the new model ID.
    def write_model(self, model: dict) -> str:
        resp = self._http.post(
            f"{self._base}/authorization-models",
            json=model,
        )
        resp.raise_for_status()
        return resp.json()["authorization_model_id"]

    # Return all tuples whose object matches the given object string.
    def read(self, object: str) -> list[dict]:
        resp = self._http.post(
            f"{self._base}/read",
            json={"tuple_key": {"object": object}},
        )
        resp.raise_for_status()
        return [t["key"] for t in resp.json().get("tuples", [])]

    # Return all object IDs of the given type that the user has the given relation on.
    def list_objects(
        self, user: str, relation: str, object_type: str, context: dict | None = None
    ) -> list[str]:
        try:
            body: dict = {"user": user, "relation": relation, "type": object_type}
            if context:
                body["context"] = context
            resp = self._http.post(f"{self._base}/list-objects", json=body)
            resp.raise_for_status()
            return resp.json().get("objects", [])
        except Exception:
            logger.exception(
                "OpenFGA list_objects failed: user=%s relation=%s type=%s",
                user,
                relation,
                object_type,
            )
            return []


# Module-level singleton; shared by FGAAdapter and setup scripts.
fga_client = FGAClient()
