"""
Low-level HTTP client for the OpenFGA authorization service.
Wraps the /check, /write, /read, and /list-objects REST endpoints.
"""
import logging
import json
import threading
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# Thin HTTP wrapper around the OpenFGA API; one instance is shared application-wide.
class FGAClient:
    # Initialize a shared synchronous httpx client with a short timeout.
    def __init__(self):
        self._http = httpx.Client(timeout=5.0)
        self._configure_lock = threading.Lock()
        self._verified_config: tuple[str, str] | None = None

    def _config_path(self) -> Path:
        return Path(settings.openfga_config_file)

    def _load_persisted_config(self) -> None:
        configured_store_id = str(settings.openfga_store_id or "").strip()
        configured_model_id = str(settings.openfga_model_id or "").strip()
        try:
            data = json.loads(self._config_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            settings.openfga_store_id = configured_store_id
            settings.openfga_model_id = configured_model_id
            return

        persisted_store_id = str(data.get("store_id") or "").strip()
        persisted_model_id = str(data.get("model_id") or "").strip()
        settings.openfga_store_id = configured_store_id or persisted_store_id
        settings.openfga_model_id = configured_model_id or persisted_model_id

    def _persist_config(self) -> None:
        path = self._config_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "store_id": settings.openfga_store_id,
                        "model_id": settings.openfga_model_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            # Read-only API/worker mounts are expected after bootstrap. Their
            # settings are still populated from the shared file or environment.
            logger.debug("OpenFGA config file is not writable: %s", path, exc_info=True)

    # Build the base URL for the configured store.
    @property
    def _base(self) -> str:
        return f"{settings.openfga_url}/stores/{settings.openfga_store_id}"

    def list_stores(self) -> list[dict[str, Any]]:
        resp = self._http.get(f"{settings.openfga_url}/stores")
        resp.raise_for_status()
        return resp.json().get("stores", [])

    def list_authorization_models(self) -> list[dict[str, Any]]:
        self._load_persisted_config()
        resp = self._http.get(f"{self._base}/authorization-models")
        resp.raise_for_status()
        return resp.json().get("authorization_models", [])

    def ensure_configured(self) -> tuple[str, str]:
        """Ensure a reusable store/model exists and load its IDs.

        Production uses a shared config volume written by the bootstrap service.
        The fallback discovery path also makes local/dev startup self-contained.
        """
        self._load_persisted_config()
        current_config = (settings.openfga_store_id, settings.openfga_model_id)
        if all(current_config) and self._verified_config == current_config:
            return settings.openfga_store_id, settings.openfga_model_id

        with self._configure_lock:
            self._load_persisted_config()

            # The OpenFGA database and the shared config volume may be removed
            # independently during a deployment. Verify persisted IDs once per
            # process so stale IDs are repaired automatically instead of
            # leaving the API pointed at a store that no longer exists.
            if settings.openfga_store_id:
                configured_store = settings.openfga_store_id
                stores = self.list_stores()
                store_exists = any(
                    str(store.get("id") or "") == configured_store for store in stores
                )
                if not store_exists:
                    logger.warning(
                        "Persisted OpenFGA store is stale; provisioning a new configuration"
                    )
                    settings.openfga_store_id = ""
                    settings.openfga_model_id = ""
                elif settings.openfga_model_id:
                    model_ids = {
                        str(model.get("id") or model.get("authorization_model_id") or "")
                        for model in self.list_authorization_models()
                    }
                    if settings.openfga_model_id in model_ids:
                        self._verified_config = (
                            settings.openfga_store_id,
                            settings.openfga_model_id,
                        )
                        return settings.openfga_store_id, settings.openfga_model_id
                    logger.warning(
                        "Persisted OpenFGA model is stale; selecting a valid model"
                    )
                    settings.openfga_model_id = ""

            if not settings.openfga_store_id:
                stores = self.list_stores()
                existing = next(
                    (store for store in stores if store.get("name") == settings.openfga_store_name),
                    None,
                )
                if existing is None:
                    # Older deployments used names such as
                    # ``rag-enterprise-6``. Reuse a single legacy store so
                    # existing document tuples are not silently orphaned.
                    legacy_stores = [
                        store
                        for store in stores
                        if str(store.get("name") or "").startswith(
                            f"{settings.openfga_store_name}-"
                        )
                    ]
                    if len(legacy_stores) == 1:
                        existing = legacy_stores[0]
                        logger.warning(
                            "Reusing legacy OpenFGA store name=%s",
                            existing.get("name"),
                        )
                if existing and existing.get("id"):
                    settings.openfga_store_id = str(existing["id"])
                else:
                    settings.openfga_store_id = self.create_store(settings.openfga_store_name)
                # A model ID belongs to a specific store and cannot be reused
                # after creating or discovering a different store.
                settings.openfga_model_id = ""

            if not settings.openfga_model_id:
                models = self.list_authorization_models()
                if models:
                    model_id = models[0].get("id") or models[0].get("authorization_model_id")
                    if model_id:
                        settings.openfga_model_id = str(model_id)
                if not settings.openfga_model_id:
                    model_path = Path(__file__).with_name("model.json")
                    model = json.loads(model_path.read_text(encoding="utf-8"))
                    settings.openfga_model_id = self.write_model(model)

            self._persist_config()
            self._verified_config = (
                settings.openfga_store_id,
                settings.openfga_model_id,
            )

        logger.info(
            "OpenFGA configured store=%s model=%s",
            settings.openfga_store_id,
            settings.openfga_model_id,
        )
        return settings.openfga_store_id, settings.openfga_model_id

    def _authorization_model_id(self, body: dict[str, Any]) -> dict[str, Any]:
        self.ensure_configured()
        if settings.openfga_model_id:
            body["authorization_model_id"] = settings.openfga_model_id
        return body

    # Return True if the user has the given relation on the object; False on any error.
    def check(self, user: str, relation: str, object: str, context: dict | None = None) -> bool:
        try:
            body: dict[str, Any] = {"tuple_key": {"user": user, "relation": relation, "object": object}}
            if context:
                body["context"] = context
            self._authorization_model_id(body)
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
        self.ensure_configured()
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
        self.ensure_configured()
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
        self.ensure_configured()
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
            body: dict[str, Any] = {"user": user, "relation": relation, "type": object_type}
            if context:
                body["context"] = context
            self._authorization_model_id(body)
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
