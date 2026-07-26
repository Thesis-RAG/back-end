"""
Digital signature service for document version content — Ed25519.

Threat model this protects against: an attacker who compromises the MySQL
database directly (leaked credentials, SQL injection, a rogue DBA) and
silently rewrites already-ingested chunk content without going through the
app's upload/approval flow. The signing key lives in a file outside the
database (settings.document_signing_key_path), so tampering with DB rows
alone can no longer produce a signature that still verifies.

This does NOT protect against: a legitimate user uploading malicious content
through the normal upload flow — the signature would faithfully authenticate
that content, since it proves origin/integrity, not that the content is
benign (that's an authorization/content-moderation problem, see the ingest
approval gate). It also does not protect against an attacker with full
host/container filesystem access, who could read this same key file —
a proper deployment would move the key into an HSM/KMS instead.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentSigningService:
    def __init__(self) -> None:
        self._private_key: Ed25519PrivateKey | None = None
        self._key_id: str | None = None

    # Load the signing key from disk, generating+persisting a new one on
    # first use. Not thread-safe against concurrent first-run key creation,
    # but that race only matters once, at fresh-deploy time.
    def _load_or_create_key(self) -> Ed25519PrivateKey:
        if self._private_key is not None:
            return self._private_key

        path = settings.document_signing_key_path
        if os.path.exists(path):
            with open(path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
            return self._private_key

        os.makedirs(os.path.dirname(path), exist_ok=True)
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(path, "wb") as f:
            f.write(pem)
        os.chmod(path, 0o600)
        logger.warning("Generated new document-signing key at %s", path)
        self._private_key = key
        return key

    def _public_key(self) -> Ed25519PublicKey:
        return self._load_or_create_key().public_key()

    # Short fingerprint of the current public key, stored alongside each
    # signature so a future key rotation doesn't silently misattribute old
    # signatures to a new key.
    def key_id(self) -> str:
        if self._key_id is None:
            raw = self._public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            self._key_id = hashlib.sha256(raw).hexdigest()[:16]
        return self._key_id

    # Deterministic aggregate hash over a version's chunk_hash values —
    # order-independent (sorted) so re-querying chunks in a different order
    # still reproduces the same hash.
    @staticmethod
    def compute_content_hash(chunk_hashes: list[str]) -> str:
        joined = "\n".join(sorted(chunk_hashes))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    # Sign a version's content_hash. Returns (signature_b64, key_id, signed_at).
    def sign(self, content_hash: str) -> tuple[str, str, datetime]:
        key = self._load_or_create_key()
        signature = key.sign(content_hash.encode("utf-8"))
        return (
            base64.b64encode(signature).decode("ascii"),
            self.key_id(),
            datetime.now(timezone.utc),
        )

    # Verify a stored signature against a freshly recomputed content_hash.
    # Returns False (never raises) on any mismatch, malformed input, or
    # unknown key_id — callers treat "not verified" as "drop this chunk",
    # not as a crash.
    def verify(self, content_hash: str, signature_b64: str | None, key_id: str | None) -> bool:
        if not signature_b64 or not content_hash:
            return False
        if key_id is not None and key_id != self.key_id():
            logger.warning(
                "Document signature key_id mismatch: current=%s stored=%s — cannot verify (no key rotation registry yet)",
                self.key_id(), key_id,
            )
            return False
        try:
            signature = base64.b64decode(signature_b64)
        except (ValueError, TypeError):
            return False
        try:
            self._public_key().verify(signature, content_hash.encode("utf-8"))
            return True
        except InvalidSignature:
            return False


document_signing_service = DocumentSigningService()
