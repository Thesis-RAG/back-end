"""
Hybrid entity extraction pipeline with three layers:
  Layer 1 (Regex)  — structured PII (email, phone, national_id, ...)
  Layer 2 (GLiNER) — free-text entities from the shared baseline and file actions
  Layer 3 (Rule)   — boolean summary labels and chunk sensitivity scoring

Boolean flags (fixed vocabulary used for sensitivity metadata):
  has_pii        — personal identifiable information
  has_financial  — financial / quantitative business data
  has_credential — authentication secrets (passwords, tokens, API keys)
  has_legal      — legal / regulatory / contractual content
  has_strategic  — strategic / competitive plans
  has_hr         — HR-specific sensitive data (salary, employment records)

Entity type → flag mapping:
  Regex-detected types → hardcoded _BUILTIN_ENTITY_FLAGS
  GLiNER-detected types → loaded from document entity-action metadata
  Both are merged into a single TTL cache refreshed every 5 minutes.
"""
from __future__ import annotations

import logging
import re
import threading
import time

logger = logging.getLogger(__name__)

# ── Regex patterns (Layer 1) ──────────────────────────────────────────────────

REGEX_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b0\d{2,3}[.\s]?\d{3}[.\s]?\d{3,4}\b"),
    "national_id": re.compile(
        r"(?:CCCD|CMND|Số CMND/CCCD|Số CMND|Số CCCD)[:\s/]*?(\d{9}|\d{12})", re.IGNORECASE
    ),
    "tax_id": re.compile(r"(?:mã số thuế)[^\d]{0,15}(\d{10,13})", re.IGNORECASE),
    "social_insurance": re.compile(
        r"(?:số BHXH|số sổ BHXH)[:\s]*?([A-Z]{2}\d{8,12}|\d{8,12})", re.IGNORECASE
    ),
    "bank_account": re.compile(r"(?:tài khoản|TK)[^\d]{0,20}(\d{9,16})", re.IGNORECASE),
    "dob": re.compile(
        r"(?:ngày sinh|sinh ngày|DOB)[:\s]*?(\d{1,2}/\d{1,2}/\d{2,4})", re.IGNORECASE
    ),
    "date_generic": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    "money": re.compile(r"\b[\d.,]+\s?(?:VND|đồng|VNĐ)\b", re.IGNORECASE),
    "percentage": re.compile(r"\b\d{1,3}\s?%"),
    "credential": re.compile(
        r"(?i)\b(?:password|mật khẩu|api[_\s-]?key|token|secret|otp)\b\s*[:=]\s*[^\s,;]+"
    ),
}

# ── Builtin flags for regex-detected entity types ─────────────────────────────
# These types are captured by Layer 1 regex patterns (not from DB domains),
# so their flag mapping is hardcoded here.

_BUILTIN_ENTITY_FLAGS: dict[str, list[str]] = {
    "email":            ["has_pii"],
    "phone":            ["has_pii"],
    "national_id":      ["has_pii", "has_hr"],
    "tax_id":           ["has_pii", "has_hr"],
    "social_insurance": ["has_pii", "has_hr"],
    "bank_account":     ["has_pii", "has_financial"],
    "dob":              ["has_pii", "has_hr"],
    "money":            ["has_financial"],
    "percentage":       ["has_financial"],
    "date_generic":     [],
    "credential":       ["has_credential"],
}

# Available before a newly uploaded file has persisted action rows. This is
# the shared baseline used by upload preview, ingestion, and query detection.
_DEFAULT_GLINER_LABELS: tuple[str, ...] = (
    "person_name", "organization", "address", "salary", "money", "project",
    "contract", "email", "phone", "account_number", "credential", "date",
)

_DEFAULT_GLINER_FLAGS: dict[str, list[str]] = {
    "person_name": ["has_pii"],
    "address": ["has_pii"],
    "salary": ["has_financial", "has_hr"],
    "project": ["has_strategic"],
    "contract": ["has_legal"],
    "account_number": ["has_pii", "has_financial"],
}

# Only these entity types are safe enough to justify lowering a chunk's
# sensitivity. This is intentionally a very small allowlist: a missing
# sensitive entity must never turn a chunk into a less protected one.
COMMON_PUBLIC_ENTITY_TYPES = frozenset({"date", "date_generic"})
SENSITIVITY_PROTECTIVE_FLAGS = frozenset({
    "has_pii", "has_financial", "has_credential", "has_legal",
    "has_strategic", "has_hr",
})

# Keyword-based augmentation: catches in-text patterns GLiNER might miss
_CREDENTIAL_RE = re.compile(r"(?i)\b(mật khẩu|password|api[_\s]?key|token|secret|otp)\b")
_LEGAL_RE      = re.compile(r"(?i)\b(nghị định|thông tư|điều\s+\d+|luật|hợp đồng|quyết định số)\b")
_STRATEGIC_RE  = re.compile(r"(?i)\b(chiến lược|kế hoạch mở rộng|sáp nhập|m&a|định hướng|roadmap)\b")

# ── Sensitivity scoring ───────────────────────────────────────────────────────

_FLAG_SENSITIVITY_WEIGHTS: dict[str, int] = {
    "has_credential": 2,
    "has_pii":        1,
    "has_hr":         1,
    "has_strategic":  1,
    "has_financial":  1,
    "has_legal":      0,
}


# Derive chunk-level sensitivity from doc sensitivity and detected boolean flags; result clamped to [1, 5].
def compute_chunk_sensitivity(doc_sensitivity: int, labels: dict[str, bool]) -> int:
    if not any(labels.values()):
        # No entity evidence means no downgrade.  Lowering is handled by
        # adjust_chunk_sensitivity and requires the strict public allowlist.
        delta = 0
    else:
        raw = sum(_FLAG_SENSITIVITY_WEIGHTS.get(f, 0) for f, v in labels.items() if v)
        delta = min(raw, 2)
    return max(1, min(5, doc_sensitivity + delta))


def _normalize_entity_label(value: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def has_common_public_entity(
    entities: list[dict] | None,
    flags: set[str] | dict[str, bool] | None = None,
) -> bool:
    """Return whether a chunk is safe enough for the one-level reduction."""
    detected = [
        entity for entity in (entities or [])
        if _normalize_entity_label(entity.get("label"))
    ]
    if not detected:
        return False

    labels = {_normalize_entity_label(entity.get("label")) for entity in detected}
    if not labels.issubset(COMMON_PUBLIC_ENTITY_TYPES):
        return False
    # The downgrade must be based on GLiNER evidence, not only a regex hit.
    if not any(str(entity.get("source") or "").lower() == "gliner" for entity in detected):
        return False

    if isinstance(flags, dict):
        active_flags = {key for key, value in flags.items() if value}
    else:
        active_flags = set(flags or set())
    return not (active_flags & SENSITIVITY_PROTECTIVE_FLAGS)


def adjust_chunk_sensitivity(
    doc_sensitivity: int,
    original_chunk_sensitivity: int | None,
    entities: list[dict] | None = None,
    flags: set[str] | dict[str, bool] | None = None,
) -> int:
    """Apply GLiNER lowering without weakening the existing raise logic.

    The original value is produced by the existing chunker/LLM logic. Values
    above the document level are preserved exactly. Values below or equal to
    the document level are lowered only for the strict common-public-entity
    case; otherwise they are restored to the document level.
    """
    document_level = max(1, min(5, int(doc_sensitivity or 1)))
    original = document_level if original_chunk_sensitivity is None else max(
        1, min(5, int(original_chunk_sensitivity))
    )

    if original > document_level:
        return original
    if has_common_public_entity(entities, flags):
        return max(1, document_level - 1)
    return document_level


# ── Combined entity cache (labels + flag mapping) ─────────────────────────────
# One DB call serves both Layer 2 (GLiNER label list) and Layer 3 (entity→flags map).

_cache: dict[str, list] = {}
_cache_ts: float = 0.0
_cache_lock: threading.Lock = threading.Lock()
_CACHE_TTL = 300.0  # 5 minutes


# Return (active_gliner_labels, entity_flags_map) from the TTL cache, refreshing from DB when stale.
def _refresh_cache(db=None) -> tuple[list[str], dict[str, list[str]]]:
    global _cache, _cache_ts
    now = time.monotonic()

    if now - _cache_ts < _CACHE_TTL and _cache:
        return _cache["labels"], _cache["flags"]

    if db is None:
        return (
            _cache.get("labels", list(_DEFAULT_GLINER_LABELS)),
            _cache.get("flags", {**_BUILTIN_ENTITY_FLAGS, **_DEFAULT_GLINER_FLAGS}),
        )

    with _cache_lock:
        # Double-check after acquiring lock
        if now - _cache_ts < _CACHE_TTL and _cache:
            return _cache["labels"], _cache["flags"]
        try:
            from app.models.document_entity_action import DocumentEntityAction
            rows = (
                db.query(DocumentEntityAction.entity_type, DocumentEntityAction.metadata_json)
                .filter(DocumentEntityAction.enabled.is_(True))
                .all()
            )

            labels = sorted({*(_DEFAULT_GLINER_LABELS), *(str(entity_type) for entity_type, _ in rows if entity_type)})
            db_flags = {
                str(entity_type): list((metadata or {}).get("boolean_labels") or [])
                for entity_type, metadata in rows
                if entity_type
            }

            # Builtins are baseline; DB values override if same key exists
            combined_flags = {**_BUILTIN_ENTITY_FLAGS, **_DEFAULT_GLINER_FLAGS, **db_flags}

            _cache = {"labels": labels, "flags": combined_flags}
            _cache_ts = now
            logger.debug(
                "Entity cache refreshed: %d GLiNER labels, %d flag mappings",
                len(labels), len(combined_flags),
            )
        except Exception as exc:
            logger.warning("Failed to refresh entity cache: %s", exc)

    return (
        _cache.get("labels", list(_DEFAULT_GLINER_LABELS)),
        _cache.get("flags", {**_BUILTIN_ENTITY_FLAGS, **_DEFAULT_GLINER_FLAGS}),
    )


# Force a cache refresh on the next call; invoke after creating or deleting entity types.
def invalidate_label_cache() -> None:
    global _cache_ts
    _cache_ts = 0.0


# ── GLiNER lazy loader ────────────────────────────────────────────────────────

_gliner_model = None
_gliner_lock  = threading.Lock()


# Lazily load and cache the GLiNER model; returns None if loading fails.
def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        with _gliner_lock:
            if _gliner_model is None:
                try:
                    from gliner import GLiNER
                    from app.core.config import settings
                    logger.info("Loading GLiNER model %s ...", settings.gliner_model_name)
                    _gliner_model = GLiNER.from_pretrained(settings.gliner_model_name)
                    logger.info("GLiNER model loaded.")
                except Exception as exc:
                    logger.error("Failed to load GLiNER: %s", exc)
                    _gliner_model = None
    return _gliner_model


# ── Layer 1: Regex extraction ─────────────────────────────────────────────────

# Extract structured PII entities from text using regex patterns (Layer 1).
def extract_structured_entities(text: str) -> list[dict]:
    results = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            value = m.group(1) if m.groups() else m.group(0)
            results.append({
                "text":   value,
                "label":  label,
                "start":  m.start(1) if m.groups() else m.start(),
                "end":    m.end(1)   if m.groups() else m.end(),
                "score":  1.0,
                "source": "regex",
            })
    return results


# ── Layer 2: GLiNER extraction ────────────────────────────────────────────────

# Extract free-text entities using GLiNER with the given label list (Layer 2).
def extract_freetext_entities(text: str, labels: list[str], threshold: float = 0.3) -> list[dict]:
    if not labels:
        return []
    model = _get_gliner()
    if model is None:
        return []
    try:
        raw = model.predict_entities(text, labels, threshold=threshold)
        for e in raw:
            e["source"] = "gliner"
        return raw
    except Exception as exc:
        logger.error("GLiNER inference error: %s", exc)
        return []


# ── Layer 3: Boolean labels ───────────────────────────────────────────────────

# Map entity types to boolean flags and augment with keyword rules for patterns GLiNER may miss (Layer 3).
def detect_boolean_labels(
    text: str, all_entities: list[dict], entity_flags: dict[str, list[str]]
) -> dict[str, bool]:
    active_flags: set[str] = set()

    for entity in all_entities:
        for flag in entity_flags.get(entity["label"], []):
            active_flags.add(flag)

    if _CREDENTIAL_RE.search(text):
        active_flags.add("has_credential")
    if _LEGAL_RE.search(text):
        active_flags.add("has_legal")
    if _STRATEGIC_RE.search(text):
        active_flags.add("has_strategic")

    return {
        "has_pii":        "has_pii"        in active_flags,
        "has_financial":  "has_financial"  in active_flags,
        "has_credential": "has_credential" in active_flags,
        "has_legal":      "has_legal"      in active_flags,
        "has_strategic":  "has_strategic"  in active_flags,
        "has_hr":         "has_hr"         in active_flags,
    }


# ── Realtime extraction (retrieval time) ──────────────────────────────────────

# Run GLiNER on a single text. Returns (entities, detected_entity_types).
def extract_realtime(
    text: str,
    *,
    db=None,
    threshold: float = 0.3,
) -> tuple[list[dict], set[str]]:
    gliner_labels, _ = _refresh_cache(db)
    entities = extract_freetext_entities(text, gliner_labels, threshold=threshold)
    return entities, {e["label"] for e in entities}


# Run GLiNER on multiple texts in one sequential batch — avoids model reload overhead.
# Returns one set[str] of detected entity types per text.
def extract_realtime_batch(
    texts: list[str],
    *,
    db=None,
    threshold: float = 0.3,
) -> list[set[str]]:
    if not texts:
        return []
    gliner_labels, _ = _refresh_cache(db)
    if not gliner_labels:
        return [set() for _ in texts]
    model = _get_gliner()
    if model is None:
        return [set() for _ in texts]

    results: list[set[str]] = []
    try:
        for text in texts:
            raw = model.predict_entities(text, gliner_labels, threshold=threshold)
            results.append({e["label"] for e in raw})
    except Exception as exc:
        logger.error("GLiNER batch inference error: %s", exc)
        while len(results) < len(texts):
            results.append(set())
    return results


def extract_realtime_batch_detailed(
    texts: list[str],
    *,
    db=None,
    threshold: float = 0.3,
) -> list[dict]:
    """Return structured entities, GLiNER entities, types and boolean flags.

    Policy enforcement needs spans, not just a set of labels, in order to mask
    only the salary/PII field inside a mixed chunk.  This keeps the existing
    lightweight batch API intact and adds a detailed variant for that path.
    """
    if not texts:
        return []
    gliner_labels, entity_flags = _refresh_cache(db)
    model = _get_gliner() if gliner_labels else None
    results: list[dict] = []
    for text in texts:
        structured = extract_structured_entities(text)
        freetext: list[dict] = []
        if model is not None:
            try:
                freetext = model.predict_entities(text, gliner_labels, threshold=threshold)
                for entity in freetext:
                    entity["source"] = "gliner"
            except Exception as exc:
                logger.debug("GLiNER detailed extraction failed: %s", exc)
        entities = structured + freetext
        for entity in entities:
            entity["flags"] = list(entity_flags.get(entity.get("label"), []))
        labels = detect_boolean_labels(text, entities, entity_flags)
        results.append({
            "entities": entities,
            "entity_types": {str(e.get("label")) for e in entities if e.get("label")},
            "flags": {key for key, value in labels.items() if value},
        })
    return results


# ── Pipeline ──────────────────────────────────────────────────────────────────

# Run all three extraction layers and return entities, boolean labels, and deduplicated entity types.
def run_pipeline(
    text: str,
    *,
    db=None,
    gliner_threshold: float = 0.3,
) -> dict:
    gliner_labels, entity_flags = _refresh_cache(db)

    structured = extract_structured_entities(text)
    freetext   = extract_freetext_entities(text, gliner_labels, threshold=gliner_threshold)
    all_entities = structured + freetext

    booleans = detect_boolean_labels(text, all_entities, entity_flags)

    return {
        "entities":     all_entities,
        "labels":       booleans,
        "entity_types": list({e["label"] for e in all_entities}),
    }
