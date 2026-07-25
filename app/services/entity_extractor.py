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
_FALLBACK_GLINER_LABELS: tuple[str, ...] = (
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

# Keyword-based augmentation: catches in-text patterns GLiNER might miss
_CREDENTIAL_RE = re.compile(r"(?i)\b(mật khẩu|password|api[_\s]?key|token|secret|otp)\b")
_LEGAL_RE      = re.compile(r"(?i)\b(nghị định|thông tư|điều\s+\d+|luật|hợp đồng|quyết định số)\b")
_STRATEGIC_RE  = re.compile(r"(?i)\b(chiến lược|kế hoạch mở rộng|sáp nhập|m&a|định hướng|roadmap)\b")

# ── Sensitivity scoring ───────────────────────────────────────────────────────
# chunk_sensitivity is derived purely from which entity types were detected
# in the chunk and each type's own (action, sensitivity) — not from the old
# LLM per-chunk guess or boolean-flag weights.
#
#   entity_snapshot: {entity_type: {"action": "full"|"mask"|"block", "sensitivity": 1-5}}
#   — one entry per confirmed entity type detected in the chunk. This is the
#   exact shape persisted as chunk metadata_json["entity_policy_snapshot"] so
#   a later doc-level sensitivity change can recompute without re-detecting.
#
# Rules (evaluated in order):
#   1. No entity detected                       -> doc_sensitivity - 1
#   2. Only "full"-action entities detected      -> doc_sensitivity
#   3. Any "mask"/"block"-action entity detected -> max(sensitivity of every
#      detected entity type, including "full" ones present in the same chunk)
#      — independent of doc_sensitivity, so it may end up higher OR lower.
def compute_chunk_sensitivity(doc_sensitivity: int, entity_snapshot: dict[str, dict] | None) -> int:
    document_level = max(1, min(5, int(doc_sensitivity or 1)))
    snapshot = entity_snapshot or {}
    if not snapshot:
        return max(1, document_level - 1)

    actions = {str(info.get("action") or "full") for info in snapshot.values()}
    if actions <= {"full"}:
        return document_level

    levels = [max(1, min(5, int(info.get("sensitivity") or 1))) for info in snapshot.values()]
    return max(1, min(5, max(levels)))


# ── Combined entity cache (labels + flag mapping) ─────────────────────────────
# One DB call serves both Layer 2 (GLiNER label list) and Layer 3 (entity→flags map).

_cache: dict[str, list] = {}
_cache_ts: float = 0.0
_cache_source: str | None = None
_cache_lock: threading.Lock = threading.Lock()
_CACHE_TTL = 300.0  # 5 minutes


# Return (active_gliner_labels, entity_flags_map) from the TTL cache, refreshing from DB when stale.
def _refresh_cache(db=None) -> tuple[list[str], dict[str, list[str]]]:
    global _cache, _cache_ts, _cache_source
    now = time.monotonic()

    requested_source = "db" if db is not None else "fallback"
    if now - _cache_ts < _CACHE_TTL and _cache and _cache_source == requested_source:
        return _cache["labels"], _cache["flags"]

    if db is None:
        _cache_source = "fallback"
        return (
            _cache.get("labels", list(_FALLBACK_GLINER_LABELS)),
            _cache.get("flags", {**_BUILTIN_ENTITY_FLAGS, **_DEFAULT_GLINER_FLAGS}),
        )

    with _cache_lock:
        # Double-check after acquiring lock
        if now - _cache_ts < _CACHE_TTL and _cache and _cache_source == "db":
            return _cache["labels"], _cache["flags"]
        try:
            from app.models.entity_policy_rule import EntityPolicyRule
            rows = (
                db.query(EntityPolicyRule.entity_key, EntityPolicyRule.metadata_json)
                .filter(
                    EntityPolicyRule.policy_profile == "enterprise_secure",
                    EntityPolicyRule.enabled.is_(True),
                )
                .all()
            )

            # The database is the source of truth. No global hard-coded labels
            # are merged when a DB session is available.
            labels = sorted({str(entity_type) for entity_type, _ in rows if entity_type})
            db_flags = {
                str(entity_type): list((metadata or {}).get("boolean_labels") or [])
                for entity_type, metadata in rows
                if entity_type
            }

            # Builtins are baseline; DB values override if same key exists
            combined_flags = {**_BUILTIN_ENTITY_FLAGS, **_DEFAULT_GLINER_FLAGS, **db_flags}

            _cache = {"labels": labels, "flags": combined_flags}
            _cache_ts = now
            _cache_source = "db"
            logger.debug(
                "Entity cache refreshed: %d GLiNER labels, %d flag mappings",
                len(labels), len(combined_flags),
            )
        except Exception as exc:
            logger.warning("Failed to refresh entity cache: %s", exc)

    return (
            _cache.get("labels", list(_FALLBACK_GLINER_LABELS)),
        _cache.get("flags", {**_BUILTIN_ENTITY_FLAGS, **_DEFAULT_GLINER_FLAGS}),
    )


# Force a cache refresh on the next call; invoke after creating or deleting entity types.
def invalidate_label_cache() -> None:
    global _cache_ts, _cache_source
    _cache_ts = 0.0
    _cache_source = None


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
def extract_structured_entities(text: str, allowed_labels: set[str] | None = None) -> list[dict]:
    results = []
    for label, pattern in REGEX_PATTERNS.items():
        if allowed_labels is not None and label not in allowed_labels:
            continue
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


# Run GLiNER's own batched inference() over ALL texts in a single forward
# pass, instead of one model call per text. predict_entities(text, ...) is
# just inference([text], ...)[0] under the hood — calling it once per chunk
# pays the full per-call overhead (tokenize/collate/eval/dispatch) N times
# instead of once, which is what made detection latency scale linearly with
# top_k. Falls back to per-text predict_entities if the installed gliner
# version lacks inference() (older releases only expose predict_entities).
def _batch_predict(model, texts: list[str], labels: list[str], threshold: float) -> list[list[dict]]:
    if hasattr(model, "inference"):
        return model.inference(texts, labels, threshold=threshold, batch_size=max(1, len(texts)))
    return [model.predict_entities(text, labels, threshold=threshold) for text in texts]


# Run GLiNER on multiple texts in one batched forward pass.
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

    try:
        batched = _batch_predict(model, texts, gliner_labels, threshold)
        return [{e["label"] for e in raw} for raw in batched]
    except Exception as exc:
        logger.error("GLiNER batch inference error: %s", exc)
        return [set() for _ in texts]


def extract_realtime_batch_detailed(
    texts: list[str],
    *,
    db=None,
    threshold: float = 0.3,
    labels: list[str] | set[str] | None = None,
) -> list[dict]:
    """Return structured entities, GLiNER entities, types and boolean flags.

    Policy enforcement needs spans, not just a set of labels, in order to mask
    only the salary/PII field inside a mixed chunk.  This keeps the existing
    lightweight batch API intact and adds a detailed variant for that path.
    """
    if not texts:
        return []
    gliner_labels, entity_flags = _refresh_cache(db) if labels is None else (
        sorted(set(labels)), _refresh_cache(db)[1]
    )
    model = _get_gliner() if gliner_labels else None

    # One batched GLiNER call across all chunks (see _batch_predict), then a
    # cheap per-text loop for regex extraction + assembly.
    freetext_by_text: list[list[dict]] = [[] for _ in texts]
    if model is not None:
        try:
            batched = _batch_predict(model, texts, gliner_labels, threshold)
            for i, freetext in enumerate(batched):
                for entity in freetext:
                    entity["source"] = "gliner"
                freetext_by_text[i] = freetext
        except Exception as exc:
            logger.debug("GLiNER batched detailed extraction failed: %s", exc)

    allowed_labels = set(gliner_labels)
    results: list[dict] = []
    for text, freetext in zip(texts, freetext_by_text):
        structured = extract_structured_entities(text, allowed_labels)
        entities = structured + freetext
        for entity in entities:
            entity["flags"] = list(entity_flags.get(entity.get("label"), []))
        boolean_summary = detect_boolean_labels(text, entities, entity_flags)
        results.append({
            "entities": entities,
            "entity_types": {str(e.get("label")) for e in entities if e.get("label")},
            "flags": {key for key, value in boolean_summary.items() if value},
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

    structured = extract_structured_entities(text, set(gliner_labels))
    freetext   = extract_freetext_entities(text, gliner_labels, threshold=gliner_threshold)
    all_entities = structured + freetext

    booleans = detect_boolean_labels(text, all_entities, entity_flags)

    return {
        "entities":     all_entities,
        "labels":       booleans,
        "entity_types": list({e["label"] for e in all_entities}),
    }
