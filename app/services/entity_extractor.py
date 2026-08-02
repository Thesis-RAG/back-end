"""
Hybrid entity extraction pipeline with three layers:
  Layer 1 (Regex)       — structured PII (email, phone, national_id, ...)
  Layer 2 (GLiNER2/LLM) — free-text entities from the shared baseline and file actions.
                           Uses GLiNER2 (local model) by default; if the
                           "entity_extraction.use_llm" system setting is on,
                           routes chunk text to the configured LLM instead
                           (slower/costs tokens, but can generalize to entity
                           types GLiNER2 was never fine-tuned on).
  Layer 3 (Rule)         — boolean summary labels and chunk sensitivity scoring

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

import json
import logging
import re
import threading
import time

logger = logging.getLogger(__name__)

# ── Regex patterns (Layer 1) ──────────────────────────────────────────────────

# GLiNER2 empirically cannot detect Vietnamese street addresses at all — under
# any label wording (address/employee_address/location), confidence stays
# ~0 for the full "số nhà + đường + phường + quận + tỉnh/thành phố" span; it
# only ever catches isolated city/district name fragments under "location".
# This regex exists specifically to cover that gap as a Layer-1 fallback.
#
# Enumerable list of VN provinces/centrally-run cities — the one part of a
# Vietnamese address that's a closed, reliable vocabulary (street/ward/
# district names are open-ended and can't be enumerated this way).
_VN_PROVINCES = (
    r"(?:Hà\s*Nội|TP\.?\s*Hồ\s*Chí\s*Minh|Thành\s*phố\s*Hồ\s*Chí\s*Minh|Hồ\s*Chí\s*Minh|TP\.?\s*HCM|TPHCM|"
    r"Hải\s*Phòng|Đà\s*Nẵng|Cần\s*Thơ|An\s*Giang|Bà\s*Rịa[\s\-]*Vũng\s*Tàu|Bạc\s*Liêu|Bắc\s*Giang|"
    r"Bắc\s*Kạn|Bắc\s*Ninh|Bến\s*Tre|Bình\s*Định|Bình\s*Dương|Bình\s*Phước|Bình\s*Thuận|"
    r"Cà\s*Mau|Cao\s*Bằng|Đắk\s*Lắk|Đắk\s*Nông|Điện\s*Biên|Đồng\s*Nai|Đồng\s*Tháp|Gia\s*Lai|"
    r"Hà\s*Giang|Hà\s*Nam|Hà\s*Tĩnh|Hải\s*Dương|Hậu\s*Giang|Hòa\s*Bình|Hưng\s*Yên|"
    r"Khánh\s*Hòa|Kiên\s*Giang|Kon\s*Tum|Lai\s*Châu|Lâm\s*Đồng|Lạng\s*Sơn|Lào\s*Cai|"
    r"Long\s*An|Nam\s*Định|Nghệ\s*An|Ninh\s*Bình|Ninh\s*Thuận|Phú\s*Thọ|Phú\s*Yên|"
    r"Quảng\s*Bình|Quảng\s*Nam|Quảng\s*Ngãi|Quảng\s*Ninh|Quảng\s*Trị|Sóc\s*Trăng|"
    r"Sơn\s*La|Tây\s*Ninh|Thái\s*Bình|Thái\s*Nguyên|Thanh\s*Hóa|Thừa\s*Thiên[\s\-]*Huế|"
    r"Tiền\s*Giang|Trà\s*Vinh|Tuyên\s*Quang|Vĩnh\s*Long|Vĩnh\s*Phúc|Yên\s*Bái)"
)

# Street/ward/district-level keywords. Deliberately used only to VALIDATE a
# candidate span (via the lookahead below), not to anchor every segment —
# real addresses routinely drop the "Quận"/"Thành phố" prefix in casual
# writing (eg "Cầu Giấy, Hà Nội" with no "Quận" before "Cầu Giấy"), so
# requiring a keyword on every segment would silently under-match.
_VN_ADDRESS_KEYWORDS = (
    r"(?:Số\s*nhà|Số\s*\d|đường|phố|ngõ|hẻm|ngách|kiệt|thôn|xóm|ấp|tổ\s*\d|"
    r"khu\s*phố|căn\s*hộ|chung\s*cư|tòa\s*nhà|Phường|Xã|Thị\s*trấn|Quận|Huyện|Thị\s*xã)"
)

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
    "birth_date": re.compile(
        r"(?:ngày sinh|sinh ngày|DOB)[:\s]*?(\d{1,2}/\d{1,2}/\d{2,4})", re.IGNORECASE
    ),
    "percentage": re.compile(r"\b\d{1,3}\s?%"),
    "credential": re.compile(
        r"(?i)\b(?:password|mật khẩu|api[_\s-]?key|token|secret|otp)\b\s*[:=]\s*[^\s,;]+"
    ),
}

# Not registered under a single fixed key in REGEX_PATTERNS — applied
# dynamically (see extract_structured_entities below) to EVERY active
# entity_key whose name contains "address" (employee_address,
# contract_address, ...), so new address-type labels added later in
# Settings pick this up for free with no code change.
#
# Bounded window ending at a known province/city name (start-of-line or
# after a strong delimiter | . ; :), gated by a lookahead requiring at
# least one street/ward/district keyword to appear somewhere before that
# province name — without the lookahead, a bare "... tại Hà Nội" mention
# with no real address nearby would false-positive on the province name
# alone. Captured span (group 1) is the whole matched address, not just
# the province tail — verified against every real address chunk seen in
# this project's own documents.
_VN_ADDRESS_RE = re.compile(
    r"(?:^|[|.;:\n])\s*"
    r"(?=[^|.;:\n]{0,120}?" + _VN_ADDRESS_KEYWORDS + r"[^|.;:\n]{0,120}?" + _VN_PROVINCES + r")"
    r"([^|.;:\n]{0,120}?" + _VN_PROVINCES + r")",
    re.IGNORECASE,
)
_ADDRESS_LABEL_RE = re.compile(r"address", re.IGNORECASE)

# No regex-detected "money" type anymore — deliberately not in
# REGEX_PATTERNS, so Layer 1 never emits a "money" entity on its own.
# Money-shaped numbers are found ONLY to annotate them for GLiNER (below);
# whatever label (money/salary/revenue/bonus/financial_data/...) actually
# gets applied is entirely GLiNER's zero-shot call over the active policy
# label set, not a hardcoded regex classification.
_MONEY_SHAPE_RE = re.compile(
    r"\b[\d.,]+\s?(?:VND|đồng|VNĐ)\b|\b\d{1,3}(?:\.\d{3}){2,}\b", re.IGNORECASE
)

# ── Money-hint annotation for GLiNER (query-time only) ────────────────────────
# GLiNER's zero-shot scoring leans heavily on nearby lexical context — a bare
# grouped number ("22.000.000") reads far more ambiguously to it than the
# same number written "22.000.000 $" does. The hint doesn't need to be the
# real currency — it's never stored, embedded, or shown to anyone
# (chunk_text/document_text handed to the LLM and the sources panel always
# stay the original, unannotated string), so the only thing that matters is
# picking whatever token GLiNER's own training data associates MOST
# strongly and unambiguously with "this is a monetary amount". "$" is about
# as globally common a money marker as text gets, likely a stronger signal
# than a VNĐ-specific token for a multilingual zero-shot model. Swap freely
# if empirical testing finds a different token scores better.
#
# Detected span offsets come back in annotated-text coordinates and are
# mapped onto the original text (via _remap_offset) before anything else
# touches them, so masking still edits the real chunk, not the hint.
_MONEY_HINT = " $"


def _annotate_bare_money(text: str) -> tuple[str, list[tuple[int, int]]]:
    if not text:
        return text, []
    insertions: list[tuple[int, int]] = []
    pieces: list[str] = []
    cursor = 0
    for m in _MONEY_SHAPE_RE.finditer(text):
        matched = m.group(0)
        if matched[-1].isalpha():
            continue  # already ends in a unit (VND/đồng/VNĐ) — no hint needed
        pieces.append(text[cursor:m.end()])
        pieces.append(_MONEY_HINT)
        insertions.append((m.end(), len(_MONEY_HINT)))
        cursor = m.end()
    if not insertions:
        return text, []
    pieces.append(text[cursor:])
    return "".join(pieces), insertions


# Translate a span offset found in annotated-text coordinates back onto the
# original (pre-annotation) text, given the insertions _annotate_bare_money
# recorded (each: original-text position -> length of hint spliced in there).
def _remap_offset(annotated_pos: int, insertions: list[tuple[int, int]]) -> int:
    shift = 0
    for orig_pos, length in insertions:
        if annotated_pos >= orig_pos + shift + length:
            shift += length
        else:
            break
    return annotated_pos - shift


def _remap_entity_span(entity: dict, insertions: list[tuple[int, int]], original_text: str) -> None:
    if not insertions:
        return
    try:
        start, end = int(entity.get("start", 0)), int(entity.get("end", 0))
    except (TypeError, ValueError):
        return
    new_start = max(0, min(_remap_offset(start, insertions), len(original_text)))
    new_end = max(new_start, min(_remap_offset(end, insertions), len(original_text)))
    entity["start"] = new_start
    entity["end"] = new_end
    # GLiNER's own "text" may include the synthetic hint (eg matched
    # "22.000.000 VNĐ") — replace it with the real substring at the
    # remapped, original-text position.
    entity["text"] = original_text[new_start:new_end]

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
    "money": ["has_financial"],
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


# ── GLiNER2 lazy loader ───────────────────────────────────────────────────────

_gliner_model = None
_gliner_lock  = threading.Lock()


# Lazily load and cache the GLiNER2 model; returns None if loading fails.
# Only actually loaded the first time it's needed — if
# "entity_extraction.use_llm" stays on for the life of the process, this
# never gets called and no GLiNER2 weights are loaded into memory at all.
def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        with _gliner_lock:
            if _gliner_model is None:
                try:
                    from gliner2 import GLiNER2
                    from app.core.config import settings
                    logger.info("Loading GLiNER2 model %s ...", settings.gliner_model_name)
                    _gliner_model = GLiNER2.from_pretrained(settings.gliner_model_name)
                    logger.info("GLiNER2 model loaded.")
                except Exception as exc:
                    logger.error("Failed to load GLiNER2: %s", exc)
                    _gliner_model = None
    return _gliner_model


# Read the "entity_extraction.use_llm" system setting (default False —
# see system_setting_repository.DEFAULTS). Fails closed to GLiNER2 (False)
# on any error, since GLiNER2 is the cheap/local/always-available path.
def _use_llm_for_entities(db) -> bool:
    if db is None:
        return False
    try:
        from app.repositories.system_setting_repository import system_setting_repository
        return bool(system_setting_repository.get(db, "entity_extraction.use_llm"))
    except Exception:
        return False


# Real batched forward pass — model.batch_extract_entities() encodes every
# text in the group through the encoder in one pass (DataLoader-based),
# not once per text. Measured ~2.3x faster than looping extract_entities()
# per text (12-text batch, CPU). Output is nested by label per text
# ({"entities": {label: [{"text","confidence","start","end"}]}}), unlike
# the old `gliner` package's flat list — flatten each to the same
# {"text","label","start","end","score","source"} shape every caller in
# this module already expects, so nothing downstream has to change.
#
# One try/except around the whole batch, not per-text: a real single-pass
# call fails or succeeds as a unit (eg OOM, corrupt state), so partial
# per-text recovery isn't meaningful here — same fail-open contract as
# every other engine in this module, just applied to the group as a whole.
def _gliner2_predict(model, texts: list[str], labels: list[str], threshold: float) -> list[list[dict]]:
    gliner_prompts = [_to_gliner_prompt(label) for label in labels]
    try:
        raw_results = model.batch_extract_entities(
            texts, gliner_prompts, batch_size=max(1, len(texts)), threshold=threshold,
            include_confidence=True, include_spans=True,
        )
    except Exception as exc:
        logger.error("GLiNER2 batch inference error: %s", exc)
        return [[] for _ in texts]

    results: list[list[dict]] = []
    for raw in raw_results:
        entities: list[dict] = []
        for label, items in (raw.get("entities") or {}).items():
            for item in items or []:
                entities.append({
                    "text": item.get("text"),
                    "label": _from_gliner_prompt(label),
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "score": item.get("confidence", 0.0),
                    "source": "gliner",
                })
        results.append(entities)
    return results


_ENTITY_LLM_SYSTEM_PROMPT = (
    "Bạn là hệ thống nhận diện thực thể (named entity recognition) trong văn bản "
    "doanh nghiệp tiếng Việt. Chỉ trả về JSON, không kèm giải thích."
)


def _build_entity_llm_prompt(texts: list[str], labels: list[str]) -> str:
    label_list = "\n".join(f"- {label}" for label in labels)
    chunks_block = "\n\n".join(
        f'<chunk id="{i}">\n{text}\n</chunk>' for i, text in enumerate(texts)
    )
    return (
        "Danh sách loại thực thể cần tìm (entity_key):\n"
        f"{label_list}\n\n"
        "Với MỖI đoạn văn bản dưới đây, tìm mọi cụm từ khớp với 1 trong các "
        "entity_key trên. Chỉ trả về entity thực sự xuất hiện NGUYÊN VĂN trong "
        "đoạn văn bản đó (copy chính xác từng ký tự, không diễn giải/rút gọn/dịch), "
        "vì hệ thống sẽ tự tìm lại vị trí bằng cách so khớp chuỗi con nguyên văn.\n\n"
        f"{chunks_block}\n\n"
        "Trả về đúng JSON theo dạng sau, không kèm gì khác:\n"
        '{"chunks": [{"chunk_id": "<id đúng như trong thẻ chunk>", '
        '"entities": [{"entity_key": "<entity_key nguyên văn>", "text": "<cụm từ nguyên văn trong đoạn>"}]}]}'
    )


# Alternative to GLiNER2: ask the configured LLM to find entities instead,
# gated by the "entity_extraction.use_llm" system setting. Mirrors the
# fail-open JSON-prompt pattern already used by
# entity_policy_service._select_relevant_labels() (llm_service.generate_json
# with use_default_instructions=False, temperature=0.0, wrapped try/except).
#
# The LLM is deliberately NEVER asked for character offsets — models are
# unreliable at counting characters, and a wrong span would mask/reveal the
# wrong text. Instead it must copy the matched text verbatim, and this
# function locates the real start/end itself via a literal substring search
# on the original chunk. If the LLM's text doesn't appear verbatim, that
# entity is dropped rather than guessed at.
def _extract_via_llm(texts: list[str], labels: list[str], threshold: float) -> list[list[dict]]:
    empty: list[list[dict]] = [[] for _ in texts]
    if not texts or not labels:
        return empty

    from app.services.llm_service import llm_service

    if not llm_service.is_configured():
        return empty

    prompt = _build_entity_llm_prompt(texts, labels)
    try:
        text_json, _, _ = llm_service.generate_json(
            prompt,
            system=_ENTITY_LLM_SYSTEM_PROMPT,
            use_default_instructions=False,
            temperature=0.0,
        )
        data = json.loads(text_json)
    except Exception:
        logger.warning(
            "[entity-extractor] LLM entity detection failed, returning no entities",
            exc_info=True,
        )
        return empty

    allowed_labels = set(labels)
    results: list[list[dict]] = [[] for _ in texts]
    for item in data.get("chunks") or []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("chunk_id"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(texts):
            continue
        original = texts[idx]
        entities: list[dict] = []
        for raw_entity in item.get("entities") or []:
            if not isinstance(raw_entity, dict):
                continue
            label = str(raw_entity.get("entity_key") or "")
            value = str(raw_entity.get("text") or "")
            if not label or not value or label not in allowed_labels:
                continue
            start = original.find(value)
            if start < 0:
                continue  # LLM didn't copy the span verbatim - unsafe to guess an offset
            entities.append({
                "text": value,
                "label": label,
                "start": start,
                "end": start + len(value),
                "score": 0.95,  # LLM has no real confidence score to report
                "source": "llm",
            })
        results[idx] = entities
    return results


# Single choke point for "detect entities in these texts": routes to the LLM
# path or the GLiNER2 path based on the "entity_extraction.use_llm" system
# setting. Both paths return the identical flat shape, so every caller below
# stays agnostic to which one actually ran.
def _detect_entities_batch(
    texts: list[str], labels: list[str], threshold: float, db=None,
) -> list[list[dict]]:
    if not texts or not labels:
        return [[] for _ in texts]
    if _use_llm_for_entities(db):
        return _extract_via_llm(texts, labels, threshold)
    model = _get_gliner()
    if model is None:
        return [[] for _ in texts]
    return _gliner2_predict(model, texts, labels, threshold)


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

    # VN address regex fallback (see _VN_ADDRESS_RE) — applies to every
    # active label whose entity_key names it an address, not just one
    # hardcoded key. Requires allowed_labels because a matched span needs
    # a concrete label to be emitted under; with no restriction given
    # there's no way to know which address-type key(s) are even active.
    if allowed_labels:
        for label in allowed_labels:
            if not _ADDRESS_LABEL_RE.search(label):
                continue
            for m in _VN_ADDRESS_RE.finditer(text):
                results.append({
                    "text":   m.group(1),
                    "label":  label,
                    "start":  m.start(1),
                    "end":    m.end(1),
                    "score":  1.0,
                    "source": "regex",
                })
    return results


# ── Layer 2: GLiNER extraction ────────────────────────────────────────────────

# GLiNER is a zero-shot NER model trained against natural-language label
# prompts ("employee salary"), not snake_case identifiers ("employee_salary")
# — the underscore is an odd token that hurts its label embedding quality.
# entity_key stays snake_case everywhere else in this module (DB matching,
# entity_flags lookup, allowed_labels/per_text_labels comparisons) — this
# conversion only wraps the actual model call, converted back on the way out
# so no downstream code ever sees the space form.
def _to_gliner_prompt(label: str) -> str:
    return label.replace("_", " ")


def _from_gliner_prompt(label: str) -> str:
    return label.replace(" ", "_")


# Extract free-text entities (GLiNER2 or LLM — see _detect_entities_batch)
# for the given label list (Layer 2). Money-hint annotated (see
# _annotate_bare_money) so ingest-time detection (run_pipeline) gets the
# same VNĐ-context boost as query-time — without this, removing the regex
# "money" pattern would leave ingest-time chunk_sensitivity scoring blind
# to bare numeric amounts entirely.
def extract_freetext_entities(text: str, labels: list[str], threshold: float = 0.3, db=None) -> list[dict]:
    if not labels:
        return []
    annotated, insertions = _annotate_bare_money(text)
    batched = _detect_entities_batch([annotated], labels, threshold, db=db)
    entities = batched[0] if batched else []
    for e in entities:
        _remap_entity_span(e, insertions, text)
    return entities


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

# Run entity detection on a single text. Returns (entities, detected_entity_types).
def extract_realtime(
    text: str,
    *,
    db=None,
    threshold: float = 0.3,
) -> tuple[list[dict], set[str]]:
    gliner_labels, _ = _refresh_cache(db)
    entities = extract_freetext_entities(text, gliner_labels, threshold=threshold, db=db)
    return entities, {e["label"] for e in entities}


# Thin wrapper kept for its name/callers below (extract_realtime_batch,
# extract_realtime_batch_detailed) — the actual GLiNER2-vs-LLM routing lives
# in _detect_entities_batch. Kept as a separate function (rather than
# inlining) because callers group texts by per_text_labels first (see
# _group_by_labels/_MAX_LABEL_GROUPS below) and call this once per group,
# so one call here still corresponds to "one batch of texts sharing a label
# set", regardless of which underlying engine handles it.
def _batch_predict(
    texts: list[str], labels: list[str], threshold: float, db=None,
) -> list[list[dict]]:
    return _detect_entities_batch(texts, labels, threshold, db=db)


# Cap on distinct per-text label subsets extract_realtime_batch_detailed will
# honor (see per_text_labels below) before collapsing back to one shared
# label list for the whole batch. Each distinct group costs its own
# _batch_predict call — on CPU (no GPU configured in this deployment) the
# fixed per-call overhead _batch_predict's docstring warns about (tokenize/
# collate/eval/dispatch) adds up fast if every text ends up in its own
# group, which would silently re-create the "one model call per chunk"
# latency problem batching was introduced to avoid.
_MAX_LABEL_GROUPS = 4


# Bucket text indices by identical relevant-label subset (order-independent —
# {"a","b"} and {"b","a"} are the same group) so callers that narrowed
# GLiNER's active labels per-text (eg an LLM pre-filter step) still get one
# _batch_predict call per distinct subset instead of per text. Falls back to
# a single group covering the union of every subset once there are more than
# _MAX_LABEL_GROUPS distinct ones, trading exact per-text filtering for
# bounded latency.
def _group_by_labels(per_text_labels: list[list[str]]) -> list[tuple[list[str], list[int]]]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, labels in enumerate(per_text_labels):
        key = tuple(sorted(set(labels)))
        groups.setdefault(key, []).append(index)
    if len(groups) > _MAX_LABEL_GROUPS:
        union = sorted({label for labels in per_text_labels for label in labels})
        return [(union, list(range(len(per_text_labels))))]
    return [(list(key), indices) for key, indices in groups.items()]


# Run entity detection on multiple texts in one batch (GLiNER2 or LLM — see
# _detect_entities_batch). Returns one set[str] of detected entity types per text.
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

    try:
        batched = _batch_predict(texts, gliner_labels, threshold, db=db)
        return [{e["label"] for e in raw} for raw in batched]
    except Exception as exc:
        logger.error("Entity batch detection error: %s", exc)
        return [set() for _ in texts]


def extract_realtime_batch_detailed(
    texts: list[str],
    *,
    db=None,
    threshold: float = 0.3,
    labels: list[str] | set[str] | None = None,
    per_text_labels: list[list[str]] | None = None,
) -> list[dict]:
    """Return structured entities, GLiNER entities, types and boolean flags.

    Policy enforcement needs spans, not just a set of labels, in order to mask
    only the salary/PII field inside a mixed chunk.  This keeps the existing
    lightweight batch API intact and adds a detailed variant for that path.

    per_text_labels: optional, one relevant-label subset per text (same
    length/order as `texts`) — eg from an upstream LLM topic-filter — used to
    narrow GLiNER's active labels per text instead of running every text
    against the full `labels`/cache list. Grouped via _group_by_labels so
    texts sharing a subset still batch together; ignored (falls back to the
    single shared list) if its length doesn't match `texts`.
    """
    if not texts:
        return []
    gliner_labels, entity_flags = _refresh_cache(db) if labels is None else (
        sorted(set(labels)), _refresh_cache(db)[1]
    )
    allowed_labels = set(gliner_labels)

    # Annotated-for-GLiNER2 copies only — never used for regex, never
    # returned/stored (see _annotate_bare_money's docstring above).
    annotated_texts: list[str] = []
    insertions_by_text: list[list[tuple[int, int]]] = []
    for t in texts:
        annotated, insertions = _annotate_bare_money(t)
        annotated_texts.append(annotated)
        insertions_by_text.append(insertions)

    # One batched call per label-group across all chunks (see
    # _batch_predict/_detect_entities_batch — GLiNER2 or LLM depending on
    # the entity_extraction.use_llm setting), then a cheap per-text loop for
    # regex extraction + assembly. entity["source"] is already set correctly
    # ("gliner" or "llm") by whichever engine actually ran.
    freetext_by_text: list[list[dict]] = [[] for _ in texts]
    if gliner_labels:
        try:
            if per_text_labels is not None and len(per_text_labels) == len(texts):
                for group_labels, indices in _group_by_labels(per_text_labels):
                    # A per-text subset may reference labels that are no
                    # longer active/valid (eg from a stale LLM suggestion) —
                    # clip to what this run is actually allowed to look for.
                    group_labels = [label for label in group_labels if label in allowed_labels]
                    if not group_labels:
                        continue
                    group_texts = [annotated_texts[i] for i in indices]
                    batched = _batch_predict(group_texts, group_labels, threshold, db=db)
                    for i, freetext in zip(indices, batched):
                        for entity in freetext:
                            _remap_entity_span(entity, insertions_by_text[i], texts[i])
                        freetext_by_text[i] = freetext
            else:
                batched = _batch_predict(annotated_texts, gliner_labels, threshold, db=db)
                for i, freetext in enumerate(batched):
                    for entity in freetext:
                        _remap_entity_span(entity, insertions_by_text[i], texts[i])
                    freetext_by_text[i] = freetext
        except Exception as exc:
            logger.debug("Entity batched detailed extraction failed: %s", exc)

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
    freetext   = extract_freetext_entities(text, gliner_labels, threshold=gliner_threshold, db=db)
    all_entities = structured + freetext

    booleans = detect_boolean_labels(text, all_entities, entity_flags)

    return {
        "entities":     all_entities,
        "labels":       booleans,
        "entity_types": list({e["label"] for e in all_entities}),
    }
