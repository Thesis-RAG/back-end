"""Per-document entity detection, action resolution, masking and access gates."""
from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from typing import Iterable

from sqlalchemy.orm import Session

from app.models.document_entity_action import DocumentEntityAction
from app.models.document_version import DocumentVersion
from app.repositories.document_entity_repository import document_entity_repository
from app.services.entity_extractor import extract_realtime_batch_detailed, run_pipeline
from app.services.parser_service import parser_service
from app.services.policy_rule_service import (
    DEFAULT_POLICY_PROFILE,
    expand_oui_ids_via_graph,
    policy_rule_service,
    resolve_tier,
)
from app.services.user_service import user_service


VALID_ACTIONS = {"block", "full", "mask"}
_SEMANTIC_ALIASES = {
    "salary": {"salary", "income", "wage", "pay", "lương", "thu nhập", "tiền công"},
    "person_name": {"name", "employee", "person", "tên", "nhân viên", "người"},
    "email": {"email", "mail", "e-mail", "thư điện tử"},
    "phone": {"phone", "mobile", "telephone", "số điện thoại", "điện thoại"},
    "address": {"address", "location", "địa chỉ", "nơi ở"},
    "money": {"money", "amount", "price", "cost", "tiền", "số tiền", "chi phí"},
}


def normalize_entity_type(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    return value.strip("_")[:128]


def normalize_actions(actions: Iterable[dict]) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    for item in actions or []:
        entity_type = normalize_entity_type(item.get("entity_type", ""))
        if not entity_type or entity_type in seen:
            continue
        action = str(item.get("action") or "full").lower()
        if action not in VALID_ACTIONS:
            action = "full"
        seen.add(entity_type)
        scope_oui_ids = list(dict.fromkeys(
            str(value).strip()
            for value in (item.get("scope_oui_ids") or [])
            if str(value).strip()
        ))
        scope_position_ids = list(dict.fromkeys(
            str(value).strip()
            for value in (item.get("scope_position_ids") or [])
            if str(value).strip()
        ))
        result.append({
            "entity_type": entity_type,
            "label": str(item.get("label") or entity_type)[:255],
            "action": action,
            "source": str(item.get("source") or "manual")[:16],
            "enabled": bool(item.get("enabled", True)),
            "detection_count": int(item.get("detection_count") or 0),
            "scope_oui_ids": scope_oui_ids,
            "scope_position_ids": scope_position_ids,
            "metadata_json": dict(item.get("metadata_json") or {}),
        })
    return result


# Mask character shown to the LLM/user for hidden values — kept in one place
# so the whole masking scheme (fixed placeholder + partial mask) stays
# consistent if this ever changes again.
_MASK_CHAR = "•"

# Fixed-length placeholder for fully-masked ("mask_style=full") values — a
# constant length so the mask itself never leaks how long the real value was.
_MASK_PLACEHOLDER = _MASK_CHAR * 16


# Mask part of a value's text, keeping the rest visible for context/verification.
# position="head" masks the start (keeps the tail visible), "tail" masks the
# end (keeps the head visible), "center" keeps both ends and masks the middle.
def _partial_mask(text: str, position: str | None) -> str:
    text = text or ""
    n = len(text)
    if n <= 2:
        return _MASK_CHAR * n
    keep = max(1, n // 4)
    if position == "head":
        return _MASK_CHAR * (n - keep) + text[-keep:]
    if position == "center":
        half = max(1, keep // 2)
        return text[:half] + _MASK_CHAR * (n - 2 * half) + text[-half:]
    # default: tail
    return text[:keep] + _MASK_CHAR * (n - keep)


def _replace_entity_spans(
    text: str, entities: list[dict], replacements: dict[str, "str | callable"]
) -> str:
    edits: list[tuple[int, int, str]] = []
    for entity in entities:
        label = normalize_entity_type(entity.get("label", ""))
        replacement = replacements.get(label)
        if replacement is None:
            continue
        if callable(replacement):
            replacement = replacement(entity)
        try:
            start, end = int(entity.get("start", 0)), int(entity.get("end", 0))
        except (TypeError, ValueError):
            continue
        if 0 <= start < end <= len(text):
            edits.append((start, end, replacement))

    output = text
    occupied_start = len(text) + 1
    for start, end, replacement in sorted(edits, key=lambda value: (value[0], -value[1]), reverse=True):
        if end > occupied_start:
            continue
        output = output[:start] + replacement + output[end:]
        occupied_start = start
    return output


_TABLE_ROW_RE = re.compile(r"^\|([^|\n]*)\|([^|\n]*)\|$")
# A value cell counts as "empty" once only separator punctuation/whitespace
# is left in it — the actual content (all of it, or every comma-joined
# sub-value) was erased by a block-tier replacement.
_SEPARATOR_ONLY_RE = re.compile(r"^[,.\s]*$")
_REPEAT_SEPARATOR_RE = re.compile(r"(?:[,.]\s*){2,}")


def _tidy_masked_table_rows(text: str) -> str:
    """Clean up table rows after entity erasure/masking.

    Block-tier erasure removes only the matched span, not the comma/pipe
    punctuation around it — a row like "| Địa chỉ | ••••, , ,  |" (some
    sub-values masked, others erased) or "| Số điện thoại |  |" (erased
    entirely) is technically valid markdown but reads as broken. Drop rows
    whose value is now nothing but leftover separators (no label revealed
    either — consistent with "erase all trace"), and collapse repeated
    separators in rows that still have real (or masked) content.
    """
    lines: list[str] = []
    for line in text.split("\n"):
        match = _TABLE_ROW_RE.match(line.strip())
        if not match:
            lines.append(line)
            continue
        label, value = match.group(1).strip(), match.group(2).strip()
        if _SEPARATOR_ONLY_RE.match(value):
            continue
        value = _REPEAT_SEPARATOR_RE.sub(", ", value).strip(" ,.")
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


class EntityPolicyService:
    def policy_snapshot(self, db: Session) -> dict:
        """Resolve the active global policy for a new document version."""
        return policy_rule_service.snapshot(db, DEFAULT_POLICY_PROFILE)

    # Recompute a 'require' entity's real value for display, scoped to this
    # exact message and this exact user — never written back into
    # Message.content, never reused by any other message or user.
    def reveal_for_message(self, db: Session, user, message) -> str:
        content = message.content or ""
        masks = message.entity_masks_json or []
        if not masks:
            return content

        from app.repositories.message_entity_repository import message_entity_repository
        granted = message_entity_repository.granted_types(db, str(user.id), message.id)
        if not granted:
            return content

        from app.models.document_chunk import DocumentChunk
        chunk_ids = {
            mask.get("chunk_id") for mask in masks
            if normalize_entity_type(mask.get("entity_type") or "") in granted and mask.get("chunk_id")
        }
        if not chunk_ids:
            return content
        chunks = db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()
        if not chunks:
            return content
        details = extract_realtime_batch_detailed([c.chunk_text or "" for c in chunks], db=db)

        revealed = content
        for chunk, detail in zip(chunks, details):
            wanted_types = {
                normalize_entity_type(mask.get("entity_type") or "")
                for mask in masks
                if mask.get("chunk_id") == chunk.id
                and normalize_entity_type(mask.get("entity_type") or "") in granted
            }
            if not wanted_types:
                continue
            rules = policy_rule_service.rules_by_key(db, wanted_types)
            for entity in detail.get("entities") or []:
                label = normalize_entity_type(entity.get("label") or "")
                if label not in wanted_types:
                    continue
                real_text = str(entity.get("text") or "")
                if not real_text:
                    continue
                rule = rules.get(label)
                if rule and rule.mask_style == "partial":
                    masked_repr = _partial_mask(real_text, rule.mask_position)
                else:
                    masked_repr = _MASK_PLACEHOLDER
                if masked_repr in revealed:
                    revealed = revealed.replace(masked_repr, real_text)
        return revealed

    @staticmethod
    def action_applies_to_user(action: DocumentEntityAction, user) -> bool:
        """Match a scoped action against any current user unit/role assignment."""
        scoped_oui_ids = {str(value) for value in (action.scope_oui_ids or [])}
        scoped_position_ids = {str(value) for value in (action.scope_position_ids or [])}
        if not scoped_oui_ids and not scoped_position_ids:
            return True

        for assignment in getattr(user, "oui_positions", []) or []:
            oui_id = str(getattr(assignment, "oui_id", "") or "")
            position_id = str(getattr(assignment, "position_id", "") or "")
            position = getattr(assignment, "position", None)
            if not position_id and position is not None:
                position_id = str(getattr(position, "id", "") or "")
            if scoped_oui_ids and oui_id not in scoped_oui_ids:
                continue
            if scoped_position_ids and position_id not in scoped_position_ids:
                continue
            return True
        return False

    def preview(self, raw_bytes: bytes, filename: str, mime_type: str, db: Session) -> dict:
        parsed = parser_service.parse(raw_bytes, filename, mime_type)
        from app.core.config import settings
        details = run_pipeline(parsed.full_text, db=db, gliner_threshold=settings.gliner_threshold)
        entities = details.get("entities") or []
        confirmed_labels = sorted({normalize_entity_type(item.get("label")) for item in entities if item.get("label")})
        policy = self.policy_snapshot(db)
        rules = {
            str(item["entity_key"]): item
            for item in policy["resolved_rules"]
            if item.get("entity_key") in confirmed_labels
        }
        applied_rules = []
        for entity_type in confirmed_labels:
            rule = rules.get(entity_type)
            if rule:
                applied_rules.append({
                    "entity_key": entity_type,
                    "display_name": rule.get("display_name") or entity_type,
                    "action": rule.get("action") or "full",
                    "detection_count": sum(1 for entity in entities if normalize_entity_type(entity.get("label")) == entity_type),
                })
        action_summary = {"block": 0, "mask": 0, "full": 0}
        for rule in applied_rules:
            action_summary[rule["action"]] = action_summary.get(rule["action"], 0) + rule["detection_count"]
        serializable = []
        for entity in entities:
            serializable.append({
                "text": str(entity.get("text") or ""),
                "label": normalize_entity_type(entity.get("label") or ""),
                "start": int(entity.get("start") or 0),
                "end": int(entity.get("end") or 0),
                "score": float(entity.get("score") or 0.0),
                "source": entity.get("source") or "gliner",
                "flags": list(entity.get("flags") or []),
            })
        return {
            "file_name": filename,
            "text_preview": parsed.full_text[:12000],
            "text_truncated": len(parsed.full_text) > 12000,
            "entities": serializable,
            "entity_types": confirmed_labels,
            "confirmed_labels": confirmed_labels,
            "policy_profile": policy["policy_profile"],
            "policy_version": policy["policy_version"],
            "applied_rules": applied_rules,
            "action_summary": action_summary,
        }

    def detect_full_text(self, text: str, db: Session) -> dict:
        from app.core.config import settings
        details = run_pipeline(text, db=db, gliner_threshold=settings.gliner_threshold)
        entities = []
        for entity in details.get("entities") or []:
            item = dict(entity)
            item["label"] = normalize_entity_type(item.get("label") or "")
            item["flags"] = list(item.get("flags") or [])
            entities.append(item)
        return {
            "entities": entities,
            "entity_types": sorted({item["label"] for item in entities if item.get("label")}),
        }

    def configured_labels(self, db: Session) -> list[str]:
        return sorted({
            normalize_entity_type(rule.entity_key)
            for rule in policy_rule_service.active_rules(db, DEFAULT_POLICY_PROFILE)
            if rule.entity_key
        })

    def query_entities(self, query: str, db: Session) -> set[str]:
        from app.core.config import settings
        details = run_pipeline(query, db=db, gliner_threshold=settings.gliner_threshold)
        found = {normalize_entity_type(value) for value in (details.get("entity_types") or [])}
        lowered = query.lower()
        for label in set(self.configured_labels(db)) | set(_SEMANTIC_ALIASES):
            aliases = _SEMANTIC_ALIASES.get(label, set())
            label_tokens = set(label.split("_"))
            if any(alias in lowered for alias in aliases) or any(token and token in lowered for token in label_tokens):
                found.add(label)
        return found

    def replace_actions(self, db: Session, version_id: str, actions: list[dict]) -> list[DocumentEntityAction]:
        normalized = normalize_actions(actions)
        # Saving the action configuration must not re-run detection and must
        # not reset the counters produced during ingest. The UI only edits
        # the action field, so restore the persisted counters (and metadata)
        # for existing entity types before replacing the rows.
        existing_rows = document_entity_repository.list_actions(db, version_id)
        existing_by_type = {
            normalize_entity_type(row.entity_type): row
            for row in existing_rows
        }
        for item in normalized:
            previous = existing_by_type.get(item["entity_type"])
            if previous is None:
                continue
            item["detection_count"] = int(previous.detection_count or 0)
            if not item.get("metadata_json"):
                item["metadata_json"] = dict(previous.metadata_json or {})
        rows = document_entity_repository.replace_actions(db, version_id, normalized)
        from app.services.entity_extractor import invalidate_label_cache
        invalidate_label_cache()
        return rows

    # Full normalized text for a version, used only for the raw-file gate's
    # live detection — never cached, always re-read at call time.
    def _version_full_text(self, db: Session, version: DocumentVersion) -> str:
        if not version.normalized_object:
            return ""
        from app.services.storage_service import storage_service
        try:
            raw = storage_service.download(
                version.normalized_object.bucket, version.normalized_object.object_key
            )
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    # Entity types that gate the RAW file (whole-document, not per-span):
    # detected live against the current central policy, every call — a
    # label added after this document was ingested applies immediately,
    # with no reprocessing step required.
    def blocked_types_for_version(self, db: Session, version_id: str, user=None) -> set[str]:
        if user is None:
            return set()
        version = db.get(DocumentVersion, version_id)
        if not version:
            return set()
        text = self._version_full_text(db, version)
        if not text:
            return set()
        detail = extract_realtime_batch_detailed([text], db=db)
        if not detail:
            return set()
        detected_types = {normalize_entity_type(value) for value in detail[0].get("entity_types") or []}
        if not detected_types:
            return set()
        rules = policy_rule_service.rules_by_key(db, detected_types)
        expanded_oui_ids = expand_oui_ids_via_graph(user)
        return {
            label for label in detected_types
            if label in rules and resolve_tier(rules[label], user, expanded_oui_ids)["tier"] == "block"
        }

    def has_entity_access(self, db: Session, user, document_id: str, version_id: str, entity_types: set[str]) -> bool:
        if not entity_types:
            return True
        try:
            if user_service.build_user_response(db, user).is_corp_member:
                return True
        except Exception:
            pass
        granted = document_entity_repository.active_granted_types(
            db, str(user.id), document_id, version_id
        )
        return {normalize_entity_type(value) for value in entity_types}.issubset(granted)

    def apply_to_retrieved(self, db: Session, user, query: str, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
        if not chunks:
            return [], []

        query_entities = self.query_entities(query, db)
        # Always detect against the LIVE central policy label set (no frozen
        # per-chunk snapshot) — a label added today must apply to documents
        # ingested long before it existed, without any reprocessing step.
        details = extract_realtime_batch_detailed(
            [str(chunk.get("document_text") or "") for chunk in chunks], db=db,
        )
        processed: list[dict] = []
        contracts: list[dict] = []
        # One graph round-trip per chat turn, reused for every chunk/entity
        # below — see expand_oui_ids_via_graph's docstring for why this
        # can't just be an exact match on the user's own OUI.
        expanded_oui_ids = expand_oui_ids_via_graph(user)

        for chunk, detail in zip(chunks, details):
            result = dict(chunk)
            metadata = dict(chunk.get("metadata") or {})
            document_id = str(metadata.get("document_id") or "")
            version_id = str(metadata.get("document_version_id") or "")
            chunk_id = str(chunk.get("chunk_id") or "")
            entities = detail.get("entities") or []
            detected_types = {normalize_entity_type(value) for value in detail.get("entity_types") or []}

            values_by_label: dict[str, list[str]] = {}
            for entity in entities:
                label = normalize_entity_type(entity.get("label") or "")
                text_value = str(entity.get("text") or "")
                if label and text_value:
                    values_by_label.setdefault(label, []).append(text_value)

            rules = policy_rule_service.rules_by_key(db, detected_types)
            resolved = {
                label: resolve_tier(rules[label], user, expanded_oui_ids)
                for label in detected_types
                if label in rules
            }
            blocked_types = {label for label, res in resolved.items() if res["tier"] == "block"}
            require_types = {label for label, res in resolved.items() if res["tier"] == "require"}

            # block: erase the span entirely — the LLM never sees any trace
            # that a value was even there (no placeholder, no appeal path).
            replacements: dict[str, object] = {label: "" for label in blocked_types}

            # require: never revealed at generation time (no message_id yet
            # to check a per-message grant against) — always masked as a
            # run of asterisks here. Reveal happens later, per-message, when
            # serving a persisted message via reveal_for_message().
            for label in require_types:
                if label in replacements:
                    continue
                res = resolved[label]
                if res.get("mask_style") == "partial":
                    position = res.get("mask_position")
                    replacements[label] = (
                        lambda entity, _pos=position: _partial_mask(str(entity.get("text") or ""), _pos)
                    )
                else:
                    replacements[label] = _MASK_PLACEHOLDER

            result["document_text"] = _tidy_masked_table_rows(_replace_entity_spans(
                str(chunk.get("document_text") or ""), entities, replacements
            ))
            metadata.update({
                # No appeal exists for block tier anymore, so this chunk
                # never needs the document-level access-request UI.
                "entity_access_required": False,
                "entity_access_granted": False,
                "blocked_entity_types": sorted(blocked_types),
                "require_entity_types": sorted(require_types),
                "detected_entity_types": sorted(detected_types),
            })
            result["metadata"] = metadata
            result["entity_access_required"] = False
            result["entity_access_granted"] = False
            result["blocked_entity_types"] = sorted(blocked_types)
            result["require_entity_types"] = sorted(require_types)
            result["chunk_id"] = chunk_id
            # Only require-tier content is appeal-eligible now; block-tier
            # masking is permanent and shouldn't invite a request.
            result["doc_restricted"] = bool(chunk.get("doc_restricted") or require_types)

            if blocked_types:
                decision = "block"
            elif require_types:
                decision = "require"
            else:
                decision = "full"

            contract = {
                "contract_id": f"PC-{uuid.uuid4().hex[:12]}",
                "document_id": document_id,
                "document_version_id": version_id,
                "chunk_id": chunk_id,
                "query_entities": sorted(query_entities),
                "detected_entities": sorted(detected_types),
                "matched_entities": [
                    {
                        "entity_type": label,
                        "tier": res["tier"],
                        "display_name": rules[label].display_name,
                        # Raw detected values (GLiNER/regex), for debugging/audit —
                        # never sent to the LLM, only ever surfaced in logs/traces.
                        "values": values_by_label.get(label, []),
                    }
                    for label, res in sorted(resolved.items())
                ],
                "decision": decision,
                "requires_access_request": bool(require_types),
                "require_entity_types": sorted(require_types),
            }
            result["policy_contract"] = contract
            contracts.append(contract)
            processed.append(result)

        return processed, contracts


entity_policy_service = EntityPolicyService()
