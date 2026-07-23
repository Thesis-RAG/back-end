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
from app.services.policy_rule_service import DEFAULT_POLICY_PROFILE, policy_rule_service
from app.services.user_service import user_service


VALID_ACTIONS = {"block", "full", "mask"}
# Clearance 4 (Mật) and 5 (Tuyệt mật) can inspect entity values directly.
ENTITY_POLICY_BYPASS_CLEARANCE = 4
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


def _replace_entity_spans(text: str, entities: list[dict], replacements: dict[str, str]) -> str:
    edits: list[tuple[int, int, str]] = []
    for entity in entities:
        label = normalize_entity_type(entity.get("label", ""))
        replacement = replacements.get(label)
        if replacement is None:
            continue
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


class EntityPolicyService:
    def policy_snapshot(self, db: Session) -> dict:
        """Resolve the active global policy for a new document version."""
        return policy_rule_service.snapshot(db, DEFAULT_POLICY_PROFILE)

    @staticmethod
    def max_user_clearance(user) -> int:
        """Return the highest clearance available on either user shape."""
        if user is None:
            return 1

        max_clearance = 1
        try:
            max_clearance = max(max_clearance, int(getattr(user, "max_clearance", 1) or 1))
        except (TypeError, ValueError):
            pass

        # API dependencies provide the SQLAlchemy User model, while some
        # internal callers/tests provide the serialized user response. Support
        # both shapes so the policy bypass cannot depend on representation.
        for position_holder in getattr(user, "oui_positions", []) or []:
            position = getattr(position_holder, "position", position_holder)
            try:
                max_clearance = max(
                    max_clearance,
                    int(getattr(position, "clearance", 1) or 1),
                )
            except (TypeError, ValueError):
                continue
        return max_clearance

    @classmethod
    def bypasses_entity_actions(cls, user) -> bool:
        """High-clearance users do not receive entity-rule transformations."""
        return cls.max_user_clearance(user) >= ENTITY_POLICY_BYPASS_CLEARANCE

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

    def blocked_types_for_version(self, db: Session, version_id: str, user=None) -> set[str]:
        actions = document_entity_repository.list_actions(db, version_id)
        version = db.get(DocumentVersion, version_id)
        detected = set((version.entity_detection_json or {}).get("entity_types") or []) if version else set()
        blocked = {
            normalize_entity_type(row.entity_type)
            for row in actions
            if row.enabled
            and row.action == "block"
            and (user is None or self.action_applies_to_user(row, user))
        }
        return blocked.intersection({normalize_entity_type(value) for value in detected})

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
        if self.bypasses_entity_actions(user):
            return chunks, []

        query_entities = self.query_entities(query, db)
        # Use each chunk's snapshotted label set. This prevents a later global
        # rule edit from changing how an older document is interpreted.
        details: list[dict] = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            raw_labels = metadata.get("confirmed_labels")
            if isinstance(raw_labels, str):
                snapshot_labels = [value for value in raw_labels.split(",") if value]
            elif isinstance(raw_labels, (list, tuple, set)):
                snapshot_labels = [str(value) for value in raw_labels]
            else:
                snapshot_labels = None
            details.extend(extract_realtime_batch_detailed(
                [str(chunk.get("document_text") or "")],
                db=db,
                labels=snapshot_labels,
            ))
        version_ids = [
            str((chunk.get("metadata") or {}).get("document_version_id"))
            for chunk in chunks
            if (chunk.get("metadata") or {}).get("document_version_id")
        ]
        action_map = document_entity_repository.get_action_map(db, version_ids)
        processed: list[dict] = []
        contracts: list[dict] = []

        for chunk, detail in zip(chunks, details):
            result = dict(chunk)
            metadata = dict(chunk.get("metadata") or {})
            document_id = str(metadata.get("document_id") or "")
            version_id = str(metadata.get("document_version_id") or "")
            actions = {
                normalize_entity_type(row.entity_type): row
                for row in action_map.get(version_id, [])
                if row.enabled and self.action_applies_to_user(row, user)
            }
            entities = detail.get("entities") or []
            detected_types = {normalize_entity_type(value) for value in detail.get("entity_types") or []}
            blocked_types = {
                label for label, row in actions.items()
                if row.action == "block" and label in detected_types
            }
            masked_types = {
                label for label, row in actions.items()
                if row.action == "mask" and label in detected_types
            }
            approved = self.has_entity_access(db, user, document_id, version_id, blocked_types)
            hidden_blocked_types = set() if approved else blocked_types

            replacements: dict[str, str] = {
                label: f"[Nội dung thực thể '{label}' cần quyền xem]"
                for label in hidden_blocked_types
            }
            replacements.update({
                label: f"[Đã che thực thể '{label}']"
                for label in masked_types
                if label not in replacements
            })
            result["document_text"] = _replace_entity_spans(
                str(chunk.get("document_text") or ""), entities, replacements
            )
            metadata.update({
                "entity_access_required": bool(hidden_blocked_types),
                "entity_access_granted": approved,
                "blocked_entity_types": sorted(hidden_blocked_types),
                "masked_entity_types": sorted(masked_types),
                "detected_entity_types": sorted(detected_types),
            })
            result["metadata"] = metadata
            result["entity_access_required"] = bool(hidden_blocked_types)
            result["entity_access_granted"] = approved
            result["blocked_entity_types"] = sorted(hidden_blocked_types)
            result["masked_entity_types"] = sorted(masked_types)
            result["doc_restricted"] = bool(chunk.get("doc_restricted") or hidden_blocked_types)

            if hidden_blocked_types:
                decision = "block"
            elif masked_types:
                decision = "mask"
            else:
                decision = "full"

            contract = {
                "contract_id": f"PC-{uuid.uuid4().hex[:12]}",
                "document_id": document_id,
                "document_version_id": version_id,
                "chunk_id": chunk.get("chunk_id") or "",
                "query_entities": sorted(query_entities),
                "detected_entities": sorted(detected_types),
                "matched_entities": [
                    {"entity_type": label, "action": actions[label].action}
                    for label in sorted(actions.keys() & detected_types)
                ],
                "decision": decision,
                "requires_access_request": bool(hidden_blocked_types),
            }
            result["policy_contract"] = contract
            contracts.append(contract)
            processed.append(result)

        return processed, contracts


entity_policy_service = EntityPolicyService()
