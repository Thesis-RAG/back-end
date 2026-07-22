"""
Chat orchestration service: non-streaming and streaming RAG pipeline,
Guard enforcement (intent/PII/post-LLM), and policy-contract transformations.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.message_source import MessageSource
from app.models.trace import Trace
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.message_source_repository import MessageSourceRepository
from app.repositories.system_setting_repository import system_setting_repository
from app.repositories.trace_repository import TraceRepository
from app.services.answer_service import answer_service
from app.services.audit_service import audit_service
from app.services.guard_service import guard_service
from app.services.guard_service import sanitize_masked_markers
from app.services.intent_classifier import intent_classifier
from app.services.llm_service import llm_service
from app.services.memory_service import memory_service
from app.services.policy_agent import policy_contract_agent
from app.services.prompt_injection_guard import prompt_injection_guard
from app.services.retrieval_service import retrieval_service
from app.utils.status_answer import is_no_answer

# Toggle Guard 1/2/3 enforcement; set True to enable intent and PII checks.
GUARDS_ENABLED = True
# Toggle policy-contract per-chunk enforcement.
POLICY_ENABLED = True
# Terminal assistant message statuses considered complete for listing.
DONE_STATUSES = {"success", "fallback", "no_answer", "llm_error", "blocked"}

logger = logging.getLogger(__name__)


def _safe_stream_fragment(buffer: str, fragment: str, *, final: bool = False) -> tuple[str, list[str]]:
    """Emit completed lines after removing internal redaction markers.

    Keeping an incomplete line buffered prevents a marker split across model
    tokens from reaching the client. Table rows containing a masked value are
    omitted entirely.
    """
    buffer += fragment
    parts = buffer.split("\n")
    remainder = parts.pop() if not final else ""
    if final and parts == [] and buffer:
        parts = [buffer]

    emitted: list[str] = []
    for line in parts:
        cleaned = sanitize_masked_markers(line, drop_masked_lines=True)
        if cleaned:
            emitted.append(cleaned + "\n")

    if final:
        remainder = ""
    return remainder, emitted

CHATBOT_SYSTEM_PROMPT = """
Bạn là trợ lý AI thông minh, thân thiện như ChatGPT, Gemini, Claude.
Trả lời mọi câu hỏi của người dùng một cách tự nhiên, chính xác và hữu ích.
""".strip()

# ------------------------------------------------------------------------------
# Guard — blocked-message map
# ------------------------------------------------------------------------------

# Vietnamese user-facing messages returned when Guard 1 blocks a request.
_BLOCK_MESSAGES = {
    "PROMPT_INJECTION":  "Câu hỏi của bạn có dấu hiệu cố tình can thiệp vào hệ thống. Vui lòng đặt câu hỏi bình thường.",
    "JAILBREAK":         "Câu hỏi của bạn vi phạm chính sách sử dụng. Vui lòng thử lại với câu hỏi khác.",
    "HARMFUL_INTENT":    "Yêu cầu này không thể được xử lý do vi phạm chính sách an toàn.",
    "DATA_EXFILTRATION": "Không thể thực hiện yêu cầu trích xuất dữ liệu hàng loạt.",
    "OFF_TOPIC":         "Câu hỏi này nằm ngoài phạm vi hỗ trợ của hệ thống. Vui lòng hỏi về các tài liệu nội bộ của công ty.",
    "_DEFAULT":          "Không thể xử lý yêu cầu này. Vui lòng thử lại với câu hỏi khác.",
}

# ------------------------------------------------------------------------------
# Policy — shared notice string
# ------------------------------------------------------------------------------

_POLICY_NOTICE = "[Nội dung này đã được ẩn theo chính sách phân quyền.]"


# Return the user-facing block message for the given Guard 1 classification.
def _block_message(class_: str) -> str:
    return _BLOCK_MESSAGES.get(class_, _BLOCK_MESSAGES["_DEFAULT"])


class ChatService:
    # Initialize repository instances used throughout the chat pipeline.
    def __init__(self):
        self.convs       = ConversationRepository()
        self.msgs        = MessageRepository()
        self.traces      = TraceRepository()
        self.msg_sources = MessageSourceRepository()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Create and persist a new conversation for the user.
    def create_conversation(self, db: Session, user, title: str | None) -> Conversation:
        conv = Conversation(user_id=user.id, title=title)
        self.convs.create(db, conv)
        db.commit()
        db.refresh(conv)
        return conv

    # Return the provided trace ID or generate a new one.
    def _get_trace_id(self, trace_id: str | None) -> str:
        return trace_id or uuid.uuid4().hex

    # Extract a numeric relevance score from a retrieved chunk dict.
    def _safe_score(self, item: dict) -> float | None:
        score = item.get("score")
        if score is None:
            score = item.get("relevance")
        try:
            return float(score) if score is not None else None
        except Exception:
            return None

    # Filter, dedupe, and sort retrieved chunks by score; cap at limit.
    def _normalize_retrieved(
        self, retrieved: list[dict], limit: int = 5, min_score: float = 0.0
    ) -> list[dict]:
        cleaned: list[dict] = []
        for r in retrieved or []:
            doc_text = (r.get("document_text") or "").strip()
            if not doc_text:
                continue
            score = self._safe_score(r)
            if min_score > 0.0 and score is not None and score < min_score:
                continue
            md = r.get("metadata") or {}
            cleaned.append({
                "chunk_id":       r.get("chunk_id"),
                "document_text":  doc_text,
                "metadata":       md,
                "score":          score,
                "semantic_score": r.get("semantic_score"),
                "keyword_score":  r.get("keyword_score"),
                "distance":       r.get("distance"),
            })
        cleaned.sort(
            key=lambda x: (x["score"] is not None, x["score"] if x["score"] is not None else -1.0),
            reverse=True,
        )
        return cleaned[:limit]

    # Load conversation history via memory_service using a fresh DB session.
    def _load_history(self, _db: Session, conversation_id: str, query: str) -> list[dict]:
        try:
            fresh_db = SessionLocal()
            try:
                return memory_service.load_history(fresh_db, conversation_id, query)
            finally:
                fresh_db.close()
        except Exception:
            logger.exception("memory_service.load_history failed, returning empty")
            return []

    # Load the last user/assistant pair for a conversation directly from DB.
    # Uses parent_message_id to guarantee correct user→assistant order.
    # Only returns completed (non-streaming) assistant messages.
    def _load_recent_turns_direct(self, conversation_id: str) -> list[dict]:
        try:
            from app.models.message import Message as MsgModel
            fresh_db = SessionLocal()
            try:
                last_assistants = (
                    fresh_db.query(MsgModel)
                    .filter(
                        MsgModel.conversation_id == conversation_id,
                        MsgModel.role == "assistant",
                        MsgModel.status.in_(DONE_STATUSES),
                        MsgModel.content.isnot(None),
                        MsgModel.content != "",
                    )
                    .order_by(MsgModel.created_at.desc())
                    .limit(3)
                    .all()
                )
                if not last_assistants:
                    return []
                turns: list[dict] = []
                for asst in reversed(last_assistants):
                    if asst.parent_message_id:
                        user_msg = fresh_db.get(MsgModel, asst.parent_message_id)
                        if user_msg and user_msg.content:
                            turns.append({"role": "user", "content": user_msg.content})
                    turns.append({"role": "assistant", "content": asst.content or ""})
                return turns
            finally:
                fresh_db.close()
        except Exception:
            logger.warning("_load_recent_turns_direct failed", exc_info=True)
            return []

    # Rewrite a follow-up query into a standalone question using recent conversation turns.
    # Returns the original query unchanged if rewriting is unnecessary or fails.
    def _contextualize_query(self, _db: Session, conversation_id: str, query: str) -> str:
        if not llm_service.is_configured():
            return query
        try:
            # Try memory_service first; fall back to direct DB load if empty.
            history = self._load_history(_db, conversation_id, query)
            qa_turns = [h for h in history if h.get("role") in ("user", "assistant")]
            if not qa_turns:
                qa_turns = self._load_recent_turns_direct(conversation_id)
            print(f"[CONTEXTUALIZE] qa_turns={len(qa_turns)} conv={conversation_id[:8]}", flush=True)
            if not qa_turns:
                return query

            turns: list[str] = []
            for h in qa_turns[-6:]:
                role = "Người dùng" if h["role"] == "user" else "Trợ lý"
                turns.append(f"{role}: {(h.get('content') or '')[:300]}")

            history_text = "\n".join(turns)
            print(f"[CONTEXTUALIZE] history_text={history_text[:300]!r}", flush=True)
            prompt = (
                "Dưới đây là lịch sử hội thoại gần nhất và câu hỏi tiếp theo của người dùng.\n"
                "Nhiệm vụ: Viết lại câu hỏi tiếp theo thành một câu hỏi độc lập, không cần "
                "ngữ cảnh hội thoại để hiểu. Giải quyết mọi đại từ ('người đó', 'họ', 'nó', "
                "'đó', 'anh ấy', 'cô ấy', 'điều này', 'vấn đề đó'...) thành tên/khái niệm "
                "cụ thể từ lịch sử.\n"
                "Nếu câu hỏi đã rõ ràng, không cần đại từ, trả về nguyên văn.\n"
                "Chỉ trả về câu hỏi đã viết lại, không giải thích.\n\n"
                f"Lịch sử:\n{history_text}\n\n"
                f"Câu hỏi tiếp theo: {query}\n\n"
                "Câu hỏi độc lập:"
            )
            rewritten, _, _ = llm_service.generate(
                prompt=prompt,
                system="Bạn là hệ thống xử lý ngôn ngữ. Chỉ trả về câu hỏi đã viết lại.",
                max_tokens=200,
                temperature=0.0,
            )
            print(f"[CONTEXTUALIZE] raw_rewritten={rewritten!r}", flush=True)
            rewritten = (rewritten or "").strip().strip('"').strip()
            if rewritten and rewritten != query:
                logger.info("Query contextualized: %r → %r", query, rewritten)
                print(f"[CONTEXTUALIZE] rewritten: {query!r} → {rewritten!r}", flush=True)
            else:
                print(f"[CONTEXTUALIZE] no change (rewritten={rewritten!r})", flush=True)
            return rewritten or query
        except Exception:
            logger.warning("Query contextualization failed, using original query", exc_info=True)
            return query

    # Persist source attribution records for an assistant message.
    def _persist_sources(self, db: Session, assistant_message_id: str, sources: list[dict]) -> None:
        for s in sources or []:
            src = MessageSource(
                message_id=assistant_message_id,
                document_id=s.get("documentId"),
                document_title=s.get("documentTitle"),
                version_id=s.get("versionId"),
                section_path=s.get("sectionPath"),
                relevance=s.get("relevance"),
                excerpt=s.get("excerpt"),
            )
            self.msg_sources.create(db, src)

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    # Rewrite chunk text at a higher abstraction level, removing specific sensitive values.
    def _generalize_chunk(self, chunk: dict) -> dict:
        original = chunk.get("document_text", "")
        if not original:
            return chunk

        contract            = chunk.get("_policy_contract", {})
        numeric_granularity = (contract.get("numeric_granularity") or "aggregated").lower()
        _numeric_desc = {
            "hidden":     "ẩn hoàn toàn tất cả số liệu (lương, ngân sách, KPI, mã số, ...)",
            "aggregated": "chỉ dùng số liệu tổng hợp/ước lượng (ví dụ: 'khoảng vài triệu')",
            "range_only": "chỉ dùng dạng khoảng (ví dụ: '10–20 triệu')",
            "exact":      "giữ nguyên số liệu chính xác",
        }
        numeric_hint = _numeric_desc.get(numeric_granularity, "chỉ dùng số liệu tổng hợp/ước lượng")

        prompt = (
            "Bạn là agent viết lại nội dung theo chính sách phân quyền dữ liệu.\n\n"
            "## QUY TẮC KHÁI QUÁT HÓA\n"
            "- Thay thế thông tin định danh cụ thể (tên người, mã số, địa chỉ, liên lạc) "
            "bằng mô tả cấp cao hơn (ví dụ: 'một nhân viên', 'phòng ban liên quan').\n"
            f"- Số liệu: {numeric_hint}.\n"
            "Chỉ trả về đoạn văn đã viết lại, không giải thích.\n\n"
            f"Đoạn văn gốc:\n{original[:1500]}"
        )
        try:
            text, _, _ = llm_service.generate(prompt=prompt, max_tokens=1024, temperature=0.0)
            if text and text.strip():
                result = dict(chunk)
                result["document_text"] = text.strip()
                return result
        except Exception as exc:
            logger.warning("Generalize chunk failed chunk=%s: %s", chunk.get("chunk_id"), exc)
        # LLM error — return policy notice as fallback.
        result = dict(chunk)
        result["document_text"] = "[Nội dung đã được khái quát hóa theo chính sách phân quyền.]"
        return result

    # Remove sensitive values in-place, keeping the surrounding structure intact.
    def _redact_chunk(self, chunk: dict) -> dict:
        original = chunk.get("document_text", "")
        if not original:
            return chunk
        contract = chunk.get("_policy_contract", {})
        numeric_granularity = (contract.get("numeric_granularity") or "hidden").lower()
        _NUMERIC_DESC = {
            "hidden":     "ẩn hoàn toàn tất cả số liệu",
            "aggregated": "chỉ dùng số liệu tổng hợp/ước lượng",
            "range_only": "chỉ dùng dạng khoảng",
            "exact":      "giữ nguyên số liệu chính xác",
        }
        numeric_hint = _NUMERIC_DESC.get(numeric_granularity, "ẩn hoàn toàn tất cả số liệu")
        prompt = (
            "Bạn là agent che thông tin nhạy cảm theo chính sách phân quyền dữ liệu.\n"
            "Thay thế TẤT CẢ thông tin nhạy cảm (tên người, mã số, địa chỉ, thông tin liên lạc, "
            "dữ liệu nội bộ) bằng '[ẨN]'. Giữ nguyên cấu trúc và ngữ cảnh của đoạn văn.\n"
            f"Số liệu: {numeric_hint}.\n"
            "Chỉ trả về đoạn văn đã xử lý, không giải thích.\n\n"
            f"Đoạn văn gốc:\n{original[:1500]}"
        )
        try:
            text, _, _ = llm_service.generate(prompt=prompt, max_tokens=1024, temperature=0.0)
            if text and text.strip():
                result = dict(chunk)
                result["document_text"] = text.strip()
                return result
        except Exception as exc:
            logger.warning("Redact chunk failed chunk=%s: %s", chunk.get("chunk_id"), exc)
        result = dict(chunk)
        result["document_text"] = _POLICY_NOTICE
        return result

    # Replace personal identifiers with consistent aliases (Employee A, Department X, …).
    def _anonymize_chunk(self, chunk: dict) -> dict:
        original = chunk.get("document_text", "")
        if not original:
            return chunk
        contract = chunk.get("_policy_contract", {})
        numeric_granularity = (contract.get("numeric_granularity") or "aggregated").lower()
        _NUMERIC_DESC = {
            "hidden":     "ẩn hoàn toàn tất cả số liệu",
            "aggregated": "chỉ dùng số liệu tổng hợp/ước lượng",
            "range_only": "chỉ dùng dạng khoảng",
            "exact":      "giữ nguyên số liệu chính xác",
        }
        numeric_hint = _NUMERIC_DESC.get(numeric_granularity, "chỉ dùng số liệu tổng hợp/ước lượng")
        prompt = (
            "Bạn là agent ẩn danh hóa dữ liệu.\n"
            "Thay thế TẤT CẢ định danh cá nhân và mã số cụ thể bằng alias nhất quán "
            "(VD: tên người → 'Nhân viên A', mã số → 'ID-001', ...).\n"
            f"Số liệu: {numeric_hint}.\n"
            "Chỉ trả về đoạn văn đã viết lại, không giải thích.\n\n"
            f"Đoạn văn gốc:\n{original[:1500]}"
        )
        try:
            text, _, _ = llm_service.generate(prompt=prompt, max_tokens=1024, temperature=0.0)
            if text and text.strip():
                result = dict(chunk)
                result["document_text"] = text.strip()
                return result
        except Exception as exc:
            logger.warning("Anonymize chunk failed chunk=%s: %s", chunk.get("chunk_id"), exc)
        result = dict(chunk)
        result["document_text"] = "[Nội dung đã được ẩn danh hóa theo chính sách phân quyền.]"
        return result

    # Summarize chunk into 1-2 sentences without revealing specific values.
    def _summarize_chunk(self, chunk: dict) -> dict:
        original = chunk.get("document_text", "")
        if not original:
            return chunk
        prompt = (
            "Bạn là agent tóm tắt nội dung theo chính sách phân quyền dữ liệu.\n"
            "Hãy tóm tắt đoạn văn sau thành 1-2 câu ngắn gọn, "
            "chỉ nêu chủ đề và loại thông tin có trong đoạn. "
            "TUYỆT ĐỐI không tiết lộ giá trị cụ thể (tên người, số liệu, mã số, địa chỉ, ...).\n"
            "Ví dụ đầu ra: 'Thông tin nhân viên bao gồm thông tin cá nhân và liên hệ.'\n"
            "Chỉ trả về câu tóm tắt, không giải thích.\n\n"
            f"Đoạn văn:\n{original[:1500]}"
        )
        try:
            text, _, _ = llm_service.generate(prompt=prompt, max_tokens=256, temperature=0.0)
            if text and text.strip():
                result = dict(chunk)
                result["document_text"] = text.strip()
                return result
        except Exception as exc:
            logger.warning("Summarize chunk failed chunk=%s: %s", chunk.get("chunk_id"), exc)
        result = dict(chunk)
        result["document_text"] = "[Nội dung đã được tóm tắt theo chính sách phân quyền.]"
        return result

    # Dispatch each chunk to the appropriate transform based on policy decision flags.
    def _apply_transforms(self, chunks: list[dict]) -> list[dict]:
        result = []
        for c in chunks:
            if c.get("_needs_field_policy"):
                result.append(self._apply_field_scoped_policy(c))
            elif c.get("_needs_redact"):
                result.append(self._redact_chunk(c))
            elif c.get("_needs_anonymize"):
                result.append(self._anonymize_chunk(c))
            elif c.get("_needs_generalize"):
                result.append(self._generalize_chunk(c))
            elif c.get("_needs_summarize"):
                result.append(self._summarize_chunk(c))
            elif c.get("_needs_watermark"):
                wc = dict(c)
                wc["document_text"] = (c.get("document_text") or "") + "\n\n[Watermark: nội dung được kiểm soát theo chính sách phân quyền]"
                result.append(wc)
            else:
                result.append(c)
        return result

    def _apply_field_scoped_policy(self, chunk: dict) -> dict:
        """Mask only entity spans covered by field-scoped policy rules.

        This is intentionally deterministic.  A second LLM rewrite could
        accidentally reconstruct a blocked salary while rewriting an allowed
        personal field, so salary/PII replacements happen before answer
        generation and never leave the protected value in the RAG context.
        """
        text = chunk.get("document_text") or ""
        contract = chunk.get("_policy_contract") or {}
        entities = chunk.get("_policy_entities") or []
        field_rules = contract.get("field_rules") or []

        def clear_internal_fields(value: dict) -> dict:
            cleaned = dict(value)
            cleaned.pop("_policy_entities", None)
            cleaned.pop("_needs_field_policy", None)
            return cleaned

        if not text or not field_rules or not entities:
            return clear_internal_fields(chunk)

        builtin_flags = {
            "email": {"has_pii"}, "phone": {"has_pii"},
            "national_id": {"has_pii", "has_hr"}, "tax_id": {"has_pii", "has_hr"},
            "social_insurance": {"has_pii", "has_hr"}, "bank_account": {"has_pii", "has_financial"},
            "dob": {"has_pii", "has_hr"}, "money": {"has_financial"},
            "percentage": {"has_financial"},
        }
        detail_order = {"redact": 4, "anonymize": 3, "generalize": 2, "summarize": 1}
        replacements: list[tuple[int, int, str]] = []

        for entity in sorted(entities, key=lambda item: (int(item.get("start", 0)), -int(item.get("end", 0)))):
            try:
                start, end = int(entity.get("start", 0)), int(entity.get("end", 0))
            except (TypeError, ValueError):
                continue
            if start < 0 or end <= start or end > len(text):
                continue
            label = str(entity.get("label") or "").lower()
            flags = set(entity.get("flags") or builtin_flags.get(label, set()))
            matches = []
            for rule in field_rules:
                targets = {str(v).lower() for v in (rule.get("target_entity_types") or [])}
                target_flags = {str(v).lower() for v in (rule.get("target_flags") or [])}
                if (targets and label in targets) or (target_flags and flags.intersection(target_flags)):
                    matches.append(rule)
            if not matches:
                continue

            normalized_actions = {
                str(rule.get("action") or (rule.get("contract") or {}).get("violation_action") or "allow").lower()
                for rule in matches
            }

            # A targeted ALLOW is a no-op for this entity. It must not inherit
            # the default max_detail=generalize and accidentally redact it.
            if normalized_actions <= {"allow"}:
                continue

            if "block" in normalized_actions or "deny" in normalized_actions:
                replacement = "[ĐÃ ẨN THEO CHÍNH SÁCH]"
            else:
                conditional_matches = [
                    rule for rule in matches
                    if str(rule.get("action") or (rule.get("contract") or {}).get("violation_action") or "allow").lower()
                    in {"conditional", "redact", "anonymize", "generalize", "summarize"}
                ]
                # Watermark and ALLOW do not rewrite a field. If no
                # conditional rule remains, preserve the entity as-is.
                if not conditional_matches:
                    continue
                rule = max(
                    conditional_matches,
                    key=lambda item: (
                        detail_order.get(str((item.get("contract") or {}).get("max_detail", "generalize")).lower(), 2),
                        int(item.get("priority") or 0),
                    ),
                )
                detail = str((rule.get("contract") or {}).get("max_detail", "generalize")).lower()
                if detail == "redact":
                    replacement = "[ĐÃ ẨN]"
                elif detail == "anonymize":
                    replacement = "[ĐỐI TƯỢNG ĐÃ ẨN DANH]"
                elif detail == "summarize":
                    replacement = "[THÔNG TIN ĐÃ TÓM TẮT]"
                elif label in {"money", "percentage", "bank_account"} or "financial" in label:
                    replacement = "[GIÁ TRỊ TÀI CHÍNH ĐÃ KHÁI QUÁT]"
                else:
                    replacement = "[THÔNG TIN CÁ NHÂN ĐÃ KHÁI QUÁT]"
            # Keep field redaction user-facing and never expose internal
            # bracketed entity labels in the answer context.
            replacement = "giá trị đã được ẩn theo chính sách"
            replacements.append((start, end, replacement))

        if not replacements:
            return clear_internal_fields(chunk)
        # Apply right-to-left and avoid overlapping replacements.
        output = text
        occupied_end = len(text) + 1
        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            if end > occupied_end:
                continue
            output = output[:start] + replacement + output[end:]
            occupied_end = start
        result = dict(chunk)
        result["document_text"] = output
        return clear_internal_fields(result)

    # Run the policy-contract agent per chunk; return (approved_chunks, contracts).
    # Corp members (admin) bypass all policy constraints.
    def _apply_policy_contracts(
        self,
        db: Session,
        chunks: list[dict],
        user,
        raw_query: str,
        intent_class: str,
    ) -> tuple[list[dict], list[dict]]:
        approved: list[dict] = []
        contracts: list[dict] = []

        declared_sensitivity = 2

        # ── Corp member bypass ────────────────────────────────────────────
        # Corp members (users with a position in the root OrgUnit) bypass all policy.
        try:
            from sqlalchemy import text as _sql
            _corp_check = db.execute(_sql("""
                SELECT 1
                FROM user_oui_positions uop
                JOIN org_unit_instances oui ON uop.oui_id = oui.id
                JOIN org_units ou ON oui.ou_id = ou.id
                WHERE uop.user_id = :uid AND ou.parent_id IS NULL
                LIMIT 1
            """), {"uid": user.id}).fetchone()
            if _corp_check:
                print(f"[POLICY] corp member uid={user.id[:8]} → bypass policy, {len(chunks)} chunks approved")
                return list(chunks), []
        except Exception as exc:
            logger.warning("Corp member check failed: %s", exc)

        user_positions: list[dict] = []
        for up in (getattr(user, "oui_positions", None) or []):
            pos = getattr(up, "position", None)
            if pos and up.oui_id:
                user_positions.append({
                    "oui_id":    up.oui_id,
                    "clearance": pos.clearance,
                })

        # ── Batch GLiNER for all chunks at once ───────────────────────────
        chunk_texts = [c.get("document_text") or "" for c in chunks]
        try:
            from app.services.entity_extractor import extract_realtime_batch_detailed
            batch_details = extract_realtime_batch_detailed(chunk_texts, db=db)
        except Exception as exc:
            logger.warning("Batch entity extraction failed: %s", exc)
            batch_details = [{"entities": [], "entity_types": set(), "flags": set()} for _ in chunks]

        # Shared across chunks — LLM relevance filter fires at most once per unique rule set.
        rule_filter_cache: dict = {}

        for i, chunk in enumerate(chunks):
            md = chunk.get("metadata") or {}
            chunk_id = chunk.get("chunk_id") or "unknown"
            chunk_text = chunk.get("document_text") or ""
            chunk_sensitivity = int(md.get("sensitivity") or declared_sensitivity)

            try:
                contract = policy_contract_agent.generate_contract(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    chunk_metadata=md,
                    declared_sensitivity=chunk_sensitivity,
                    user_role=getattr(user, "role", "Employee"),
                    user_level=getattr(user, "clearance_level", 1),
                    user_department=getattr(user, "department", ""),
                    user_id=user.id,
                    intent_class=intent_class,
                    raw_query=raw_query,
                    user_positions=user_positions,
                    detected_entity_types=batch_details[i].get("entity_types", set()),
                    detected_flags=batch_details[i].get("flags", set()),
                    detected_entities=batch_details[i].get("entities", []),
                    rule_filter_cache=rule_filter_cache,
                    db=db,
                )
                contracts.append(contract)

                decision   = str(contract.get("decision", "allow")).lower()
                max_detail = contract.get("max_detail", "generalize")
                _ctr = {
                    "max_detail":          max_detail,
                    "numeric_granularity": contract.get("numeric_granularity", "aggregated"),
                }
                print(f"[POLICY] apply chunk={chunk_id[:8]} decision={decision} max_detail={max_detail}")
                if decision == "block":
                    print(f"[POLICY]   → BLOCK: chunk bị loại bỏ")
                    logger.info("Policy block chunk=%s", chunk_id)
                    continue
                elif decision == "conditional":
                    chunk = dict(chunk)
                    chunk["_policy_contract"] = _ctr
                    if max_detail == "redact":
                        chunk["_needs_redact"] = True
                    elif max_detail == "anonymize":
                        chunk["_needs_anonymize"] = True
                    elif max_detail == "summarize":
                        chunk["_needs_summarize"] = True
                    else:  # generalize (default)
                        chunk["_needs_generalize"] = True
                    print(f"[POLICY]   → CONDITIONAL: transform={max_detail} numeric={_ctr['numeric_granularity']}")
                    logger.info("Policy conditional chunk=%s max_detail=%s", chunk_id, max_detail)
                elif decision == "field_scoped":
                    chunk = dict(chunk)
                    policy_contract = contract.get("contract") or {}
                    chunk["_policy_contract"] = policy_contract
                    chunk["_policy_entities"] = batch_details[i].get("entities", [])
                    chunk["_needs_field_policy"] = True
                    logger.info("Policy field-scoped chunk=%s rules=%s", chunk_id, len(policy_contract.get("field_rules") or []))
                elif decision == "watermark":
                    chunk = dict(chunk)
                    chunk["_needs_watermark"] = True
                    print(f"[POLICY]   → WATERMARK")
                    logger.info("Policy watermark chunk=%s", chunk_id)
                else:
                    print(f"[POLICY]   → ALLOW: chunk đi qua không đổi")
                # allow → pass through as-is

                approved.append(chunk)
            except Exception as exc:
                logger.warning("Policy contract failed chunk=%s: %s", chunk_id, exc)
                approved.append(chunk)

        return approved, contracts

    # Trigger a conversation summary update in a background thread.
    def _update_summary_background(self, conversation_id: str) -> None:
        try:
            with SessionLocal() as db:
                memory_service.update_summary(db, conversation_id)
                db.commit()
        except Exception:
            logger.exception("Background summary update failed conv_id=%s", conversation_id)

    # Parse all [N] citation markers from the LLM answer and return their indices.
    def _extract_cited_indices(self, answer_text: str) -> set[int]:
        if not answer_text:
            return set()
        found = re.findall(r"\[(\d+)\]", answer_text)
        return {int(n) for n in found}

    # Build a source entry dict from a retrieved chunk for the API response.
    @staticmethod
    def _make_source_entry(r: dict) -> dict:
        md = r.get("metadata", {}) or {}
        return {
            "documentId":    md.get("document_id"),
            "documentTitle": md.get("document_title") or md.get("document_id"),
            "versionId":     md.get("document_version_id"),
            "sectionPath":   md.get("section_heading"),
            "relevance":     r.get("score") if r.get("score") is not None else r.get("relevance"),
            "excerpt":       r.get("document_text") or md.get("excerpt"),
            "docRestricted": r.get("doc_restricted", False),
        }

    # Renumber [N] citation markers to be sequential (1-based) and return matched sources.
    # No [N] found → answer unchanged, sources = restricted chunks only.
    def _normalize_citations(
        self,
        answer_text: str | None,
        retrieved: list[dict],
    ) -> tuple[str | None, list[dict]]:
        if not answer_text:
            return answer_text, []

        cited = self._extract_cited_indices(answer_text)

        if not cited:
            # LLM cited nothing — surface only restricted chunks so the user
            # knows to request access for potentially relevant content.
            seen: set[str] = set()
            restricted: list[dict] = []
            for r in (retrieved or []):
                if not r.get("doc_restricted", False):
                    continue
                md = r.get("metadata", {}) or {}
                doc_id = md.get("document_id")
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                entry = self._make_source_entry(r)
                entry["docRestricted"] = True
                restricted.append(entry)
            return answer_text, restricted

        # Sort cited indices → stable sequential mapping: old→new
        sorted_cited = sorted(cited)
        old_to_new = {old: new for new, old in enumerate(sorted_cited, start=1)}

        # Renumber every [N] in the answer text
        def _replace(m: re.Match) -> str:
            new_n = old_to_new.get(int(m.group(1)))
            return f"[{new_n}]" if new_n is not None else m.group(0)

        normalized_text = re.sub(r"\[(\d+)\]", _replace, answer_text)

        # Build sources in sorted citation order (exactly one entry per cited chunk)
        retrieved_list = list(retrieved or [])
        sources: list[dict] = []
        for old_idx in sorted_cited:
            if 1 <= old_idx <= len(retrieved_list):
                sources.append(self._make_source_entry(retrieved_list[old_idx - 1]))

        return normalized_text, sources

    # Legacy wrapper — prefer _normalize_citations for LLM answers.
    def _build_sources_from_retrieved(self, retrieved: list[dict], answer_text: str | None = None) -> list[dict]:
        _, sources = self._normalize_citations(answer_text, retrieved)
        return sources

    # ------------------------------------------------------------------
    # Guard helpers
    # ------------------------------------------------------------------

    # Persist a blocked assistant message and audit log when Guard 1 blocks a request.
    def _make_blocked_assistant_message(
        self,
        db: Session,
        conversation_id: str,
        user_msg: Message,
        block_text: str,
        trace: Trace,
        tid: str,
        user,
    ) -> tuple[Message, Message, list]:
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=block_text,
            status="blocked",
            trace_id=tid,
            parent_message_id=user_msg.id,
        )
        self.msgs.create(db, assistant_msg)
        db.flush()
        db.refresh(assistant_msg)

        trace.assistant_output_summary = block_text
        trace.status = "blocked"

        audit_service.log_action(
            db, trace_id=tid, user_id=user.id,
            action="chat.message.blocked", resource_type="conversation",
            resource_id=conversation_id, decision="deny",
            input_json={"message": user_msg.content},
            output_json={"reason": block_text},
        )

        db.commit()
        db.refresh(assistant_msg)
        return user_msg, assistant_msg, []

    # ------------------------------------------------------------------
    # RAG pipeline helpers  (shared by post_message + post_message_stream)
    # ------------------------------------------------------------------

    # Run Retrieval → Guard 2 → Policy → history; return context dict for LLM generation.
    def _run_rag_pipeline(
        self,
        db: Session,
        user,
        conversation_id: str,
        effective_query: str,
        tid: str,
        oui_ids: list[str] | None = None,
        chat_mode: str = "rag",
    ) -> dict:
        _top_k     = int(system_setting_repository.get(db, "rag.top_k") or 5)
        _min_score = float(system_setting_repository.get(db, "rag.similarity_threshold") or 0.0)

        try:
            retrieved_raw = retrieval_service.retrieve(
                query=effective_query,
                user=user,
                top_k=_top_k,
                oui_ids=oui_ids,
                chat_mode=chat_mode,
                db=db,
            )
        except Exception:
            logger.exception("Retrieval failed trace_id=%s chat_mode=%s", tid, chat_mode)
            retrieved_raw = []

        retrieved = self._normalize_retrieved(retrieved_raw, limit=_top_k, min_score=_min_score)
        if not retrieved and _min_score > 0.0:
            retrieved = self._normalize_retrieved(retrieved_raw, limit=1, min_score=0.0)
            if retrieved:
                logger.info("Retrieval fallback best-effort: top1 score=%.3f", retrieved[0].get("score") or 0)

        # Retrieved documents are untrusted data.  Keep an audit marker for
        # suspicious instructions; llm_service also wraps the text with a
        # hard data boundary before sending it to the model.
        retrieved = prompt_injection_guard.annotate_chunks(retrieved)

        # ── [GUARD 2] PII scan on retrieved chunks ─────────────────────
        logger.info("USER ROLE: %s USER ID: %s", getattr(user, "role", None), getattr(user, "id", None))
        if GUARDS_ENABLED:
            retrieved = guard_service.scan_chunks(retrieved, user=user)

        # ── [POLICY] Policy-contract enforcement ──────────────────────
        policy_contracts: list[dict] = []
        has_watermark = False
        if POLICY_ENABLED and retrieved:
            query_intent = intent_classifier.classify(effective_query)
            retrieved, policy_contracts = self._apply_policy_contracts(
                db, retrieved, user, effective_query, query_intent
            )
            has_watermark = any(str(c.get("decision", "")).lower() in {"watermark", "allow_with_watermark"} for c in policy_contracts)
            print(f"[POLICY] intent={query_intent} approved={len(retrieved)}/{len(retrieved_raw)} watermark={has_watermark}")
            for _c in policy_contracts:
                rules = [r.get("rule_code") for r in _c.get("applied_rules", [])]
                domains = [d.get("code") for d in _c.get("domains", [])]
                print(f"[POLICY] chunk={_c.get('chunk_id','?')[:8]} domains={domains} decision={_c.get('decision')} rules={rules}")
            retrieved = self._apply_transforms(retrieved)

        # Collect unique applied rules (non-allow only) for SSE streaming to client.
        seen_rule_codes: set[str] = set()
        applied_rules: list[dict] = []
        for contract in policy_contracts:
            final_decision = str(contract.get("decision", "allow")).lower()
            if final_decision == "allow":
                continue
            for rule in contract.get("applied_rules", []):
                # Only surface rules whose own action determined the final outcome.
                # e.g. block+conditional → only block rules shown; conditional+conditional → all shown.
                if final_decision != "field_scoped" and str(rule.get("action", "")).lower() != final_decision:
                    continue
                code = rule.get("rule_code", "")
                if code and code not in seen_rule_codes:
                    seen_rule_codes.add(code)
                    applied_rules.append({
                        "rule_code": code,
                        "name":      rule.get("name", code),
                        "action":    rule.get("action", final_decision) if final_decision == "field_scoped" else final_decision,
                        "domain":    rule.get("domain", ""),
                    })

        has_restricted = bool(retrieved and any(
            c.get("document_text") == _POLICY_NOTICE for c in retrieved
        )) or bool(
            policy_contracts and any(str(c.get("decision", "")).lower() in ("block", "deny", "redact") for c in policy_contracts)
        )
        all_restricted = bool(retrieved and all(
            c.get("document_text") == _POLICY_NOTICE for c in retrieved
        ))

        history = self._load_history(db, conversation_id, effective_query)
        safe_history: list = [] if has_restricted else history

        return {
            "retrieved":        retrieved,
            "retrieved_raw":    retrieved_raw,
            "policy_contracts": policy_contracts,
            "applied_rules":    applied_rules,
            "has_watermark":    has_watermark,
            "has_restricted":   has_restricted,
            "all_restricted":   all_restricted,
            "safe_history":     safe_history,
        }

    # Guard 3: post-LLM PII and secret scan; returns (possibly redacted) answer text.
    def _apply_guard3(self, answer_text: str, user, tid: str) -> str:
        if not answer_text:
            return answer_text
        if not GUARDS_ENABLED:
            return sanitize_masked_markers(answer_text, drop_masked_lines=True)

        post_scan = guard_service.scan_response(answer_text, user=user)

        if post_scan.judge and post_scan.judge.should_block:
            logger.warning("Guard3b BLOCK: reason=%s trace_id=%s", post_scan.judge.reason, tid)
            return "Xin lỗi, nội dung này không thể hiển thị do vi phạm chính sách bảo mật."

        if post_scan.has_pii:
            logger.warning("Guard3 REDACT: entities=%s trace_id=%s",
                           [e.entity_type for e in post_scan.entities], tid)
            answer_text = post_scan.redacted_text

        if post_scan.has_pii or post_scan.has_secret:
            logger.warning("Guard3 POST-LLM: has_pii=%s has_secret=%s trace_id=%s",
                           post_scan.has_pii, post_scan.has_secret, tid)

        return sanitize_masked_markers(answer_text, drop_masked_lines=True)

    # ------------------------------------------------------------------
    # list_messages_flat
    # ------------------------------------------------------------------

    # Return all user/assistant message pairs for a conversation as a flat list.
    def list_messages_flat(self, db: Session, conversation_id: str, limit: int = 1000) -> list[dict]:
        from app.models.message import Message as MsgModel

        user_msgs = (
            db.query(MsgModel)
            .filter(
                MsgModel.conversation_id == conversation_id,
                MsgModel.role == "user",
            )
            .order_by(MsgModel.created_at.asc())
            .limit(limit)
            .all()
        )

        assistant_msgs_raw = (
            db.query(MsgModel)
            .filter(
                MsgModel.conversation_id == conversation_id,
                MsgModel.role == "assistant",
                MsgModel.status.in_(DONE_STATUSES),
                MsgModel.content.isnot(None),
                MsgModel.content != "",
            )
            .order_by(MsgModel.created_at.asc())
            .all()
        )

        assistant_by_parent: dict[str, MsgModel] = {}
        for a in assistant_msgs_raw:
            if a.parent_message_id:
                assistant_by_parent[a.parent_message_id] = a

        out = []
        for user_msg in user_msgs:
            assistant_msg = assistant_by_parent.get(user_msg.id)
            srcs = self.msg_sources.list_by_message(db, assistant_msg.id) if assistant_msg else []

            out.append({
                "conversationId": conversation_id,
                "messageId": user_msg.id,
                "content": user_msg.content,
                "createdAt": user_msg.created_at,
                "attachedFileName": user_msg.attached_file_name,
                "traceId": (assistant_msg.trace_id if assistant_msg else None) or user_msg.trace_id,
                "assistantMessage": {
                    "id": assistant_msg.id,
                    "content": assistant_msg.content,
                    "status": assistant_msg.status,
                    "createdAt": assistant_msg.created_at,
                } if assistant_msg else None,
                "sources": [
                    {
                        "documentId": s.document_id,
                        "documentTitle": s.document_title,
                        "versionId": s.version_id,
                        "sectionPath": s.section_path,
                        "relevance": s.relevance,
                        "excerpt": s.excerpt,
                        "surroundingContext": s.surrounding_context,
                    }
                    for s in (srcs or [])
                ],
            })

        return out

    # ------------------------------------------------------------------
    # post_message  (non-streaming)
    # ------------------------------------------------------------------

    # Handle a non-streaming chat turn: Guard 1 → RAG pipeline → LLM → Guard 3 → persist.
    def post_message(
        self,
        db: Session,
        user,
        conversation_id: str,
        content: str,
        client_message_id: str | None,
        trace_id: str,
    ):
        tid = self._get_trace_id(trace_id)
        _start = time.perf_counter()
        content = prompt_injection_guard.normalize(content)

        if client_message_id:
            existing = self.msgs.find_by_client_id(db, conversation_id, client_message_id)
            if existing:
                return existing, None, None

        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            client_message_id=client_message_id,
            trace_id=tid,
        )
        self.msgs.create(db, user_msg)
        db.flush()
        db.refresh(user_msg)

        tr = Trace(
            trace_id=tid,
            conversation_id=conversation_id,
            message_id=user_msg.id,
            user_id=user.id,
            user_input=content[:2000],
            status="running",
        )
        self.traces.create(db, tr)
        db.flush()

        # ── [GUARD 1] Intent classification ───────────────────────────
        if GUARDS_ENABLED:
            intent = guard_service.check_intent(content)
            if intent.blocked:
                block_text = _block_message(intent.class_)
                return self._make_blocked_assistant_message(
                    db, conversation_id, user_msg, block_text, tr, tid, user,
                )
            effective_query = intent.rewrite if intent.should_rewrite else content
        else:
            effective_query = content

        effective_query = self._contextualize_query(db, conversation_id, effective_query)
        ctx = self._run_rag_pipeline(db, user, conversation_id, effective_query, tid)
        retrieved        = ctx["retrieved"]
        retrieved_raw    = ctx["retrieved_raw"]
        policy_contracts = ctx["policy_contracts"]
        has_watermark    = ctx["has_watermark"]
        all_restricted   = ctx["all_restricted"]
        safe_history     = ctx["safe_history"]

        answer_text: str | None = None
        llm_raw: Any = None
        prompt: str | None = None
        sources: list[dict] = []
        assistant_status = "fallback"
        llm_text: str | None = None

        # All chunks were REDACT'd — respond directly without calling the LLM.
        if all_restricted:
            answer_text = "Thông tin này bị hạn chế theo chính sách phân quyền của hệ thống và không thể hiển thị."
            sources = []
            assistant_status = "success"

        elif llm_service.is_configured():
            try:
                _has_transform = bool(policy_contracts and any(
                    c.get("decision") in ("ANONYMIZE", "GENERALIZE", "SUMMARIZE")
                    for c in policy_contracts
                ))
                _extra_instructions = (
                    "Dữ liệu trong context đã được ẩn danh hóa theo chính sách phân quyền "
                    "(tên người thay bằng alias như 'Nhân viên A', mã số thay bằng ID đại diện, "
                    "số liệu thể hiện dạng khoảng hoặc tổng hợp). "
                    "Hãy trả lời dựa trên thông tin ẩn danh đó. "
                    "Không xác định danh tính cụ thể. "
                    "Không nói 'không có trong tài liệu' nếu thông tin liên quan (dù đã ẩn danh) thực sự có trong context."
                ) if _has_transform else (
                    "Nếu câu hỏi là dạng hỏi trực tiếp về một người, "
                    "hãy trả lời đúng thông tin đó, ngắn gọn, không kèm dữ liệu thừa."
                )
                prompt = llm_service.build_prompt(
                    question=effective_query,
                    contexts=retrieved,
                    chat_history=safe_history,
                    extra_instructions=_extra_instructions,
                )
                llm_text, llm_raw, _ = llm_service.generate(
                    prompt=prompt, max_tokens=512, temperature=0.0,
                )
                if llm_text and llm_text.strip():
                    answer_text = llm_text.strip()
                assistant_status = "no_answer" if is_no_answer(answer_text) else "success"
                if assistant_status == "success":
                    answer_text, sources = self._normalize_citations(answer_text, retrieved)
            except Exception:
                logger.exception("LLM generation failed trace_id=%s", tid)
                answer_text = None
                assistant_status = "llm_error"

        if not answer_text:
            answer_text, sources = answer_service.generate(user_input=effective_query, retrieved=retrieved)
            assistant_status = "fallback"
            if not sources:
                answer_text, sources = self._normalize_citations(answer_text, retrieved)

        # ── [GUARD 3] PII + secret scan on LLM response ───────────────
        answer_text = self._apply_guard3(answer_text, user, tid)

        # ── Watermark notice ──────────────────────────────────────────
        if has_watermark and answer_text:
            answer_text = (
                answer_text
                + "\n\n---\nNội dung này được truy cập theo điều kiện kiểm soát phân quyền. Hoạt động truy vấn đã được ghi nhận."
            )

        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer_text,
            status=assistant_status,
            trace_id=tid,
            parent_message_id=user_msg.id,
        )
        self.msgs.create(db, assistant_msg)
        db.flush()
        db.refresh(assistant_msg)

        self._persist_sources(db, assistant_msg.id, sources)

        _latency_ms = int((time.perf_counter() - _start) * 1000)
        tr.assistant_output_summary = answer_text[:2000] if answer_text else None
        tr.retrieved_sources = retrieved_raw
        tr.llm_prompt  = prompt
        tr.llm_response = {
            "text":        llm_text,
            "response_id": getattr(llm_raw, "id", None),
            "model":       getattr(llm_raw, "model", None),
        }
        tr.timings = {"total_ms": _latency_ms}
        tr.status = "completed"

        audit_service.log_action(
            db, trace_id=tid, user_id=user.id,
            action="chat.message", resource_type="conversation",
            resource_id=conversation_id, decision="allow",
            input_json={"message": content, "effective_query": effective_query},
            output_json={"assistant_message": answer_text, "sources": sources},
            latency_ms=_latency_ms,
        )

        db.commit()
        db.refresh(assistant_msg)

        threading.Thread(
            target=self._update_summary_background,
            args=(conversation_id,),
            daemon=True,
        ).start()

        return user_msg, assistant_msg, sources

    # ------------------------------------------------------------------
    # post_message_stream  (streaming with Guards)
    # ------------------------------------------------------------------

    # Handle a streaming chat turn: Guard 1 → RAG/chatbot pipeline → Guard 3 → persist; yields SSE events.
    def post_message_stream(
        self,
        db: Session,
        user,
        conversation_id: str,
        content: str,
        client_message_id: str | None,
        trace_id: str,
        oui_ids: list[str] | None = None,
        mode: str = "rag",
        chat_source="rag",
        file_content: str | None = None,
        file_name: str | None = None,
    ):
        tid = self._get_trace_id(trace_id)
        _start = time.perf_counter()
        content = prompt_injection_guard.normalize(content)
        llm_extra_context = ""
        if file_content:
            label = file_name or "file"
            file_content = prompt_injection_guard.wrap_untrusted_context(file_content[:40000])
            llm_extra_context = f"[Tài liệu đính kèm: {label}]\n\n{file_content[:40000]}\n\n---\n"

        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            client_message_id=client_message_id,
            trace_id=tid,
            attached_file_name=file_name,
        )
        tr = Trace(
            trace_id=tid,
            conversation_id=conversation_id,
            message_id=user_msg.id,
            user_id=user.id,
            user_input=content[:2000],
            status="running",
        )
        self.traces.create(db, tr)
        self.msgs.create(db, user_msg)
        db.flush()
        db.refresh(user_msg)

        # ── [GUARD 1] Intent classification (stream) ───────────────────
        if GUARDS_ENABLED:
            intent = guard_service.check_intent(content)

            if intent.blocked:
                # Create the blocked assistant message immediately without a streaming placeholder.
                block_text = _block_message(intent.class_)

                assistant_msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=block_text,
                    status="blocked",
                    trace_id=tid,
                    parent_message_id=user_msg.id,
                )
                self.msgs.create(db, assistant_msg)
                db.flush()
                db.refresh(assistant_msg)

                tr.assistant_output_summary = block_text
                tr.timings = {"total_ms": int((time.perf_counter() - _start) * 1000)}
                tr.status = "blocked"

                audit_service.log_action(
                    db, trace_id=tid, user_id=user.id,
                    action="chat.message.blocked", resource_type="conversation",
                    resource_id=conversation_id, decision="deny",
                    input_json={"message": content},
                    output_json={"reason": block_text},
                    latency_ms=int((time.perf_counter() - _start) * 1000),
                )
                db.commit()

                # Emit stream events then terminate.
                yield {"type": "message_start", "messageId": assistant_msg.id, "userMessageId": user_msg.id}
                yield {"type": "token", "text": block_text}
                yield {"type": "done", "content": block_text, "sources": [], "messageId": assistant_msg.id, "blocked": True, "blockClass": intent.class_}
                return

            # Use rewritten query for non-privileged users if Guard 1 suggests a rewrite.
            is_corp = getattr(user, "is_corp_member", False)
            max_clearance = getattr(user, "max_clearance", 1)
            is_privileged = is_corp and max_clearance >= 4
            effective_query = content
            if intent.should_rewrite and not is_privileged:
                effective_query = intent.rewrite

        else:
            effective_query = content

        effective_query = self._contextualize_query(db, conversation_id, effective_query)

        # Create a streaming placeholder assistant message to return the ID immediately.
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content="",
            status="streaming",
            trace_id=tid,
            parent_message_id=user_msg.id,
        )
        self.msgs.create(db, assistant_msg)
        db.flush()
        db.refresh(assistant_msg)
        db.commit()

        yield {"type": "message_start", "messageId": assistant_msg.id, "userMessageId": user_msg.id}

        full_text            = ""
        sources              = []
        retrieved_raw:  list[dict] = []
        stream_applied_rules: list[dict] = []

        # ── CHATBOT MODE ──────────────────────────────────────────────
        if mode == "chatbot":
            if llm_service.is_configured():
                try:
                    history = self._load_history(db, conversation_id, effective_query)
                    history_messages = [h for h in history if h["role"] in ("user", "assistant")]
                    summary_items    = [h for h in history if h["role"] == "system"]
                    summary_note     = summary_items[0]["content"] if summary_items else ""

                    # Build proper multi-turn messages array
                    api_messages: list[dict] = []
                    for i, h in enumerate(history_messages):
                        content = h["content"]
                        if i == 0 and summary_note:
                            content = f"{summary_note}\n\n{content}"
                        api_messages.append({"role": h["role"], "content": content})
                    api_messages.append({"role": "user", "content": (llm_extra_context + effective_query).strip()})

                    stream_buffer = ""
                    for token in llm_service.generate_stream(
                        messages=api_messages,
                        max_tokens=1024,
                        temperature=0.7,
                        system=CHATBOT_SYSTEM_PROMPT,
                    ):
                        full_text += token
                        stream_buffer, safe_parts = _safe_stream_fragment(stream_buffer, token)
                        for safe_part in safe_parts:
                            yield {"type": "token", "text": safe_part}

                    _, safe_parts = _safe_stream_fragment(stream_buffer, "", final=True)
                    for safe_part in safe_parts:
                        yield {"type": "token", "text": safe_part}

                    full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()

                except Exception:
                    logger.exception("Chatbot stream failed")
                    full_text = "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại."
                    yield {"type": "token", "text": full_text}
            else:
                full_text = "LLM chưa được cấu hình."
                yield {"type": "token", "text": full_text}

        # ── RAG MODE ──────────────────────────────────────────────────
        else:
            ctx = self._run_rag_pipeline(
                db, user, conversation_id, effective_query, tid,
                oui_ids=oui_ids, chat_mode=chat_source,
            )
            retrieved               = ctx["retrieved"]
            retrieved_raw           = ctx["retrieved_raw"]
            stream_policy_contracts = ctx["policy_contracts"]
            stream_applied_rules    = ctx.get("applied_rules", [])
            stream_has_watermark    = ctx["has_watermark"]
            stream_all_restricted   = ctx["all_restricted"]
            safe_history            = ctx["safe_history"]

            if stream_applied_rules:
                yield {"type": "policy_rules", "rules": stream_applied_rules}
            logger.info("STREAM RETRIEVED COUNT: %d", len(retrieved_raw))

            if not retrieved:
                full_text = "Xin lỗi, không tìm thấy thông tin liên quan. Vui lòng thử diễn đạt lại câu hỏi."
                for token in full_text:
                    yield {"type": "token", "text": token}

            elif stream_all_restricted:
                full_text = "Thông tin này bị hạn chế theo chính sách phân quyền của hệ thống và không thể hiển thị."
                for token in full_text:
                    yield {"type": "token", "text": token}

            elif llm_service.is_configured():
                try:
                    _stream_has_transform = bool(stream_policy_contracts and any(
                        c.get("decision") in ("ANONYMIZE", "GENERALIZE", "SUMMARIZE")
                        for c in stream_policy_contracts
                    ))
                    _stream_extra = (
                        "Dữ liệu trong context đã được ẩn danh hóa theo chính sách phân quyền "
                        "(tên người thay bằng alias như 'Nhân viên A', mã số thay bằng ID đại diện, "
                        "số liệu thể hiện dạng khoảng hoặc tổng hợp). "
                        "Hãy trả lời dựa trên thông tin ẩn danh đó. "
                        "Không xác định danh tính cụ thể. "
                        "Không nói 'không có trong tài liệu' nếu thông tin liên quan (dù đã ẩn danh) thực sự có trong context."
                    ) if _stream_has_transform else None
                    prompt = llm_service.build_prompt(
                        question=llm_extra_context + effective_query,
                        contexts=retrieved,
                        chat_history=safe_history,
                        extra_instructions=_stream_extra,
                    )
                    logger.info("LLM stream prompt trace_id=%s", tid)

                    stream_buffer = ""
                    for token in llm_service.generate_stream(prompt=prompt, max_tokens=2048):
                        full_text += token
                        stream_buffer, safe_parts = _safe_stream_fragment(stream_buffer, token)
                        for safe_part in safe_parts:
                            yield {"type": "token", "text": safe_part}

                    _, safe_parts = _safe_stream_fragment(stream_buffer, "", final=True)
                    for safe_part in safe_parts:
                        yield {"type": "token", "text": safe_part}

                    full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
                    logger.info("LLM stream result len=%d", len(full_text))

                except Exception:
                    logger.exception("LLM stream failed")
                    full_text = ""

            if not full_text:
                full_text, _ = answer_service.generate(user_input=effective_query, retrieved=retrieved)

            # ── Watermark notice (stream) ──────────────────────────────
            if stream_has_watermark and full_text:
                watermark_text = "\n\n---\n⚠️ Nội dung này được truy cập theo điều kiện kiểm soát phân quyền. Hoạt động truy vấn đã được ghi nhận."
                yield {"type": "token", "text": watermark_text}
                full_text += watermark_text

            full_text, sources = self._normalize_citations(full_text, retrieved)

        # ── [GUARD 3] PII + secret scan on stream response ────────────
        full_text = self._apply_guard3(full_text, user, tid)

        # ── Persist ───────────────────────────────────────────────────
        _latency_ms = int((time.perf_counter() - _start) * 1000)
        assistant_msg.content = full_text
        assistant_msg.status  = "success"
        self._persist_sources(db, assistant_msg.id, sources)
        tr.assistant_output_summary = full_text[:2000] if full_text else None
        tr.retrieved_sources = [] if mode == "chatbot" else retrieved_raw
        tr.llm_response      = {"text": full_text}
        tr.timings           = {"total_ms": _latency_ms}
        tr.status            = "completed"
        audit_service.log_action(
            db, trace_id=tid, user_id=user.id,
            action="chat.message", resource_type="conversation",
            resource_id=conversation_id, decision="allow",
            input_json={"message": content, "effective_query": effective_query},
            output_json={"assistant_message": full_text[:500], "sources": sources},
            latency_ms=_latency_ms,
        )
        db.commit()

        threading.Thread(
            target=self._update_summary_background,
            args=(conversation_id,),
            daemon=True,
        ).start()

        yield {"type": "done", "content": full_text, "sources": sources, "messageId": assistant_msg.id, "applied_rules": stream_applied_rules}


# Module-level singleton; imported by the chat API router.
chat_service = ChatService()
