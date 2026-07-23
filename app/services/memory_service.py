"""
Hybrid conversation memory: history = rolling summary + relevant Q&A pairs + recent Q&A pairs.

  load_history(conv_id, query)   — build the history list to inject into the LLM prompt.
  update_summary(conv_id)        — update the rolling summary via LLM after each assistant reply.

Design:
  Recent:   last RECENT_PAIRS Q&A pairs — preserves immediate conversational flow.
  Relevant: top RELEVANT_PAIRS pairs most similar to the current query (cosine on embeddings),
            deduplicated against recent.
  Summary:  one short paragraph (~150 tokens) covering all messages older than recent,
            refreshed every SUMMARY_UPDATE_EVERY turns.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.message import Message
from app.models.conversation import Conversation
from app.repositories.message_repository import MessageRepository
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────
RECENT_PAIRS    = 3    # most-recent Q&A pairs always included verbatim
RELEVANT_PAIRS  = 3    # top similar Q&A pairs retrieved by embedding
SUMMARY_UPDATE_EVERY = 6   # refresh summary every N new Q&A pairs
MAX_SUMMARY_TOKENS   = 150 # token budget for the LLM-generated summary
DONE_STATUSES = {"success", "fallback", "no_answer", "llm_error", "blocked"}
# ──────────────────────────────────────────────────────────────────────────


# Compute cosine similarity between two embedding vectors.
def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MemoryService:
    def __init__(self):
        self.msg_repo = MessageRepository()

    # ------------------------------------------------------------------
    # Public: load history for prompt
    # ------------------------------------------------------------------

    # Build the prompt history list: summary → relevant pairs → recent pairs.
    def load_history(
        self,
        db: Session,
        conversation_id: str,
        query: str,
    ) -> list[dict]:
        # Load all messages (max 200 to avoid OOM)
        all_msgs = self.msg_repo.list_by_conversation(db, conversation_id, limit=200)

        # Build Q&A pairs: [(user_msg, assistant_msg), ...]
        pairs = self._build_pairs(all_msgs)

        if not pairs:
            return []

        # Split: recent vs older
        recent_pairs  = pairs[-RECENT_PAIRS:]
        older_pairs   = pairs[:-RECENT_PAIRS] if len(pairs) > RECENT_PAIRS else []

        # Get summary from conversation
        conv = db.get(Conversation, conversation_id)
        summary = (conv.summary or "").strip() if conv else ""

        # Find relevant pairs from older_pairs
        relevant_pairs = self._find_relevant(query, older_pairs, top_k=RELEVANT_PAIRS)

        # Dedup: remove relevant pairs already in recent (compare by message ID)
        recent_ids = {u.id for u, a in recent_pairs}
        relevant_pairs = [p for p in relevant_pairs if p[0].id not in recent_ids]

        # Build history list
        history: list[dict] = []

        if summary:
            history.append({
                "role":    "system",
                "content": f"[Tóm tắt cuộc trò chuyện trước]\n{summary}",
            })

        # relevant first (older context), then recent (immediate context)
        for u_msg, a_msg in relevant_pairs:
            history.append({"role": "user",      "content": u_msg.content or ""})
            history.append({"role": "assistant",  "content": a_msg.content or ""})

        for u_msg, a_msg in recent_pairs:
            history.append({"role": "user",      "content": u_msg.content or ""})
            history.append({"role": "assistant",  "content": a_msg.content or ""})

        return history

    # ------------------------------------------------------------------
    # Public: update rolling summary after assistant replies
    # ------------------------------------------------------------------

    # Regenerate the rolling summary via LLM if the conversation is long enough.
    def update_summary(
        self,
        db: Session,
        conversation_id: str,
    ) -> None:
        all_msgs = self.msg_repo.list_by_conversation(db, conversation_id, limit=200)
        pairs    = self._build_pairs(all_msgs)

        if len(pairs) < RECENT_PAIRS + SUMMARY_UPDATE_EVERY:
            return

        older_pairs = pairs[:-RECENT_PAIRS]

        # Call the LLM to generate a new summary.
        try:
            from app.services.llm_service import llm_service
            if not llm_service.is_configured():
                return

            history_text = "\n".join(
                f"User: {u.content or ''}\nAssistant: {a.content or ''}"
                for u, a in older_pairs
            )

            prompt = (
                f"Hãy tóm tắt ngắn gọn (tối đa {MAX_SUMMARY_TOKENS} từ) "
                f"nội dung cuộc trò chuyện sau, giữ lại các thông tin quan trọng:\n\n"
                f"{history_text}\n\n"
                f"Tóm tắt:"
            )

            summary_text, _, _ = llm_service.generate(
                prompt=prompt,
                max_tokens=MAX_SUMMARY_TOKENS,
                temperature=0.0,
                include_markdown_instructions=False,
            )

            if summary_text and summary_text.strip():
                conv = db.get(Conversation, conversation_id)
                if conv:
                    conv.summary = summary_text.strip()
                    conv.summary_updated_at = datetime.now(timezone.utc)
                    db.flush()
                    logger.info("Summary updated conv_id=%s len=%d", conversation_id, len(summary_text))

        except Exception:
            logger.exception("Failed to update summary conv_id=%s", conversation_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Group a flat message list into consecutive (user, assistant) Q&A pairs.
    def _build_pairs(self, messages):
        pairs = []
        i = 0
        while i < len(messages):
            m = messages[i]
            if (
                m.role == "user"
                and i + 1 < len(messages)
                and messages[i + 1].role == "assistant"
            ):
                assistant = messages[i + 1]
                is_done = assistant.status in DONE_STATUSES
                is_streaming_with_content = (
                    assistant.status == "streaming"
                    and assistant.content
                    and len(assistant.content.strip()) > 10  # có nội dung thật
                )
                if is_done or is_streaming_with_content:
                    pairs.append((m, assistant))
                i += 2
            else:
                i += 1
        return pairs

    # Return the top_k pairs most relevant to the query via embedding similarity, falling back to keyword overlap.
    def _find_relevant(
        self,
        query: str,
        pairs: list[tuple[Message, Message]],
        top_k: int,
    ) -> list[tuple[Message, Message]]:
        if not pairs or not query.strip():
            return []

        # Try embedding-based
        if embedding_service.is_configured():
            try:
                return self._relevant_by_embedding(query, pairs, top_k)
            except Exception:
                logger.warning("Embedding relevance failed, falling back to keyword")

        # Fallback: keyword overlap
        return self._relevant_by_keyword(query, pairs, top_k)

    # Rank pairs by cosine similarity between the query embedding and each user-message embedding.
    def _relevant_by_embedding(
        self,
        query: str,
        pairs: list[tuple[Message, Message]],
        top_k: int,
    ) -> list[tuple[Message, Message]]:
        # Embed query + all user messages in one batch
        texts = [query] + [u.content or "" for u, _ in pairs]
        vectors = embedding_service.embed_many(texts)

        query_vec = vectors[0]
        pair_vecs  = vectors[1:]

        scored = []
        for i, (pair, vec) in enumerate(zip(pairs, pair_vecs)):
            sim = _cosine(query_vec, vec)
            scored.append((sim, i, pair))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [pair for _, _, pair in scored[:top_k]]

    # Rank pairs by token-overlap count between the query and each user message.
    def _relevant_by_keyword(
        self,
        query: str,
        pairs: list[tuple[Message, Message]],
        top_k: int,
    ) -> list[tuple[Message, Message]]:
        q_tokens = set(query.lower().split())
        scored = []
        for pair in pairs:
            u_msg = pair[0]
            d_tokens = set((u_msg.content or "").lower().split())
            overlap = len(q_tokens & d_tokens)
            if overlap > 0:
                scored.append((overlap, pair))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [pair for _, pair in scored[:top_k]]


# Module-level singleton; imported by the chat service.
memory_service = MemoryService()
