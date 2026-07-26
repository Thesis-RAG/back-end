"""
Chat endpoints: conversation lifecycle management, message posting (sync and streaming),
document search, and conversation title generation.
"""
import json as jsonlib
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, get_db
from app.db.session import SessionLocal
from app.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.trace_repository import TraceRepository
from app.schemas.chat import (
    ConversationCreateRequest,
    ConversationMessageRead,
    ConversationRead,
    MessageCreateRequest,
    MessagePostResponse,
    MessageRead,
    SourceRead,
    TraceRead,
)
from app.services.chat_service import chat_service
from app.services.llm_service import llm_service
from app.services.retrieval_service import retrieval_service
from app.services.entity_policy_service import entity_policy_service

logger = logging.getLogger(__name__)

router = APIRouter()


# Create a new conversation for the current user.
@router.post("/conversations", response_model=ConversationRead)
def create_conversation(payload: ConversationCreateRequest, request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    conv = chat_service.create_conversation(db, current_user, payload.title)
    return ConversationRead.model_validate(conv)


# Send a user message and return the assistant response with retrieved sources.
@router.post("/conversations/{conversation_id}/messages", response_model=MessagePostResponse)
def post_message(conversation_id: str, payload: MessageCreateRequest, request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # basic permission: ensure conversation exists and belongs to user (or allow)
    repo = ConversationRepository()
    conv = repo.get(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.status != "open":
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msg, assistant_msg, sources = chat_service.post_message(
        db, current_user, conversation_id, payload.content, payload.clientMessageId, request.state.trace_id
    )

    src_objs = [SourceRead(**s) for s in (sources or [])]
    require_types = sorted({t for s in src_objs for t in (s.requireEntityTypes or [])})

    return MessagePostResponse(
        conversationId=conversation_id,
        messageId=user_msg.id,
        assistantMessage={
            "id": assistant_msg.id,
            "content": assistant_msg.content,
            "status": assistant_msg.status,
            "createdAt": assistant_msg.created_at,
            "appliedRules": assistant_msg.applied_rules_json or [],
            "requireEntityTypes": require_types,
        },
        traceId=request.state.trace_id,
        sources=src_objs,
    )


# List all messages in a conversation (up to 1000), flattened into a unified list.
@router.get("/conversations/{conversation_id}/messages", response_model=list[ConversationMessageRead])
def get_messages(conversation_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Delegate formatting/aggregation to the chat service; return unified messages
    # ensure conversation is open
    repo = ConversationRepository()
    conv = repo.get(db, conversation_id)
    if not conv or conv.status != "open":
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = chat_service.list_messages_flat(db, conversation_id, limit=1000, user=current_user)
    return msgs


# Update the title of an existing conversation.
@router.patch("/conversations/{conversation_id}", response_model=ConversationRead)
def rename_conversation(conversation_id: str, payload: ConversationCreateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> Any:
    repo = ConversationRepository()
    conv = repo.get(db, conversation_id)
    if not conv or conv.status != "open":
        raise HTTPException(status_code=404, detail="Conversation not found")

    updated = repo.update_title(db, conversation_id, payload.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.commit()
    db.refresh(updated)
    return ConversationRead.model_validate(updated)


# Soft-delete a conversation by setting its status to closed.
@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> dict:
    repo = ConversationRepository()
    conv = repo.get(db, conversation_id)
    if not conv or conv.status != "open":
        raise HTTPException(status_code=404, detail="Conversation not found")

    updated = repo.set_status(db, conversation_id, "close")
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.commit()
    return {"ok": True}


# Retrieve a request trace by its trace ID.
@router.get("/traces/{trace_id}", response_model=TraceRead)
def get_trace(trace_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = TraceRepository()
    tr = repo.get_by_trace_id(db, trace_id)
    if not tr:
        raise HTTPException(status_code=404, detail="Trace not found")
    return TraceRead.model_validate(tr)


# List all conversations belonging to a specific user.
@router.get("/users/{user_id}/conversations", response_model=list[ConversationRead])
def list_conversations_by_user(user_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    repo = ConversationRepository()
    convs = repo.list_by_user(db, user_id)
    return [ConversationRead.model_validate(c) for c in (convs or [])]


# Stream the assistant response as Server-Sent Events (SSE).
@router.post("/conversations/{conversation_id}/messages/stream")
def post_message_stream(
    conversation_id: str,
    payload: MessageCreateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    db_check = SessionLocal()
    try:
        repo = ConversationRepository()
        conv = repo.get(db_check, conversation_id)
        if not conv or conv.status != "open":
            raise HTTPException(status_code=404, detail="Conversation not found")
    finally:
        db_check.close()

    # Inner generator: opens its own DB session to avoid conflicts with the request session.
    def generate():
        stream_db = SessionLocal()
        try:
            for event in chat_service.post_message_stream(
                stream_db, current_user, conversation_id,
                payload.content, payload.clientMessageId,
                request.state.trace_id,
                oui_ids=payload.oui_ids,
                mode=payload.mode,
                chat_source=payload.chat_source,
                file_content=payload.file_content,
                file_name=payload.file_name,
            ):
                yield f"data: {jsonlib.dumps(event)}\n\n"
        finally:
            stream_db.close()

    return StreamingResponse(generate(), media_type="text/event-stream")


# Perform a hybrid semantic/keyword search over documents accessible to the current user.
@router.post("/search")
def search_documents(
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = payload.get("query", "").strip()
    mode = payload.get("mode", "hybrid")
    top_k = min(int(payload.get("top_k", 10)), 20)
    if not query:
        return []
    results = retrieval_service.retrieve(
        query=query, user=current_user, top_k=top_k, mode=mode, db=db
    )
    results, _contracts = entity_policy_service.apply_to_retrieved(
        db, current_user, query, results
    )

    # Enrich document_type from DB for chunks whose Chroma metadata still has "general".
    stale_doc_ids = {
        (r.get("metadata") or {}).get("document_id")
        for r in results
        if (r.get("metadata") or {}).get("document_type") == "general"
    } - {None}
    if stale_doc_ids:
        from app.models.document import Document as _Doc
        from app.schemas.document import DocumentRead as _DocRead
        doc_type_map = {
            d.id: _DocRead.model_validate(d).document_type
            for d in db.query(_Doc).filter(_Doc.id.in_(stale_doc_ids)).all()
        }
        for r in results:
            meta = r.get("metadata") or {}
            if meta.get("document_type") == "general":
                enriched = doc_type_map.get(meta.get("document_id"))
                if enriched:
                    meta["document_type"] = enriched

    return results


# Generate a short conversation title from the first user message via LLM.
@router.post("/conversations/{conversation_id}/generate-title")
def generate_title(
    conversation_id: str,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    first_message = (payload.get("first_message") or "").strip()
    if not first_message:
        return {"title": "Cuộc trò chuyện"}

    try:
        title, _, _ = llm_service.generate(
            prompt=f'Tạo tiêu đề ngắn gọn (tối đa 6 từ, không dấu ngoặc kép) cho cuộc trò chuyện bắt đầu bằng câu hỏi sau:\n"{first_message}"\n\nChỉ trả về tiêu đề, không giải thích thêm.',
            system="Bạn là trợ lý tạo tiêu đề ngắn gọn. Chỉ trả về tiêu đề, không có gì khác.",
            max_tokens=30,
            temperature=0.3,
            fallback_to_ollama=False,
            include_markdown_instructions=False,
        )
        title = (title or "").strip().strip('"').strip("'")
        if not title:
            title = first_message[:40]
    except Exception:
        logger.exception("generate_title failed conv_id=%s", conversation_id)
        title = first_message[:40]

    repo = ConversationRepository()
    repo.update_title(db, conversation_id, title)
    db.commit()

    return {"title": title}
