"""
System settings endpoints. Read and update key-value configuration entries
such as RAG parameters and query scope mode.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_db
from app.repositories.system_setting_repository import system_setting_repository

router = APIRouter()

CHAT_MODEL_OPTIONS = [
    {"id": "gpt-5.6", "label": "GPT-5.6 (alias Sol)", "tier": "frontier"},
    {"id": "gpt-5.6-sol", "label": "GPT-5.6 Sol", "tier": "frontier"},
    {"id": "gpt-5.6-terra", "label": "GPT-5.6 Terra", "tier": "balanced"},
    {"id": "gpt-5.6-luna", "label": "GPT-5.6 Luna", "tier": "cost"},
    {"id": "gpt-5.5", "label": "GPT-5.5", "tier": "frontier"},
    {"id": "gpt-5.4", "label": "GPT-5.4", "tier": "balanced"},
    {"id": "gpt-5.4-mini", "label": "GPT-5.4 mini", "tier": "cost"},
    {"id": "gpt-5.4-nano", "label": "GPT-5.4 nano", "tier": "cost"},
    {"id": "gpt-5", "label": "GPT-5", "tier": "previous"},
    {"id": "gpt-5-mini", "label": "GPT-5 mini", "tier": "cost"},
    {"id": "gpt-5-nano", "label": "GPT-5 nano", "tier": "cost"},
]
EMBEDDING_MODEL_OPTIONS = ["text-embedding-3-small", "text-embedding-3-large"]


# Return all system settings as a flat key-value map.
@router.get("")
def get_settings(
    db: Session = Depends(get_db),
):
    return system_setting_repository.get_all(db)


@router.get("/models")
def get_model_options():
    return {
        "chat_models": CHAT_MODEL_OPTIONS,
        "embedding_models": EMBEDDING_MODEL_OPTIONS,
        "reasoning_efforts": ["none", "low", "medium", "high", "xhigh", "max"],
    }


# Update one or more system settings. Only whitelisted keys are accepted.
@router.put("")
def update_settings(
    payload: dict,
    db: Session = Depends(get_db),
):
    # Whitelist of mutable keys to prevent unintended config overrides.
    allowed_keys = {
        "rag.top_k",
        "rag.similarity_threshold",
        "rag.hybrid_search",
        "query_scope_mode",
        "llm.provider",
        "llm.chat_model",
        "llm.reasoning_effort",
        "llm.embedding_model",
    }
    filtered = {k: v for k, v in payload.items() if k in allowed_keys}
    if "llm.provider" in filtered and filtered["llm.provider"] not in {None, "openai", "ollama"}:
        raise HTTPException(status_code=422, detail="llm.provider must be openai or ollama")
    if "llm.chat_model" in filtered and filtered["llm.chat_model"] not in {None, *(item["id"] for item in CHAT_MODEL_OPTIONS)}:
        raise HTTPException(status_code=422, detail="Unsupported chat model")
    if "llm.embedding_model" in filtered and filtered["llm.embedding_model"] not in {None, *EMBEDDING_MODEL_OPTIONS}:
        raise HTTPException(status_code=422, detail="Unsupported embedding model")
    if "llm.reasoning_effort" in filtered and filtered["llm.reasoning_effort"] not in {"none", "low", "medium", "high", "xhigh", "max"}:
        raise HTTPException(status_code=422, detail="Unsupported reasoning effort")
    system_setting_repository.set_many(db, filtered)
    return system_setting_repository.get_all(db)
