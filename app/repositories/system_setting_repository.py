"""
Repository for system-wide key-value settings with JSON serialisation and built-in defaults.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.system_setting import SystemSetting

# Default values used when a key has never been saved.
DEFAULTS: dict[str, Any] = {
    "rag.top_k": 5,
    "rag.similarity_threshold": 0.0,
    "rag.hybrid_search": True,
    "query_scope_mode": "full_db",
    "llm.provider": None,
    "llm.chat_model": None,
    "llm.reasoning_effort": "medium",
    "llm.embedding_model": None,
}


class SystemSettingRepository:

    # Return the decoded value for a key, falling back to the built-in default.
    def get(self, db: Session, key: str) -> Any:
        row = db.get(SystemSetting, key)
        if row is None:
            return DEFAULTS.get(key)
        try:
            return json.loads(row.value)
        except (ValueError, TypeError):
            return row.value

    # Return all settings merged with built-in defaults.
    def get_all(self, db: Session) -> dict[str, Any]:
        rows = db.query(SystemSetting).all()
        result = dict(DEFAULTS)
        for row in rows:
            try:
                result[row.key] = json.loads(row.value)
            except (ValueError, TypeError):
                result[row.key] = row.value
        return result

    # Upsert a single setting, serialising the value as JSON.
    def set(self, db: Session, key: str, value: Any) -> None:
        serialized = json.dumps(value)
        row = db.get(SystemSetting, key)
        if row is None:
            db.add(SystemSetting(key=key, value=serialized, updated_at=datetime.utcnow()))
        else:
            row.value = serialized
            row.updated_at = datetime.utcnow()

    # Upsert multiple settings and commit.
    def set_many(self, db: Session, data: dict[str, Any]) -> None:
        for key, value in data.items():
            self.set(db, key, value)
        db.commit()


# Module-level singleton; imported across services that read/write system settings.
system_setting_repository = SystemSettingRepository()
