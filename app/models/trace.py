from sqlalchemy import Column, JSON, String, Text

from app.db.base import Base, TimestampMixin
from app.utils.ids import new_uuid


class Trace(Base, TimestampMixin):
    __tablename__ = "traces"

    id = Column(String(36), primary_key=True, default=new_uuid)
    trace_id = Column(String(128), nullable=False, unique=True, index=True)
    conversation_id = Column(String(36), nullable=True, index=True)
    message_id = Column(String(36), nullable=True)
    user_id = Column(String(36), nullable=True, index=True)
    user_input = Column(String(4000), nullable=True)
    assistant_output_summary = Column(String(4000), nullable=True)
    retrieved_sources = Column(JSON, nullable=True)
    llm_prompt = Column(Text, nullable=True)
    llm_response = Column(JSON, nullable=True)
    timings = Column(JSON, nullable=True)
    token_usage = Column(JSON, nullable=True)
    status = Column(String(32), nullable=True)

