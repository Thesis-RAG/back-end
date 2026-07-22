from sqlalchemy import Column, ForeignKey, JSON, String, Text
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin
from app.utils.ids import new_uuid


class Message(Base, TimestampMixin):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=new_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # user | assistant
    content = Column(Text, nullable=True)
    token_usage = Column(JSON, nullable=True)
    # Snapshot of the policy rules that affected this assistant response.
    # Stored with the message so historical conversations remain explainable
    # even after rules are edited or removed.
    applied_rules_json = Column(JSON, nullable=True)
    parent_message_id = Column(String(36), nullable=True)
    client_message_id = Column(String(36), nullable=True, index=True)
    status = Column(String(36), nullable=True, index=True)
    trace_id = Column(String(64), nullable=True, index=True)
    attached_file_name = Column(String(512), nullable=True)

    conversation = relationship("Conversation")
