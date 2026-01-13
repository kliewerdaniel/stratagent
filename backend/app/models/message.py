"""
Message model
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.db.base import Base

class MessageType(str, enum.Enum):
    """Message type enumeration"""
    USER_MESSAGE = "user_message"
    PERSONA_RESPONSE = "persona_response"
    SYSTEM_MESSAGE = "system_message"
    ERROR_MESSAGE = "error_message"

class MessageStatus(str, enum.Enum):
    """Message status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Message(Base):
    """Message model for chat interactions"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    message_type = Column(Enum(MessageType), nullable=False)
    status = Column(Enum(MessageStatus), default=MessageStatus.PENDING)

    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=True)  # Null for user messages
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Metadata
    tokens_used = Column(Integer)
    response_time = Column(Float)  # Response time in seconds
    model_used = Column(String)  # LLM model used
    metadata = Column(JSON)  # Additional metadata (confidence scores, etc.)

    # Relationships
    project = relationship("Project", back_populates="messages")
    persona = relationship("Persona", back_populates="messages")
    user = relationship("User")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())