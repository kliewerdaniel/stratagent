"""
Persona model
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Enum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.db.base import Base

class PersonaRole(str, enum.Enum):
    """Persona role enumeration"""
    ARCHITECT = "architect"
    FRONTEND_DEV = "frontend_dev"
    BACKEND_DEV = "backend_dev"
    TESTER = "tester"
    REVIEWER = "reviewer"
    SPEC_INTERPRETER = "spec_interpreter"
    GENERATOR = "generator"
    VALIDATOR = "validator"

class PersonaStatus(str, enum.Enum):
    """Persona status enumeration"""
    ACTIVE = "active"
    TRAINING = "training"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"

class Persona(Base):
    """Persona model - AI agent with specific role and capabilities"""
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    role = Column(Enum(PersonaRole), nullable=False)
    description = Column(Text)
    system_prompt = Column(Text, nullable=False)
    personality_traits = Column(JSON)  # Dict of personality characteristics
    capabilities = Column(JSON)  # List of capabilities and skills
    status = Column(Enum(PersonaStatus), default=PersonaStatus.ACTIVE)

    # Performance metrics
    success_rate = Column(Float, default=0.0)
    total_interactions = Column(Integer, default=0)
    average_response_time = Column(Float)

    # Learning parameters
    learning_rate = Column(Float, default=0.01)
    adaptation_score = Column(Float, default=0.0)

    # Relationships
    project_personas = relationship("ProjectPersona", back_populates="persona")
    messages = relationship("Message", back_populates="persona")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ProjectPersona(Base):
    """Association table for projects and personas"""
    __tablename__ = "project_personas"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=False)

    # Project-specific persona configuration
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=0)  # Higher priority personas are used first

    # Relationships
    project = relationship("Project", back_populates="personas")
    persona = relationship("Persona", back_populates="project_personas")

    created_at = Column(DateTime(timezone=True), server_default=func.now())