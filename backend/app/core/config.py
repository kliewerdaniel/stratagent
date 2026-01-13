"""
Core configuration for StratAgent backend
"""
import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # Server Configuration
    SERVER_NAME: str = os.getenv("SERVER_NAME", "StratAgent")
    SERVER_HOST: str = os.getenv("SERVER_HOST", "http://localhost")

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./stratagent.db")
    DATABASE_TYPE: str = os.getenv("DATABASE_TYPE", "sqlite")  # sqlite or postgresql

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:8000",  # FastAPI
        "http://localhost:8080",  # Alternative port
    ]

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "llama2")

    # Redis Configuration (for caching and sessions)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Vector Database Configuration
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

    # Knowledge Graph Configuration
    GRAPH_DATA_PATH: str = os.getenv("GRAPH_DATA_PATH", "./data/knowledge_graph")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        case_sensitive = True
        env_file = ".env"

# Create global settings instance
settings = Settings()