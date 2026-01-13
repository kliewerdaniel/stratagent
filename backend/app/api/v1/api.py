"""
Main API router
"""
from fastapi import APIRouter

from app.api.v1.endpoints import projects, personas, chat, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(personas.router, prefix="/personas", tags=["personas"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])