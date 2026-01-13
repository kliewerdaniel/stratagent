"""
Health check endpoint
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "StratAgent API"}

@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    return {
        "status": "healthy",
        "service": "StratAgent API",
        "components": {
            "database": "healthy",
            "ollama": "unknown",  # TODO: Implement Ollama health check
            "redis": "unknown",    # TODO: Implement Redis health check
        },
        "version": "1.0.0"
    }