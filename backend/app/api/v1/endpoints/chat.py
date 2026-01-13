"""
Chat endpoint for handling AI persona interactions
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio

from app.services.ollama_service import OllamaService
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize Ollama service
ollama_service = OllamaService()

class ChatMessage(BaseModel):
    """Chat message model"""
    content: str
    persona_id: Optional[int] = None
    user_id: int

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    project_id: int
    user_id: int
    active_personas: List[str] = []

class ChatResponse(BaseModel):
    """Chat response model"""
    responses: List[dict] = []
    status: str = "success"

@router.post("/send", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Send a chat message and get responses from active personas
    """
    try:
        # Save user message to database
        # TODO: Implement message persistence

        # Get responses from active personas
        responses = []

        for persona_role in request.active_personas:
            try:
                # Generate persona-specific prompt
                system_prompt = get_persona_prompt(persona_role)

                # Get response from Ollama
                response = await ollama_service.generate_response(
                    prompt=request.message,
                    system_prompt=system_prompt,
                    model="llama2"  # Default model
                )

                responses.append({
                    "persona_role": persona_role,
                    "content": response,
                    "status": "success"
                })

            except Exception as e:
                logger.error(f"Error generating response for {persona_role}: {str(e)}")
                responses.append({
                    "persona_role": persona_role,
                    "content": f"Error: {str(e)}",
                    "status": "error"
                })

        # Save AI responses to database
        # TODO: Implement response persistence

        return ChatResponse(responses=responses)

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_available_models():
    """List available Ollama models"""
    try:
        models = await ollama_service.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_persona_prompt(persona_role: str) -> str:
    """Get system prompt for a specific persona role"""
    prompts = {
        "architect": """You are a Software Architect. Your role is to design system architecture,
technical specifications, and provide high-level technical guidance. Focus on:
- System design and architecture patterns
- Technology stack recommendations
- Scalability and performance considerations
- Security architecture
- Database design
- API design principles

Provide detailed, technical responses with clear reasoning.""",

        "frontend_dev": """You are a Frontend Developer. Your role is to build user interfaces
and client-side applications. Focus on:
- React/Next.js development
- UI/UX implementation
- Responsive design
- Component architecture
- State management
- CSS/styling best practices

Provide practical, implementable code suggestions.""",

        "backend_dev": """You are a Backend Developer. Your role is to implement server-side logic
and APIs. Focus on:
- API development and design
- Database operations
- Server architecture
- Authentication and authorization
- Performance optimization
- Error handling

Provide robust, scalable backend solutions.""",

        "tester": """You are a Quality Assurance Engineer. Your role is to ensure code quality
through testing and validation. Focus on:
- Test case development
- Testing strategies (unit, integration, e2e)
- Bug identification and reporting
- Quality assurance processes
- Performance testing
- Security testing considerations

Provide comprehensive testing approaches.""",

        "reviewer": """You are a Code Reviewer. Your role is to conduct code reviews and provide
constructive feedback. Focus on:
- Code quality assessment
- Best practices compliance
- Security vulnerabilities
- Performance issues
- Maintainability concerns
- Documentation requirements

Provide detailed, actionable review comments.""",

        "spec_interpreter": """You are a Requirements Analyst. Your role is to analyze requirements
and create detailed specifications. Focus on:
- Requirements analysis and clarification
- Functional and non-functional requirements
- User story creation
- Acceptance criteria definition
- Technical specification writing
- Scope and boundary definition

Provide clear, unambiguous specifications.""",

        "generator": """You are a Code Generator. Your role is to generate code based on
specifications and patterns. Focus on:
- Code generation from requirements
- Design pattern implementation
- Boilerplate code creation
- Template-based development
- Code structure and organization
- Best practices application

Provide clean, well-structured, production-ready code.""",

        "validator": """You are a Code Validator. Your role is to validate code quality and ensure
standards compliance. Focus on:
- Code quality metrics
- Standards compliance checking
- Static analysis
- Code smell detection
- Performance validation
- Security validation

Provide validation results with specific recommendations.""",
    }

    return prompts.get(persona_role, "You are an AI assistant helping with software development.")