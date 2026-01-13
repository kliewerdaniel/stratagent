"""
Base agent class with MCP protocol support
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import uuid

from app.core.config import settings
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class MCPMessage:
    """MCP protocol message"""

    def __init__(self, message_type: str, sender: str, recipient: str, content: Any, metadata: Optional[Dict] = None):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.message_type = message_type
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "metadata": self.metadata
        }

class BaseAgent(ABC):
    """Base class for all AI agents in StratAgent"""

    def __init__(self, agent_id: str, name: str, role: str, ollama_service: OllamaService):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.ollama_service = ollama_service

        # Agent state
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.last_active = datetime.utcnow()

        # Performance metrics
        self.message_count = 0
        self.success_count = 0
        self.error_count = 0
        self.average_response_time = 0.0

        # MCP protocol
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}

        # Memory and context
        self.conversation_history: List[MCPMessage] = []
        self.context_window_size = 10

        # Setup default message handlers
        self._setup_message_handlers()

    def _setup_message_handlers(self):
        """Setup default MCP message handlers"""
        self.register_handler("ping", self._handle_ping)
        self.register_handler("status", self._handle_status)
        self.register_handler("shutdown", self._handle_shutdown)

    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler

    async def receive_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Receive and process an MCP message"""
        logger.info(f"Agent {self.name} received message: {message.message_type}")

        self.message_count += 1
        self.last_active = datetime.utcnow()

        # Add to conversation history
        self.conversation_history.append(message)

        # Keep only recent messages in context
        if len(self.conversation_history) > self.context_window_size:
            self.conversation_history = self.conversation_history[-self.context_window_size:]

        # Handle the message
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                start_time = asyncio.get_event_loop().time()
                response = await handler(message)
                end_time = asyncio.get_event_loop().time()

                # Update performance metrics
                response_time = end_time - start_time
                self.average_response_time = (
                    (self.average_response_time * (self.message_count - 1)) + response_time
                ) / self.message_count
                self.success_count += 1

                return response
            except Exception as e:
                logger.error(f"Error handling message {message.message_type}: {str(e)}")
                self.error_count += 1
                return MCPMessage(
                    "error",
                    self.agent_id,
                    message.sender,
                    {"error": str(e), "original_message": message.to_dict()},
                    {"error_type": "handler_error"}
                )
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
            return MCPMessage(
                "error",
                self.agent_id,
                message.sender,
                {"error": f"Unknown message type: {message.message_type}"},
                {"error_type": "unknown_message_type"}
            )

    async def send_message(self, recipient: str, message_type: str, content: Any, metadata: Optional[Dict] = None) -> str:
        """Send an MCP message to another agent"""
        message = MCPMessage(message_type, self.agent_id, recipient, content, metadata)
        message_id = message.id

        # Create a future for the response
        self.pending_responses[message_id] = asyncio.Future()

        # In a real implementation, this would be sent through the MCP protocol
        # For now, we'll simulate by calling receive_message directly
        logger.info(f"Agent {self.name} sending message to {recipient}: {message_type}")

        return message_id

    async def wait_for_response(self, message_id: str, timeout: float = 30.0) -> Any:
        """Wait for a response to a sent message"""
        if message_id not in self.pending_responses:
            raise ValueError(f"No pending response for message {message_id}")

        try:
            response = await asyncio.wait_for(self.pending_responses[message_id], timeout=timeout)
            del self.pending_responses[message_id]
            return response
        except asyncio.TimeoutError:
            del self.pending_responses[message_id]
            raise TimeoutError(f"Timeout waiting for response to message {message_id}")

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a response using the Ollama service"""
        try:
            return await self.ollama_service.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task specific to this agent's role"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this agent"""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "performance": {
                "message_count": self.message_count,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate": self.success_count / max(self.message_count, 1),
                "average_response_time": self.average_response_time
            },
            "capabilities": self.get_capabilities()
        }

    # Default message handlers

    async def _handle_ping(self, message: MCPMessage) -> MCPMessage:
        """Handle ping messages"""
        return MCPMessage("pong", self.agent_id, message.sender, {"status": "ok"})

    async def _handle_status(self, message: MCPMessage) -> MCPMessage:
        """Handle status requests"""
        return MCPMessage("status_response", self.agent_id, message.sender, self.get_status())

    async def _handle_shutdown(self, message: MCPMessage) -> MCPMessage:
        """Handle shutdown requests"""
        self.is_active = False
        logger.info(f"Agent {self.name} shutting down")
        return MCPMessage("shutdown_ack", self.agent_id, message.sender, {"status": "shutting_down"})

    def cleanup(self):
        """Cleanup agent resources"""
        # Cancel any pending responses
        for future in self.pending_responses.values():
            if not future.done():
                future.cancel()
        self.pending_responses.clear()
        self.conversation_history.clear()