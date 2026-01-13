"""
SpecInterpreter agent - analyzes requirements and creates detailed specifications
"""
import logging
from typing import Dict, List, Any
from datetime import datetime

from app.agents.base_agent import BaseAgent
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class SpecInterpreterAgent(BaseAgent):
    """Agent responsible for interpreting user requirements and creating specifications"""

    def __init__(self, agent_id: str, ollama_service: OllamaService):
        super().__init__(agent_id, "SpecInterpreter", "spec_interpreter", ollama_service)

        # Register additional message handlers
        self.register_handler("analyze_requirements", self._handle_analyze_requirements)
        self.register_handler("clarify_requirements", self._handle_clarify_requirements)

    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this agent"""
        return [
            "requirements_analysis",
            "functional_requirements_extraction",
            "non_functional_requirements_identification",
            "user_story_creation",
            "acceptance_criteria_definition",
            "requirements_clarification"
        ]

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specification interpretation task"""
        task_type = task.get("type", "analyze_requirements")

        if task_type == "analyze_requirements":
            return await self._analyze_requirements(task)
        elif task_type == "clarify_requirements":
            return await self._clarify_requirements(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _analyze_requirements(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user requirements and extract specifications"""
        user_input = task.get("user_input", "")
        context = task.get("context", {})

        system_prompt = """You are a Requirements Analyst specializing in software specification analysis.
Your role is to analyze user requirements and create clear, unambiguous specifications.

Focus on:
- Extracting functional requirements
- Identifying non-functional requirements (performance, security, usability, etc.)
- Creating user stories in standard format: "As a [user], I want [functionality] so that [benefit]"
- Defining clear acceptance criteria
- Identifying dependencies and constraints
- Asking clarifying questions when requirements are ambiguous

Provide structured output with clear sections and actionable specifications."""

        prompt = f"""Analyze the following user requirements and create detailed specifications:

User Input: {user_input}

Context: {context}

Please provide:
1. Summary of understood requirements
2. Functional requirements (numbered list)
3. Non-functional requirements (categorized)
4. User stories (in standard format)
5. Acceptance criteria for each user story
6. Questions for clarification (if any)
7. Assumptions made
8. Dependencies and constraints identified

Be thorough and ask for clarification when needed."""

        response = await self.generate_response(prompt, system_prompt)

        # Parse and structure the response
        try:
            # In a real implementation, you'd parse the structured response
            # For now, return the raw response with metadata
            return {
                "status": "completed",
                "specifications": {
                    "raw_analysis": response,
                    "structured_requirements": self._parse_requirements(response),
                    "confidence_score": 0.85,  # Would be calculated based on clarity
                    "clarification_needed": self._check_clarification_needed(response)
                },
                "metadata": {
                    "analyzed_at": datetime.utcnow().isoformat(),
                    "input_length": len(user_input),
                    "analysis_method": "llm_structured"
                }
            }
        except Exception as e:
            logger.error(f"Error parsing requirements analysis: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": response
            }

    async def _clarify_requirements(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clarification questions for ambiguous requirements"""
        requirements = task.get("requirements", "")
        identified_issues = task.get("issues", [])

        system_prompt = """You are an expert requirements analyst focused on clarity and precision.
Your goal is to identify ambiguities, inconsistencies, and missing information in requirements."""

        prompt = f"""Review these requirements and identify areas that need clarification:

Requirements: {requirements}

Known Issues: {identified_issues}

Please provide:
1. Specific questions that need answers
2. Ambiguities found
3. Assumptions that should be confirmed
4. Missing constraints or dependencies
5. Prioritized list of clarification items"""

        response = await self.generate_response(prompt, system_prompt)

        return {
            "status": "completed",
            "clarifications": {
                "questions": self._extract_questions(response),
                "issues": self._extract_issues(response),
                "priority_order": ["high", "medium", "low"]  # Would be determined by analysis
            }
        }

    async def _handle_analyze_requirements(self, message) -> Any:
        """Handle analyze requirements message"""
        task = message.content
        result = await self._analyze_requirements(task)
        return self._create_response_message(message, "requirements_analysis_complete", result)

    async def _handle_clarify_requirements(self, message) -> Any:
        """Handle clarify requirements message"""
        task = message.content
        result = await self._clarify_requirements(task)
        return self._create_response_message(message, "clarification_complete", result)

    def _create_response_message(self, original_message, response_type: str, content: Any):
        """Create a response message"""
        from app.agents.base_agent import MCPMessage
        return MCPMessage(
            response_type,
            self.agent_id,
            original_message.sender,
            content,
            {"original_message_id": original_message.id}
        )

    def _parse_requirements(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured requirements"""
        # This would use more sophisticated parsing in a real implementation
        # For now, return a basic structure
        return {
            "functional_requirements": [],
            "non_functional_requirements": {},
            "user_stories": [],
            "acceptance_criteria": [],
            "questions": [],
            "assumptions": [],
            "dependencies": []
        }

    def _check_clarification_needed(self, response: str) -> bool:
        """Check if clarification is needed based on the response"""
        clarification_indicators = [
            "clarify", "unclear", "ambiguous", "need more", "question",
            "confirm", "assume", "unsure", "specify"
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in clarification_indicators)

    def _extract_questions(self, response: str) -> List[str]:
        """Extract questions from the response"""
        # Simple extraction - would be more sophisticated in practice
        lines = response.split('\n')
        questions = [line.strip() for line in lines if line.strip().endswith('?')]
        return questions

    def _extract_issues(self, response: str) -> List[str]:
        """Extract identified issues from the response"""
        # Simple extraction - would be more sophisticated in practice
        issues = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['issue', 'problem', 'concern', 'ambiguity']):
                issues.append(line)
        return issues