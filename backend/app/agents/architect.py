"""
Architect agent - designs system architecture and technical specifications
"""
import logging
from typing import Dict, List, Any
from datetime import datetime

from app.agents.base_agent import BaseAgent
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class ArchitectAgent(BaseAgent):
    """Agent responsible for designing system architecture and technical specifications"""

    def __init__(self, agent_id: str, ollama_service: OllamaService):
        super().__init__(agent_id, "Architect", "architect", ollama_service)

        # Register additional message handlers
        self.register_handler("design_architecture", self._handle_design_architecture)
        self.register_handler("review_architecture", self._handle_review_architecture)
        self.register_handler("optimize_architecture", self._handle_optimize_architecture)

    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this agent"""
        return [
            "system_architecture_design",
            "technology_stack_recommendation",
            "scalability_analysis",
            "security_architecture",
            "database_design",
            "api_design",
            "infrastructure_planning",
            "performance_optimization",
            "architecture_review"
        ]

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an architecture design task"""
        task_type = task.get("type", "design_architecture")

        if task_type == "design_architecture":
            return await self._design_architecture(task)
        elif task_type == "review_architecture":
            return await self._review_architecture(task)
        elif task_type == "optimize_architecture":
            return await self._optimize_architecture(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _design_architecture(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Design system architecture based on requirements"""
        requirements = task.get("requirements", {})
        constraints = task.get("constraints", {})
        preferences = task.get("preferences", {})

        system_prompt = """You are a Senior Software Architect with extensive experience in designing scalable, maintainable systems.
Your expertise includes:
- System architecture patterns (microservices, monolithic, serverless, etc.)
- Technology stack selection and evaluation
- Scalability and performance considerations
- Security architecture and best practices
- Database design and data modeling
- API design and integration patterns
- Infrastructure and deployment strategies
- Cost optimization and resource planning

Design architectures that are:
- Scalable and maintainable
- Secure and reliable
- Cost-effective and efficient
- Technology-appropriate for the problem domain"""

        prompt = f"""Design a comprehensive system architecture based on these requirements:

Requirements: {requirements}

Constraints: {constraints}

Technology Preferences: {preferences}

Please provide a detailed architectural design including:

1. **Architecture Overview**
   - Recommended architecture pattern (microservices, monolithic, etc.)
   - High-level system components and their interactions
   - Data flow and integration points

2. **Technology Stack**
   - Backend technologies and frameworks
   - Frontend technologies (if applicable)
   - Database technologies and storage solutions
   - Infrastructure and deployment technologies
   - Supporting tools and services

3. **Component Design**
   - Detailed breakdown of system components
   - API specifications and contracts
   - Database schema design
   - Security measures and authentication/authorization

4. **Scalability & Performance**
   - Horizontal and vertical scaling strategies
   - Caching strategies
   - Performance optimization recommendations
   - Load balancing and redundancy

5. **Security Architecture**
   - Authentication and authorization mechanisms
   - Data protection strategies
   - Security best practices implementation
   - Compliance considerations

6. **Infrastructure & Deployment**
   - Cloud provider recommendations
   - Containerization and orchestration
   - CI/CD pipeline design
   - Monitoring and logging architecture

7. **Risks & Mitigations**
   - Identified technical risks
   - Mitigation strategies
   - Contingency plans

8. **Cost Analysis**
   - Infrastructure cost estimates
   - Development effort estimates
   - Maintenance cost considerations

Provide specific, actionable recommendations with reasoning for each choice."""

        response = await self.generate_response(prompt, system_prompt)

        return {
            "status": "completed",
            "architecture_design": {
                "raw_design": response,
                "structured_design": self._parse_architecture_design(response),
                "confidence_score": 0.9,
                "design_complexity": self._assess_complexity(requirements)
            },
            "metadata": {
                "designed_at": datetime.utcnow().isoformat(),
                "requirements_hash": hash(str(requirements)),  # For caching/versioning
                "design_method": "llm_comprehensive"
            }
        }

    async def _review_architecture(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review an existing architecture design"""
        architecture_design = task.get("architecture_design", "")
        requirements = task.get("requirements", {})

        system_prompt = """You are a Senior Architecture Reviewer specializing in evaluating system designs for:
- Technical soundness and best practices compliance
- Scalability and performance considerations
- Security vulnerabilities and mitigations
- Maintainability and extensibility
- Cost-effectiveness and resource optimization
- Risk assessment and mitigation strategies"""

        prompt = f"""Review this architecture design against the requirements:

Architecture Design: {architecture_design}

Requirements: {requirements}

Provide a comprehensive review covering:
1. Strengths of the design
2. Weaknesses and concerns
3. Security vulnerabilities or gaps
4. Scalability and performance issues
5. Maintainability concerns
6. Cost optimization opportunities
7. Compliance and regulatory considerations
8. Recommended improvements and alternatives
9. Risk assessment (High/Medium/Low for each concern)
10. Overall recommendation (Accept/Reject/Revise)"""

        response = await self.generate_response(prompt, system_prompt)

        return {
            "status": "completed",
            "architecture_review": {
                "raw_review": response,
                "structured_feedback": self._parse_architecture_review(response),
                "overall_recommendation": self._extract_recommendation(response),
                "critical_issues": self._identify_critical_issues(response)
            }
        }

    async def _optimize_architecture(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize an existing architecture for specific goals"""
        current_architecture = task.get("current_architecture", "")
        optimization_goals = task.get("optimization_goals", [])
        constraints = task.get("constraints", {})

        system_prompt = """You are an Architecture Optimization Specialist focused on:
- Performance optimization
- Cost reduction
- Scalability improvements
- Security hardening
- Maintainability enhancements
- Technology modernization"""

        prompt = f"""Optimize this architecture for the specified goals:

Current Architecture: {current_architecture}

Optimization Goals: {optimization_goals}

Constraints: {constraints}

Provide optimization recommendations including:
1. Specific changes to improve performance/scalability
2. Cost reduction opportunities
3. Security enhancements
4. Technology modernization options
5. Process and tooling improvements
6. Risk-benefit analysis for each recommendation
7. Implementation priority and effort estimates
8. Expected outcomes and metrics for success"""

        response = await self.generate_response(prompt, system_prompt)

        return {
            "status": "completed",
            "architecture_optimization": {
                "raw_optimization": response,
                "optimization_recommendations": self._parse_optimizations(response),
                "expected_benefits": self._estimate_benefits(response),
                "implementation_effort": self._assess_effort(response)
            }
        }

    async def _handle_design_architecture(self, message) -> Any:
        """Handle design architecture message"""
        task = message.content
        result = await self._design_architecture(task)
        return self._create_response_message(message, "architecture_design_complete", result)

    async def _handle_review_architecture(self, message) -> Any:
        """Handle review architecture message"""
        task = message.content
        result = await self._review_architecture(task)
        return self._create_response_message(message, "architecture_review_complete", result)

    async def _handle_optimize_architecture(self, message) -> Any:
        """Handle optimize architecture message"""
        task = message.content
        result = await self._optimize_architecture(task)
        return self._create_response_message(message, "architecture_optimization_complete", result)

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

    def _parse_architecture_design(self, response: str) -> Dict[str, Any]:
        """Parse architecture design response into structured format"""
        # Would implement more sophisticated parsing in practice
        return {
            "architecture_pattern": "microservices",  # Extract from response
            "technology_stack": {},
            "components": [],
            "security_measures": [],
            "scalability_strategy": "",
            "infrastructure_plan": {}
        }

    def _parse_architecture_review(self, response: str) -> Dict[str, Any]:
        """Parse architecture review into structured feedback"""
        return {
            "strengths": [],
            "weaknesses": [],
            "security_concerns": [],
            "scalability_issues": [],
            "recommendations": [],
            "risk_assessment": {}
        }

    def _parse_optimizations(self, response: str) -> List[Dict[str, Any]]:
        """Parse optimization recommendations"""
        return []

    def _assess_complexity(self, requirements: Dict[str, Any]) -> str:
        """Assess the complexity of the requirements"""
        # Simple assessment based on requirements size
        req_str = str(requirements)
        if len(req_str) > 2000:
            return "high"
        elif len(req_str) > 1000:
            return "medium"
        else:
            return "low"

    def _extract_recommendation(self, response: str) -> str:
        """Extract overall recommendation from review"""
        response_lower = response.lower()
        if "accept" in response_lower and "reject" not in response_lower:
            return "accept"
        elif "reject" in response_lower:
            return "reject"
        else:
            return "revise"

    def _identify_critical_issues(self, response: str) -> List[str]:
        """Identify critical issues from the response"""
        issues = []
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(severity in line_lower for severity in ['critical', 'high risk', 'severe']):
                issues.append(line.strip())
        return issues

    def _estimate_benefits(self, response: str) -> Dict[str, Any]:
        """Estimate benefits from optimization"""
        return {
            "performance_improvement": "estimated",
            "cost_reduction": "estimated",
            "scalability_gain": "estimated"
        }

    def _assess_effort(self, response: str) -> str:
        """Assess implementation effort"""
        # Simple assessment
        if "significant" in response.lower() or "major" in response.lower():
            return "high"
        elif "moderate" in response.lower():
            return "medium"
        else:
            return "low"