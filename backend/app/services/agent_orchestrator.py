"""
Agent Orchestrator - coordinates interactions between AI personas
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from app.agents.base_agent import BaseAgent, MCPMessage
from app.agents.spec_interpreter import SpecInterpreterAgent
from app.agents.architect import ArchitectAgent
from app.agents.generator import GeneratorAgent
from app.agents.validator import ValidatorAgent
from app.agents.persona_generator import PersonaGenerator
from app.services.ollama_service import OllamaService
from app.services.graph_rag_service import GraphRAGService

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrates interactions between AI agents in the StratAgent system"""

    def __init__(self, ollama_service: OllamaService, graph_rag_service: GraphRAGService):
        self.ollama_service = ollama_service
        self.graph_rag_service = graph_rag_service

        # Initialize core agents
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents()

        # Interaction tracking
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.interaction_history: List[Dict[str, Any]] = []

        # Orchestration configuration
        self.max_concurrent_interactions = 5
        self.response_timeout = 30.0

    def _initialize_agents(self):
        """Initialize all core agents"""
        # SpecGen Pipeline Agents
        self.agents["spec_interpreter"] = SpecInterpreterAgent("spec_interpreter_001", self.ollama_service)
        self.agents["architect"] = ArchitectAgent("architect_001", self.ollama_service)
        self.agents["generator"] = GeneratorAgent("generator_001", self.ollama_service)
        self.agents["validator"] = ValidatorAgent("validator_001", self.ollama_service)

        # PersonaGen System
        self.agents["persona_generator"] = PersonaGenerator("persona_gen_001", self.ollama_service)

        logger.info(f"Initialized {len(self.agents)} core agents")

    async def process_user_request(self, user_input: str, project_context: Dict[str, Any], active_personas: List[str]) -> Dict[str, Any]:
        """Process a user request by coordinating multiple agents"""
        conversation_id = f"conv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(user_input) % 10000}"

        # Initialize conversation tracking
        self.active_conversations[conversation_id] = {
            "user_input": user_input,
            "project_context": project_context,
            "active_personas": active_personas,
            "start_time": datetime.utcnow(),
            "status": "processing",
            "responses": [],
            "coordination_log": []
        }

        try:
            # Step 1: Retrieve relevant context from knowledge graph
            context = await self.graph_rag_service.retrieve_context(user_input, max_nodes=3)

            # Step 2: Analyze requirements with SpecInterpreter
            spec_analysis = await self._coordinate_spec_analysis(user_input, context, project_context)

            # Step 3: Generate coordinated responses from active personas
            responses = await self._coordinate_persona_responses(
                user_input, spec_analysis, active_personas, context, project_context
            )

            # Step 4: Validate and consolidate responses
            validated_responses = await self._validate_responses(responses, user_input)

            # Step 5: Learn from interaction
            await self._learn_from_interaction(user_input, responses, validated_responses)

            # Update conversation status
            self.active_conversations[conversation_id].update({
                "status": "completed",
                "end_time": datetime.utcnow(),
                "final_responses": validated_responses
            })

            # Record interaction in history
            self.interaction_history.append({
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "response_count": len(validated_responses),
                "active_personas": active_personas,
                "success": True
            })

            return {
                "conversation_id": conversation_id,
                "responses": validated_responses,
                "context_used": len(context),
                "processing_time": (datetime.utcnow() - self.active_conversations[conversation_id]["start_time"]).total_seconds()
            }

        except Exception as e:
            logger.error(f"Error processing user request: {str(e)}")

            # Update conversation status on error
            self.active_conversations[conversation_id].update({
                "status": "error",
                "error": str(e),
                "end_time": datetime.utcnow()
            })

            # Record failed interaction
            self.interaction_history.append({
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "error": str(e),
                "success": False
            })

            return {
                "conversation_id": conversation_id,
                "error": str(e),
                "responses": []
            }

    async def _coordinate_spec_analysis(self, user_input: str, context: List[Dict[str, Any]], project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate requirements analysis across agents"""
        try:
            # Use SpecInterpreter to analyze requirements
            spec_agent = self.agents["spec_interpreter"]

            analysis_task = {
                "user_input": user_input,
                "context": [ctx["content"] for ctx in context],
                "project_context": project_context
            }

            analysis_result = await spec_agent.process_task({
                "type": "analyze_requirements",
                **analysis_task
            })

            return analysis_result

        except Exception as e:
            logger.warning(f"Spec analysis failed: {str(e)}, continuing without analysis")
            return {"specifications": {}, "confidence_score": 0.0}

    async def _coordinate_persona_responses(
        self,
        user_input: str,
        spec_analysis: Dict[str, Any],
        active_personas: List[str],
        context: List[Dict[str, Any]],
        project_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Coordinate responses from multiple active personas"""
        responses = []
        coordination_log = []

        # Create tasks for concurrent processing
        tasks = []
        for persona_role in active_personas:
            if persona_role in self.agents:
                task = self._generate_persona_response(
                    persona_role, user_input, spec_analysis, context, project_context
                )
                tasks.append(task)
            else:
                logger.warning(f"Unknown persona role: {persona_role}")

        # Process responses with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_interactions)

        async def process_with_semaphore(task):
            async with semaphore:
                return await task

        # Execute tasks concurrently
        task_results = await asyncio.gather(*[process_with_semaphore(task) for task in tasks], return_exceptions=True)

        # Process results
        for i, result in enumerate(task_results):
            persona_role = active_personas[i]

            if isinstance(result, Exception):
                logger.error(f"Error from {persona_role}: {str(result)}")
                responses.append({
                    "persona_role": persona_role,
                    "content": f"Error: {str(result)}",
                    "status": "error",
                    "confidence": 0.0
                })
                coordination_log.append(f"ERROR: {persona_role} failed with {str(result)}")
            else:
                responses.append(result)
                coordination_log.append(f"SUCCESS: {persona_role} responded")

        # Update coordination log
        if hasattr(self, 'active_conversations') and self.active_conversations:
            current_conv_id = list(self.active_conversations.keys())[-1]
            self.active_conversations[current_conv_id]["coordination_log"].extend(coordination_log)

        return responses

    async def _generate_persona_response(
        self,
        persona_role: str,
        user_input: str,
        spec_analysis: Dict[str, Any],
        context: List[Dict[str, Any]],
        project_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response from a specific persona"""
        try:
            agent = self.agents[persona_role]

            # Create persona-specific context
            persona_context = self._create_persona_context(
                persona_role, user_input, spec_analysis, context, project_context
            )

            # Generate response using agent's system prompt
            system_prompt = self._get_persona_system_prompt(persona_role)

            response = await agent.generate_response(
                prompt=user_input,
                system_prompt=system_prompt,
                temperature=0.7
            )

            return {
                "persona_role": persona_role,
                "content": response,
                "status": "success",
                "confidence": 0.85,  # Would be calculated based on response quality
                "context_used": len(context),
                "processing_time": 0.0  # Would be measured
            }

        except Exception as e:
            logger.error(f"Error generating response from {persona_role}: {str(e)}")
            return {
                "persona_role": persona_role,
                "content": f"Error generating response: {str(e)}",
                "status": "error",
                "confidence": 0.0
            }

    async def _validate_responses(self, responses: List[Dict[str, Any]], original_input: str) -> List[Dict[str, Any]]:
        """Validate and potentially consolidate responses"""
        try:
            validator = self.agents["validator"]

            # Use validator to check response quality
            validation_tasks = []
            for response in responses:
                if response["status"] == "success":
                    validation_task = {
                        "type": "validate_code",  # Could be general validation
                        "code": response["content"],
                        "language": "natural_language",  # Since these are text responses
                        "criteria": ["quality", "relevance", "safety"]
                    }
                    validation_tasks.append(validator.process_task(validation_task))

            # Run validations concurrently
            if validation_tasks:
                validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

                # Apply validation results
                validated_responses = []
                for i, (response, validation_result) in enumerate(zip(responses, validation_results)):
                    if isinstance(validation_result, Exception):
                        logger.warning(f"Validation failed for response {i}: {str(validation_result)}")
                        validated_responses.append(response)
                    else:
                        # Enhance response with validation data
                        validated_response = response.copy()
                        validated_response["validation"] = validation_result
                        validated_response["quality_score"] = validation_result.get("validation_report", {}).get("code_quality", {}).get("overall_score", 0.8)
                        validated_responses.append(validated_response)

                return validated_responses

        except Exception as e:
            logger.warning(f"Response validation failed: {str(e)}, returning original responses")

        return responses

    async def _learn_from_interaction(self, user_input: str, responses: List[Dict[str, Any]], validated_responses: List[Dict[str, Any]]):
        """Learn from the interaction and update knowledge base"""
        try:
            # Create learning data
            interaction_data = {
                "user_input": user_input,
                "responses_generated": len(responses),
                "successful_responses": len([r for r in responses if r["status"] == "success"]),
                "average_confidence": sum(r.get("confidence", 0) for r in responses) / len(responses) if responses else 0,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add to knowledge graph
            await self.graph_rag_service.add_knowledge(
                content=f"User query: {user_input}\nResponses: {len(responses)} generated",
                metadata={
                    "type": "interaction",
                    "interaction_data": interaction_data,
                    "learning_opportunity": True
                },
                node_type="interaction"
            )

            # Extract key learnings from successful responses
            for response in validated_responses:
                if response.get("quality_score", 0) > 0.8:
                    await self.graph_rag_service.add_knowledge(
                        content=response["content"],
                        metadata={
                            "type": "response_pattern",
                            "persona_role": response["persona_role"],
                            "quality_score": response.get("quality_score", 0),
                            "source": "generated_response"
                        },
                        node_type="pattern"
                    )

        except Exception as e:
            logger.warning(f"Learning from interaction failed: {str(e)}")

    def _create_persona_context(
        self,
        persona_role: str,
        user_input: str,
        spec_analysis: Dict[str, Any],
        context: List[Dict[str, Any]],
        project_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create context specific to a persona's role"""
        base_context = {
            "user_input": user_input,
            "project_context": project_context,
            "available_context": [ctx["content"] for ctx in context]
        }

        # Add role-specific context
        if persona_role == "architect":
            base_context["spec_analysis"] = spec_analysis
            base_context["focus_areas"] = ["system_design", "technology_selection", "scalability"]
        elif persona_role == "generator":
            base_context["spec_analysis"] = spec_analysis
            base_context["focus_areas"] = ["code_generation", "best_practices", "implementation"]
        elif persona_role == "validator":
            base_context["spec_analysis"] = spec_analysis
            base_context["focus_areas"] = ["quality_assurance", "security_review", "testing"]
        # Add other persona-specific contexts as needed

        return base_context

    def _get_persona_system_prompt(self, persona_role: str) -> str:
        """Get the system prompt for a persona role"""
        # This would ideally come from the persona profile, but for now use defaults
        prompts = {
            "architect": """You are a Senior Software Architect. Focus on system design, technology recommendations, and technical leadership.""",
            "frontend_dev": """You are a Frontend Developer specializing in React/Next.js. Focus on UI/UX implementation and user experience.""",
            "backend_dev": """You are a Backend Developer. Focus on server-side logic, APIs, and data management.""",
            "tester": """You are a Quality Assurance Engineer. Focus on testing strategies, bug prevention, and quality assurance.""",
            "reviewer": """You are a Code Reviewer. Focus on code quality assessment, best practices, and constructive feedback.""",
            "spec_interpreter": """You are a Requirements Analyst. Focus on clarifying requirements and creating detailed specifications.""",
            "generator": """You are a Code Generator. Focus on producing clean, maintainable, production-ready code.""",
            "validator": """You are a Code Validator. Focus on quality assurance, security validation, and standards compliance."""
        }

        return prompts.get(persona_role, "You are an AI assistant helping with software development.")

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get the current status of the orchestrator"""
        agent_statuses = {}
        for agent_name, agent in self.agents.items():
            agent_statuses[agent_name] = agent.get_status()

        return {
            "total_agents": len(self.agents),
            "active_conversations": len(self.active_conversations),
            "total_interactions": len(self.interaction_history),
            "agent_statuses": agent_statuses,
            "system_health": "healthy" if all(status.get("is_active", False) for status in agent_statuses.values()) else "degraded"
        }

    async def shutdown(self):
        """Gracefully shutdown the orchestrator and all agents"""
        logger.info("Shutting down agent orchestrator...")

        # Shutdown all agents
        shutdown_tasks = []
        for agent in self.agents.values():
            shutdown_tasks.append(agent.receive_message(
                MCPMessage("shutdown", "orchestrator", agent.agent_id, {"reason": "system_shutdown"})
            ))

        # Wait for all agents to shutdown
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Save any pending data
        self.graph_rag_service.save_graph()
        self.graph_rag_service.save_index()

        logger.info("Agent orchestrator shutdown complete")