"""
Orthos Service - Dual-validation system for agent safety and reliability
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import json

from app.services.ollama_service import OllamaService
from app.services.graph_rag_service import GraphRAGService

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationResult(Enum):
    """Validation result outcomes"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"

class OrthosValidator:
    """Individual validator in the Orthos dual-validation system"""

    def __init__(self, validator_id: str, name: str, ollama_service: OllamaService, validation_type: str = "primary"):
        self.validator_id = validator_id
        self.name = name
        self.ollama_service = ollama_service
        self.validation_type = validation_type  # "primary" or "secondary"

        # Validation configuration
        self.confidence_threshold = 0.8
        self.timeout_seconds = 30.0

        # Performance tracking
        self.validations_performed = 0
        self.success_rate = 0.0
        self.average_response_time = 0.0

    async def validate(self, content: str, validation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform validation on content"""
        start_time = asyncio.get_event_loop().time()

        try:
            # Generate validation prompt based on content type
            validation_prompt = self._create_validation_prompt(content, validation_context)

            # Get validation response from LLM
            response = await asyncio.wait_for(
                self.ollama_service.generate_response(
                    prompt=validation_prompt,
                    system_prompt=self._get_validation_system_prompt(),
                    temperature=0.1  # Low temperature for consistent validation
                ),
                timeout=self.timeout_seconds
            )

            # Parse validation result
            validation_result = self._parse_validation_response(response)

            # Update performance metrics
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time

            self.validations_performed += 1
            self.average_response_time = (
                (self.average_response_time * (self.validations_performed - 1)) + response_time
            ) / self.validations_performed

            return {
                "validator_id": self.validator_id,
                "validator_name": self.name,
                "validation_type": self.validation_type,
                "result": validation_result["result"],
                "severity": validation_result["severity"],
                "confidence": validation_result["confidence"],
                "issues": validation_result["issues"],
                "recommendations": validation_result["recommendations"],
                "processing_time": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }

        except asyncio.TimeoutError:
            logger.error(f"Validation timeout for {self.name}")
            return self._create_error_result("timeout", "Validation timed out")
        except Exception as e:
            logger.error(f"Validation error for {self.name}: {str(e)}")
            return self._create_error_result("error", str(e))

    def _create_validation_prompt(self, content: str, context: Dict[str, Any]) -> str:
        """Create appropriate validation prompt based on content and context"""
        content_type = context.get("content_type", "general")
        validation_criteria = context.get("criteria", ["quality", "safety", "accuracy"])

        base_prompt = f"""Please validate the following content:

Content: {content}

Content Type: {content_type}
Validation Criteria: {', '.join(validation_criteria)}

"""

        if content_type == "code":
            base_prompt += """
Code Validation Checklist:
1. Syntax correctness
2. Logic soundness
3. Security vulnerabilities
4. Best practices compliance
5. Error handling adequacy
6. Performance considerations
7. Maintainability assessment

"""
        elif content_type == "text_response":
            base_prompt += """
Text Response Validation Checklist:
1. Accuracy and factual correctness
2. Clarity and coherence
3. Appropriateness and tone
4. Completeness of response
5. Safety and ethical considerations
6. Relevance to the query
7. Professional quality

"""
        elif content_type == "design_specification":
            base_prompt += """
Design Specification Validation Checklist:
1. Completeness of requirements
2. Technical feasibility
3. Consistency and coherence
4. Scalability considerations
5. Security implications
6. Implementation practicality
7. Risk assessment

"""

        base_prompt += """
Validation Response Format:
RESULT: [PASS/FAIL/WARNING]
SEVERITY: [LOW/MEDIUM/HIGH/CRITICAL]
CONFIDENCE: [0.0-1.0]

ISSUES:
- [List specific issues found]

RECOMMENDATIONS:
- [List specific recommendations for improvement]

EXPLANATION:
[Detailed explanation of validation reasoning]
"""

        return base_prompt

    def _get_validation_system_prompt(self) -> str:
        """Get the system prompt for validation"""
        if self.validation_type == "primary":
            return """You are a Primary Validator in the Orthos dual-validation system.
Your role is to perform comprehensive, thorough validation of content for quality, safety, and compliance.
You must be meticulous, identifying even minor issues that could impact quality or safety.
When in doubt, err on the side of caution and recommend improvements."""
        else:
            return """You are a Secondary Validator in the Orthos dual-validation system.
Your role is to provide a second opinion and catch any issues missed by primary validation.
Focus on different aspects than primary validation and provide complementary analysis.
Confirm primary findings and identify additional concerns or mitigations."""

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse the validation response into structured format"""
        result = {
            "result": ValidationResult.WARNING.value,
            "severity": ValidationSeverity.MEDIUM.value,
            "confidence": 0.5,
            "issues": [],
            "recommendations": [],
            "explanation": ""
        }

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse structured sections
            if line.startswith('RESULT:'):
                result_text = line.split(':', 1)[1].strip().upper()
                if result_text in [r.value for r in ValidationResult]:
                    result["result"] = result_text
                else:
                    result["result"] = ValidationResult.WARNING.value

            elif line.startswith('SEVERITY:'):
                severity_text = line.split(':', 1)[1].strip().upper()
                if severity_text in [s.value for s in ValidationSeverity]:
                    result["severity"] = severity_text
                else:
                    result["severity"] = ValidationSeverity.MEDIUM.value

            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                    result["confidence"] = max(0.0, min(1.0, confidence))
                except ValueError:
                    result["confidence"] = 0.5

            elif line.startswith('ISSUES:'):
                current_section = "issues"
            elif line.startswith('RECOMMENDATIONS:'):
                current_section = "recommendations"
            elif line.startswith('EXPLANATION:'):
                current_section = "explanation"

            elif current_section and line.startswith('-'):
                item = line[1:].strip()
                if current_section in ["issues", "recommendations"]:
                    result[current_section].append(item)
                elif current_section == "explanation":
                    result["explanation"] += item + " "

        # Clean up explanation
        result["explanation"] = result["explanation"].strip()

        return result

    def _create_error_result(self, error_type: str, message: str) -> Dict[str, Any]:
        """Create an error validation result"""
        return {
            "validator_id": self.validator_id,
            "validator_name": self.name,
            "validation_type": self.validation_type,
            "result": ValidationResult.ERROR.value,
            "severity": ValidationSeverity.HIGH.value,
            "confidence": 0.0,
            "issues": [f"Validation {error_type}: {message}"],
            "recommendations": ["Retry validation or check system status"],
            "processing_time": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }

class OrthosService:
    """Dual-validation system for agent safety and reliability"""

    def __init__(self, ollama_service: OllamaService, graph_rag_service: GraphRAGService):
        self.ollama_service = ollama_service
        self.graph_rag_service = graph_rag_service

        # Initialize validators
        self.primary_validator = OrthosValidator(
            "orthos_primary_001",
            "Orthos Primary Validator",
            ollama_service,
            "primary"
        )

        self.secondary_validator = OrthosValidator(
            "orthos_secondary_001",
            "Orthos Secondary Validator",
            ollama_service,
            "secondary"
        )

        # Consensus configuration
        self.consensus_threshold = 0.7  # Minimum agreement required
        self.critical_severity_block = True  # Block content with critical issues

        # Validation history
        self.validation_history: List[Dict[str, Any]] = []

    async def validate_content(self, content: str, validation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform dual validation on content"""
        validation_id = f"val_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(content) % 10000}"

        # Perform primary validation
        primary_result = await self.primary_validator.validate(content, validation_context)

        # Perform secondary validation
        secondary_result = await self.secondary_validator.validate(content, validation_context)

        # Calculate consensus
        consensus_result = self._calculate_consensus([primary_result, secondary_result])

        # Determine final validation outcome
        final_result = self._determine_final_result(primary_result, secondary_result, consensus_result)

        # Record validation
        validation_record = {
            "validation_id": validation_id,
            "content_hash": hash(content),
            "content_length": len(content),
            "primary_validation": primary_result,
            "secondary_validation": secondary_result,
            "consensus": consensus_result,
            "final_result": final_result,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.validation_history.append(validation_record)

        # Learn from validation results
        await self._learn_from_validation(content, validation_context, final_result)

        return {
            "validation_id": validation_id,
            "final_result": final_result,
            "primary_validation": primary_result,
            "secondary_validation": secondary_result,
            "consensus_score": consensus_result["agreement_score"],
            "processing_time": primary_result["processing_time"] + secondary_result["processing_time"],
            "recommendations": self._compile_recommendations([primary_result, secondary_result])
        }

    def _calculate_consensus(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus between validators"""
        if len(validation_results) < 2:
            return {"agreement_score": 1.0, "consensus_level": "single_validator"}

        results = [r["result"] for r in validation_results]
        severities = [r["severity"] for r in validation_results]
        confidences = [r["confidence"] for r in validation_results]

        # Calculate agreement score
        result_agreement = len(set(results)) == 1  # All same result
        severity_agreement = len(set(severities)) <= 2  # At most 2 different severity levels

        base_agreement = 1.0 if result_agreement else 0.5
        severity_bonus = 0.3 if severity_agreement else 0.0
        confidence_avg = sum(confidences) / len(confidences)

        agreement_score = min(1.0, base_agreement + severity_bonus + (confidence_avg * 0.2))

        # Determine consensus level
        if agreement_score >= 0.8:
            consensus_level = "strong_agreement"
        elif agreement_score >= 0.6:
            consensus_level = "moderate_agreement"
        else:
            consensus_level = "disagreement"

        return {
            "agreement_score": agreement_score,
            "consensus_level": consensus_level,
            "result_distribution": self._count_occurrences(results),
            "severity_distribution": self._count_occurrences(severities),
            "average_confidence": confidence_avg
        }

    def _determine_final_result(self, primary: Dict[str, Any], secondary: Dict[str, Any], consensus: Dict[str, Any]) -> str:
        """Determine the final validation result"""
        # Critical issues always fail
        if self.critical_severity_block and (
            primary["severity"] == ValidationSeverity.CRITICAL.value or
            secondary["severity"] == ValidationSeverity.CRITICAL.value
        ):
            return ValidationResult.FAIL.value

        # Use consensus for final decision
        if consensus["agreement_score"] >= self.consensus_threshold:
            # Return the most common result
            results = [primary["result"], secondary["result"]]
            if results[0] == results[1]:
                return results[0]
            elif ValidationResult.FAIL.value in results:
                return ValidationResult.FAIL.value
            elif ValidationResult.WARNING.value in results:
                return ValidationResult.WARNING.value
            else:
                return ValidationResult.PASS.value
        else:
            # Low consensus - be conservative
            if ValidationResult.FAIL.value in [primary["result"], secondary["result"]]:
                return ValidationResult.WARNING.value
            else:
                return ValidationResult.WARNING.value

    def _compile_recommendations(self, validation_results: List[Dict[str, Any]]) -> List[str]:
        """Compile recommendations from all validators"""
        all_recommendations = []
        seen_recommendations = set()

        for result in validation_results:
            for recommendation in result.get("recommendations", []):
                if recommendation not in seen_recommendations:
                    all_recommendations.append(recommendation)
                    seen_recommendations.add(recommendation)

        return all_recommendations

    async def _learn_from_validation(self, content: str, context: Dict[str, Any], final_result: str):
        """Learn from validation results to improve future validations"""
        try:
            # Add validation patterns to knowledge graph
            learning_content = f"Validation Result: {final_result}\nContent Type: {context.get('content_type', 'unknown')}"

            await self.graph_rag_service.add_knowledge(
                content=learning_content,
                metadata={
                    "type": "validation_pattern",
                    "validation_result": final_result,
                    "content_type": context.get("content_type"),
                    "learning_opportunity": True
                },
                node_type="validation_learning"
            )

        except Exception as e:
            logger.warning(f"Failed to learn from validation: {str(e)}")

    def _count_occurrences(self, items: List[Any]) -> Dict[str, int]:
        """Count occurrences of items in a list"""
        counts = {}
        for item in items:
            counts[str(item)] = counts.get(str(item), 0) + 1
        return counts

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {"total_validations": 0}

        total_validations = len(self.validation_history)
        result_counts = {}

        for record in self.validation_history[-100:]:  # Last 100 validations
            result = record["final_result"]
            result_counts[result] = result_counts.get(result, 0) + 1

        # Calculate pass rate
        pass_count = result_counts.get(ValidationResult.PASS.value, 0)
        pass_rate = pass_count / sum(result_counts.values()) if result_counts else 0

        return {
            "total_validations": total_validations,
            "recent_result_distribution": result_counts,
            "pass_rate": pass_rate,
            "primary_validator_stats": self.primary_validator.__dict__,
            "secondary_validator_stats": self.secondary_validator.__dict__,
            "average_consensus_score": sum(r["consensus"]["agreement_score"] for r in self.validation_history[-50:]) / min(50, len(self.validation_history)) if self.validation_history else 0
        }

    async def validate_with_context(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content with additional context from knowledge graph"""
        # Retrieve relevant validation context
        validation_context = await self.graph_rag_service.retrieve_context(
            f"validation patterns for {context.get('content_type', 'content')}",
            max_nodes=3
        )

        # Enhance validation context
        enhanced_context = context.copy()
        enhanced_context["historical_validation_context"] = [ctx["content"] for ctx in validation_context]

        # Perform validation
        return await self.validate_content(content, enhanced_context)