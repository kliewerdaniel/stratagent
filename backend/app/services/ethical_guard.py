"""
Ethical Guard - Boundary enforcement and safety checks
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import json

from app.services.ollama_service import OllamaService
from app.services.graph_rag_service import GraphRAGService

logger = logging.getLogger(__name__)

class EthicalBoundary:
    """Defines an ethical boundary with rules and constraints"""

    def __init__(self, boundary_id: str, name: str, category: str, rules: List[Dict[str, Any]], severity: str = "high"):
        self.boundary_id = boundary_id
        self.name = name
        self.category = category
        self.rules = rules
        self.severity = severity

    def check_violation(self, content: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if content violates this boundary"""
        for rule in self.rules:
            if self._evaluate_rule(rule, content, context):
                return {
                    "boundary_id": self.boundary_id,
                    "boundary_name": self.name,
                    "category": self.category,
                    "severity": self.severity,
                    "rule": rule,
                    "violated_content": content[:200] + "..." if len(content) > 200 else content,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                }
        return None

    def _evaluate_rule(self, rule: Dict[str, Any], content: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single rule against content and context"""
        rule_type = rule.get("type")

        if rule_type == "keyword_match":
            return self._check_keyword_match(rule, content)
        elif rule_type == "pattern_match":
            return self._check_pattern_match(rule, content)
        elif rule_type == "semantic_check":
            return self._check_semantic_rule(rule, content, context)
        elif rule_type == "context_check":
            return self._check_context_rule(rule, context)
        else:
            return False

    def _check_keyword_match(self, rule: Dict[str, Any], content: str) -> bool:
        """Check for keyword matches"""
        keywords = rule.get("keywords", [])
        case_sensitive = rule.get("case_sensitive", False)

        content_check = content if case_sensitive else content.lower()
        keywords_check = keywords if case_sensitive else [k.lower() for k in keywords]

        return any(keyword in content_check for keyword in keywords_check)

    def _check_pattern_match(self, rule: Dict[str, Any], content: str) -> bool:
        """Check for regex pattern matches"""
        patterns = rule.get("patterns", [])

        for pattern in patterns:
            try:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")

        return False

    def _check_semantic_rule(self, rule: Dict[str, Any], content: str, context: Dict[str, Any]) -> bool:
        """Check semantic rules (would require ML model)"""
        # Placeholder for semantic analysis
        # In practice, this would use a classifier or embedding similarity
        return False

    def _check_context_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check context-based rules"""
        required_context = rule.get("required_context", [])
        forbidden_context = rule.get("forbidden_context", [])

        for req_ctx in required_context:
            if req_ctx not in context:
                return True  # Violation - required context missing

        for forbid_ctx in forbidden_context:
            if forbid_ctx in context:
                return True  # Violation - forbidden context present

        return False

class EthicalGuard:
    """Ethical boundary enforcement and safety checking system"""

    def __init__(self, ollama_service: OllamaService, graph_rag_service: GraphRAGService):
        self.ollama_service = ollama_service
        self.graph_rag_service = graph_rag_service

        # Ethical boundaries
        self.boundaries = self._load_ethical_boundaries()

        # Safety configuration
        self.block_critical_violations = True
        self.warn_medium_violations = True
        self.log_all_checks = True

        # Violation tracking
        self.violation_history: List[Dict[str, Any]] = []

    def _load_ethical_boundaries(self) -> List[EthicalBoundary]:
        """Load predefined ethical boundaries"""
        return [
            # Harm Prevention
            EthicalBoundary(
                "harm_prevention_001",
                "Harm Prevention",
                "safety",
                [
                    {
                        "type": "keyword_match",
                        "keywords": ["harm", "damage", "injure", "hurt", "kill", "destroy"],
                        "case_sensitive": False
                    },
                    {
                        "type": "pattern_match",
                        "patterns": [r"how to.*harm", r"ways to.*damage", r"methods to.*injure"]
                    }
                ],
                "critical"
            ),

            # Illegal Activities
            EthicalBoundary(
                "illegal_activities_001",
                "Illegal Activities",
                "legality",
                [
                    {
                        "type": "keyword_match",
                        "keywords": ["hack", "exploit", "crack", "steal", "fraud", "scam", "illegal"],
                        "case_sensitive": False
                    },
                    {
                        "type": "pattern_match",
                        "patterns": [r"how to.*hack", r"ways to.*steal", r"methods to.*fraud"]
                    }
                ],
                "critical"
            ),

            # Personal Information Protection
            EthicalBoundary(
                "privacy_protection_001",
                "Privacy Protection",
                "privacy",
                [
                    {
                        "type": "pattern_match",
                        "patterns": [
                            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                            r"\b\d{4} \d{4} \d{4} \d{4}\b",  # Credit card pattern
                            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email pattern
                        ]
                    }
                ],
                "high"
            ),

            # Misinformation Prevention
            EthicalBoundary(
                "misinformation_prevention_001",
                "Misinformation Prevention",
                "accuracy",
                [
                    {
                        "type": "keyword_match",
                        "keywords": ["false", "lie", "deceive", "mislead", "fabricate"],
                        "case_sensitive": False
                    },
                    {
                        "type": "context_check",
                        "required_context": ["factual_check"],
                        "forbidden_context": ["speculative_content"]
                    }
                ],
                "medium"
            ),

            # Bias and Discrimination
            EthicalBoundary(
                "bias_prevention_001",
                "Bias Prevention",
                "fairness",
                [
                    {
                        "type": "keyword_match",
                        "keywords": ["discriminate", "bias", "prejudice", "stereotype"],
                        "case_sensitive": False
                    }
                ],
                "medium"
            ),

            # Intellectual Property Respect
            EthicalBoundary(
                "ip_respect_001",
                "Intellectual Property Respect",
                "legality",
                [
                    {
                        "type": "keyword_match",
                        "keywords": ["copyright", "pirate", "steal code", "plagiarize"],
                        "case_sensitive": False
                    }
                ],
                "high"
            )
        ]

    async def check_content(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check content against all ethical boundaries"""
        violations = []
        warnings = []

        for boundary in self.boundaries:
            violation = boundary.check_violation(content, context)
            if violation:
                if violation["severity"] == "critical":
                    violations.append(violation)
                elif violation["severity"] in ["high", "medium"] and self.warn_medium_violations:
                    warnings.append(violation)

        # Record check in history
        check_record = {
            "content_hash": hash(content),
            "content_length": len(content),
            "violations_found": len(violations),
            "warnings_found": len(warnings),
            "boundaries_checked": len(self.boundaries),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.violation_history.append(check_record)

        # Determine overall safety status
        if violations and self.block_critical_violations:
            safety_status = "blocked"
            action_required = "Content blocked due to critical ethical violations"
        elif warnings:
            safety_status = "warning"
            action_required = "Content approved with warnings"
        else:
            safety_status = "approved"
            action_required = "Content approved"

        return {
            "safety_status": safety_status,
            "action_required": action_required,
            "violations": violations,
            "warnings": warnings,
            "checked_boundaries": len(self.boundaries),
            "recommendations": self._generate_safety_recommendations(violations, warnings)
        }

    async def enforce_boundaries(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce ethical boundaries with correction suggestions"""
        safety_check = await self.check_content(content, context)

        if safety_check["safety_status"] == "blocked":
            # Attempt to sanitize content for critical violations
            sanitized_content = await self._sanitize_content(content, safety_check["violations"])
            safety_check["sanitized_content"] = sanitized_content
            safety_check["sanitization_attempted"] = True

        elif safety_check["safety_status"] == "warning":
            # Provide improvement suggestions
            improvement_suggestions = await self._suggest_improvements(content, safety_check["warnings"])
            safety_check["improvement_suggestions"] = improvement_suggestions

        # Learn from this enforcement
        await self._learn_from_enforcement(content, safety_check)

        return safety_check

    async def _sanitize_content(self, content: str, violations: List[Dict[str, Any]]) -> str:
        """Attempt to sanitize content by removing or modifying violating elements"""
        sanitized = content

        for violation in violations:
            boundary = self._get_boundary_by_id(violation["boundary_id"])
            if boundary:
                # Apply sanitization rules based on boundary type
                if boundary.category == "privacy":
                    sanitized = self._sanitize_privacy_data(sanitized)
                elif boundary.category == "safety":
                    sanitized = self._sanitize_harmful_content(sanitized, violation)
                # Add more sanitization rules as needed

        # Add warning header
        warning_header = f"<!-- CONTENT SANITIZED: {len(violations)} ethical violations detected and addressed -->\n"
        sanitized = warning_header + sanitized

        return sanitized

    async def _suggest_improvements(self, content: str, warnings: List[Dict[str, Any]]) -> List[str]:
        """Suggest improvements for content with warnings"""
        suggestions = []

        for warning in warnings:
            boundary = self._get_boundary_by_id(warning["boundary_id"])
            if boundary:
                suggestions.extend(self._get_boundary_suggestions(boundary.category))

        # Get LLM-based suggestions for general improvements
        if suggestions:
            llm_suggestions = await self._get_llm_improvement_suggestions(content, warnings)
            suggestions.extend(llm_suggestions)

        return list(set(suggestions))  # Remove duplicates

    async def _get_llm_improvement_suggestions(self, content: str, warnings: List[Dict[str, Any]]) -> List[str]:
        """Get improvement suggestions from LLM"""
        try:
            prompt = f"""Review this content and suggest ethical improvements:

Content: {content[:500]}...

Warnings: {json.dumps(warnings, indent=2)}

Provide 3-5 specific, actionable suggestions for improving the ethical quality of this content."""

            response = await self.ollama_service.generate_response(
                prompt=prompt,
                system_prompt="You are an ethical content reviewer. Provide constructive suggestions for improving content safety and appropriateness.",
                temperature=0.3
            )

            # Extract suggestions from response
            suggestions = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('-', '•', '*')) or (line[0].isdigit() and line[1] in '.)'):
                    suggestions.append(line.lstrip('123456789.-•* ').strip())

            return suggestions[:5]  # Limit to 5 suggestions

        except Exception as e:
            logger.warning(f"Failed to get LLM improvement suggestions: {str(e)}")
            return ["Consider reviewing content for ethical compliance"]

    def _sanitize_privacy_data(self, content: str) -> str:
        """Sanitize personal information"""
        # Remove potential SSNs
        content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', content)
        # Remove potential credit cards
        content = re.sub(r'\b\d{4} \d{4} \d{4} \d{4}\b', '[REDACTED CARD]', content)
        # Remove emails
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED EMAIL]', content)

        return content

    def _sanitize_harmful_content(self, content: str, violation: Dict[str, Any]) -> str:
        """Sanitize harmful content"""
        # Add warning and context
        warning = f"[CONTENT WARNING: {violation['boundary_name']}]"
        return f"{warning}\n\n{content}\n\n{warning}"

    def _get_boundary_by_id(self, boundary_id: str) -> Optional[EthicalBoundary]:
        """Get boundary by ID"""
        for boundary in self.boundaries:
            if boundary.boundary_id == boundary_id:
                return boundary
        return None

    def _get_boundary_suggestions(self, category: str) -> List[str]:
        """Get suggestions for boundary category"""
        suggestions_map = {
            "safety": [
                "Consider the potential impact on user safety",
                "Avoid providing instructions for harmful activities",
                "Include safety warnings where appropriate"
            ],
            "privacy": [
                "Ensure personal information is not exposed",
                "Use data anonymization techniques",
                "Follow privacy best practices"
            ],
            "fairness": [
                "Avoid biased or discriminatory content",
                "Ensure inclusive language",
                "Consider diverse perspectives"
            ],
            "accuracy": [
                "Verify information accuracy",
                "Include sources where appropriate",
                "Avoid spreading misinformation"
            ]
        }

        return suggestions_map.get(category, ["Review content for ethical compliance"])

    async def _learn_from_enforcement(self, content: str, safety_check: Dict[str, Any]):
        """Learn from ethical enforcement outcomes"""
        try:
            # Add ethical pattern to knowledge graph
            learning_content = f"Ethical Pattern: {safety_check['safety_status']} - {len(safety_check.get('violations', []))} violations"

            await self.graph_rag_service.add_knowledge(
                content=learning_content,
                metadata={
                    "type": "ethical_pattern",
                    "safety_status": safety_check["safety_status"],
                    "violations_count": len(safety_check.get("violations", [])),
                    "warnings_count": len(safety_check.get("warnings", [])),
                    "learning_opportunity": True
                },
                node_type="ethical_learning"
            )

        except Exception as e:
            logger.warning(f"Failed to learn from ethical enforcement: {str(e)}")

    def get_ethical_statistics(self) -> Dict[str, Any]:
        """Get ethical enforcement statistics"""
        if not self.violation_history:
            return {"total_checks": 0}

        total_checks = len(self.violation_history)
        total_violations = sum(check["violations_found"] for check in self.violation_history)
        total_warnings = sum(check["warnings_found"] for check in self.violation_history)

        return {
            "total_checks": total_checks,
            "total_violations": total_violations,
            "total_warnings": total_warnings,
            "average_violations_per_check": total_violations / total_checks,
            "average_warnings_per_check": total_warnings / total_checks,
            "active_boundaries": len(self.boundaries)
        }

    def add_custom_boundary(self, boundary: EthicalBoundary):
        """Add a custom ethical boundary"""
        self.boundaries.append(boundary)
        logger.info(f"Added custom ethical boundary: {boundary.name}")

    def remove_boundary(self, boundary_id: str) -> bool:
        """Remove an ethical boundary"""
        for i, boundary in enumerate(self.boundaries):
            if boundary.boundary_id == boundary_id:
                removed = self.boundaries.pop(i)
                logger.info(f"Removed ethical boundary: {removed.name}")
                return True
        return False