"""
Correction Engine - Content correction with multiple strategies
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import re
import ast
import json

from app.services.ollama_service import OllamaService
from app.services.graph_rag_service import GraphRAGService

logger = logging.getLogger(__name__)

class CorrectionStrategy(Enum):
    """Correction strategy types"""
    LLM_REWRITE = "llm_rewrite"
    RULE_BASED = "rule_based"
    TEMPLATE_BASED = "template_based"
    HYBRID_CORRECTION = "hybrid_correction"
    ITERATIVE_IMPROVEMENT = "iterative_improvement"

class CorrectionEngine:
    """Engine for correcting and improving content using multiple strategies"""

    def __init__(self, ollama_service: OllamaService, graph_rag_service: GraphRAGService):
        self.ollama_service = ollama_service
        self.graph_rag_service = graph_rag_service

        # Correction configuration
        self.max_correction_attempts = 3
        self.improvement_threshold = 0.1  # Minimum improvement required
        self.confidence_threshold = 0.7

        # Correction rules and patterns
        self.correction_rules = self._load_correction_rules()

        # Correction history
        self.correction_history: List[Dict[str, Any]] = []

    async def correct_content(
        self,
        content: str,
        issues: List[Dict[str, Any]],
        context: Dict[str, Any],
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Correct content based on identified issues"""
        correction_id = f"corr_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(content) % 10000}"

        # Determine best correction strategy
        if strategy is None:
            strategy = self._select_correction_strategy(issues, context)

        # Apply correction strategy
        correction_result = await self._apply_correction_strategy(
            content, issues, context, strategy
        )

        # Validate correction effectiveness
        validation_result = await self._validate_correction(
            content, correction_result["corrected_content"], issues, context
        )

        # Record correction outcome
        correction_record = {
            "correction_id": correction_id,
            "original_content_hash": hash(content),
            "strategy": strategy,
            "issues_addressed": len(issues),
            "success": validation_result["effectiveness"] > 0.5,
            "improvement_score": validation_result["effectiveness"],
            "timestamp": datetime.utcnow().isoformat()
        }

        self.correction_history.append(correction_record)

        # Learn from correction
        await self._learn_from_correction(content, correction_result, validation_result)

        return {
            "correction_id": correction_id,
            "original_content": content,
            "corrected_content": correction_result["corrected_content"],
            "strategy_used": strategy,
            "issues_resolved": correction_result["issues_resolved"],
            "remaining_issues": correction_result["remaining_issues"],
            "confidence": correction_result["confidence"],
            "validation_score": validation_result["effectiveness"],
            "processing_time": correction_result["processing_time"],
            "recommendations": correction_result["recommendations"]
        }

    def _select_correction_strategy(self, issues: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Select the most appropriate correction strategy based on issues and context"""
        content_type = context.get("content_type", "general")
        severity_levels = [issue.get("severity", "medium") for issue in issues]

        # Rule-based corrections for simple, well-defined issues
        if all(severity in ["low", "medium"] for severity in severity_levels) and \
           all(issue.get("type") in ["style_violation", "formatting_issue", "simple_error"] for issue in issues):
            return CorrectionStrategy.RULE_BASED.value

        # LLM rewrite for complex issues requiring understanding and creativity
        if any(severity == "high" for severity in severity_levels) or \
           any(issue.get("type") in ["logic_error", "complex_issue", "architectural_problem"] for issue in issues):
            return CorrectionStrategy.LLM_REWRITE.value

        # Template-based for standardized corrections
        if content_type in ["api_response", "error_message", "log_entry"] and \
           len(issues) <= 3:
            return CorrectionStrategy.TEMPLATE_BASED.value

        # Hybrid approach for mixed complexity
        return CorrectionStrategy.HYBRID_CORRECTION.value

    async def _apply_correction_strategy(
        self,
        content: str,
        issues: List[Dict[str, Any]],
        context: Dict[str, Any],
        strategy: str
    ) -> Dict[str, Any]:
        """Apply the selected correction strategy"""

        if strategy == CorrectionStrategy.LLM_REWRITE.value:
            return await self._llm_rewrite_correction(content, issues, context)
        elif strategy == CorrectionStrategy.RULE_BASED.value:
            return await self._rule_based_correction(content, issues, context)
        elif strategy == CorrectionStrategy.TEMPLATE_BASED.value:
            return await self._template_based_correction(content, issues, context)
        elif strategy == CorrectionStrategy.HYBRID_CORRECTION.value:
            return await self._hybrid_correction(content, issues, context)
        elif strategy == CorrectionStrategy.ITERATIVE_IMPROVEMENT.value:
            return await self._iterative_improvement_correction(content, issues, context)
        else:
            return await self._llm_rewrite_correction(content, issues, context)

    async def _llm_rewrite_correction(
        self,
        content: str,
        issues: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to rewrite and correct content"""
        start_time = asyncio.get_event_loop().time()

        # Create correction prompt
        correction_prompt = self._create_correction_prompt(content, issues, context)

        # Get corrected content from LLM
        corrected_content = await self.ollama_service.generate_response(
            prompt=correction_prompt,
            system_prompt=self._get_correction_system_prompt(context),
            temperature=0.3  # Lower temperature for more consistent corrections
        )

        processing_time = asyncio.get_event_loop().time() - start_time

        # Extract corrected content (remove any extra formatting)
        corrected_content = self._extract_corrected_content(corrected_content)

        return {
            "corrected_content": corrected_content,
            "issues_resolved": len(issues),  # Assume all addressed
            "remaining_issues": [],
            "confidence": 0.8,
            "processing_time": processing_time,
            "method": "llm_rewrite",
            "recommendations": ["Review the rewritten content for accuracy"]
        }

    async def _rule_based_correction(
        self,
        content: str,
        issues: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply rule-based corrections"""
        start_time = asyncio.get_event_loop().time()

        corrected_content = content
        resolved_issues = []
        remaining_issues = []

        for issue in issues:
            rule_applied = False

            # Apply relevant correction rules
            if issue.get("type") == "style_violation":
                corrected_content, rule_applied = self._apply_style_rules(corrected_content, issue)
            elif issue.get("type") == "security_vulnerability":
                corrected_content, rule_applied = self._apply_security_rules(corrected_content, issue)
            elif issue.get("type") == "syntax_error":
                corrected_content, rule_applied = self._apply_syntax_rules(corrected_content, issue)
            elif issue.get("type") == "formatting_issue":
                corrected_content, rule_applied = self._apply_formatting_rules(corrected_content, issue)

            if rule_applied:
                resolved_issues.append(issue)
            else:
                remaining_issues.append(issue)

        processing_time = asyncio.get_event_loop().time() - start_time

        return {
            "corrected_content": corrected_content,
            "issues_resolved": len(resolved_issues),
            "remaining_issues": remaining_issues,
            "confidence": len(resolved_issues) / len(issues) if issues else 1.0,
            "processing_time": processing_time,
            "method": "rule_based",
            "recommendations": [f"Could not auto-correct {len(remaining_issues)} issues"] if remaining_issues else []
        }

    async def _template_based_correction(
        self,
        content: str,
        issues: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply template-based corrections for standardized content"""
        start_time = asyncio.get_event_loop().time()

        content_type = context.get("content_type", "general")

        # Get appropriate template
        template = self._get_correction_template(content_type)
        if template:
            corrected_content = template.format(content=content, **context)
        else:
            corrected_content = content

        processing_time = asyncio.get_event_loop().time() - start_time

        return {
            "corrected_content": corrected_content,
            "issues_resolved": len(issues),
            "remaining_issues": [],
            "confidence": 0.9,
            "processing_time": processing_time,
            "method": "template_based",
            "recommendations": ["Template applied - verify content fits requirements"]
        }

    async def _hybrid_correction(
        self,
        content: str,
        issues: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply hybrid correction combining multiple strategies"""
        start_time = asyncio.get_event_loop().time()

        # Step 1: Apply rule-based corrections first
        rule_result = await self._rule_based_correction(content, issues, context)

        # Step 2: Use LLM to handle remaining complex issues
        if rule_result["remaining_issues"]:
            llm_result = await self._llm_rewrite_correction(
                rule_result["corrected_content"],
                rule_result["remaining_issues"],
                context
            )

            final_content = llm_result["corrected_content"]
            total_resolved = rule_result["issues_resolved"] + llm_result["issues_resolved"]
            remaining = llm_result["remaining_issues"]
            confidence = (rule_result["confidence"] + llm_result["confidence"]) / 2
        else:
            final_content = rule_result["corrected_content"]
            total_resolved = rule_result["issues_resolved"]
            remaining = rule_result["remaining_issues"]
            confidence = rule_result["confidence"]

        processing_time = asyncio.get_event_loop().time() - start_time

        return {
            "corrected_content": final_content,
            "issues_resolved": total_resolved,
            "remaining_issues": remaining,
            "confidence": confidence,
            "processing_time": processing_time,
            "method": "hybrid_correction",
            "recommendations": ["Hybrid approach applied - review for consistency"]
        }

    async def _iterative_improvement_correction(
        self,
        content: str,
        issues: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply iterative improvement through multiple correction cycles"""
        current_content = content
        total_resolved = 0
        all_recommendations = []
        iteration = 0

        while iteration < self.max_correction_attempts and issues:
            iteration += 1

            # Apply LLM-based correction for current issues
            correction_result = await self._llm_rewrite_correction(current_content, issues, context)

            # Check if improvement occurred
            if correction_result["confidence"] > self.confidence_threshold:
                current_content = correction_result["corrected_content"]
                total_resolved += correction_result["issues_resolved"]
                issues = correction_result["remaining_issues"]
                all_recommendations.extend(correction_result["recommendations"])
            else:
                # No significant improvement, stop iterating
                break

        return {
            "corrected_content": current_content,
            "issues_resolved": total_resolved,
            "remaining_issues": issues,
            "confidence": 0.7,  # Conservative estimate
            "processing_time": 0.0,  # Would need to track across iterations
            "method": "iterative_improvement",
            "recommendations": all_recommendations + [f"Completed after {iteration} iterations"]
        }

    def _create_correction_prompt(self, content: str, issues: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Create a correction prompt for LLM"""
        issues_text = "\n".join([f"- {issue.get('message', 'Issue')} (Severity: {issue.get('severity', 'medium')})"
                                for issue in issues])

        return f"""Please correct the following content to address the identified issues:

CONTENT TO CORRECT:
{content}

ISSUES TO ADDRESS:
{issues_text}

CONTENT TYPE: {context.get('content_type', 'general')}
ADDITIONAL CONTEXT: {json.dumps(context, indent=2)}

Please provide the corrected content that:
1. Addresses all identified issues
2. Maintains the original intent and meaning
3. Improves clarity and correctness
4. Follows best practices for the content type

CORRECTED CONTENT:"""

    def _get_correction_system_prompt(self, context: Dict[str, Any]) -> str:
        """Get system prompt for correction"""
        content_type = context.get("content_type", "general")

        prompts = {
            "code": """You are an expert code reviewer and corrector. Focus on fixing syntax errors, logic issues, security vulnerabilities, and improving code quality while maintaining functionality.""",
            "text_response": """You are a professional editor. Focus on improving clarity, grammar, tone, and effectiveness of written communication.""",
            "design_specification": """You are a technical specification reviewer. Focus on improving completeness, accuracy, clarity, and technical correctness of specifications.""",
            "general": """You are a content correction specialist. Focus on improving quality, accuracy, and effectiveness while preserving the original intent."""
        }

        return prompts.get(content_type, prompts["general"])

    def _extract_corrected_content(self, llm_response: str) -> str:
        """Extract the corrected content from LLM response"""
        # Look for content between markers or after specific phrases
        if "CORRECTED CONTENT:" in llm_response:
            content = llm_response.split("CORRECTED CONTENT:", 1)[1].strip()
        elif "```" in llm_response:
            # Extract from code blocks
            parts = llm_response.split("```")
            if len(parts) >= 3:
                content = parts[1].split("\n", 1)[1] if "\n" in parts[1] else parts[1]
            else:
                content = llm_response
        else:
            content = llm_response

        return content.strip()

    def _load_correction_rules(self) -> Dict[str, Any]:
        """Load correction rules and patterns"""
        return {
            "style_rules": {
                "max_line_length": 88,
                "indentation": "spaces",
                "quote_style": "double"
            },
            "security_patterns": [
                (r"eval\s*\(", "eval() usage detected - high security risk"),
                (r"exec\s*\(", "exec() usage detected - high security risk"),
                (r"os\.system\s*\(", "os.system() usage detected - potential security risk")
            ],
            "formatting_rules": {
                "trailing_whitespace": True,
                "empty_lines": "single",
                "imports_order": True
            }
        }

    def _apply_style_rules(self, content: str, issue: Dict[str, Any]) -> Tuple[str, bool]:
        """Apply style correction rules"""
        corrected = content
        applied = False

        # Fix line length issues
        if "line too long" in issue.get("message", "").lower():
            lines = content.split('\n')
            corrected_lines = []
            for line in lines:
                if len(line) > self.correction_rules["style_rules"]["max_line_length"]:
                    # Simple line breaking (could be more sophisticated)
                    corrected_lines.append(line[:88] + " \\")
                    corrected_lines.append(line[88:])
                    applied = True
                else:
                    corrected_lines.append(line)
            corrected = '\n'.join(corrected_lines)

        return corrected, applied

    def _apply_security_rules(self, content: str, issue: Dict[str, Any]) -> Tuple[str, bool]:
        """Apply security correction rules"""
        # For security issues, we typically can't auto-fix them safely
        # Instead, we add comments or warnings
        if "security" in issue.get("type", "").lower():
            corrected = f"# SECURITY WARNING: {issue.get('message', '')}\n{content}"
            return corrected, True

        return content, False

    def _apply_syntax_rules(self, content: str, issue: Dict[str, Any]) -> Tuple[str, bool]:
        """Apply syntax correction rules"""
        # Attempt simple syntax fixes
        corrected = content
        applied = False

        # Try to parse and fix basic Python syntax
        if "python" in issue.get("language", "").lower():
            try:
                ast.parse(content)
            except SyntaxError as e:
                # Attempt simple fixes
                if "EOF" in str(e):
                    corrected = content.rstrip() + "\n"  # Add missing newline
                    applied = True
                elif "indentation" in str(e):
                    # Attempt to fix indentation (very basic)
                    lines = content.split('\n')
                    corrected_lines = []
                    for i, line in enumerate(lines):
                        if line.startswith(' ') and not line.strip():
                            corrected_lines.append(line[4:])  # Remove 4 spaces
                            applied = True
                        else:
                            corrected_lines.append(line)
                    corrected = '\n'.join(corrected_lines)

        return corrected, applied

    def _apply_formatting_rules(self, content: str, issue: Dict[str, Any]) -> Tuple[str, bool]:
        """Apply formatting correction rules"""
        corrected = content
        applied = False

        # Remove trailing whitespace
        if self.correction_rules["formatting_rules"]["trailing_whitespace"]:
            lines = [line.rstrip() for line in content.split('\n')]
            corrected = '\n'.join(lines)
            if corrected != content:
                applied = True

        return corrected, applied

    def _get_correction_template(self, content_type: str) -> Optional[str]:
        """Get correction template for content type"""
        templates = {
            "error_message": """# Error Response
## Issue
{content}

## Recommended Actions
1. Review the error details
2. Check system logs
3. Verify configuration
4. Contact support if needed

## Error Code: {error_code}
## Timestamp: {timestamp}""",
            "api_response": """{
  "status": "success",
  "data": {content},
  "metadata": {
    "timestamp": "{timestamp}",
    "version": "1.0"
  }
}""",
            "log_entry": """[{timestamp}] {level}: {content}
Context: {context}
User: {user_id}"""
        }

        return templates.get(content_type)

    async def _validate_correction(
        self,
        original_content: str,
        corrected_content: str,
        original_issues: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate the effectiveness of the correction"""
        # Simple validation - check if content changed and issues might be resolved
        content_changed = original_content != corrected_content

        # Estimate effectiveness based on content change and issue types
        effectiveness = 0.5  # Base effectiveness

        if content_changed:
            effectiveness += 0.3

        # Check if correction addressed known issue types
        for issue in original_issues:
            if issue.get("type") == "style_violation" and len(corrected_content.split('\n')) != len(original_content.split('\n')):
                effectiveness += 0.1
            elif issue.get("type") == "security_vulnerability" and "WARNING" in corrected_content:
                effectiveness += 0.1

        effectiveness = min(1.0, effectiveness)

        return {
            "effectiveness": effectiveness,
            "content_changed": content_changed,
            "estimated_issues_resolved": int(effectiveness * len(original_issues)),
            "validation_method": "heuristic"
        }

    async def _learn_from_correction(
        self,
        original_content: str,
        correction_result: Dict[str, Any],
        validation_result: Dict[str, Any]
    ):
        """Learn from correction outcomes to improve future corrections"""
        try:
            # Add correction pattern to knowledge graph
            correction_content = f"Correction Pattern: {correction_result.get('method', 'unknown')} strategy"

            await self.graph_rag_service.add_knowledge(
                content=correction_content,
                metadata={
                    "type": "correction_pattern",
                    "strategy": correction_result.get("method"),
                    "effectiveness": validation_result.get("effectiveness", 0),
                    "issues_resolved": correction_result.get("issues_resolved", 0),
                    "learning_opportunity": True
                },
                node_type="correction_learning"
            )

        except Exception as e:
            logger.warning(f"Failed to learn from correction: {str(e)}")

    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get correction statistics"""
        if not self.correction_history:
            return {"total_corrections": 0}

        total_corrections = len(self.correction_history)
        successful_corrections = sum(1 for c in self.correction_history if c["success"])
        avg_improvement = sum(c["improvement_score"] for c in self.correction_history) / total_corrections

        strategy_counts = {}
        for correction in self.correction_history:
            strategy = correction["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "total_corrections": total_corrections,
            "successful_corrections": successful_corrections,
            "success_rate": successful_corrections / total_corrections,
            "average_improvement_score": avg_improvement,
            "strategy_distribution": strategy_counts
        }