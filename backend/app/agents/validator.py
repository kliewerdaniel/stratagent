"""
Validator agent - validates code quality and ensures standards compliance
"""
import logging
from typing import Dict, List, Any
from datetime import datetime
import re
import ast

from app.agents.base_agent import BaseAgent
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class ValidatorAgent(BaseAgent):
    """Agent responsible for validating code quality and ensuring standards compliance"""

    def __init__(self, agent_id: str, ollama_service: OllamaService):
        super().__init__(agent_id, "Validator", "validator", ollama_service)

        # Register additional message handlers
        self.register_handler("validate_code", self._handle_validate_code)
        self.register_handler("security_audit", self._handle_security_audit)
        self.register_handler("performance_analysis", self._handle_performance_analysis)
        self.register_handler("compliance_check", self._handle_compliance_check)

        # Validation rules and patterns
        self.validation_rules = self._load_validation_rules()

    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this agent"""
        return [
            "code_quality_validation",
            "security_vulnerability_scanning",
            "performance_analysis",
            "standards_compliance_checking",
            "static_code_analysis",
            "code_smell_detection",
            "best_practices_enforcement",
            "maintainability_assessment",
            "technical_debt_identification"
        ]

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a validation task"""
        task_type = task.get("type", "validate_code")

        if task_type == "validate_code":
            return await self._validate_code(task)
        elif task_type == "security_audit":
            return await self._security_audit(task)
        elif task_type == "performance_analysis":
            return await self._performance_analysis(task)
        elif task_type == "compliance_check":
            return await self._compliance_check(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _validate_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code quality and standards compliance"""
        code_to_validate = task.get("code", "")
        language = task.get("language", "python")
        validation_criteria = task.get("criteria", ["quality", "security", "performance"])

        system_prompt = f"""You are a Senior Code Validator specializing in {language} code quality assessment.
You excel at identifying:
- Code quality issues and improvements
- Security vulnerabilities and best practices violations
- Performance bottlenecks and optimization opportunities
- Maintainability concerns and technical debt
- Standards compliance and convention adherence

Your validation should be:
- Thorough and systematic
- Specific with actionable recommendations
- Prioritized by severity and impact
- Supported by reasoning and examples
- Focused on practical improvements"""

        prompt = f"""Perform a comprehensive code validation on this {language} code:

Code to Validate:
```python
{code_to_validate}
```

Validation Criteria: {validation_criteria}
Language: {language}

Please provide a detailed validation report covering:

1. **Code Quality Assessment**
   - Readability and structure evaluation
   - Naming convention compliance
   - Documentation adequacy
   - Code organization and modularity

2. **Security Analysis**
   - Potential security vulnerabilities
   - Input validation and sanitization
   - Authentication and authorization issues
   - Data protection concerns

3. **Performance Evaluation**
   - Potential performance bottlenecks
   - Resource usage optimization opportunities
   - Scalability considerations
   - Memory and CPU efficiency

4. **Maintainability Review**
   - Code complexity assessment
   - Technical debt identification
   - Testability evaluation
   - Future extensibility considerations

5. **Standards Compliance**
   - Language-specific best practices
   - Framework conventions (if applicable)
   - Industry standards adherence
   - Code style guide compliance

6. **Recommendations**
   - Prioritized improvement suggestions
   - Specific code changes with examples
   - Effort estimates for fixes
   - Risk-benefit analysis

For each issue identified, include:
- Severity level (Critical/High/Medium/Low)
- Location in code (line numbers, functions)
- Description of the problem
- Suggested fix with code example
- Rationale for the recommendation"""

        response = await self.generate_response(prompt, system_prompt)

        # Perform automated validation as well
        automated_issues = self._automated_validation(code_to_validate, language)

        validation_report = self._parse_validation_report(response)
        validation_report["automated_issues"] = automated_issues

        return {
            "status": "completed",
            "validation_report": validation_report,
            "metadata": {
                "validated_at": datetime.utcnow().isoformat(),
                "language": language,
                "criteria": validation_criteria,
                "validation_method": "hybrid_llm_automated"
            }
        }

    async def _security_audit(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a security audit on code"""
        code_to_audit = task.get("code", "")
        audit_scope = task.get("scope", ["input_validation", "authentication", "data_protection"])

        system_prompt = """You are a Security Auditor specializing in identifying security vulnerabilities and recommending mitigations.
You focus on:
- Common security vulnerabilities (OWASP Top 10, etc.)
- Secure coding practices and patterns
- Data protection and privacy concerns
- Authentication and authorization weaknesses
- Input validation and sanitization issues

Security assessments should:
- Identify specific vulnerabilities with CVEs where applicable
- Provide severity ratings and exploitability analysis
- Include remediation steps with code examples
- Consider compliance requirements (GDPR, HIPAA, etc.)"""

        prompt = f"""Perform a comprehensive security audit on this code:

Code to Audit:
```python
{code_to_audit}
```

Audit Scope: {audit_scope}

Please provide a detailed security assessment covering:

1. **Vulnerability Assessment**
   - Injection vulnerabilities (SQL, NoSQL, Command, etc.)
   - Broken authentication and session management
   - Cross-site scripting (XSS) vulnerabilities
   - Insecure direct object references
   - Security misconfigurations

2. **Data Protection Analysis**
   - Sensitive data handling
   - Encryption implementation
   - Data leakage prevention
   - Privacy compliance

3. **Access Control Review**
   - Authentication mechanisms
   - Authorization checks
   - Session management
   - Privilege escalation risks

4. **Input Validation**
   - Input sanitization adequacy
   - Boundary checking
   - Type safety validation

5. **Security Recommendations**
   - Prioritized remediation steps
   - Secure coding best practices
   - Security testing suggestions
   - Compliance considerations

For each vulnerability found, include:
- Severity (Critical/High/Medium/Low)
- CVSS score estimate
- Affected code locations
- Attack vectors and impact
- Remediation steps with code examples"""

        response = await self.generate_response(prompt, system_prompt)

        security_findings = self._parse_security_findings(response)

        return {
            "status": "completed",
            "security_audit": {
                "raw_report": response,
                "findings": security_findings,
                "overall_risk_level": self._calculate_risk_level(security_findings),
                "compliance_status": self._assess_compliance(security_findings)
            }
        }

    async def _performance_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code performance"""
        code_to_analyze = task.get("code", "")
        performance_metrics = task.get("metrics", ["time_complexity", "space_complexity", "resource_usage"])

        system_prompt = """You are a Performance Analyst specializing in code optimization and efficiency analysis.
You focus on:
- Algorithm complexity analysis (Big O notation)
- Memory usage optimization
- CPU efficiency improvements
- Database query optimization
- Caching strategy evaluation
- Concurrency and parallelism opportunities

Performance analysis should:
- Identify computational bottlenecks
- Suggest algorithmic improvements
- Recommend optimization techniques
- Consider scalability implications
- Balance performance with maintainability"""

        prompt = f"""Perform a comprehensive performance analysis on this code:

Code to Analyze:
```python
{code_to_analyze}
```

Performance Metrics: {performance_metrics}

Please provide a detailed performance assessment covering:

1. **Algorithm Complexity**
   - Time complexity analysis (Big O)
   - Space complexity evaluation
   - Best/worst/average case scenarios

2. **Resource Usage**
   - Memory allocation patterns
   - CPU utilization assessment
   - I/O operation efficiency
   - Network resource usage (if applicable)

3. **Performance Bottlenecks**
   - Identified slow operations
   - Inefficient algorithms or data structures
   - Resource contention issues
   - Scalability limitations

4. **Optimization Opportunities**
   - Algorithm improvements
   - Data structure optimizations
   - Caching strategy recommendations
   - Parallelization opportunities

5. **Scalability Assessment**
   - Performance under load
   - Horizontal/vertical scaling considerations
   - Database scaling requirements
   - Caching and CDN recommendations

6. **Recommendations**
   - Prioritized optimization suggestions
   - Expected performance improvements
   - Implementation effort estimates
   - Trade-off considerations (performance vs. complexity)

Provide specific recommendations with code examples where applicable."""

        response = await self.generate_response(prompt, system_prompt)

        performance_analysis = self._parse_performance_analysis(response)

        return {
            "status": "completed",
            "performance_analysis": {
                "raw_report": response,
                "analysis": performance_analysis,
                "bottlenecks_identified": len(performance_analysis.get("bottlenecks", [])),
                "optimization_potential": self._assess_optimization_potential(performance_analysis)
            }
        }

    async def _compliance_check(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check code compliance with standards and regulations"""
        code_to_check = task.get("code", "")
        compliance_standards = task.get("standards", ["pep8", "security_best_practices"])

        system_prompt = """You are a Compliance Auditor specializing in code standards and regulatory requirements.
You ensure adherence to:
- Language-specific style guides and conventions
- Industry security standards
- Regulatory compliance requirements
- Organizational coding standards
- Accessibility guidelines

Compliance checks should:
- Identify specific violations
- Provide remediation guidance
- Assess compliance levels
- Recommend enforcement mechanisms"""

        prompt = f"""Perform a compliance check on this code:

Code to Check:
```python
{code_to_check}
```

Compliance Standards: {compliance_standards}

Please evaluate compliance with:

1. **Coding Standards**
   - Naming conventions
   - Code formatting and style
   - Documentation requirements
   - Structure and organization

2. **Security Standards**
   - Secure coding practices
   - Data protection requirements
   - Access control standards
   - Encryption requirements

3. **Regulatory Compliance**
   - Data privacy regulations (GDPR, CCPA)
   - Industry-specific requirements
   - Accessibility standards (WCAG)
   - Audit and logging requirements

4. **Quality Standards**
   - Code coverage requirements
   - Testing standards
   - Code review requirements
   - Maintainability standards

5. **Compliance Report**
   - Overall compliance score
   - Critical violations
   - Recommended corrective actions
   - Compliance improvement plan

For each violation, include:
- Standard violated
- Severity and impact
- Specific code location
- Remediation steps"""

        response = await self.generate_response(prompt, system_prompt)

        compliance_report = self._parse_compliance_report(response)

        return {
            "status": "completed",
            "compliance_check": {
                "raw_report": response,
                "report": compliance_report,
                "overall_compliance_score": self._calculate_compliance_score(compliance_report),
                "critical_violations": self._identify_critical_violations(compliance_report)
            }
        }

    async def _handle_validate_code(self, message) -> Any:
        """Handle validate code message"""
        task = message.content
        result = await self._validate_code(task)
        return self._create_response_message(message, "code_validation_complete", result)

    async def _handle_security_audit(self, message) -> Any:
        """Handle security audit message"""
        task = message.content
        result = await self._security_audit(task)
        return self._create_response_message(message, "security_audit_complete", result)

    async def _handle_performance_analysis(self, message) -> Any:
        """Handle performance analysis message"""
        task = message.content
        result = await self._performance_analysis(task)
        return self._create_response_message(message, "performance_analysis_complete", result)

    async def _handle_compliance_check(self, message) -> Any:
        """Handle compliance check message"""
        task = message.content
        result = await self._compliance_check(task)
        return self._create_response_message(message, "compliance_check_complete", result)

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

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules and patterns"""
        return {
            "python": {
                "security_patterns": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"pickle\.loads?\s*\(",
                    r"subprocess\.(call|Popen|run)\s*\(",
                    r"os\.system\s*\(",
                    r"sqlalchemy\.text\s*\("
                ],
                "code_quality_checks": [
                    "line_length",
                    "function_length",
                    "complexity",
                    "documentation"
                ]
            }
        }

    def _automated_validation(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Perform automated validation checks"""
        issues = []

        if language == "python":
            issues.extend(self._python_syntax_check(code))
            issues.extend(self._python_security_check(code))
            issues.extend(self._python_quality_check(code))

        return issues

    def _python_syntax_check(self, code: str) -> List[Dict[str, Any]]:
        """Check Python syntax"""
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "severity": "critical",
                "message": f"Syntax error: {e.msg}",
                "line": e.lineno,
                "column": e.offset
            })
        return issues

    def _python_security_check(self, code: str) -> List[Dict[str, Any]]:
        """Check for security issues in Python code"""
        issues = []
        security_patterns = self.validation_rules["python"]["security_patterns"]

        for pattern in security_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                issues.append({
                    "type": "security_vulnerability",
                    "severity": "high",
                    "message": f"Potential security vulnerability: {pattern}",
                    "line": code[:match.start()].count('\n') + 1,
                    "code_snippet": match.group()
                })

        return issues

    def _python_quality_check(self, code: str) -> List[Dict[str, Any]]:
        """Check Python code quality"""
        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:  # PEP 8 recommendation
                issues.append({
                    "type": "style_violation",
                    "severity": "low",
                    "message": f"Line too long ({len(line)} characters)",
                    "line": i
                })

            # Check for TODO comments
            if "TODO" in line.upper():
                issues.append({
                    "type": "maintainability_issue",
                    "severity": "medium",
                    "message": "TODO comment found",
                    "line": i
                })

        return issues

    def _parse_validation_report(self, response: str) -> Dict[str, Any]:
        """Parse validation report from LLM response"""
        # Would implement more sophisticated parsing in practice
        return {
            "code_quality": {},
            "security_analysis": {},
            "performance_evaluation": {},
            "maintainability_review": {},
            "standards_compliance": {},
            "recommendations": []
        }

    def _parse_security_findings(self, response: str) -> List[Dict[str, Any]]:
        """Parse security findings"""
        return []

    def _calculate_risk_level(self, findings: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level"""
        critical_count = sum(1 for finding in findings if finding.get("severity") == "critical")
        high_count = sum(1 for finding in findings if finding.get("severity") == "high")

        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "high"
        elif high_count > 0:
            return "medium"
        else:
            return "low"

    def _assess_compliance(self, findings: List[Dict[str, Any]]) -> str:
        """Assess compliance status"""
        # Simple assessment
        if any(finding.get("severity") == "critical" for finding in findings):
            return "non_compliant"
        else:
            return "compliant"

    def _parse_performance_analysis(self, response: str) -> Dict[str, Any]:
        """Parse performance analysis"""
        return {
            "algorithm_complexity": {},
            "resource_usage": {},
            "bottlenecks": [],
            "optimization_opportunities": [],
            "scalability_assessment": {}
        }

    def _assess_optimization_potential(self, analysis: Dict[str, Any]) -> str:
        """Assess optimization potential"""
        bottlenecks = analysis.get("bottlenecks", [])
        opportunities = analysis.get("optimization_opportunities", [])

        if len(bottlenecks) > 5 or len(opportunities) > 5:
            return "high"
        elif len(bottlenecks) > 2 or len(opportunities) > 2:
            return "medium"
        else:
            return "low"

    def _parse_compliance_report(self, response: str) -> Dict[str, Any]:
        """Parse compliance report"""
        return {
            "coding_standards": {},
            "security_standards": {},
            "regulatory_compliance": {},
            "quality_standards": {},
            "violations": []
        }

    def _calculate_compliance_score(self, report: Dict[str, Any]) -> float:
        """Calculate compliance score"""
        violations = report.get("violations", [])
        critical_violations = sum(1 for v in violations if v.get("severity") == "critical")
        high_violations = sum(1 for v in violations if v.get("severity") == "high")

        # Simple scoring algorithm
        score = 100 - (critical_violations * 20) - (high_violations * 10)
        return max(0, score)

    def _identify_critical_violations(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical violations"""
        violations = report.get("violations", [])
        return [v for v in violations if v.get("severity") == "critical"]