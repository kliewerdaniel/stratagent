"""
Generator agent - generates code based on specifications and patterns
"""
import logging
from typing import Dict, List, Any
from datetime import datetime
import re

from app.agents.base_agent import BaseAgent
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class GeneratorAgent(BaseAgent):
    """Agent responsible for generating code based on specifications and design patterns"""

    def __init__(self, agent_id: str, ollama_service: OllamaService):
        super().__init__(agent_id, "Generator", "generator", ollama_service)

        # Register additional message handlers
        self.register_handler("generate_code", self._handle_generate_code)
        self.register_handler("generate_tests", self._handle_generate_tests)
        self.register_handler("generate_documentation", self._handle_generate_documentation)
        self.register_handler("refactor_code", self._handle_refactor_code)

        # Code generation patterns and templates
        self.code_patterns = self._load_code_patterns()

    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this agent"""
        return [
            "code_generation",
            "boilerplate_code_creation",
            "api_endpoint_generation",
            "database_model_generation",
            "test_case_generation",
            "documentation_generation",
            "code_refactoring",
            "pattern_implementation",
            "template_based_development"
        ]

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a code generation task"""
        task_type = task.get("type", "generate_code")

        if task_type == "generate_code":
            return await self._generate_code(task)
        elif task_type == "generate_tests":
            return await self._generate_tests(task)
        elif task_type == "generate_documentation":
            return await self._generate_documentation(task)
        elif task_type == "refactor_code":
            return await self._refactor_code(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def _generate_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on specifications"""
        specifications = task.get("specifications", {})
        language = task.get("language", "python")
        framework = task.get("framework", "")
        component_type = task.get("component_type", "general")

        system_prompt = f"""You are an expert {language} developer specializing in clean, maintainable code generation.
You excel at:
- Writing production-ready, well-documented code
- Following language-specific best practices and conventions
- Implementing design patterns correctly
- Creating modular, testable code
- Adding appropriate error handling and logging
- Following security best practices

Generate code that is:
- Functionally correct and complete
- Well-structured and readable
- Properly documented with comments and docstrings
- Following language-specific style guides (PEP 8 for Python, etc.)
- Including error handling and edge cases
- Ready for production use"""

        prompt = f"""Generate {language} code for the following specifications:

Specifications: {specifications}
Language: {language}
Framework: {framework}
Component Type: {component_type}

Please generate complete, production-ready code that includes:

1. **Main Implementation**
   - Complete function/class implementation
   - Proper imports and dependencies
   - Error handling and validation

2. **Documentation**
   - Comprehensive docstrings/comments
   - Usage examples
   - Parameter and return value documentation

3. **Best Practices**
   - Following language conventions
   - Security considerations
   - Performance optimizations where applicable

4. **Testing Considerations**
   - Code structure that supports testing
   - Clear interfaces for mocking/stubbing

If this is an API endpoint, include:
- Request/response models
- Validation logic
- Error responses
- Authentication/authorization checks

If this is a database model, include:
- Proper field definitions
- Relationships and constraints
- Validation rules
- Indexing recommendations

Provide the complete code implementation with all necessary imports and dependencies."""

        response = await self.generate_response(prompt, system_prompt)

        # Extract code blocks from response
        generated_code = self._extract_code_blocks(response, language)

        return {
            "status": "completed",
            "generated_code": {
                "raw_response": response,
                "code_blocks": generated_code,
                "language": language,
                "framework": framework,
                "quality_score": self._assess_code_quality(generated_code),
                "testability_score": self._assess_testability(generated_code)
            },
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "specifications_hash": hash(str(specifications)),
                "generation_method": "llm_structured"
            }
        }

    async def _generate_tests(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test cases for code"""
        code_to_test = task.get("code", "")
        language = task.get("language", "python")
        test_framework = task.get("test_framework", "pytest")

        system_prompt = f"""You are a Testing Specialist expert in {language} testing with {test_framework}.
You focus on:
- Comprehensive test coverage (unit, integration, edge cases)
- Testing best practices and patterns
- Mocking and stubbing external dependencies
- Property-based and fuzz testing where applicable
- Performance and load testing considerations

Generate tests that:
- Cover happy path and error scenarios
- Include edge cases and boundary conditions
- Use appropriate fixtures and setup/teardown
- Follow testing best practices
- Are maintainable and readable"""

        prompt = f"""Generate comprehensive test cases for this {language} code:

Code to Test:
```python
{code_to_test}
```

Test Framework: {test_framework}
Language: {language}

Please generate:

1. **Unit Tests**
   - Test individual functions/methods
   - Cover all branches and edge cases
   - Mock external dependencies

2. **Integration Tests**
   - Test component interactions
   - Database integration if applicable
   - API endpoint testing if applicable

3. **Error Handling Tests**
   - Invalid inputs and edge cases
   - Exception handling
   - Error propagation

4. **Test Fixtures and Setup**
   - Test data preparation
   - Mock objects and services
   - Database test setup

5. **Test Organization**
   - Proper test file structure
   - Test naming conventions
   - Test categorization and marking

Provide complete, runnable test code with all necessary imports and dependencies."""

        response = await self.generate_response(prompt, system_prompt)

        test_code = self._extract_code_blocks(response, language)

        return {
            "status": "completed",
            "generated_tests": {
                "raw_response": response,
                "test_code": test_code,
                "test_framework": test_framework,
                "coverage_estimate": self._estimate_test_coverage(test_code),
                "test_quality_score": self._assess_test_quality(test_code)
            }
        }

    async def _generate_documentation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate documentation for code"""
        code = task.get("code", "")
        documentation_type = task.get("documentation_type", "api")

        system_prompt = """You are a Technical Documentation Specialist who creates clear, comprehensive documentation.
You excel at:
- Writing clear, concise explanations
- Creating API documentation with examples
- Writing user guides and tutorials
- Creating architectural documentation
- Writing maintenance and deployment guides

Documentation should be:
- Clear and accessible to the target audience
- Comprehensive but not overwhelming
- Well-structured with proper formatting
- Including practical examples and use cases"""

        prompt = f"""Generate comprehensive documentation for this code:

Code:
```python
{code}
```

Documentation Type: {documentation_type}

Please create:

1. **Overview**
   - Purpose and functionality
   - Key features and capabilities
   - Architecture and design decisions

2. **API Documentation** (if applicable)
   - Endpoint specifications
   - Request/response formats
   - Authentication requirements
   - Error codes and handling

3. **Usage Examples**
   - Basic usage scenarios
   - Advanced use cases
   - Integration examples

4. **Configuration**
   - Required settings and parameters
   - Environment variables
   - Configuration options

5. **Deployment & Maintenance**
   - Installation instructions
   - Configuration steps
   - Monitoring and troubleshooting
   - Backup and recovery procedures

6. **Security Considerations**
   - Security features implemented
   - Known vulnerabilities and mitigations
   - Best practices for secure usage

Format the documentation in Markdown with proper headings, code examples, and clear structure."""

        response = await self.generate_response(prompt, system_prompt)

        return {
            "status": "completed",
            "generated_documentation": {
                "raw_response": response,
                "documentation_type": documentation_type,
                "formatted_docs": self._format_documentation(response),
                "completeness_score": self._assess_documentation_completeness(response)
            }
        }

    async def _refactor_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Refactor existing code for improvements"""
        code_to_refactor = task.get("code", "")
        refactoring_goals = task.get("refactoring_goals", [])
        constraints = task.get("constraints", {})

        system_prompt = """You are a Code Refactoring Specialist who improves code quality while maintaining functionality.
You focus on:
- Improving readability and maintainability
- Applying design patterns appropriately
- Reducing complexity and technical debt
- Enhancing testability and modularity
- Optimizing performance where beneficial
- Following language-specific best practices

Refactoring should:
- Preserve existing functionality
- Improve code structure and organization
- Enhance readability and maintainability
- Follow established patterns and conventions
- Include appropriate comments and documentation"""

        prompt = f"""Refactor this code according to the specified goals:

Code to Refactor:
```python
{code_to_refactor}
```

Refactoring Goals: {refactoring_goals}

Constraints: {constraints}

Please provide:

1. **Refactored Code**
   - Improved structure and organization
   - Better naming and documentation
   - Applied design patterns where appropriate

2. **Changes Made**
   - List of specific improvements
   - Rationale for each change
   - Benefits achieved

3. **Quality Improvements**
   - Readability enhancements
   - Maintainability improvements
   - Performance optimizations (if applicable)

4. **Testing Considerations**
   - Impact on existing tests
   - New test cases needed
   - Testability improvements

Ensure the refactored code maintains the same functionality while being more maintainable and readable."""

        response = await self.generate_response(prompt, system_prompt)

        refactored_code = self._extract_code_blocks(response, "python")  # Assume Python for now

        return {
            "status": "completed",
            "refactored_code": {
                "original_code": code_to_refactor,
                "refactored_code": refactored_code,
                "changes_summary": self._extract_changes_summary(response),
                "quality_improvement": self._assess_refactoring_quality(refactored_code)
            }
        }

    async def _handle_generate_code(self, message) -> Any:
        """Handle generate code message"""
        task = message.content
        result = await self._generate_code(task)
        return self._create_response_message(message, "code_generation_complete", result)

    async def _handle_generate_tests(self, message) -> Any:
        """Handle generate tests message"""
        task = message.content
        result = await self._generate_tests(task)
        return self._create_response_message(message, "test_generation_complete", result)

    async def _handle_generate_documentation(self, message) -> Any:
        """Handle generate documentation message"""
        task = message.content
        result = await self._generate_documentation(task)
        return self._create_response_message(message, "documentation_generation_complete", result)

    async def _handle_refactor_code(self, message) -> Any:
        """Handle refactor code message"""
        task = message.content
        result = await self._refactor_code(task)
        return self._create_response_message(message, "code_refactoring_complete", result)

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

    def _load_code_patterns(self) -> Dict[str, Any]:
        """Load common code patterns and templates"""
        return {
            "python": {
                "api_endpoint": """
def {function_name}({parameters}):
    \"\"\"{docstring}\"\"\"
    try:
        # Input validation
        {validation_code}

        # Business logic
        {business_logic}

        # Response
        return {response}

    except {exception_type} as e:
        logger.error(f"Error in {function_name}: {str(e)}")
        raise HTTPException(status_code={status_code}, detail=str(e))
""",
                "database_model": """
class {ModelName}(Base):
    __tablename__ = "{table_name}"

    id = Column(Integer, primary_key=True, index=True)
    {field_definitions}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
"""
            }
        }

    def _extract_code_blocks(self, response: str, language: str) -> List[str]:
        """Extract code blocks from LLM response"""
        # Simple regex-based extraction - would be more sophisticated in practice
        code_blocks = []
        # Look for ```language or ``` blocks
        pattern = r'```(?:' + language + r')?(.*?)(?:```|$)'
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            code_blocks.append(match.strip())
        return code_blocks

    def _assess_code_quality(self, code_blocks: List[str]) -> float:
        """Assess the quality of generated code"""
        # Simple quality assessment - would use more sophisticated analysis in practice
        if not code_blocks:
            return 0.0

        total_score = 0.0
        for code in code_blocks:
            score = 0.0
            # Check for basic quality indicators
            if "try:" in code and "except" in code:
                score += 0.3  # Error handling
            if "def " in code or "class " in code:
                score += 0.2  # Proper structure
            if '"""' in code or "'''" in code:
                score += 0.2  # Documentation
            if "import " in code:
                score += 0.1  # Proper imports
            if len(code.strip()) > 50:  # Substantial code
                score += 0.2
            total_score += min(score, 1.0)  # Cap at 1.0 per block

        return total_score / len(code_blocks)

    def _assess_testability(self, code_blocks: List[str]) -> float:
        """Assess how testable the generated code is"""
        # Simple assessment
        if not code_blocks:
            return 0.0

        testable_score = 0.0
        for code in code_blocks:
            score = 0.0
            if "def " in code:  # Has functions that can be tested
                score += 0.5
            if "class " in code:  # Has classes that can be tested
                score += 0.3
            if "return " in code:  # Has return values to assert
                score += 0.2
            testable_score += score

        return min(testable_score / len(code_blocks), 1.0)

    def _estimate_test_coverage(self, test_code: List[str]) -> float:
        """Estimate test coverage from generated tests"""
        # Very simple estimation
        coverage_hints = ["test_", "assert", "pytest", "unittest"]
        coverage_score = 0.0
        total_lines = sum(len(code.split('\n')) for code in test_code)

        if total_lines > 0:
            hint_count = sum(code.count(hint) for code in test_code for hint in coverage_hints)
            coverage_score = min(hint_count / total_lines, 1.0)

        return coverage_score

    def _assess_test_quality(self, test_code: List[str]) -> float:
        """Assess the quality of generated tests"""
        if not test_code:
            return 0.0

        quality_score = 0.0
        for code in test_code:
            score = 0.0
            if "assert" in code:
                score += 0.3
            if "test_" in code:
                score += 0.2
            if "mock" in code.lower() or "patch" in code.lower():
                score += 0.2
            if "fixture" in code.lower():
                score += 0.1
            if "parametrize" in code.lower():
                score += 0.1
            if "@" in code:  # Decorators
                score += 0.1
            quality_score += score

        return min(quality_score / len(test_code), 1.0)

    def _format_documentation(self, response: str) -> str:
        """Format documentation response"""
        # Basic formatting - would be more sophisticated in practice
        return response.strip()

    def _assess_documentation_completeness(self, response: str) -> float:
        """Assess documentation completeness"""
        sections = ["overview", "usage", "api", "configuration", "deployment"]
        found_sections = sum(1 for section in sections if section.lower() in response.lower())
        return found_sections / len(sections)

    def _extract_changes_summary(self, response: str) -> List[str]:
        """Extract summary of changes from refactoring response"""
        changes = []
        lines = response.split('\n')
        in_changes_section = False

        for line in lines:
            line = line.strip()
            if "changes" in line.lower() and ("made" in line.lower() or "summary" in line.lower()):
                in_changes_section = True
                continue
            elif in_changes_section and line.startswith('#') or not line:
                continue
            elif in_changes_section and line:
                if line.startswith('-') or line[0].isdigit():
                    changes.append(line)
                elif len(changes) > 0 and not line.startswith(('1.', '2.', '3.', '-')):
                    # Continuation of previous change
                    changes[-1] += " " + line
                else:
                    changes.append(line)

        return changes

    def _assess_refactoring_quality(self, refactored_code: List[str]) -> float:
        """Assess the quality of refactored code"""
        return self._assess_code_quality(refactored_code)  # Reuse code quality assessment