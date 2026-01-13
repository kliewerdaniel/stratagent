"""
Code Generation Pipeline - Advanced code generation with RAG and pattern-based development
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import ast
import json

from app.services.ollama_service import OllamaService
from app.services.graph_rag_service import GraphRAGService
from app.services.orthos_service import OrthosService
from app.services.correction_engine import CorrectionEngine

logger = logging.getLogger(__name__)

class CodePattern:
    """Represents a reusable code pattern"""

    def __init__(self, pattern_id: str, name: str, category: str, language: str, complexity: str, code: str, metadata: Dict[str, Any]):
        self.pattern_id = pattern_id
        self.name = name
        self.category = category
        self.language = language
        self.complexity = complexity
        self.code = code
        self.metadata = metadata
        self.usage_count = 0
        self.success_rate = 0.0
        self.created_at = datetime.utcnow()
        self.last_used = None

class CodeGenerationPipeline:
    """Advanced code generation pipeline with RAG and pattern-based development"""

    def __init__(
        self,
        ollama_service: OllamaService,
        graph_rag_service: GraphRAGService,
        orthos_service: OrthosService,
        correction_engine: CorrectionEngine
    ):
        self.ollama_service = ollama_service
        self.graph_rag_service = graph_rag_service
        self.orthos_service = orthos_service
        self.correction_engine = correction_engine

        # Code patterns and templates
        self.code_patterns: Dict[str, CodePattern] = {}
        self.pattern_embeddings = {}

        # Generation configuration
        self.max_generation_attempts = 3
        self.quality_threshold = 0.8
        self.enable_rag = True
        self.enable_patterns = True

        # Generation history
        self.generation_history: List[Dict[str, Any]] = []

        # Initialize with common patterns
        self._initialize_code_patterns()

    def _initialize_code_patterns(self):
        """Initialize the pipeline with common code patterns"""
        # Python patterns
        python_patterns = [
            CodePattern(
                "api_endpoint_fastapi",
                "FastAPI REST Endpoint",
                "api",
                "python",
                "intermediate",
                """
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.schemas.{resource} import {Resource}Create, {Resource}Update, {Resource}
from app.crud.{resource} import {resource}_crud

router = APIRouter()

@router.get("/{resource_id}", response_model={Resource})
def read_{resource}({resource}_id: int, db: Session = Depends(get_db)):
    \"\"\"Get {resource} by ID\"\"\"
    {resource} = {resource}_crud.get(db, id={resource}_id)
    if {resource} is None:
        raise HTTPException(status_code=404, detail="{Resource} not found")
    return {resource}

@router.get("/", response_model=List[{Resource}])
def read_{resource}s(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    \"\"\"Get list of {resource}s\"\"\"
    {resource}s = {resource}_crud.get_multi(db, skip=skip, limit=limit)
    return {resource}s

@router.post("/", response_model={Resource})
def create_{resource}({resource}: {Resource}Create, db: Session = Depends(get_db)):
    \"\"\"Create new {resource}\"\"\"
    return {resource}_crud.create(db, obj_in={resource})

@router.put("/{resource_id}", response_model={Resource})
def update_{resource}({resource}_id: int, {resource}_in: {Resource}Update, db: Session = Depends(get_db)):
    \"\"\"Update {resource}\"\"\"
    {resource} = {resource}_crud.get(db, id={resource}_id)
    if {resource} is None:
        raise HTTPException(status_code=404, detail="{Resource} not found")
    return {resource}_crud.update(db, db_obj={resource}, obj_in={resource}_in)

@router.delete("/{resource_id}")
def delete_{resource}({resource}_id: int, db: Session = Depends(get_db)):
    \"\"\"Delete {resource}\"\"\"
    {resource} = {resource}_crud.get(db, id={resource}_id)
    if {resource} is None:
        raise HTTPException(status_code=404, detail="{Resource} not found")
    {resource}_crud.remove(db, id={resource}_id)
    return {{"ok": True}}
""",
                {"framework": "fastapi", "includes_crud": True, "includes_validation": True}
            ),

            CodePattern(
                "database_model_sqlalchemy",
                "SQLAlchemy Database Model",
                "database",
                "python",
                "beginner",
                """
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base

class {ModelName}(Base):
    \"\"\"{ModelName} database model\"\"\"

    __tablename__ = "{table_name}"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    # user_id = Column(Integer, ForeignKey("users.id"))
    # user = relationship("User", back_populates="{model_names}")

    def __repr__(self):
        return f"<{ModelName}(id={{self.id}}, name='{{self.name}}')>"
""",
                {"framework": "sqlalchemy", "includes_relationships": False, "includes_validation": False}
            ),

            CodePattern(
                "pydantic_schema",
                "Pydantic Data Schema",
                "schema",
                "python",
                "beginner",
                """
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class {SchemaName}Base(BaseModel):
    \"\"\"Base {SchemaName} schema\"\"\"

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class {SchemaName}Create({SchemaName}Base):
    \"\"\"{SchemaName} creation schema\"\"\"
    pass

class {SchemaName}Update({SchemaName}Base):
    \"\"\"{SchemaName} update schema\"\"\"
    name: Optional[str] = Field(None, min_length=1, max_length=100)

class {SchemaName}InDBBase({SchemaName}Base):
    \"\"\"{SchemaName} database schema\"\"\"
    id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class {SchemaName}({SchemaName}InDBBase):
    \"\"\"Complete {SchemaName} schema\"\"\"
    pass

class {SchemaName}InDB({SchemaName}InDBBase):
    \"\"\"{SchemaName} in database\"\"\"
    pass
""",
                {"framework": "pydantic", "includes_validation": True, "includes_examples": False}
            )
        ]

        # JavaScript/React patterns
        js_patterns = [
            CodePattern(
                "react_component_functional",
                "React Functional Component",
                "frontend",
                "javascript",
                "intermediate",
                """
import React, {{ useState, useEffect }} from 'react';
import PropTypes from 'prop-types';

const {ComponentName} = ({{ {props} }}) => {{
    const [state, setState] = useState({initialState});

    useEffect(() => {{
        // Component initialization logic
        {initializationCode}

        return () => {{
            // Cleanup logic
            {cleanupCode}
        }};
    }}, [{dependencies}]);

    const handleAction = () => {{
        {actionLogic}
    }};

    return (
        <div className="{componentName}-container">
            {renderContent}
        </div>
    );
}};

{ComponentName}.propTypes = {{
    {propTypes}
}};

{ComponentName}.defaultProps = {{
    {defaultProps}
}};

export default {ComponentName};
""",
                {"framework": "react", "hooks_used": ["useState", "useEffect"], "includes_propTypes": True}
            ),

            CodePattern(
                "express_route",
                "Express.js Route Handler",
                "backend",
                "javascript",
                "beginner",
                """
const express = require('express');
const router = express.Router();
const {{ validationResult }} = require('express-validator');

// Validation middleware
const validateRequest = [
    // Add validation rules here
];

// Routes
router.get('/{resource}s', async (req, res) => {{
    try {{
        const { page = 1, limit = 10 } = req.query;

        // Fetch data logic
        const {resources} = await {Resource}.find()
            .limit(limit * 1)
            .skip((page - 1) * limit)
            .exec();

        const count = await {Resource}.countDocuments();

        res.json({{
            {resources},
            totalPages: Math.ceil(count / limit),
            currentPage: page,
            totalCount: count
        }});
    }} catch (error) {{
        console.error('Error fetching {resources}:', error);
        res.status(500).json({{ error: 'Internal server error' }});
    }}
}});

router.post('/{resource}s', validateRequest, async (req, res) => {{
    try {{
        const errors = validationResult(req);
        if (!errors.isEmpty()) {{
            return res.status(400).json({{ errors: errors.array() }});
        }}

        const new{Resource} = new {Resource}(req.body);
        const saved{Resource} = await new{Resource}.save();

        res.status(201).json(saved{Resource});
    }} catch (error) {{
        console.error('Error creating {resource}:', error);
        if (error.name === 'ValidationError') {{
            return res.status(400).json({{ error: error.message }});
        }}
        res.status(500).json({{ error: 'Internal server error' }});
    }}
}});

module.exports = router;
""",
                {"framework": "express", "includes_validation": True, "database": "mongodb"}
            )
        ]

        # Store all patterns
        for pattern in python_patterns + js_patterns:
            self.code_patterns[pattern.pattern_id] = pattern

    async def generate_code(
        self,
        specifications: Dict[str, Any],
        language: str = "python",
        framework: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate code using the complete pipeline"""
        generation_id = f"gen_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(str(specifications)) % 10000}"

        # Step 1: Retrieve relevant context and patterns
        relevant_context = []
        relevant_patterns = []

        if self.enable_rag:
            relevant_context = await self.graph_rag_service.retrieve_context(
                f"code generation patterns for {language} {framework or ''} {specifications.get('type', '')}",
                max_nodes=5
            )

        if self.enable_patterns:
            relevant_patterns = self._find_relevant_patterns(specifications, language, framework)

        # Step 2: Generate initial code using LLM with context
        initial_code = await self._generate_code_with_context(
            specifications, language, framework, relevant_context, relevant_patterns
        )

        # Step 3: Validate and correct the generated code
        validation_context = {
            "content_type": "code",
            "language": language,
            "framework": framework,
            "specifications": specifications
        }

        validation_result = await self.orthos_service.validate_content(initial_code, validation_context)

        # Step 4: Apply corrections if needed
        final_code = initial_code
        corrections_applied = []

        if validation_result["final_result"] != "pass":
            correction_result = await self.correction_engine.correct_content(
                initial_code,
                [{"message": str(validation_result), "type": "validation_error", "severity": "medium"}],
                validation_context,
                "llm_rewrite"
            )
            final_code = correction_result["corrected_content"]
            corrections_applied = correction_result["recommendations"]

        # Step 5: Final quality assessment
        quality_score = self._assess_code_quality(final_code, language)

        # Step 6: Generate documentation and tests
        documentation = await self._generate_documentation(final_code, specifications, language)
        tests = await self._generate_tests(final_code, specifications, language)

        # Record generation
        generation_record = {
            "generation_id": generation_id,
            "specifications": specifications,
            "language": language,
            "framework": framework,
            "patterns_used": [p.pattern_id for p in relevant_patterns],
            "context_used": len(relevant_context),
            "validation_result": validation_result["final_result"],
            "corrections_applied": len(corrections_applied),
            "quality_score": quality_score,
            "code_length": len(final_code),
            "timestamp": datetime.utcnow().isoformat()
        }

        self.generation_history.append(generation_record)

        # Learn from this generation
        await self._learn_from_generation(specifications, final_code, quality_score)

        return {
            "generation_id": generation_id,
            "code": final_code,
            "documentation": documentation,
            "tests": tests,
            "quality_score": quality_score,
            "validation_status": validation_result["final_result"],
            "patterns_used": [p.name for p in relevant_patterns],
            "metadata": {
                "language": language,
                "framework": framework,
                "generated_at": datetime.utcnow().isoformat(),
                "generation_method": "pipeline_with_rag"
            }
        }

    async def _generate_code_with_context(
        self,
        specifications: Dict[str, Any],
        language: str,
        framework: Optional[str],
        context: List[Dict[str, Any]],
        patterns: List[CodePattern]
    ) -> str:
        """Generate code using retrieved context and patterns"""
        # Build context string
        context_str = "\n".join([f"- {ctx['content'][:200]}..." for ctx in context])

        # Build patterns string
        patterns_str = "\n".join([f"Pattern: {p.name}\n{p.code}\n---" for p in patterns[:3]])  # Limit to top 3

        system_prompt = f"""You are an expert {language} developer specializing in high-quality code generation.
You excel at:
- Writing clean, maintainable, and well-documented code
- Following language-specific best practices and conventions
- Implementing proper error handling and validation
- Creating modular and testable code structures
- Using appropriate design patterns and architectural principles

Generate production-ready code that follows industry standards and best practices."""

        prompt = f"""Generate {language} code based on the following specifications:

SPECIFICATIONS:
{json.dumps(specifications, indent=2)}

LANGUAGE: {language}
FRAMEWORK: {framework or 'None'}

RELEVANT CONTEXT:
{context_str}

RECOMMENDED PATTERNS:
{patterns_str}

Please generate complete, production-ready code that:

1. **Functionality**: Fully implements the specified requirements
2. **Structure**: Well-organized with clear separation of concerns
3. **Quality**: Clean, readable, and maintainable code
4. **Best Practices**: Follows {language} and {framework or 'general'} conventions
5. **Error Handling**: Includes appropriate error handling and validation
6. **Documentation**: Well-documented with comments and docstrings
7. **Testing**: Code structure supports comprehensive testing

If using patterns, adapt them appropriately to the specifications rather than copying verbatim.

Generate the complete implementation:"""

        return await self.ollama_service.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for more consistent code generation
        )

    def _find_relevant_patterns(
        self,
        specifications: Dict[str, Any],
        language: str,
        framework: Optional[str]
    ) -> List[CodePattern]:
        """Find relevant code patterns for the specifications"""
        relevant_patterns = []
        spec_type = specifications.get("type", "").lower()
        functionality = specifications.get("functionality", "").lower()

        for pattern in self.code_patterns.values():
            if pattern.language != language:
                continue

            # Check framework match
            if framework and pattern.metadata.get("framework") != framework:
                continue

            # Check category/type match
            if spec_type in pattern.category or any(keyword in functionality for keyword in pattern.category.split()):
                relevant_patterns.append(pattern)
                pattern.usage_count += 1
                pattern.last_used = datetime.utcnow()

        # Sort by relevance and usage success
        relevant_patterns.sort(key=lambda p: (p.usage_count, p.success_rate), reverse=True)

        return relevant_patterns[:5]  # Return top 5 most relevant

    async def _generate_documentation(self, code: str, specifications: Dict[str, Any], language: str) -> str:
        """Generate documentation for the generated code"""
        system_prompt = f"""You are a technical documentation specialist who creates clear, comprehensive documentation for {language} code.
You focus on:
- Clear explanations of functionality and purpose
- Comprehensive API documentation with examples
- Installation and setup instructions
- Usage examples and code samples
- Architecture and design decisions
- Troubleshooting and common issues"""

        prompt = f"""Generate comprehensive documentation for this {language} code:

CODE:
```python
{code}
```

SPECIFICATIONS:
{json.dumps(specifications, indent=2)}

Please create documentation that includes:

1. **Overview**
   - Purpose and functionality
   - Key features and capabilities
   - Architecture and design decisions

2. **Installation & Setup**
   - Requirements and dependencies
   - Installation instructions
   - Configuration options

3. **API Documentation**
   - Function/class signatures and parameters
   - Return values and error conditions
   - Usage examples with code samples

4. **Usage Guide**
   - Basic usage scenarios
   - Advanced features and configuration
   - Integration examples

5. **Architecture Details**
   - Component relationships and data flow
   - Design patterns used
   - Security considerations

6. **Troubleshooting**
   - Common issues and solutions
   - Error handling and debugging
   - Performance optimization tips

Format the documentation in clear Markdown with proper headings, code examples, and structure."""

        return await self.ollama_service.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2
        )

    async def _generate_tests(self, code: str, specifications: Dict[str, Any], language: str) -> str:
        """Generate comprehensive tests for the generated code"""
        system_prompt = f"""You are a testing specialist who creates comprehensive, maintainable test suites for {language} code.
You excel at:
- Writing unit tests, integration tests, and edge case coverage
- Using appropriate testing frameworks and best practices
- Creating mock objects and test fixtures
- Writing descriptive test names and documentation
- Achieving high test coverage with meaningful assertions"""

        test_framework = "pytest" if language == "python" else "jest"

        prompt = f"""Generate comprehensive tests for this {language} code using {test_framework}:

CODE TO TEST:
```python
{code}
```

SPECIFICATIONS:
{json.dumps(specifications, indent=2)}

Please generate a complete test suite that includes:

1. **Unit Tests**
   - Test individual functions/methods in isolation
   - Cover all branches and edge cases
   - Use appropriate mocking for external dependencies

2. **Integration Tests**
   - Test component interactions and data flow
   - Verify end-to-end functionality
   - Test error handling and recovery

3. **Edge Cases & Error Conditions**
   - Invalid inputs and boundary conditions
   - Error states and exception handling
   - Resource constraints and timeouts

4. **Test Fixtures & Setup**
   - Test data preparation and factories
   - Mock objects and service stubs
   - Database test setup and teardown

5. **Test Organization**
   - Logical grouping and test file structure
   - Descriptive test names and documentation
   - Test categorization and markers

6. **Test Utilities**
   - Helper functions and custom assertions
   - Test configuration and environment setup
   - Performance and load testing utilities

Ensure tests are:
- Comprehensive with high coverage
- Maintainable and readable
- Independent and reliable
- Fast to execute
- Well-documented with clear assertions

Provide complete, runnable test code with all necessary imports and dependencies."""

        return await self.ollama_service.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2
        )

    def _assess_code_quality(self, code: str, language: str) -> float:
        """Assess the quality of generated code"""
        quality_score = 0.5  # Base score

        try:
            # Basic syntax validation
            if language == "python":
                ast.parse(code)
                quality_score += 0.2

            # Length and structure checks
            lines = code.split('\n')
            if 10 <= len(lines) <= 500:  # Reasonable length
                quality_score += 0.1

            # Documentation checks
            if '"""' in code or "'''" in code or "/*" in code or "//" in code:
                quality_score += 0.1

            # Error handling checks
            if "try:" in code or "catch" in code or "except" in code:
                quality_score += 0.1

            # Function/class structure
            if "def " in code or "class " in code or "function" in code:
                quality_score += 0.1

        except SyntaxError:
            quality_score -= 0.3  # Penalize syntax errors

        return max(0.0, min(1.0, quality_score))

    async def _learn_from_generation(self, specifications: Dict[str, Any], code: str, quality_score: float):
        """Learn from code generation to improve future generations"""
        try:
            # Add generation pattern to knowledge graph
            learning_content = f"Code Generation: {specifications.get('type', 'unknown')} - Quality: {quality_score:.2f}"

            await self.graph_rag_service.add_knowledge(
                content=learning_content,
                metadata={
                    "type": "code_generation_pattern",
                    "specifications": specifications,
                    "quality_score": quality_score,
                    "code_length": len(code),
                    "learning_opportunity": True
                },
                node_type="generation_learning"
            )

            # Update pattern success rates
            for pattern in self.code_patterns.values():
                if pattern.usage_count > 0:
                    # Simple success rate update (in practice, would be more sophisticated)
                    pattern.success_rate = (pattern.success_rate + quality_score) / 2

        except Exception as e:
            logger.warning(f"Failed to learn from code generation: {str(e)}")

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get code generation statistics"""
        if not self.generation_history:
            return {"total_generations": 0}

        total_generations = len(self.generation_history)
        successful_generations = sum(1 for g in self.generation_history if g.get("validation_result") == "pass")
        avg_quality = sum(g["quality_score"] for g in self.generation_history) / total_generations

        language_counts = {}
        for generation in self.generation_history:
            lang = generation["language"]
            language_counts[lang] = language_counts.get(lang, 0) + 1

        return {
            "total_generations": total_generations,
            "successful_generations": successful_generations,
            "success_rate": successful_generations / total_generations,
            "average_quality_score": avg_quality,
            "language_distribution": language_counts,
            "patterns_available": len(self.code_patterns),
            "patterns_used": sum(len(g.get("patterns_used", [])) for g in self.generation_history)
        }

    def add_code_pattern(self, pattern: CodePattern):
        """Add a new code pattern to the library"""
        self.code_patterns[pattern.pattern_id] = pattern
        logger.info(f"Added code pattern: {pattern.name}")

    def get_pattern_suggestions(self, specifications: Dict[str, Any], language: str, framework: Optional[str] = None) -> List[CodePattern]:
        """Get pattern suggestions for given specifications"""
        return self._find_relevant_patterns(specifications, language, framework)