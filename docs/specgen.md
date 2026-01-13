# SpecGen: Deterministic Code Generation via Agentic RAG

## Overview

SpecGen is a CLI tool that transforms Markdown specifications into production-ready application skeletons using a four-agent pipeline. It replaces probabilistic guesswork with deterministic architecture.

**Repository**: `github.com/kliewerdaniel/specgen`

## Problem Statement

Conversational coding assistants hallucinate. They miss requirements, generate broken imports, and produce code that looks correct but fails validation.

## Solution Architecture

### Agent Pipeline

```
graph LR
    A[SpecInterpreter] --> B[Architect]
    B --> C[Generator]
    C --> D[Validator]
```

### Key Components

1. **SpecInterpreter Agent**
   - Parses Markdown specifications
   - Extracts requirements, dependencies, and constraints
   - Converts natural language into structured specifications

2. **Architect Agent**
   - RAG-powered design consultation
   - References knowledge base of proven design patterns
   - Enforces architectural best practices (FastAPI, Django, Next.js)
   - Generates system architecture before code generation

3. **Generator Agent**
   - Produces production-ready code based on architectural blueprint
   - Implements business logic following established patterns
   - Generates configuration files and documentation

4. **Validator Agent**
   - AST parsing and import resolution
   - Syntax validation and type checking
   - Dependency verification
   - Test generation and execution

## Technical Implementation

### Dependencies

```python
# requirements.txt
ollama>=0.1.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
astroid>=2.15.0
click>=8.1.0
rich>=13.0.0
```

### Core Classes

```python
# specgen/core/agent.py
class BaseAgent:
    def __init__(self, model_name: str = "codellama"):
        self.client = ollama.Client()
        self.model = model_name

    async def process(self, input_data: dict) -> dict:
        # Agent-specific processing logic
        pass

class SpecInterpreter(BaseAgent):
    # Parses markdown specs into structured data
    pass

class ArchitectAgent(BaseAgent):
    def __init__(self, knowledge_base_path: str):
        super().__init__()
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)

    def load_knowledge_base(self, path: str):
        # Load FAISS index and sentence transformers
        pass

    async def consult_patterns(self, requirements: dict) -> dict:
        # RAG query against design patterns
        pass

class GeneratorAgent(BaseAgent):
    # Code generation logic
    pass

class ValidatorAgent(BaseAgent):
    # AST validation and testing
    pass
```

### CLI Interface

```python
# specgen/cli.py
import click
from specgen.core.pipeline import GenerationPipeline

@click.command()
@click.argument('spec_file', type=click.Path(exists=True))
@click.option('--output', '-o', default='.', help='Output directory')
@click.option('--model', default='codellama', help='Ollama model to use')
def generate(spec_file: str, output: str, model: str):
    """Generate application from markdown specification."""
    pipeline = GenerationPipeline(model_name=model)

    with open(spec_file, 'r') as f:
        spec_content = f.read()

    result = pipeline.run(spec_content, output_dir=output)

    if result['success']:
        click.echo(f"✅ Application generated successfully in {output}")
    else:
        click.echo(f"❌ Generation failed: {result['error']}")
        raise click.Exit(1)

if __name__ == '__main__':
    generate()
```

### Knowledge Base Structure

```
specgen/knowledge_base/
├── patterns/
│   ├── fastapi_web_app.json
│   ├── django_rest_api.json
│   ├── nextjs_react_app.json
│   └── flask_microservice.json
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
└── embeddings/
    └── sentence_transformer_model/
```

### Usage Example

```bash
# Generate a FastAPI application
specgen my_app_spec.md --output ./generated_app --model codellama

# The tool creates:
# - Complete application structure
# - All necessary files
# - Dependencies and configuration
# - Basic tests and documentation
```

## Validation Mechanisms

### AST-Based Validation

```python
# specgen/core/validator.py
import ast
import importlib.util

class CodeValidator:
    def validate_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def validate_imports(self, code: str) -> list:
        tree = ast.parse(code)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)

        # Check if imports can be resolved
        missing = []
        for imp in imports:
            if not self.can_import(imp):
                missing.append(imp)

        return missing

    def can_import(self, module_name: str) -> bool:
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
```

### Test Generation

```python
# specgen/core/test_generator.py
class TestGenerator:
    def generate_unit_tests(self, code_structure: dict) -> str:
        # Generate pytest test files based on generated code
        pass

    def generate_integration_tests(self, api_endpoints: list) -> str:
        # Generate API integration tests
        pass
```

## Deployment and Distribution

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download required models
RUN ollama pull codellama

CMD ["python", "-m", "specgen.cli"]
```

### PyPI Distribution

```toml
# pyproject.toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "specgen"
version = "1.0.0"
description = "Deterministic code generation via agentic RAG"
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

dependencies = [
    "ollama>=0.1.0",
    "faiss-cpu>=1.7.0",
    "sentence-transformers>=2.2.0",
    "click>=8.1.0",
    "rich>=13.0.0",
]

[project.scripts]
specgen = "specgen.cli:generate"
```

## Impact and Benefits

- **Time Reduction**: Project boilerplate from hours to seconds
- **Quality Assurance**: Every generated project is compilable and testable
- **Consistency**: Enforces architectural best practices
- **Scalability**: Deterministic generation enables reliable automation

## Future Enhancements

- Multi-language support (Go, Rust, TypeScript)
- Plugin architecture for custom patterns
- Integration with CI/CD pipelines
- GUI interface for non-technical users