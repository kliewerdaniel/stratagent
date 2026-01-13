"""
Plugin System for StratAgent extensibility
"""
import asyncio
import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Callable, Type
from datetime import datetime
from pathlib import Path
import json
import sys

logger = logging.getLogger(__name__)

class PluginInterface:
    """Base interface for all StratAgent plugins"""

    @property
    def name(self) -> str:
        """Plugin name"""
        return self.__class__.__name__

    @property
    def version(self) -> str:
        """Plugin version"""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Plugin description"""
        return "Plugin description"

    @property
    def capabilities(self) -> List[str]:
        """Plugin capabilities"""
        return []

    async def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        return True

    async def shutdown(self) -> bool:
        """Shutdown the plugin"""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        return {
            "name": self.name,
            "version": self.version,
            "status": "active",
            "capabilities": self.capabilities
        }

class CodeAnalysisPlugin(PluginInterface):
    """Plugin for advanced code analysis"""

    @property
    def capabilities(self) -> List[str]:
        return ["code_analysis", "complexity_analysis", "dependency_analysis", "security_scanning"]

    async def analyze_code_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code complexity metrics"""
        # Placeholder implementation
        return {
            "cyclomatic_complexity": 5,
            "cognitive_complexity": 8,
            "maintainability_index": 75,
            "lines_of_code": len(code.split('\n')),
            "functions_count": code.count('def ') if language == 'python' else code.count('function ')
        }

    async def analyze_dependencies(self, code: str, language: str) -> List[str]:
        """Analyze code dependencies"""
        dependencies = []

        if language == 'python':
            # Simple import analysis
            import_lines = [line for line in code.split('\n') if line.strip().startswith('import ') or 'from ' in line and ' import ' in line]
            for line in import_lines:
                if line.strip().startswith('import '):
                    deps = line.replace('import ', '').split(',')
                    dependencies.extend([d.strip().split('.')[0] for d in deps])
                elif 'from ' in line and ' import ' in line:
                    dep = line.split('from ')[1].split(' import ')[0].strip().split('.')[0]
                    dependencies.append(dep)

        return list(set(dependencies))

class GitIntegrationPlugin(PluginInterface):
    """Plugin for Git repository integration"""

    def __init__(self):
        self.repositories = {}
        self.active_repo = None

    @property
    def capabilities(self) -> List[str]:
        return ["git_integration", "repository_management", "commit_analysis", "branch_management"]

    async def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize Git integration"""
        # In a real implementation, this would set up GitPython or similar
        logger.info("Git integration plugin initialized")
        return True

    async def clone_repository(self, url: str, local_path: str) -> bool:
        """Clone a Git repository"""
        try:
            # Placeholder for actual Git operations
            repo_id = f"repo_{hash(url)}"
            self.repositories[repo_id] = {
                "url": url,
                "local_path": local_path,
                "cloned_at": datetime.utcnow().isoformat(),
                "status": "cloned"
            }
            self.active_repo = repo_id
            logger.info(f"Repository cloned: {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to clone repository: {str(e)}")
            return False

    async def analyze_commits(self, repo_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Analyze recent commits"""
        # Placeholder implementation
        return [
            {
                "hash": f"commit_{i}",
                "message": f"Sample commit {i}",
                "author": "Developer",
                "date": datetime.utcnow().isoformat(),
                "files_changed": ["file1.py", "file2.py"],
                "lines_added": 50,
                "lines_removed": 20
            } for i in range(limit)
        ]

    async def get_repository_status(self, repo_id: str) -> Dict[str, Any]:
        """Get repository status"""
        if repo_id not in self.repositories:
            return {"error": "Repository not found"}

        repo = self.repositories[repo_id]
        return {
            "url": repo["url"],
            "status": repo["status"],
            "last_updated": repo["cloned_at"],
            "branches": ["main", "develop"],  # Placeholder
            "pending_changes": 0
        }

class TestingPlugin(PluginInterface):
    """Plugin for automated testing integration"""

    @property
    def capabilities(self) -> List[str]:
        return ["test_execution", "test_generation", "coverage_analysis", "test_reporting"]

    async def run_tests(self, test_path: str, framework: str = "pytest") -> Dict[str, Any]:
        """Run tests and return results"""
        # Placeholder implementation
        return {
            "framework": framework,
            "tests_run": 25,
            "tests_passed": 23,
            "tests_failed": 2,
            "coverage": 85.5,
            "execution_time": 45.2,
            "failed_tests": [
                {"name": "test_api_endpoint", "error": "AssertionError: expected 200, got 500"},
                {"name": "test_database_connection", "error": "ConnectionError: unable to connect"}
            ]
        }

    async def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed test report"""
        report = f"""
# Test Execution Report

## Summary
- **Framework**: {results['framework']}
- **Tests Run**: {results['tests_run']}
- **Tests Passed**: {results['tests_passed']}
- **Tests Failed**: {results['tests_failed']}
- **Coverage**: {results['coverage']:.1f}%
- **Execution Time**: {results['execution_time']:.1f}s

## Failed Tests
"""

        for failed_test in results.get('failed_tests', []):
            report += f"""
### {failed_test['name']}
**Error**: {failed_test['error']}
"""

        report += """
## Recommendations
"""

        if results['tests_failed'] > 0:
            report += f"- Address {results['tests_failed']} failing tests\n"
        if results['coverage'] < 80:
            report += f"- Improve test coverage (currently {results['coverage']:.1f}%)\n"
        if results['execution_time'] > 60:
            report += f"- Optimize test execution time (currently {results['execution_time']:.1f}s)\n"

        return report

class DocumentationPlugin(PluginInterface):
    """Plugin for documentation generation and management"""

    @property
    def capabilities(self) -> List[str]:
        return ["doc_generation", "api_documentation", "readme_generation", "doc_validation"]

    async def generate_api_docs(self, code: str, language: str) -> str:
        """Generate API documentation from code"""
        # Placeholder implementation
        return f"""
# API Documentation

## Endpoints

### GET /api/items
Retrieve all items.

**Response**: Array of item objects

### POST /api/items
Create a new item.

**Request Body**:
```json
{{
  "name": "string",
  "description": "string"
}}
```

**Response**: Created item object

### PUT /api/items/{{id}}
Update an existing item.

### DELETE /api/items/{{id}}
Delete an item.

## Data Models

### Item
- `id`: integer (primary key)
- `name`: string (required)
- `description`: string (optional)
- `created_at`: datetime
- `updated_at`: datetime
"""

    async def validate_documentation(self, docs: str) -> Dict[str, Any]:
        """Validate documentation completeness and accuracy"""
        validation_results = {
            "completeness_score": 0.85,
            "issues": [],
            "recommendations": []
        }

        # Check for required sections
        required_sections = ["Overview", "Installation", "Usage", "API"]
        missing_sections = []

        for section in required_sections:
            if section.lower() not in docs.lower():
                missing_sections.append(section)

        if missing_sections:
            validation_results["issues"].append(f"Missing sections: {', '.join(missing_sections)}")
            validation_results["recommendations"].append("Add missing documentation sections")

        # Check for code examples
        if "```" not in docs:
            validation_results["issues"].append("No code examples found")
            validation_results["recommendations"].append("Add code examples to illustrate usage")

        return validation_results

class DeploymentPlugin(PluginInterface):
    """Plugin for deployment and DevOps integration"""

    @property
    def capabilities(self) -> List[str]:
        return ["deployment", "ci_cd_integration", "container_management", "infrastructure_as_code"]

    async def deploy_application(self, app_config: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Deploy application to specified environment"""
        # Placeholder implementation
        deployment_id = f"deploy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        return {
            "deployment_id": deployment_id,
            "environment": environment,
            "status": "success",
            "deployed_at": datetime.utcnow().isoformat(),
            "version": app_config.get("version", "1.0.0"),
            "services_deployed": ["api", "frontend", "database"],
            "health_checks": {
                "api": "healthy",
                "frontend": "healthy",
                "database": "healthy"
            }
        }

    async def generate_dockerfile(self, app_type: str, dependencies: List[str]) -> str:
        """Generate Dockerfile for application"""
        if app_type == "python":
            return f"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        elif app_type == "nodejs":
            return f"""
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD ["npm", "start"]
"""
        else:
            return "# Dockerfile generation not supported for this application type"

class PluginManager:
    """Manages plugin loading, registration, and lifecycle"""

    def __init__(self):
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
        self.plugin_directory = Path("plugins")
        self.loaded_plugins: set = set()

    async def load_builtin_plugins(self):
        """Load built-in plugins"""
        builtin_plugins = [
            CodeAnalysisPlugin(),
            GitIntegrationPlugin(),
            TestingPlugin(),
            DocumentationPlugin(),
            DeploymentPlugin()
        ]

        for plugin in builtin_plugins:
            await self.register_plugin(plugin)
            logger.info(f"Loaded built-in plugin: {plugin.name}")

    async def load_external_plugins(self, plugin_directory: Optional[str] = None):
        """Load external plugins from directory"""
        if plugin_directory:
            self.plugin_directory = Path(plugin_directory)

        if not self.plugin_directory.exists():
            logger.warning(f"Plugin directory does not exist: {self.plugin_directory}")
            return

        # Scan for plugin files
        plugin_files = list(self.plugin_directory.glob("*.py"))
        plugin_files.extend(list(self.plugin_directory.glob("*/__init__.py")))

        for plugin_file in plugin_files:
            try:
                await self._load_plugin_from_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {str(e)}")

    async def _load_plugin_from_file(self, plugin_file: Path):
        """Load plugin from Python file"""
        # Add plugin directory to Python path
        plugin_dir = plugin_file.parent
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))

        # Import the module
        module_name = plugin_file.stem
        if plugin_file.name == "__init__.py":
            module_name = plugin_file.parent.name

        try:
            module = importlib.import_module(module_name)

            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginInterface) and
                    obj != PluginInterface):

                    # Instantiate and register plugin
                    plugin_instance = obj()
                    await self.register_plugin(plugin_instance)
                    logger.info(f"Loaded external plugin: {plugin_instance.name}")

        except Exception as e:
            logger.error(f"Error importing plugin module {module_name}: {str(e)}")

    async def register_plugin(self, plugin: PluginInterface) -> bool:
        """Register a plugin with the system"""
        if plugin.name in self.plugins:
            logger.warning(f"Plugin {plugin.name} is already registered")
            return False

        try:
            # Initialize plugin
            init_context = {
                "plugin_manager": self,
                "system_config": {},  # Would pass actual config
                "logger": logger
            }

            success = await plugin.initialize(init_context)
            if success:
                self.plugins[plugin.name] = plugin
                self.plugin_metadata[plugin.name] = {
                    "version": plugin.version,
                    "description": plugin.description,
                    "capabilities": plugin.capabilities,
                    "registered_at": datetime.utcnow().isoformat(),
                    "status": "active"
                }
                self.loaded_plugins.add(plugin.name)
                logger.info(f"Registered plugin: {plugin.name}")
                return True
            else:
                logger.error(f"Failed to initialize plugin: {plugin.name}")
                return False

        except Exception as e:
            logger.error(f"Error registering plugin {plugin.name}: {str(e)}")
            return False

    async def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin"""
        if plugin_name not in self.plugins:
            return False

        try:
            plugin = self.plugins[plugin_name]
            await plugin.shutdown()

            del self.plugins[plugin_name]
            del self.plugin_metadata[plugin_name]
            self.loaded_plugins.discard(plugin_name)

            logger.info(f"Unregistered plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error unregistering plugin {plugin_name}: {str(e)}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a registered plugin by name"""
        return self.plugins.get(plugin_name)

    def get_available_plugins(self) -> List[str]:
        """Get list of available plugin names"""
        return list(self.plugins.keys())

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a plugin"""
        if plugin_name not in self.plugins:
            return None

        plugin = self.plugins[plugin_name]
        return {
            "name": plugin.name,
            "version": plugin.version,
            "description": plugin.description,
            "capabilities": plugin.capabilities,
            "status": plugin.get_status(),
            "metadata": self.plugin_metadata.get(plugin_name, {})
        }

    def get_plugins_by_capability(self, capability: str) -> List[str]:
        """Get plugins that have a specific capability"""
        matching_plugins = []
        for plugin_name, metadata in self.plugin_metadata.items():
            if capability in metadata.get("capabilities", []):
                matching_plugins.append(plugin_name)
        return matching_plugins

    async def execute_plugin_method(self, plugin_name: str, method_name: str, *args, **kwargs) -> Any:
        """Execute a method on a specific plugin"""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")

        if not hasattr(plugin, method_name):
            raise ValueError(f"Method {method_name} not found on plugin {plugin_name}")

        method = getattr(plugin, method_name)
        if not callable(method):
            raise ValueError(f"{method_name} is not callable on plugin {plugin_name}")

        try:
            if inspect.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {method_name} on plugin {plugin_name}: {str(e)}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall plugin system status"""
        return {
            "total_plugins": len(self.plugins),
            "active_plugins": len(self.loaded_plugins),
            "plugin_directory": str(self.plugin_directory),
            "capabilities": self._get_all_capabilities(),
            "plugin_health": self._check_plugin_health()
        }

    def _get_all_capabilities(self) -> List[str]:
        """Get all unique capabilities across plugins"""
        capabilities = set()
        for metadata in self.plugin_metadata.values():
            capabilities.update(metadata.get("capabilities", []))
        return sorted(list(capabilities))

    def _check_plugin_health(self) -> Dict[str, str]:
        """Check health status of all plugins"""
        health_status = {}
        for plugin_name, plugin in self.plugins.items():
            try:
                status = plugin.get_status()
                health_status[plugin_name] = status.get("status", "unknown")
            except Exception as e:
                health_status[plugin_name] = "error"
                logger.error(f"Error checking health for plugin {plugin_name}: {str(e)}")

        return health_status

    async def shutdown_all_plugins(self):
        """Shutdown all registered plugins"""
        shutdown_tasks = []
        for plugin in self.plugins.values():
            shutdown_tasks.append(plugin.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self.plugins.clear()
        self.plugin_metadata.clear()
        self.loaded_plugins.clear()

        logger.info("All plugins shut down")