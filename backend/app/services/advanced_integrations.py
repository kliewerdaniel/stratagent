"""
Advanced Integrations for StratAgent - CI/CD, APIs, and Development Tools
"""
import asyncio
import logging
import aiohttp
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import base64
import hmac
import hashlib

from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class GitHubIntegration:
    """GitHub API integration for repository management and CI/CD"""

    def __init__(self, token: Optional[str] = None):
        self.base_url = "https://api.github.com"
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize HTTP session"""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        self.session = aiohttp.ClientSession(headers=headers)

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def create_repository(self, name: str, description: str = "", private: bool = False) -> Dict[str, Any]:
        """Create a new GitHub repository"""
        if not self.session or not self.token:
            return {"error": "GitHub integration not properly configured"}

        url = f"{self.base_url}/user/repos"
        data = {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": True
        }

        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 201:
                    repo_data = await response.json()
                    return {
                        "success": True,
                        "repository": {
                            "name": repo_data["name"],
                            "url": repo_data["html_url"],
                            "clone_url": repo_data["clone_url"],
                            "created_at": repo_data["created_at"]
                        }
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("message", "Failed to create repository")}
        except Exception as e:
            logger.error(f"Error creating GitHub repository: {str(e)}")
            return {"error": str(e)}

    async def create_pull_request(self, owner: str, repo: str, title: str, head: str, base: str, body: str) -> Dict[str, Any]:
        """Create a pull request"""
        if not self.session:
            return {"error": "GitHub integration not initialized"}

        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body
        }

        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 201:
                    pr_data = await response.json()
                    return {
                        "success": True,
                        "pull_request": {
                            "number": pr_data["number"],
                            "url": pr_data["html_url"],
                            "title": pr_data["title"],
                            "state": pr_data["state"]
                        }
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("message", "Failed to create pull request")}
        except Exception as e:
            logger.error(f"Error creating pull request: {str(e)}")
            return {"error": str(e)}

    async def get_repository_issues(self, owner: str, repo: str, state: str = "open", limit: int = 10) -> List[Dict[str, Any]]:
        """Get repository issues"""
        if not self.session:
            return []

        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        params = {
            "state": state,
            "per_page": min(limit, 100)
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    issues = await response.json()
                    return [{
                        "number": issue["number"],
                        "title": issue["title"],
                        "state": issue["state"],
                        "url": issue["html_url"],
                        "created_at": issue["created_at"],
                        "labels": [label["name"] for label in issue.get("labels", [])]
                    } for issue in issues]
                else:
                    logger.error(f"Failed to fetch issues: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching repository issues: {str(e)}")
            return []

class CI_CDIntegration:
    """CI/CD pipeline integration"""

    def __init__(self):
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        self.active_builds: Dict[str, asyncio.Task] = {}

    async def create_pipeline(self, name: str, config: Dict[str, Any]) -> str:
        """Create a new CI/CD pipeline"""
        pipeline_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        pipeline = {
            "id": pipeline_id,
            "name": name,
            "config": config,
            "created_at": datetime.utcnow().isoformat(),
            "status": "created",
            "builds": []
        }

        self.pipelines[pipeline_id] = pipeline
        logger.info(f"Created CI/CD pipeline: {name}")
        return pipeline_id

    async def run_pipeline(self, pipeline_id: str) -> str:
        """Execute a pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        build_id = f"build_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Create build task
        build_task = asyncio.create_task(self._execute_pipeline_build(pipeline_id, build_id))
        self.active_builds[build_id] = build_task

        # Update pipeline status
        self.pipelines[pipeline_id]["status"] = "running"

        return build_id

    async def _execute_pipeline_build(self, pipeline_id: str, build_id: str) -> Dict[str, Any]:
        """Execute pipeline build stages"""
        pipeline = self.pipelines[pipeline_id]
        config = pipeline["config"]

        build_result = {
            "build_id": build_id,
            "pipeline_id": pipeline_id,
            "status": "running",
            "stages": [],
            "started_at": datetime.utcnow().isoformat()
        }

        try:
            # Execute pipeline stages
            stages = config.get("stages", ["build", "test", "deploy"])

            for stage in stages:
                stage_result = await self._execute_stage(stage, config)
                build_result["stages"].append(stage_result)

                if not stage_result["success"]:
                    build_result["status"] = "failed"
                    break

            if build_result["status"] == "running":
                build_result["status"] = "success"

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            build_result["status"] = "error"
            build_result["error"] = str(e)

        build_result["completed_at"] = datetime.utcnow().isoformat()

        # Update pipeline
        self.pipelines[pipeline_id]["builds"].append(build_result)
        self.pipelines[pipeline_id]["status"] = "idle"

        # Remove from active builds
        if build_id in self.active_builds:
            del self.active_builds[build_id]

        return build_result

    async def _execute_stage(self, stage: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline stage"""
        # Simulate stage execution
        await asyncio.sleep(2)  # Simulate work

        # Mock success/failure based on stage
        success = True
        if stage == "test" and config.get("fail_tests", False):
            success = False

        return {
            "stage": stage,
            "success": success,
            "duration": 2.0,
            "output": f"Stage {stage} {'completed successfully' if success else 'failed'}",
            "artifacts": [] if not success else [f"{stage}_artifact.txt"]
        }

    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status"""
        return self.pipelines.get(pipeline_id)

    def get_build_status(self, build_id: str) -> Optional[Dict[str, Any]]:
        """Get build status"""
        for pipeline in self.pipelines.values():
            for build in pipeline.get("builds", []):
                if build["build_id"] == build_id:
                    return build
        return None

class ExternalAPIIntegration:
    """Integration with external APIs and services"""

    def __init__(self):
        self.api_configs: Dict[str, Dict[str, Any]] = {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    def register_api(self, name: str, config: Dict[str, Any]):
        """Register an external API configuration"""
        self.api_configs[name] = {
            "base_url": config["base_url"],
            "auth_type": config.get("auth_type", "none"),
            "auth_config": config.get("auth_config", {}),
            "headers": config.get("headers", {}),
            "timeout": config.get("timeout", 30)
        }
        logger.info(f"Registered external API: {name}")

    async def call_api(self, api_name: str, endpoint: str, method: str = "GET",
                       data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a call to an external API"""
        if api_name not in self.api_configs:
            return {"error": f"API not registered: {api_name}"}

        if not self.session:
            return {"error": "API integration not initialized"}

        config = self.api_configs[api_name]
        url = f"{config['base_url'].rstrip('/')}/{endpoint.lstrip('/')}"

        # Prepare headers
        headers = config.get("headers", {}).copy()

        # Add authentication
        if config["auth_type"] == "bearer":
            token = config["auth_config"].get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif config["auth_type"] == "basic":
            username = config["auth_config"].get("username")
            password = config["auth_config"].get("password")
            if username and password:
                auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {auth_string}"

        try:
            # Make request
            async with self.session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                json=data if data else None,
                params=params if params else None,
                timeout=aiohttp.ClientTimeout(total=config["timeout"])
            ) as response:

                result = {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "url": str(response.url)
                }

                if response.status >= 200 and response.status < 300:
                    try:
                        result["data"] = await response.json()
                    except:
                        result["data"] = await response.text()
                else:
                    result["error"] = await response.text()

                return result

        except Exception as e:
            logger.error(f"Error calling external API {api_name}: {str(e)}")
            return {"error": str(e)}

class SlackIntegration:
    """Slack integration for notifications and collaboration"""

    def __init__(self, webhook_url: Optional[str] = None, bot_token: Optional[str] = None):
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize Slack integration"""
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def send_notification(self, channel: str, message: str, attachments: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Send a notification to Slack"""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        payload = {
            "channel": channel,
            "text": message,
            "username": "StratAgent",
            "icon_emoji": ":robot_face:"
        }

        if attachments:
            payload["attachments"] = attachments

        try:
            async with self.session.post(self.webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info("Slack notification sent successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to send Slack notification: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False

    async def send_code_review_notification(self, project_name: str, pr_url: str, reviewer_feedback: Dict[str, Any]):
        """Send code review notification"""
        message = f"🤖 Code Review Complete for {project_name}"

        attachments = [{
            "color": "good" if reviewer_feedback.get("approved", False) else "warning",
            "title": "Pull Request Review",
            "title_link": pr_url,
            "fields": [
                {
                    "title": "Status",
                    "value": "Approved" if reviewer_feedback.get("approved", False) else "Changes Requested",
                    "short": True
                },
                {
                    "title": "Comments",
                    "value": str(len(reviewer_feedback.get("comments", []))),
                    "short": True
                }
            ]
        }]

        if reviewer_feedback.get("comments"):
            comments_text = "\n".join([f"• {comment}" for comment in reviewer_feedback["comments"][:5]])
            if len(reviewer_feedback["comments"]) > 5:
                comments_text += f"\n... and {len(reviewer_feedback['comments']) - 5} more comments"
            attachments[0]["text"] = comments_text

        return await self.send_notification("#code-reviews", message, attachments)

class DatabaseIntegration:
    """Integration with external databases and data sources"""

    def __init__(self):
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.active_queries: Dict[str, asyncio.Task] = {}

    async def connect_database(self, name: str, config: Dict[str, Any]) -> bool:
        """Establish database connection"""
        # Placeholder for actual database connections
        # In practice, this would use appropriate database drivers

        self.connections[name] = {
            "config": config,
            "connected_at": datetime.utcnow().isoformat(),
            "status": "connected",
            "connection_type": config.get("type", "unknown")
        }

        logger.info(f"Connected to database: {name}")
        return True

    async def execute_query(self, db_name: str, query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Execute a database query"""
        if db_name not in self.connections:
            return {"error": f"Database not connected: {db_name}"}

        # Placeholder for actual query execution
        query_id = f"query_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Simulate query execution
        await asyncio.sleep(0.1)  # Simulate network latency

        # Mock results based on query type
        if query.strip().upper().startswith("SELECT"):
            mock_results = [
                {"id": 1, "name": "Sample Record 1", "created_at": datetime.utcnow().isoformat()},
                {"id": 2, "name": "Sample Record 2", "created_at": datetime.utcnow().isoformat()},
            ]
            return {
                "query_id": query_id,
                "success": True,
                "results": mock_results,
                "row_count": len(mock_results),
                "execution_time": 0.1
            }
        else:
            return {
                "query_id": query_id,
                "success": True,
                "affected_rows": 1,
                "execution_time": 0.05
            }

    async def get_table_schema(self, db_name: str, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        if db_name not in self.connections:
            return {"error": f"Database not connected: {db_name}"}

        # Mock schema information
        return {
            "table_name": table_name,
            "columns": [
                {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
                {"name": "name", "type": "varchar(255)", "nullable": False},
                {"name": "created_at", "type": "timestamp", "nullable": False},
                {"name": "updated_at", "type": "timestamp", "nullable": True}
            ],
            "indexes": [
                {"name": "pk_table", "columns": ["id"], "unique": True},
                {"name": "idx_name", "columns": ["name"], "unique": False}
            ],
            "row_count": 1250
        }

class CloudIntegration:
    """Integration with cloud platforms (AWS, GCP, Azure)"""

    def __init__(self):
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.services: Dict[str, Any] = {}

    def configure_provider(self, provider: str, config: Dict[str, Any]):
        """Configure cloud provider credentials and settings"""
        self.providers[provider] = {
            "config": config,
            "configured_at": datetime.utcnow().isoformat(),
            "status": "configured"
        }
        logger.info(f"Configured cloud provider: {provider}")

    async def deploy_to_cloud(self, provider: str, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy application to cloud platform"""
        if provider not in self.providers:
            return {"error": f"Cloud provider not configured: {provider}"}

        deployment_id = f"deploy_{provider}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Simulate cloud deployment
        await asyncio.sleep(3)  # Simulate deployment time

        return {
            "deployment_id": deployment_id,
            "provider": provider,
            "status": "success",
            "service_url": f"https://{service_config.get('name', 'app')}.{provider}.cloud",
            "deployed_at": datetime.utcnow().isoformat(),
            "resources_created": [
                f"{provider}_instance",
                f"{provider}_load_balancer",
                f"{provider}_database"
            ]
        }

    async def get_cloud_resources(self, provider: str) -> List[Dict[str, Any]]:
        """Get list of cloud resources"""
        if provider not in self.providers:
            return []

        # Mock cloud resources
        return [
            {
                "id": f"{provider}_resource_1",
                "type": "compute_instance",
                "name": f"{provider}-app-server",
                "status": "running",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "id": f"{provider}_resource_2",
                "type": "database",
                "name": f"{provider}-app-db",
                "status": "available",
                "created_at": datetime.utcnow().isoformat()
            }
        ]

class MonitoringIntegration:
    """Integration with monitoring and logging platforms"""

    def __init__(self):
        self.monitors: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []

    def setup_monitoring(self, service_name: str, config: Dict[str, Any]):
        """Setup monitoring for a service"""
        self.monitors[service_name] = {
            "config": config,
            "setup_at": datetime.utcnow().isoformat(),
            "status": "active",
            "metrics_collected": []
        }
        logger.info(f"Setup monitoring for service: {service_name}")

    async def send_metrics(self, service_name: str, metrics: Dict[str, Any]):
        """Send metrics to monitoring service"""
        if service_name not in self.monitors:
            logger.warning(f"Monitoring not setup for service: {service_name}")
            return

        # Store metrics
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }

        self.monitors[service_name]["metrics_collected"].append(metrics_entry)

        # Check for alerts
        await self._check_metric_alerts(service_name, metrics)

        logger.info(f"Metrics sent for {service_name}: {len(metrics)} metrics")

    async def _check_metric_alerts(self, service_name: str, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds"""
        config = self.monitors[service_name]["config"]
        alert_rules = config.get("alert_rules", {})

        for metric_name, threshold in alert_rules.items():
            if metric_name in metrics:
                value = metrics[metric_name]

                if isinstance(threshold, dict):
                    warning_threshold = threshold.get("warning")
                    critical_threshold = threshold.get("critical")

                    if critical_threshold and value >= critical_threshold:
                        alert = {
                            "service": service_name,
                            "metric": metric_name,
                            "level": "critical",
                            "value": value,
                            "threshold": critical_threshold,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        self.alerts.append(alert)
                        logger.warning(f"Critical alert: {service_name} {metric_name} = {value}")

                    elif warning_threshold and value >= warning_threshold:
                        alert = {
                            "service": service_name,
                            "metric": metric_name,
                            "level": "warning",
                            "value": value,
                            "threshold": warning_threshold,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        self.alerts.append(alert)
                        logger.warning(f"Warning alert: {service_name} {metric_name} = {value}")

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-limit:] if self.alerts else []

    def get_service_metrics(self, service_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent metrics for a service"""
        if service_name not in self.monitors:
            return []

        return self.monitors[service_name]["metrics_collected"][-limit:]

class AdvancedIntegrationManager:
    """Manager for all advanced integrations"""

    def __init__(self):
        self.github = GitHubIntegration()
        self.ci_cd = CI_CDIntegration()
        self.external_api = ExternalAPIIntegration()
        self.slack = SlackIntegration()
        self.database = DatabaseIntegration()
        self.cloud = CloudIntegration()
        self.monitoring = MonitoringIntegration()

        self.integrations = {
            "github": self.github,
            "ci_cd": self.ci_cd,
            "external_api": self.external_api,
            "slack": self.slack,
            "database": self.database,
            "cloud": self.cloud,
            "monitoring": self.monitoring
        }

    async def initialize_all(self):
        """Initialize all integrations"""
        init_tasks = []

        # Initialize integrations that need it
        for name, integration in self.integrations.items():
            if hasattr(integration, 'initialize'):
                init_tasks.append(integration.initialize())

        if init_tasks:
            await asyncio.gather(*init_tasks, return_exceptions=True)

        logger.info("All advanced integrations initialized")

    async def shutdown_all(self):
        """Shutdown all integrations"""
        shutdown_tasks = []

        for name, integration in self.integrations.items():
            if hasattr(integration, 'close'):
                shutdown_tasks.append(integration.close())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        logger.info("All advanced integrations shut down")

    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        return {
            integration_name: {
                "status": "active" if hasattr(integration, 'session') and integration.session else "configured",
                "type": type(integration).__name__
            }
            for integration_name, integration in self.integrations.items()
        }

    async def execute_integration_method(self, integration_name: str, method_name: str, *args, **kwargs) -> Any:
        """Execute a method on a specific integration"""
        if integration_name not in self.integrations:
            raise ValueError(f"Integration not found: {integration_name}")

        integration = self.integrations[integration_name]

        if not hasattr(integration, method_name):
            raise ValueError(f"Method {method_name} not found on integration {integration_name}")

        method = getattr(integration, method_name)

        try:
            if asyncio.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {method_name} on {integration_name}: {str(e)}")
            raise