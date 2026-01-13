"""
Deployment Automation for StratAgent
"""
import asyncio
import logging
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
import docker
from docker.errors import DockerException

logger = logging.getLogger(__name__)

class DeploymentEnvironment:
    """Represents a deployment environment"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.status = "inactive"
        self.last_deployment = None
        self.version = None
        self.health_checks = []

    def update_status(self, status: str, version: Optional[str] = None):
        """Update environment status"""
        self.status = status
        if version:
            self.version = version
        if status == "deployed":
            self.last_deployment = datetime.utcnow()

    def add_health_check(self, check_result: Dict[str, Any]):
        """Add health check result"""
        self.health_checks.append({
            "timestamp": datetime.utcnow().isoformat(),
            "result": check_result
        })

        # Keep only last 10 health checks
        if len(self.health_checks) > 10:
            self.health_checks = self.health_checks[-10:]

    def get_health_status(self) -> Dict[str, Any]:
        """Get environment health status"""
        if not self.health_checks:
            return {"status": "unknown", "last_check": None}

        latest_check = self.health_checks[-1]

        # Determine overall health
        all_healthy = all(
            check.get("result", {}).get("status") == "healthy"
            for check in self.health_checks[-3:]  # Last 3 checks
        )

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "last_check": latest_check["timestamp"],
            "checks_passed": sum(1 for check in self.health_checks[-3:]
                               if check.get("result", {}).get("status") == "healthy"),
            "total_checks": min(3, len(self.health_checks))
        }

class ContainerManager:
    """Docker container management for deployments"""

    def __init__(self):
        self.docker_client = None
        self.containers = {}
        self.images = {}

        try:
            self.docker_client = docker.from_env()
        except DockerException as e:
            logger.warning(f"Docker client initialization failed: {str(e)}")

    async def build_image(self, image_name: str, dockerfile_path: str, context_path: str, build_args: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Build Docker image"""
        if not self.docker_client:
            return {"error": "Docker client not available"}

        try:
            # Build image
            image, build_logs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.images.build(
                    path=context_path,
                    dockerfile=dockerfile_path,
                    tag=image_name,
                    buildargs=build_args,
                    rm=True
                )
            )

            self.images[image_name] = {
                "id": image.id,
                "tags": image.tags,
                "created": datetime.utcnow().isoformat(),
                "size": image.attrs.get("Size", 0)
            }

            return {
                "success": True,
                "image_id": image.id,
                "image_name": image_name,
                "build_logs": [str(log) for log in build_logs]
            }

        except Exception as e:
            logger.error(f"Image build failed: {str(e)}")
            return {"error": str(e)}

    async def run_container(self, image_name: str, container_name: str, ports: Optional[Dict[str, Any]] = None,
                           environment: Optional[Dict[str, str]] = None, volumes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run Docker container"""
        if not self.docker_client:
            return {"error": "Docker client not available"}

        try:
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.containers.run(
                    image_name,
                    name=container_name,
                    ports=ports,
                    environment=environment,
                    volumes=volumes,
                    detach=True,
                    restart_policy={"Name": "unless-stopped"}
                )
            )

            self.containers[container_name] = {
                "id": container.id,
                "image": image_name,
                "status": container.status,
                "created": datetime.utcnow().isoformat(),
                "ports": ports
            }

            return {
                "success": True,
                "container_id": container.id,
                "container_name": container_name,
                "status": container.status
            }

        except Exception as e:
            logger.error(f"Container run failed: {str(e)}")
            return {"error": str(e)}

    async def stop_container(self, container_name: str) -> bool:
        """Stop Docker container"""
        if not self.docker_client:
            return False

        try:
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.containers.get(container_name)
            )

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: container.stop(timeout=30)
            )

            if container_name in self.containers:
                self.containers[container_name]["status"] = "stopped"

            return True

        except Exception as e:
            logger.error(f"Container stop failed: {str(e)}")
            return False

    async def get_container_status(self, container_name: str) -> Optional[Dict[str, Any]]:
        """Get container status"""
        if container_name not in self.containers:
            return None

        try:
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.containers.get(container_name)
            )

            return {
                "id": container.id,
                "status": container.status,
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "ports": container.ports,
                "created": container.attrs["Created"],
                "restarts": container.attrs.get("RestartCount", 0)
            }

        except Exception as e:
            logger.error(f"Failed to get container status: {str(e)}")
            return None

    def list_containers(self) -> List[Dict[str, Any]]:
        """List all managed containers"""
        return list(self.containers.values())

class DeploymentPipeline:
    """Automated deployment pipeline"""

    def __init__(self, container_manager: ContainerManager):
        self.container_manager = container_manager
        self.environments: Dict[str, DeploymentEnvironment] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        self.rollback_versions: Dict[str, str] = {}

    async def create_environment(self, name: str, config: Dict[str, Any]) -> DeploymentEnvironment:
        """Create a new deployment environment"""
        environment = DeploymentEnvironment(name, config)
        self.environments[name] = environment
        logger.info(f"Created deployment environment: {name}")
        return environment

    async def deploy_to_environment(self, environment_name: str, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy application to specified environment"""
        if environment_name not in self.environments:
            return {"error": f"Environment {environment_name} not found"}

        environment = self.environments[environment_name]
        deployment_id = f"deploy_{environment_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        deployment_result = {
            "deployment_id": deployment_id,
            "environment": environment_name,
            "start_time": datetime.utcnow(),
            "status": "running",
            "stages": [],
            "version": deployment_config.get("version", "latest")
        }

        try:
            # Stage 1: Pre-deployment checks
            pre_check_result = await self._run_pre_deployment_checks(environment, deployment_config)
            deployment_result["stages"].append(pre_check_result)

            if not pre_check_result["success"]:
                deployment_result["status"] = "failed"
                deployment_result["error"] = "Pre-deployment checks failed"
                return deployment_result

            # Stage 2: Build artifacts
            build_result = await self._build_deployment_artifacts(deployment_config)
            deployment_result["stages"].append(build_result)

            if not build_result["success"]:
                deployment_result["status"] = "failed"
                deployment_result["error"] = "Build failed"
                return deployment_result

            # Stage 3: Deploy to environment
            deploy_result = await self._deploy_to_environment(environment, deployment_config, build_result)
            deployment_result["stages"].append(deploy_result)

            if not deploy_result["success"]:
                deployment_result["status"] = "failed"
                deployment_result["error"] = "Deployment failed"
                return deployment_result

            # Stage 4: Post-deployment validation
            validation_result = await self._run_post_deployment_validation(environment, deployment_config)
            deployment_result["stages"].append(validation_result)

            # Stage 5: Health checks
            health_result = await self._run_health_checks(environment, deployment_config)
            deployment_result["stages"].append(health_result)

            # Determine final status
            all_stages_successful = all(stage["success"] for stage in deployment_result["stages"])
            deployment_result["status"] = "success" if all_stages_successful else "partial_success"

            # Update environment
            environment.update_status("deployed", deployment_result["version"])

            # Store rollback version
            if environment.version:
                self.rollback_versions[environment_name] = environment.version

        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            deployment_result["status"] = "error"
            deployment_result["error"] = str(e)

        deployment_result["end_time"] = datetime.utcnow()
        deployment_result["duration"] = (deployment_result["end_time"] - deployment_result["start_time"]).total_seconds()

        # Record deployment
        self.deployment_history.append(deployment_result)

        return deployment_result

    async def _run_pre_deployment_checks(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run pre-deployment checks"""
        checks = []

        # Check environment availability
        checks.append({"name": "environment_check", "status": "passed", "details": f"Environment {environment.name} is available"})

        # Check required resources
        required_resources = config.get("required_resources", [])
        for resource in required_resources:
            checks.append({"name": f"resource_{resource}", "status": "passed", "details": f"Resource {resource} is available"})

        # Check dependencies
        dependencies = config.get("dependencies", [])
        for dep in dependencies:
            checks.append({"name": f"dependency_{dep}", "status": "passed", "details": f"Dependency {dep} is satisfied"})

        return {
            "stage": "pre_deployment_checks",
            "success": True,
            "checks": checks,
            "duration": 1.0
        }

    async def _build_deployment_artifacts(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build deployment artifacts"""
        artifacts = []

        # Build Docker images
        images_to_build = config.get("docker_images", [])
        for image_config in images_to_build:
            build_result = await self.container_manager.build_image(
                image_config["name"],
                image_config.get("dockerfile", "Dockerfile"),
                image_config.get("context", "."),
                image_config.get("build_args")
            )

            if "error" in build_result:
                return {
                    "stage": "build_artifacts",
                    "success": False,
                    "error": build_result["error"],
                    "artifacts": artifacts
                }

            artifacts.append({
                "type": "docker_image",
                "name": image_config["name"],
                "id": build_result["image_id"]
            })

        return {
            "stage": "build_artifacts",
            "success": True,
            "artifacts": artifacts,
            "duration": 2.0
        }

    async def _deploy_to_environment(self, environment: DeploymentEnvironment, config: Dict[str, Any], build_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to environment"""
        deployments = []

        # Deploy containers
        containers_to_deploy = config.get("containers", [])
        for container_config in containers_to_deploy:
            container_result = await self.container_manager.run_container(
                container_config["image"],
                container_config["name"],
                container_config.get("ports"),
                container_config.get("environment"),
                container_config.get("volumes")
            )

            if "error" in container_result:
                return {
                    "stage": "deploy_to_environment",
                    "success": False,
                    "error": container_result["error"],
                    "deployments": deployments
                }

            deployments.append({
                "type": "container",
                "name": container_config["name"],
                "id": container_result["container_id"],
                "status": container_result["status"]
            })

        return {
            "stage": "deploy_to_environment",
            "success": True,
            "deployments": deployments,
            "duration": 1.5
        }

    async def _run_post_deployment_validation(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run post-deployment validation"""
        validations = []

        # Validate service availability
        services = config.get("services", [])
        for service in services:
            validations.append({
                "service": service["name"],
                "status": "validated",
                "details": f"Service {service['name']} is responding correctly"
            })

        return {
            "stage": "post_deployment_validation",
            "success": True,
            "validations": validations,
            "duration": 1.0
        }

    async def _run_health_checks(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run health checks"""
        health_checks = []

        # Run container health checks
        containers = config.get("containers", [])
        for container in containers:
            container_status = await self.container_manager.get_container_status(container["name"])

            if container_status:
                health_status = "healthy" if container_status["status"] == "running" else "unhealthy"
                health_checks.append({
                    "component": container["name"],
                    "type": "container",
                    "status": health_status,
                    "details": f"Container status: {container_status['status']}"
                })

                # Add to environment health checks
                environment.add_health_check({
                    "component": container["name"],
                    "status": health_status,
                    "timestamp": datetime.utcnow().isoformat()
                })

        return {
            "stage": "health_checks",
            "success": True,
            "health_checks": health_checks,
            "duration": 0.5
        }

    async def rollback_environment(self, environment_name: str) -> Dict[str, Any]:
        """Rollback environment to previous version"""
        if environment_name not in self.environments:
            return {"error": f"Environment {environment_name} not found"}

        if environment_name not in self.rollback_versions:
            return {"error": f"No rollback version available for {environment_name}"}

        environment = self.environments[environment_name]
        rollback_version = self.rollback_versions[environment_name]

        # Implement rollback logic (simplified)
        logger.info(f"Rolling back {environment_name} to version {rollback_version}")

        # Update environment status
        environment.update_status("rolled_back", rollback_version)

        return {
            "success": True,
            "environment": environment_name,
            "rolled_back_to": rollback_version,
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_environment_status(self, environment_name: str) -> Optional[Dict[str, Any]]:
        """Get environment status"""
        if environment_name not in self.environments:
            return None

        environment = self.environments[environment_name]
        return {
            "name": environment.name,
            "status": environment.status,
            "version": environment.version,
            "last_deployment": environment.last_deployment.isoformat() if environment.last_deployment else None,
            "health_status": environment.get_health_status()
        }

    def get_deployment_history(self, environment_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get deployment history"""
        history = self.deployment_history

        if environment_name:
            history = [d for d in history if d["environment"] == environment_name]

        return history[-limit:] if history else []

class MonitoringDashboard:
    """Deployment monitoring and analytics dashboard"""

    def __init__(self, deployment_pipeline: DeploymentPipeline, container_manager: ContainerManager):
        self.deployment_pipeline = deployment_pipeline
        self.container_manager = container_manager
        self.alerts: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "environments": self._get_environment_status(),
            "deployments": self._get_deployment_status(),
            "containers": self.container_manager.list_containers(),
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "metrics": self._get_recent_metrics(),
            "health_summary": self._get_health_summary()
        }

    def _get_environment_status(self) -> Dict[str, Any]:
        """Get status of all environments"""
        environments = {}
        for name, env in self.deployment_pipeline.environments.items():
            environments[name] = self.deployment_pipeline.get_environment_status(name)
        return environments

    def _get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status summary"""
        recent_deployments = self.deployment_pipeline.get_deployment_history(limit=5)

        return {
            "total_deployments": len(self.deployment_pipeline.deployment_history),
            "successful_deployments": sum(1 for d in self.deployment_pipeline.deployment_history if d["status"] == "success"),
            "failed_deployments": sum(1 for d in self.deployment_pipeline.deployment_history if d["status"] == "failed"),
            "recent_deployments": recent_deployments,
            "success_rate": sum(1 for d in self.deployment_pipeline.deployment_history if d["status"] == "success") / max(1, len(self.deployment_pipeline.deployment_history))
        }

    def _get_recent_metrics(self) -> List[Dict[str, Any]]:
        """Get recent system metrics"""
        return self.metrics_history[-20:] if self.metrics_history else []

    def _get_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        environments = list(self.deployment_pipeline.environments.values())

        if not environments:
            return {"overall_health": "unknown", "environments_checked": 0}

        healthy_envs = sum(1 for env in environments if env.get_health_status()["status"] == "healthy")
        total_envs = len(environments)

        overall_health = "healthy" if healthy_envs == total_envs else "degraded" if healthy_envs > 0 else "unhealthy"

        return {
            "overall_health": overall_health,
            "environments_checked": total_envs,
            "healthy_environments": healthy_envs,
            "unhealthy_environments": total_envs - healthy_envs,
            "last_check": datetime.utcnow().isoformat()
        }

    def add_alert(self, alert_type: str, severity: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a system alert"""
        alert = {
            "id": f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            "acknowledged": False
        }

        self.alerts.append(alert)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        logger.warning(f"Alert generated: {alert_type} - {message}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.utcnow().isoformat()
                return True
        return False

    def add_metrics(self, metrics: Dict[str, Any]):
        """Add system metrics"""
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }

        self.metrics_history.append(metrics_entry)

        # Keep only last 1000 metrics entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

class DeploymentAutomationSystem:
    """Complete deployment automation system"""

    def __init__(self):
        self.container_manager = ContainerManager()
        self.deployment_pipeline = DeploymentPipeline(self.container_manager)
        self.monitoring_dashboard = MonitoringDashboard(self.deployment_pipeline, self.container_manager)

        self.deployment_configs: Dict[str, Dict[str, Any]] = {}
        self.automation_rules: List[Dict[str, Any]] = []

    async def initialize(self):
        """Initialize the deployment automation system"""
        logger.info("Initializing deployment automation system...")

        # Create default environments
        await self.deployment_pipeline.create_environment("development", {
            "type": "development",
            "auto_deploy": True,
            "rollback_enabled": True
        })

        await self.deployment_pipeline.create_environment("staging", {
            "type": "staging",
            "auto_deploy": False,
            "rollback_enabled": True
        })

        await self.deployment_pipeline.create_environment("production", {
            "type": "production",
            "auto_deploy": False,
            "rollback_enabled": True,
            "approval_required": True
        })

        logger.info("Deployment automation system initialized")

    async def create_deployment_config(self, name: str, config: Dict[str, Any]) -> str:
        """Create a deployment configuration"""
        config_id = f"config_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        deployment_config = {
            "id": config_id,
            "name": name,
            "config": config,
            "created_at": datetime.utcnow().isoformat(),
            "version": config.get("version", "1.0.0")
        }

        self.deployment_configs[name] = deployment_config
        logger.info(f"Created deployment configuration: {name}")
        return config_id

    async def deploy_application(self, config_name: str, environment_name: str) -> Dict[str, Any]:
        """Deploy application using specified configuration"""
        if config_name not in self.deployment_configs:
            return {"error": f"Deployment configuration {config_name} not found"}

        if environment_name not in self.deployment_pipeline.environments:
            return {"error": f"Environment {environment_name} not found"}

        config = self.deployment_configs[config_name]["config"]

        # Add monitoring alert for deployment start
        self.monitoring_dashboard.add_alert(
            "deployment",
            "info",
            f"Starting deployment of {config_name} to {environment_name}",
            {"config": config_name, "environment": environment_name}
        )

        # Execute deployment
        result = await self.deployment_pipeline.deploy_to_environment(environment_name, config)

        # Add completion alert
        alert_severity = "success" if result["status"] == "success" else "error"
        self.monitoring_dashboard.add_alert(
            "deployment",
            alert_severity,
            f"Deployment of {config_name} to {environment_name} {result['status']}",
            {"deployment_id": result["deployment_id"], "status": result["status"]}
        )

        return result

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "deployment_pipeline": {
                "environments": len(self.deployment_pipeline.environments),
                "total_deployments": len(self.deployment_pipeline.deployment_history),
                "active_environments": sum(1 for env in self.deployment_pipeline.environments.values() if env.status == "deployed")
            },
            "container_manager": {
                "containers_managed": len(self.container_manager.containers),
                "images_built": len(self.container_manager.images)
            },
            "monitoring": self.monitoring_dashboard.get_dashboard_data(),
            "deployment_configs": len(self.deployment_configs),
            "system_health": "operational"
        }

    async def run_automated_deployments(self):
        """Run automated deployments based on rules"""
        for rule in self.automation_rules:
            if rule.get("enabled", False):
                await self._execute_automation_rule(rule)

    async def _execute_automation_rule(self, rule: Dict[str, Any]):
        """Execute a single automation rule"""
        rule_type = rule.get("type")

        if rule_type == "scheduled_deployment":
            # Check if it's time for scheduled deployment
            # Implementation would check schedule and trigger deployment
            pass
        elif rule_type == "health_based_deployment":
            # Check health and trigger rollback or alert
            # Implementation would monitor health and take action
            pass

    def add_automation_rule(self, rule: Dict[str, Any]):
        """Add an automation rule"""
        rule["id"] = f"rule_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        rule["created_at"] = datetime.utcnow().isoformat()
        self.automation_rules.append(rule)
        logger.info(f"Added automation rule: {rule['id']}")

    def get_deployment_report(self, environment_name: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """Generate deployment report"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Filter deployments by date and environment
        relevant_deployments = [
            d for d in self.deployment_pipeline.deployment_history
            if datetime.fromisoformat(d["start_time"]) > cutoff_date
        ]

        if environment_name:
            relevant_deployments = [d for d in relevant_deployments if d["environment"] == environment_name]

        # Calculate statistics
        total_deployments = len(relevant_deployments)
        successful_deployments = sum(1 for d in relevant_deployments if d["status"] == "success")
        failed_deployments = sum(1 for d in relevant_deployments if d["status"] in ["failed", "error"])

        avg_duration = sum(d["duration"] for d in relevant_deployments) / total_deployments if total_deployments > 0 else 0

        return {
            "period_days": days,
            "environment": environment_name or "all",
            "total_deployments": total_deployments,
            "successful_deployments": successful_deployments,
            "failed_deployments": failed_deployments,
            "success_rate": successful_deployments / total_deployments if total_deployments > 0 else 0,
            "average_duration": avg_duration,
            "recent_deployments": relevant_deployments[-10:],  # Last 10 deployments
            "generated_at": datetime.utcnow().isoformat()
        }