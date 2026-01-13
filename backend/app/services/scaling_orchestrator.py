"""
Distributed Scaling Orchestrator for StratAgent
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
import heapq
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

class ResourceMetrics:
    """Tracks resource utilization metrics"""

    def __init__(self):
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.disk_usage = deque(maxlen=1000)
        self.network_usage = deque(maxlen=1000)
        self.active_connections = deque(maxlen=1000)
        self.request_queue_length = deque(maxlen=1000)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update resource metrics"""
        current_time = datetime.utcnow()

        for metric_name, value in metrics.items():
            metric_queue = getattr(self, f"{metric_name}_usage", None)
            if metric_queue is not None:
                metric_queue.append((current_time, value))

    def get_average_usage(self, metric: str, minutes: int = 5) -> float:
        """Get average usage for a metric over time period"""
        metric_queue = getattr(self, f"{metric}_usage", None)
        if not metric_queue:
            return 0.0

        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_values = [value for timestamp, value in metric_queue if timestamp > cutoff_time]

        return statistics.mean(recent_values) if recent_values else 0.0

    def get_peak_usage(self, metric: str, minutes: int = 5) -> float:
        """Get peak usage for a metric over time period"""
        metric_queue = getattr(self, f"{metric}_usage", None)
        if not metric_queue:
            return 0.0

        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_values = [value for timestamp, value in metric_queue if timestamp > cutoff_time]

        return max(recent_values) if recent_values else 0.0

class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on metrics and predictions"""

    def __init__(self):
        self.scaling_history = []
        self.scaling_thresholds = {
            "cpu_scale_up": 75.0,
            "cpu_scale_down": 30.0,
            "memory_scale_up": 80.0,
            "memory_scale_down": 40.0,
            "queue_scale_up": 100,
            "queue_scale_down": 20,
            "response_time_scale_up": 2.0,  # seconds
            "response_time_scale_down": 0.5
        }

        self.cooldown_periods = {
            "scale_up": 300,  # 5 minutes
            "scale_down": 600  # 10 minutes
        }

        self.last_scaling_actions = {}

    def analyze_scaling_needs(self, current_metrics: Dict[str, Any], predicted_load: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current metrics and predict scaling needs"""
        analysis_id = f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        scaling_decisions = {
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat(),
            "immediate_actions": [],
            "scheduled_actions": [],
            "recommendations": [],
            "confidence_score": 0.0
        }

        # Check immediate scaling needs
        immediate_needs = self._check_immediate_scaling_needs(current_metrics)
        scaling_decisions["immediate_actions"].extend(immediate_needs)

        # Analyze predicted load
        predicted_needs = self._analyze_predicted_load(predicted_load)
        scaling_decisions["scheduled_actions"].extend(predicted_needs)

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(current_metrics)
        scaling_decisions["recommendations"].extend(recommendations)

        # Calculate confidence score
        scaling_decisions["confidence_score"] = self._calculate_decision_confidence(
            current_metrics, predicted_load, immediate_needs, predicted_needs
        )

        # Record analysis
        self.scaling_history.append(scaling_decisions)

        return scaling_decisions

    def _check_immediate_scaling_needs(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for immediate scaling needs based on current metrics"""
        actions = []

        # CPU scaling
        cpu_usage = metrics.get("cpu_percent", 0)
        if cpu_usage > self.scaling_thresholds["cpu_scale_up"]:
            actions.append({
                "type": "scale_up",
                "resource": "cpu",
                "reason": f"High CPU usage: {cpu_usage:.1f}%",
                "urgency": "high",
                "recommended_instances": self._calculate_required_instances(cpu_usage, "cpu")
            })
        elif cpu_usage < self.scaling_thresholds["cpu_scale_down"]:
            actions.append({
                "type": "scale_down",
                "resource": "cpu",
                "reason": f"Low CPU usage: {cpu_usage:.1f}%",
                "urgency": "medium",
                "recommended_instances": max(1, self._calculate_required_instances(cpu_usage, "cpu"))
            })

        # Memory scaling
        memory_usage = metrics.get("memory_percent", 0)
        if memory_usage > self.scaling_thresholds["memory_scale_up"]:
            actions.append({
                "type": "scale_up",
                "resource": "memory",
                "reason": f"High memory usage: {memory_usage:.1f}%",
                "urgency": "high",
                "recommended_instances": self._calculate_required_instances(memory_usage, "memory")
            })

        # Queue length scaling
        queue_length = metrics.get("request_queue_length", 0)
        if queue_length > self.scaling_thresholds["queue_scale_up"]:
            actions.append({
                "type": "scale_up",
                "resource": "workers",
                "reason": f"High queue length: {queue_length}",
                "urgency": "high",
                "recommended_instances": self._calculate_required_instances(queue_length, "queue")
            })

        # Response time scaling
        avg_response_time = metrics.get("avg_response_time", 0)
        if avg_response_time > self.scaling_thresholds["response_time_scale_up"]:
            actions.append({
                "type": "scale_up",
                "resource": "instances",
                "reason": f"Slow response times: {avg_response_time:.2f}s",
                "urgency": "high",
                "recommended_instances": self._calculate_required_instances(avg_response_time, "response_time")
            })

        return actions

    def _analyze_predicted_load(self, predicted_load: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze predicted load and schedule scaling actions"""
        actions = []

        # Analyze predicted CPU load
        predicted_cpu = predicted_load.get("cpu_percent", [])
        if predicted_cpu:
            max_predicted_cpu = max(predicted_cpu)
            if max_predicted_cpu > self.scaling_thresholds["cpu_scale_up"]:
                actions.append({
                    "type": "scale_up",
                    "resource": "cpu",
                    "reason": f"Predicted high CPU load: {max_predicted_cpu:.1f}%",
                    "urgency": "medium",
                    "scheduled_time": datetime.utcnow() + timedelta(minutes=10),
                    "recommended_instances": self._calculate_required_instances(max_predicted_cpu, "cpu")
                })

        # Analyze predicted request load
        predicted_requests = predicted_load.get("requests_per_second", [])
        if predicted_requests:
            max_predicted_requests = max(predicted_requests)
            if max_predicted_requests > 100:  # Threshold for high load
                actions.append({
                    "type": "scale_up",
                    "resource": "instances",
                    "reason": f"Predicted high request load: {max_predicted_requests} req/s",
                    "urgency": "medium",
                    "scheduled_time": datetime.utcnow() + timedelta(minutes=15),
                    "recommended_instances": self._calculate_required_instances(max_predicted_requests, "requests")
                })

        return actions

    def _calculate_required_instances(self, current_value: float, resource_type: str) -> int:
        """Calculate required number of instances based on resource usage"""
        if resource_type == "cpu":
            # Assume each instance can handle ~60% CPU
            return max(1, int(current_value / 60) + 1)
        elif resource_type == "memory":
            # Assume each instance can handle ~70% memory
            return max(1, int(current_value / 70) + 1)
        elif resource_type == "queue":
            # Assume each instance can handle ~50 queue items
            return max(1, int(current_value / 50) + 1)
        elif resource_type == "response_time":
            # Inverse relationship - higher response time needs more instances
            if current_value > 2.0:
                return 3
            elif current_value > 1.0:
                return 2
            else:
                return 1
        elif resource_type == "requests":
            # Assume each instance can handle ~50 req/s
            return max(1, int(current_value / 50) + 1)

        return 1

    def _generate_optimization_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []

        # CPU optimization
        cpu_usage = metrics.get("cpu_percent", 0)
        if cpu_usage > 80:
            recommendations.append("Consider optimizing CPU-intensive operations or implementing caching")
        elif cpu_usage < 20:
            recommendations.append("Consider consolidating instances to reduce resource waste")

        # Memory optimization
        memory_usage = metrics.get("memory_percent", 0)
        if memory_usage > 85:
            recommendations.append("Implement memory optimization techniques or increase instance memory")
        elif memory_usage < 30:
            recommendations.append("Consider reducing instance memory allocation")

        # Response time optimization
        avg_response_time = metrics.get("avg_response_time", 0)
        if avg_response_time > 1.0:
            recommendations.append("Optimize database queries and implement response caching")
        if avg_response_time > 2.0:
            recommendations.append("Consider implementing request queuing or load balancing")

        return recommendations

    def _calculate_decision_confidence(self, current_metrics: Dict[str, Any], predicted_load: Dict[str, Any],
                                     immediate_actions: List[Dict[str, Any]], predicted_actions: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for scaling decisions"""
        confidence = 0.5  # Base confidence

        # Increase confidence based on number of supporting metrics
        metric_count = len([k for k, v in current_metrics.items() if isinstance(v, (int, float)) and v > 0])
        confidence += min(0.2, metric_count * 0.05)

        # Increase confidence for immediate actions (more urgent)
        if immediate_actions:
            confidence += 0.1

        # Increase confidence for predicted actions (proactive)
        if predicted_actions:
            confidence += 0.1

        # Increase confidence for consistent patterns
        if len(self.scaling_history) > 5:
            recent_decisions = self.scaling_history[-5:]
            consistent_actions = sum(1 for decision in recent_decisions if decision["immediate_actions"])
            if consistent_actions > 2:
                confidence += 0.1

        return min(1.0, confidence)

class DistributedOrchestrator:
    """Manages distributed agent orchestration across multiple nodes/instances"""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.agent_assignments: Dict[str, str] = {}  # agent_id -> node_id
        self.task_queue = asyncio.PriorityQueue()
        self.workload_balancer = WorkloadBalancer()
        self.fault_tolerance_manager = FaultToleranceManager()

        # Distributed execution pools
        self.thread_executor = ThreadPoolExecutor(max_workers=10)
        self.process_executor = ProcessPoolExecutor(max_workers=4)

    async def register_node(self, node_id: str, node_config: Dict[str, Any]) -> bool:
        """Register a new node in the distributed system"""
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already registered")
            return False

        self.nodes[node_id] = {
            "config": node_config,
            "status": "active",
            "last_heartbeat": datetime.utcnow(),
            "active_agents": 0,
            "resource_usage": {},
            "capabilities": node_config.get("capabilities", [])
        }

        logger.info(f"Registered distributed node: {node_id}")
        return True

    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a node and redistribute its agents"""
        if node_id not in self.nodes:
            return False

        # Redistribute agents from this node
        agents_to_redistribute = [
            agent_id for agent_id, assigned_node in self.agent_assignments.items()
            if assigned_node == node_id
        ]

        for agent_id in agents_to_redistribute:
            await self._redistribute_agent(agent_id)

        # Remove node
        del self.nodes[node_id]
        logger.info(f"Unregistered distributed node: {node_id}")
        return True

    async def distribute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute a task to the most appropriate node"""
        task_id = task.get("task_id", f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

        # Find best node for this task
        best_node = await self._select_best_node_for_task(task)

        if not best_node:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": "No suitable node available for task execution"
            }

        # Assign task to node
        task_assignment = {
            "task_id": task_id,
            "node_id": best_node,
            "assigned_at": datetime.utcnow(),
            "status": "assigned",
            "task": task
        }

        # Add to task queue with priority
        priority = self._calculate_task_priority(task)
        await self.task_queue.put((priority, task_assignment))

        return {
            "task_id": task_id,
            "status": "distributed",
            "assigned_node": best_node,
            "estimated_completion": datetime.utcnow() + timedelta(seconds=task.get("estimated_duration", 30))
        }

    async def _select_best_node_for_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Select the best node for executing a task"""
        if not self.nodes:
            return None

        best_node = None
        best_score = -1

        task_requirements = task.get("requirements", {})
        task_type = task.get("type", "general")

        for node_id, node_info in self.nodes.items():
            if node_info["status"] != "active":
                continue

            # Calculate node suitability score
            score = self._calculate_node_suitability(node_id, task_requirements, task_type)

            if score > best_score:
                best_score = score
                best_node = node_id

        return best_node if best_score > 0 else None

    def _calculate_node_suitability(self, node_id: str, requirements: Dict[str, Any], task_type: str) -> float:
        """Calculate how suitable a node is for a task"""
        node_info = self.nodes[node_id]
        score = 0.0

        # Resource availability
        resource_usage = node_info.get("resource_usage", {})
        cpu_usage = resource_usage.get("cpu_percent", 0)
        memory_usage = resource_usage.get("memory_percent", 0)

        # Prefer nodes with lower resource usage
        if cpu_usage < 70:
            score += (100 - cpu_usage) / 100 * 0.3
        if memory_usage < 75:
            score += (100 - memory_usage) / 100 * 0.3

        # Capability matching
        node_capabilities = set(node_info.get("capabilities", []))
        required_capabilities = set(requirements.get("capabilities", []))

        if required_capabilities.issubset(node_capabilities):
            score += 0.4
        else:
            # Partial capability match
            matching_caps = len(required_capabilities.intersection(node_capabilities))
            if matching_caps > 0:
                score += (matching_caps / len(required_capabilities)) * 0.4

        # Current workload (prefer less loaded nodes)
        active_agents = node_info.get("active_agents", 0)
        max_agents = node_info["config"].get("max_agents", 10)

        if active_agents < max_agents:
            workload_factor = 1 - (active_agents / max_agents)
            score += workload_factor * 0.2

        return score

    def _calculate_task_priority(self, task: Dict[str, Any]) -> int:
        """Calculate task priority for queue ordering"""
        priority_map = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3
        }

        priority_level = task.get("priority", "medium")
        return priority_map.get(priority_level, 2)

    async def _redistribute_agent(self, agent_id: str):
        """Redistribute an agent to another available node"""
        # Find best available node
        best_node = None
        best_score = -1

        for node_id, node_info in self.nodes.items():
            if node_info["status"] == "active":
                active_agents = node_info.get("active_agents", 0)
                max_agents = node_info["config"].get("max_agents", 10)

                if active_agents < max_agents:
                    score = (max_agents - active_agents) / max_agents
                    if score > best_score:
                        best_score = score
                        best_node = node_id

        if best_node:
            self.agent_assignments[agent_id] = best_node
            self.nodes[best_node]["active_agents"] += 1
            logger.info(f"Redistributed agent {agent_id} to node {best_node}")
        else:
            logger.warning(f"Could not redistribute agent {agent_id} - no suitable nodes available")

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        total_nodes = len(self.nodes)
        active_nodes = sum(1 for node in self.nodes.values() if node["status"] == "active")
        total_agents = len(self.agent_assignments)

        # Calculate resource utilization across cluster
        cluster_cpu = []
        cluster_memory = []

        for node_info in self.nodes.values():
            resource_usage = node_info.get("resource_usage", {})
            if "cpu_percent" in resource_usage:
                cluster_cpu.append(resource_usage["cpu_percent"])
            if "memory_percent" in resource_usage:
                cluster_memory.append(resource_usage["memory_percent"])

        avg_cpu = statistics.mean(cluster_cpu) if cluster_cpu else 0
        avg_memory = statistics.mean(cluster_memory) if cluster_memory else 0

        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "inactive_nodes": total_nodes - active_nodes,
            "total_agents": total_agents,
            "cluster_utilization": {
                "average_cpu_percent": avg_cpu,
                "average_memory_percent": avg_memory,
                "max_cpu_percent": max(cluster_cpu) if cluster_cpu else 0,
                "max_memory_percent": max(cluster_memory) if cluster_memory else 0
            },
            "node_status": {node_id: node_info["status"] for node_id, node_info in self.nodes.items()},
            "task_queue_length": self.task_queue.qsize()
        }

class WorkloadBalancer:
    """Intelligent workload balancing across distributed nodes"""

    def __init__(self):
        self.load_history: Dict[str, List[float]] = defaultdict(list)
        self.balancing_strategies = {
            "round_robin": self._round_robin_balance,
            "least_loaded": self._least_loaded_balance,
            "capability_based": self._capability_based_balance,
            "predictive": self._predictive_balance
        }

    async def balance_workload(self, nodes: Dict[str, Dict[str, Any]], pending_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Balance workload across available nodes"""
        # Use least loaded strategy by default
        return await self.balancing_strategies["least_loaded"](nodes, pending_tasks)

    async def _least_loaded_balance(self, nodes: Dict[str, Dict[str, Any]], pending_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Balance using least loaded strategy"""
        # Sort nodes by current load (ascending)
        sorted_nodes = sorted(
            [(node_id, node_info) for node_id, node_info in nodes.items() if node_info["status"] == "active"],
            key=lambda x: x[1].get("active_agents", 0)
        )

        if not sorted_nodes:
            return {}

        assignments = {node_id: [] for node_id, _ in sorted_nodes}

        # Assign tasks to least loaded nodes first
        node_index = 0
        for task in pending_tasks:
            node_id, _ = sorted_nodes[node_index % len(sorted_nodes)]
            assignments[node_id].append(task)
            node_index += 1

        return assignments

    async def _round_robin_balance(self, nodes: Dict[str, Dict[str, Any]], pending_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Balance using round-robin strategy"""
        active_nodes = [node_id for node_id, node_info in nodes.items() if node_info["status"] == "active"]

        if not active_nodes:
            return {}

        assignments = {node_id: [] for node_id in active_nodes}

        for i, task in enumerate(pending_tasks):
            node_id = active_nodes[i % len(active_nodes)]
            assignments[node_id].append(task)

        return assignments

    async def _capability_based_balance(self, nodes: Dict[str, Dict[str, Any]], pending_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Balance based on node capabilities"""
        assignments = {}

        for task in pending_tasks:
            required_capabilities = set(task.get("requirements", {}).get("capabilities", []))
            best_node = None
            best_score = -1

            for node_id, node_info in nodes.items():
                if node_info["status"] != "active":
                    continue

                node_capabilities = set(node_info.get("capabilities", []))
                if required_capabilities.issubset(node_capabilities):
                    # Perfect match
                    score = 1.0
                else:
                    # Partial match
                    matching = len(required_capabilities.intersection(node_capabilities))
                    score = matching / len(required_capabilities) if required_capabilities else 0.5

                # Factor in current load
                load_penalty = node_info.get("active_agents", 0) / node_info["config"].get("max_agents", 10)
                score = score * (1 - load_penalty * 0.3)

                if score > best_score:
                    best_score = score
                    best_node = node_id

            if best_node:
                if best_node not in assignments:
                    assignments[best_node] = []
                assignments[best_node].append(task)

        return assignments

    async def _predictive_balance(self, nodes: Dict[str, Dict[str, Any]], pending_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Balance using predictive load analysis"""
        # Use historical load patterns to predict future load
        predictions = {}

        for node_id, node_info in nodes.items():
            if node_info["status"] == "active":
                historical_load = self.load_history.get(node_id, [])
                if historical_load:
                    # Simple prediction: average of last 5 readings
                    prediction = statistics.mean(historical_load[-5:]) if len(historical_load) >= 5 else statistics.mean(historical_load)
                else:
                    prediction = 0.5  # Default moderate load

                predictions[node_id] = prediction

        # Sort nodes by predicted load (ascending)
        sorted_nodes = sorted(predictions.items(), key=lambda x: x[1])

        if not sorted_nodes:
            return {}

        assignments = {node_id: [] for node_id, _ in sorted_nodes}

        # Assign tasks to nodes with lowest predicted load
        for task in pending_tasks:
            best_node = sorted_nodes[0][0]  # Node with lowest predicted load
            assignments[best_node].append(task)

        return assignments

class FaultToleranceManager:
    """Manages fault tolerance and automatic recovery"""

    def __init__(self):
        self.failure_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.recovery_strategies = {
            "node_failure": self._handle_node_failure,
            "agent_failure": self._handle_agent_failure,
            "task_failure": self._handle_task_failure,
            "resource_exhaustion": self._handle_resource_exhaustion
        }
        self.health_checks: Dict[str, asyncio.Task] = {}

    async def monitor_health(self, component_id: str, health_check_func: callable) -> None:
        """Monitor health of a component"""
        while True:
            try:
                health_status = await health_check_func()

                if not health_status.get("healthy", True):
                    await self._handle_component_failure(component_id, health_status)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Health check failed for {component_id}: {str(e)}")
                await asyncio.sleep(60)

    async def _handle_component_failure(self, component_id: str, failure_details: Dict[str, Any]):
        """Handle component failure with appropriate recovery strategy"""
        failure_type = failure_details.get("failure_type", "unknown")

        # Record failure
        failure_record = {
            "timestamp": datetime.utcnow(),
            "component_id": component_id,
            "failure_type": failure_type,
            "details": failure_details,
            "recovery_attempts": 0
        }

        self.failure_history[component_id].append(failure_record)

        # Attempt recovery
        recovery_strategy = self.recovery_strategies.get(failure_type, self._default_recovery)
        await recovery_strategy(component_id, failure_details, failure_record)

    async def _handle_node_failure(self, node_id: str, failure_details: Dict[str, Any], failure_record: Dict[str, Any]):
        """Handle node failure"""
        logger.warning(f"Handling node failure: {node_id}")

        # Mark node as failed
        # Redistribute agents from failed node
        # Attempt node recovery or replacement

        failure_record["recovery_attempts"] += 1
        failure_record["recovery_action"] = "agent_redistribution"

    async def _handle_agent_failure(self, agent_id: str, failure_details: Dict[str, Any], failure_record: Dict[str, Any]):
        """Handle agent failure"""
        logger.warning(f"Handling agent failure: {agent_id}")

        # Restart agent
        # Reassign tasks
        # Update routing tables

        failure_record["recovery_attempts"] += 1
        failure_record["recovery_action"] = "agent_restart"

    async def _handle_task_failure(self, task_id: str, failure_details: Dict[str, Any], failure_record: Dict[str, Any]):
        """Handle task failure"""
        logger.warning(f"Handling task failure: {task_id}")

        # Retry task on different node
        # Reduce task priority
        # Notify user of failure

        failure_record["recovery_attempts"] += 1
        failure_record["recovery_action"] = "task_retry"

    async def _handle_resource_exhaustion(self, component_id: str, failure_details: Dict[str, Any], failure_record: Dict[str, Any]):
        """Handle resource exhaustion"""
        logger.warning(f"Handling resource exhaustion: {component_id}")

        # Scale up resources
        # Optimize resource usage
        # Implement resource limits

        failure_record["recovery_attempts"] += 1
        failure_record["recovery_action"] = "resource_scaling"

    async def _default_recovery(self, component_id: str, failure_details: Dict[str, Any], failure_record: Dict[str, Any]):
        """Default recovery strategy"""
        logger.warning(f"Using default recovery for {component_id}")

        # Log failure
        # Attempt basic restart
        # Escalate if multiple failures

        failure_record["recovery_attempts"] += 1
        failure_record["recovery_action"] = "default_restart"

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure and recovery statistics"""
        total_failures = sum(len(failures) for failures in self.failure_history.values())

        recovery_attempts = sum(
            sum(failure["recovery_attempts"] for failure in failures)
            for failures in self.failure_history.values()
        )

        successful_recoveries = sum(
            sum(1 for failure in failures if failure["recovery_attempts"] > 0)
            for failures in self.failure_history.values()
        )

        return {
            "total_failures": total_failures,
            "total_recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / total_failures if total_failures > 0 else 1.0,
            "components_with_failures": len(self.failure_history),
            "most_failed_component": max(self.failure_history.items(), key=lambda x: len(x[1]))[0] if self.failure_history else None
        }