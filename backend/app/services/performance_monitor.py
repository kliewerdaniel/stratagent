"""
Performance Monitor - System monitoring and analytics dashboard
"""
import asyncio
import logging
import psutil
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects system and application metrics"""

    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 readings
        self.start_time = datetime.utcnow()

        # Initialize system metrics
        self._initialize_system_metrics()

    def _initialize_system_metrics(self):
        """Initialize system metrics collection"""
        self.system_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used_mb": 0.0,
            "memory_total_mb": 0.0,
            "disk_usage_percent": 0.0,
            "network_bytes_sent": 0,
            "network_bytes_recv": 0
        }

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # Network usage (simplified)
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "disk_usage_percent": disk_usage_percent,
                "network_bytes_sent": network_bytes_sent,
                "network_bytes_recv": network_bytes_recv,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Store in history
            for key, value in metrics.items():
                if key != "timestamp":
                    self.metrics_history[key].append((datetime.utcnow(), value))

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return {}

    def collect_application_metrics(self, app_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        metrics = {
            "active_users": app_stats.get("active_users", 0),
            "total_requests": app_stats.get("total_requests", 0),
            "error_rate": app_stats.get("error_rate", 0.0),
            "avg_response_time": app_stats.get("avg_response_time", 0.0),
            "active_conversations": app_stats.get("active_conversations", 0),
            "agents_active": app_stats.get("agents_active", 0),
            "knowledge_nodes": app_stats.get("knowledge_nodes", 0),
            "timestamp": datetime.utcnow().isoformat()
        }

        # Store in history
        for key, value in metrics.items():
            if key != "timestamp":
                self.metrics_history[key].append((datetime.utcnow(), value))

        return metrics

    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the specified time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)

        summary = {}

        for metric_name, history in self.metrics_history.items():
            # Filter data within time window
            recent_data = [(timestamp, value) for timestamp, value in history if timestamp >= cutoff_time]

            if recent_data:
                values = [value for _, value in recent_data]

                summary[metric_name] = {
                    "current": values[-1],
                    "average": statistics.mean(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "count": len(values),
                    "trend": self._calculate_trend(values)
                }

                # Add percentiles for numeric metrics
                if all(isinstance(v, (int, float)) for v in values):
                    try:
                        summary[metric_name]["percentile_95"] = statistics.quantiles(values, n=20)[18]  # 95th percentile
                        summary[metric_name]["percentile_99"] = statistics.quantiles(values, n=100)[98]  # 99th percentile
                    except statistics.StatisticsError:
                        pass

        return summary

    def _calculate_trend(self, values: List[float], window_size: int = 10) -> str:
        """Calculate trend direction"""
        if len(values) < window_size * 2:
            return "insufficient_data"

        # Compare recent values with older values
        midpoint = len(values) // 2
        recent_avg = statistics.mean(values[midpoint:])
        older_avg = statistics.mean(values[:midpoint])

        if recent_avg > older_avg * 1.05:  # 5% increase
            return "increasing"
        elif recent_avg < older_avg * 0.95:  # 5% decrease
            return "decreasing"
        else:
            return "stable"

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        summary = self.get_metrics_summary(5)  # Last 5 minutes

        health_status = "healthy"
        issues = []

        # Check CPU usage
        if summary.get("cpu_percent", {}).get("current", 0) > 90:
            health_status = "critical"
            issues.append("High CPU usage (>90%)")
        elif summary.get("cpu_percent", {}).get("current", 0) > 75:
            health_status = max(health_status, "warning")
            issues.append("Elevated CPU usage (>75%)")

        # Check memory usage
        if summary.get("memory_percent", {}).get("current", 0) > 90:
            health_status = "critical"
            issues.append("High memory usage (>90%)")
        elif summary.get("memory_percent", {}).get("current", 0) > 80:
            health_status = max(health_status, "warning")
            issues.append("Elevated memory usage (>80%)")

        # Check disk usage
        if summary.get("disk_usage_percent", {}).get("current", 0) > 95:
            health_status = "critical"
            issues.append("Critical disk usage (>95%)")
        elif summary.get("disk_usage_percent", {}).get("current", 0) > 85:
            health_status = max(health_status, "warning")
            issues.append("High disk usage (>85%)")

        # Check application health
        if summary.get("error_rate", {}).get("current", 0) > 0.1:  # 10% error rate
            health_status = max(health_status, "warning")
            issues.append("High error rate detected")

        if summary.get("avg_response_time", {}).get("current", 0) > 10.0:  # 10 seconds
            health_status = max(health_status, "warning")
            issues.append("Slow response times detected")

        return {
            "status": health_status,
            "issues": issues,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "last_check": datetime.utcnow().isoformat()
        }

class PerformanceMonitor:
    """Comprehensive performance monitoring and analytics system"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_alerts = []
        self.alert_thresholds = self._load_default_thresholds()

        # Analytics storage
        self.performance_analytics = defaultdict(list)
        self.system_events = []

        # Monitoring configuration
        self.monitoring_enabled = True
        self.alert_cooldown_minutes = 5
        self.last_alert_times = {}

    def _load_default_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Load default alert thresholds"""
        return {
            "cpu_percent": {"warning": 75, "critical": 90},
            "memory_percent": {"warning": 80, "critical": 90},
            "disk_usage_percent": {"warning": 85, "critical": 95},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "avg_response_time": {"warning": 5.0, "critical": 10.0},
            "active_users": {"warning": 1000, "critical": 5000}  # Example thresholds
        }

    async def start_monitoring(self):
        """Start continuous monitoring"""
        if not self.monitoring_enabled:
            return

        logger.info("Starting performance monitoring...")

        while self.monitoring_enabled:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()

                # Check for alerts
                await self._check_alerts(system_metrics)

                # Update analytics
                self._update_analytics(system_metrics)

                # Wait for next collection interval
                await asyncio.sleep(self.metrics_collector.collection_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    def update_application_metrics(self, app_stats: Dict[str, Any]):
        """Update application-specific metrics"""
        app_metrics = self.metrics_collector.collect_application_metrics(app_stats)

        # Check for application alerts
        asyncio.create_task(self._check_alerts(app_metrics))

        # Update analytics
        self._update_analytics(app_metrics)

    async def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds"""
        current_time = datetime.utcnow()

        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]

                # Check if we should skip due to cooldown
                last_alert_key = f"{metric_name}_warning"
                if last_alert_key in self.last_alert_times:
                    time_since_last = (current_time - self.last_alert_times[last_alert_key]).total_seconds() / 60
                    if time_since_last < self.alert_cooldown_minutes:
                        continue

                # Check warning threshold
                if value >= thresholds["critical"]:
                    alert = {
                        "type": "critical",
                        "metric": metric_name,
                        "value": value,
                        "threshold": thresholds["critical"],
                        "message": f"Critical: {metric_name} is {value} (threshold: {thresholds['critical']})",
                        "timestamp": current_time.isoformat()
                    }
                    self.performance_alerts.append(alert)
                    self.last_alert_times[last_alert_key] = current_time
                    logger.warning(alert["message"])

                elif value >= thresholds["warning"]:
                    alert = {
                        "type": "warning",
                        "metric": metric_name,
                        "value": value,
                        "threshold": thresholds["warning"],
                        "message": f"Warning: {metric_name} is {value} (threshold: {thresholds['warning']})",
                        "timestamp": current_time.isoformat()
                    }
                    self.performance_alerts.append(alert)
                    self.last_alert_times[last_alert_key] = current_time
                    logger.warning(alert["message"])

    def _update_analytics(self, metrics: Dict[str, Any]):
        """Update performance analytics"""
        timestamp = datetime.utcnow()

        for metric_name, value in metrics.items():
            if metric_name != "timestamp":
                self.performance_analytics[metric_name].append({
                    "timestamp": timestamp,
                    "value": value
                })

                # Keep only recent data (last 24 hours)
                cutoff = timestamp - timedelta(hours=24)
                self.performance_analytics[metric_name] = [
                    entry for entry in self.performance_analytics[metric_name]
                    if entry["timestamp"] > cutoff
                ]

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "system_health": self.metrics_collector.get_health_status(),
            "metrics_summary": self.metrics_collector.get_metrics_summary(),
            "recent_alerts": self.performance_alerts[-10:],  # Last 10 alerts
            "analytics": self._get_analytics_summary(),
            "performance_trends": self._calculate_performance_trends(),
            "system_events": self.system_events[-20:]  # Last 20 events
        }

    def _get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        summary = {}

        for metric_name, data_points in self.performance_analytics.items():
            if data_points:
                values = [point["value"] for point in data_points]

                # Calculate statistics
                try:
                    summary[metric_name] = {
                        "data_points": len(data_points),
                        "average": statistics.mean(values),
                        "trend": self._calculate_analytics_trend(data_points),
                        "volatility": self._calculate_volatility(values),
                        "time_range": {
                            "start": data_points[0]["timestamp"].isoformat(),
                            "end": data_points[-1]["timestamp"].isoformat()
                        }
                    }
                except Exception as e:
                    logger.warning(f"Error calculating analytics for {metric_name}: {str(e)}")

        return summary

    def _calculate_analytics_trend(self, data_points: List[Dict[str, Any]]) -> str:
        """Calculate trend from analytics data"""
        if len(data_points) < 10:
            return "insufficient_data"

        # Split into two halves and compare averages
        midpoint = len(data_points) // 2
        first_half = [point["value"] for point in data_points[:midpoint]]
        second_half = [point["value"] for point in data_points[midpoint:]]

        if not first_half or not second_half:
            return "insufficient_data"

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        ratio = second_avg / first_avg if first_avg != 0 else 1

        if ratio > 1.05:
            return "increasing"
        elif ratio < 0.95:
            return "decreasing"
        else:
            return "stable"

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (coefficient of variation)"""
        if len(values) < 2:
            return 0.0

        try:
            mean = statistics.mean(values)
            if mean == 0:
                return 0.0

            std_dev = statistics.stdev(values)
            return std_dev / mean
        except statistics.StatisticsError:
            return 0.0

    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate overall performance trends"""
        trends = {}

        # System performance trends
        system_metrics = ["cpu_percent", "memory_percent", "disk_usage_percent"]
        for metric in system_metrics:
            if metric in self.performance_analytics:
                trends[f"system_{metric}"] = self._calculate_analytics_trend(
                    self.performance_analytics[metric]
                )

        # Application performance trends
        app_metrics = ["error_rate", "avg_response_time", "active_users"]
        for metric in app_metrics:
            if metric in self.performance_analytics:
                trends[f"app_{metric}"] = self._calculate_analytics_trend(
                    self.performance_analytics[metric]
                )

        return trends

    def log_system_event(self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a system event"""
        event = {
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        self.system_events.append(event)

        # Keep only recent events
        if len(self.system_events) > 1000:
            self.system_events = self.system_events[-1000:]

        logger.info(f"System event logged: {event_type} - {message}")

    def set_alert_threshold(self, metric: str, warning_threshold: float, critical_threshold: float):
        """Set custom alert thresholds for a metric"""
        self.alert_thresholds[metric] = {
            "warning": warning_threshold,
            "critical": critical_threshold
        }
        logger.info(f"Updated alert thresholds for {metric}: warning={warning_threshold}, critical={critical_threshold}")

    def get_performance_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)

        # Filter data within time window
        filtered_analytics = {}
        for metric_name, data_points in self.performance_analytics.items():
            filtered_data = [point for point in data_points if point["timestamp"] > cutoff]
            if filtered_data:
                filtered_analytics[metric_name] = filtered_data

        # Generate report sections
        report = {
            "time_window": f"{time_window_hours} hours",
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_metrics_tracked": len(filtered_analytics),
                "total_data_points": sum(len(points) for points in filtered_analytics.values()),
                "alerts_generated": len([a for a in self.performance_alerts
                                       if datetime.fromisoformat(a["timestamp"]) > cutoff])
            },
            "system_performance": self._generate_system_performance_report(filtered_analytics),
            "application_performance": self._generate_application_performance_report(filtered_analytics),
            "recommendations": self._generate_performance_recommendations(filtered_analytics)
        }

        return report

    def _generate_system_performance_report(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system performance report section"""
        system_metrics = ["cpu_percent", "memory_percent", "disk_usage_percent"]

        report = {}
        for metric in system_metrics:
            if metric in analytics:
                data = analytics[metric]
                values = [point["value"] for point in data]

                report[metric] = {
                    "average": statistics.mean(values),
                    "peak": max(values),
                    "trend": self._calculate_analytics_trend(data),
                    "volatility": self._calculate_volatility(values),
                    "assessment": self._assess_system_metric(metric, values)
                }

        return report

    def _generate_application_performance_report(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate application performance report section"""
        app_metrics = ["error_rate", "avg_response_time", "active_users", "active_conversations"]

        report = {}
        for metric in app_metrics:
            if metric in analytics:
                data = analytics[metric]
                values = [point["value"] for point in data]

                report[metric] = {
                    "average": statistics.mean(values),
                    "peak": max(values),
                    "trend": self._calculate_analytics_trend(data),
                    "volatility": self._calculate_volatility(values),
                    "assessment": self._assess_application_metric(metric, values)
                }

        return report

    def _generate_performance_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        # System recommendations
        if "cpu_percent" in analytics:
            cpu_values = [point["value"] for point in analytics["cpu_percent"]]
            if statistics.mean(cpu_values) > 70:
                recommendations.append("Consider optimizing CPU-intensive operations or scaling compute resources")

        if "memory_percent" in analytics:
            memory_values = [point["value"] for point in analytics["memory_percent"]]
            if statistics.mean(memory_values) > 75:
                recommendations.append("Monitor memory usage patterns and consider memory optimization techniques")

        # Application recommendations
        if "error_rate" in analytics:
            error_values = [point["value"] for point in analytics["error_rate"]]
            if statistics.mean(error_values) > 0.05:
                recommendations.append("Investigate and resolve sources of application errors")

        if "avg_response_time" in analytics:
            response_values = [point["value"] for point in analytics["avg_response_time"]]
            if statistics.mean(response_values) > 3.0:
                recommendations.append("Optimize response times through caching, query optimization, or code improvements")

        return recommendations

    def _assess_system_metric(self, metric: str, values: List[float]) -> str:
        """Assess system metric health"""
        avg_value = statistics.mean(values)

        if metric == "cpu_percent":
            if avg_value < 50:
                return "excellent"
            elif avg_value < 75:
                return "good"
            elif avg_value < 90:
                return "concerning"
            else:
                return "critical"
        elif metric in ["memory_percent", "disk_usage_percent"]:
            if avg_value < 70:
                return "excellent"
            elif avg_value < 85:
                return "good"
            elif avg_value < 95:
                return "concerning"
            else:
                return "critical"

        return "unknown"

    def _assess_application_metric(self, metric: str, values: List[float]) -> str:
        """Assess application metric health"""
        avg_value = statistics.mean(values)

        if metric == "error_rate":
            if avg_value < 0.01:
                return "excellent"
            elif avg_value < 0.05:
                return "good"
            elif avg_value < 0.1:
                return "concerning"
            else:
                return "critical"
        elif metric == "avg_response_time":
            if avg_value < 1.0:
                return "excellent"
            elif avg_value < 3.0:
                return "good"
            elif avg_value < 5.0:
                return "concerning"
            else:
                return "critical"
        elif metric == "active_users":
            return "informational"  # User counts are informational, not health metrics

        return "unknown"

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_enabled = False
        logger.info("Performance monitoring stopped")