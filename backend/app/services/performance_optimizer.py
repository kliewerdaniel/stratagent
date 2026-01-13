"""
Performance Optimization Engine for StratAgent
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import time
import psutil
import gc
import sys
from functools import lru_cache, wraps
import threading
import weakref

logger = logging.getLogger(__name__)

class CacheManager:
    """Advanced caching system with multiple strategies"""

    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.caches: Dict[str, OrderedDict] = {}
        self.cache_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.memory_usage = 0

        # Cache strategies
        self.strategies = {
            "lru": self._lru_evict,
            "lfu": self._lfu_evict,
            "ttl": self._ttl_evict,
            "size_aware": self._size_aware_evict
        }

        # Background cleanup
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()

    def create_cache(self, name: str, max_size: int = 1000, strategy: str = "lru",
                    ttl_seconds: Optional[int] = None) -> None:
        """Create a new cache with specified parameters"""
        self.caches[name] = OrderedDict()
        self.cache_stats[name] = {
            "max_size": max_size,
            "current_size": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "strategy": strategy,
            "ttl_seconds": ttl_seconds,
            "created_at": datetime.utcnow()
        }
        logger.info(f"Created cache '{name}' with strategy '{strategy}'")

    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """Get value from cache"""
        if cache_name not in self.caches:
            return None

        cache = self.caches[cache_name]
        stats = self.cache_stats[cache_name]

        # Check TTL if configured
        if stats["ttl_seconds"]:
            if key in cache:
                entry = cache[key]
                if datetime.utcnow() - entry["timestamp"] > timedelta(seconds=stats["ttl_seconds"]):
                    del cache[key]
                    stats["evictions"] += 1
                    stats["current_size"] -= 1
                    return None

        if key in cache:
            stats["hits"] += 1
            # Move to end for LRU
            if stats["strategy"] == "lru":
                cache.move_to_end(key)
            return cache[key]["value"]
        else:
            stats["misses"] += 1
            return None

    def put(self, cache_name: str, key: str, value: Any, size_estimate: int = 1) -> bool:
        """Put value in cache"""
        if cache_name not in self.caches:
            return False

        cache = self.caches[cache_name]
        stats = self.cache_stats[cache_name]

        # Check if we need to evict
        if len(cache) >= stats["max_size"] or self._would_exceed_memory_limit(size_estimate):
            self._evict_entries(cache_name, 1)

        # Store entry
        entry = {
            "value": value,
            "timestamp": datetime.utcnow(),
            "size": size_estimate,
            "access_count": 0
        }

        cache[key] = entry
        stats["current_size"] += 1

        # Update memory usage estimate
        self.memory_usage += size_estimate * 1024  # Rough estimate

        return True

    def _evict_entries(self, cache_name: str, count: int) -> None:
        """Evict entries based on cache strategy"""
        cache = self.caches[cache_name]
        stats = self.cache_stats[cache_name]
        strategy = stats["strategy"]

        evict_func = self.strategies.get(strategy, self._lru_evict)
        evict_func(cache_name, count)

    def _lru_evict(self, cache_name: str, count: int) -> None:
        """Least Recently Used eviction"""
        cache = self.caches[cache_name]
        stats = self.cache_stats[cache_name]

        for _ in range(min(count, len(cache))):
            evicted_key, evicted_entry = cache.popitem(last=False)
            self.memory_usage -= evicted_entry["size"] * 1024
            stats["evictions"] += 1

    def _lfu_evict(self, cache_name: str, count: int) -> None:
        """Least Frequently Used eviction"""
        cache = self.caches[cache_name]
        stats = self.cache_stats[cache_name]

        # Sort by access count (ascending)
        sorted_items = sorted(cache.items(), key=lambda x: x[1]["access_count"])

        for evicted_key, evicted_entry in sorted_items[:count]:
            if evicted_key in cache:
                del cache[evicted_key]
                self.memory_usage -= evicted_entry["size"] * 1024
                stats["evictions"] += 1

    def _ttl_evict(self, cache_name: str, count: int) -> None:
        """Time To Live eviction - remove expired entries"""
        cache = self.caches[cache_name]
        stats = self.cache_stats[cache_name]
        ttl_seconds = stats["ttl_seconds"]

        if not ttl_seconds:
            return self._lru_evict(cache_name, count)

        current_time = datetime.utcnow()
        expired_keys = []

        for key, entry in cache.items():
            if current_time - entry["timestamp"] > timedelta(seconds=ttl_seconds):
                expired_keys.append(key)

        for key in expired_keys[:count]:
            if key in cache:
                evicted_entry = cache[key]
                del cache[key]
                self.memory_usage -= evicted_entry["size"] * 1024
                stats["evictions"] += 1

    def _size_aware_evict(self, cache_name: str, count: int) -> None:
        """Size-aware eviction - remove largest entries first"""
        cache = self.caches[cache_name]
        stats = self.cache_stats[cache_name]

        # Sort by size (descending)
        sorted_items = sorted(cache.items(), key=lambda x: x[1]["size"], reverse=True)

        for evicted_key, evicted_entry in sorted_items[:count]:
            if evicted_key in cache:
                del cache[evicted_key]
                self.memory_usage -= evicted_entry["size"] * 1024
                stats["evictions"] += 1

    def _would_exceed_memory_limit(self, size_estimate: int) -> bool:
        """Check if adding entry would exceed memory limit"""
        estimated_bytes = size_estimate * 1024  # Rough estimate
        return (self.memory_usage + estimated_bytes) > (self.max_memory_mb * 1024 * 1024)

    def _background_cleanup(self) -> None:
        """Background cleanup of expired entries"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes

                for cache_name in list(self.caches.keys()):
                    stats = self.cache_stats[cache_name]
                    if stats["ttl_seconds"]:
                        self._ttl_evict(cache_name, 10)  # Clean up to 10 expired entries

            except Exception as e:
                logger.error(f"Background cleanup error: {str(e)}")

    def get_cache_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics"""
        if cache_name:
            if cache_name not in self.cache_stats:
                return {}
            stats = self.cache_stats[cache_name].copy()
            stats["hit_rate"] = stats["hits"] / max(1, stats["hits"] + stats["misses"])
            return stats

        # Aggregate stats for all caches
        total_stats = {
            "total_caches": len(self.caches),
            "total_memory_usage_mb": self.memory_usage / (1024 * 1024),
            "max_memory_mb": self.max_memory_mb
        }

        for name, stats in self.cache_stats.items():
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    total_stats[f"total_{key}"] = total_stats.get(f"total_{key}", 0) + value

        return total_stats

    def clear_cache(self, cache_name: str) -> bool:
        """Clear all entries from a cache"""
        if cache_name not in self.caches:
            return False

        cache = self.caches[cache_name]
        stats = self.cache_stats[cache_name]

        # Reset memory usage
        for entry in cache.values():
            self.memory_usage -= entry["size"] * 1024

        cache.clear()
        stats["current_size"] = 0
        stats["evictions"] = 0

        return True

class QueryOptimizer:
    """Database query optimization and performance monitoring"""

    def __init__(self):
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        self.slow_queries: List[Dict[str, Any]] = []
        self.query_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self.slow_query_threshold = 1.0  # seconds

    def record_query(self, query_hash: str, query: str, execution_time: float,
                    rows_affected: int = 0, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Record query execution statistics"""
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                "query": query,
                "execution_times": [],
                "total_executions": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "rows_affected": [],
                "first_seen": datetime.utcnow(),
                "last_seen": datetime.utcnow()
            }

        stats = self.query_stats[query_hash]
        stats["execution_times"].append(execution_time)
        stats["total_executions"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["total_executions"]
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["last_seen"] = datetime.utcnow()

        if rows_affected > 0:
            stats["rows_affected"].append(rows_affected)

        # Keep only last 100 execution times
        if len(stats["execution_times"]) > 100:
            stats["execution_times"] = stats["execution_times"][-100:]

        # Check for slow queries
        if execution_time > self.slow_query_threshold:
            self.slow_queries.append({
                "query_hash": query_hash,
                "query": query,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow(),
                "parameters": parameters
            })

            # Keep only last 50 slow queries
            if len(self.slow_queries) > 50:
                self.slow_queries = self.slow_queries[-50:]

    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns for optimization opportunities"""
        analysis = {
            "total_queries": len(self.query_stats),
            "slow_queries_count": len(self.slow_queries),
            "optimization_opportunities": []
        }

        # Find frequently executed slow queries
        slow_query_hashes = {sq["query_hash"] for sq in self.slow_queries}
        frequent_slow_queries = []

        for query_hash in slow_query_hashes:
            if query_hash in self.query_stats:
                stats = self.query_stats[query_hash]
                if stats["total_executions"] > 5:  # Executed more than 5 times
                    frequent_slow_queries.append({
                        "query_hash": query_hash,
                        "query": stats["query"][:100] + "..." if len(stats["query"]) > 100 else stats["query"],
                        "avg_time": stats["avg_time"],
                        "total_executions": stats["total_executions"],
                        "total_time": stats["total_time"]
                    })

        # Sort by total time impact
        frequent_slow_queries.sort(key=lambda x: x["total_time"], reverse=True)
        analysis["frequent_slow_queries"] = frequent_slow_queries[:10]  # Top 10

        # Generate optimization recommendations
        recommendations = []

        for slow_query in frequent_slow_queries[:5]:
            if "SELECT" in slow_query["query"].upper():
                if "JOIN" in slow_query["query"].upper():
                    recommendations.append("Consider adding indexes on JOIN columns")
                if "WHERE" in slow_query["query"].upper():
                    recommendations.append("Review WHERE clause conditions for index usage")
                recommendations.append("Consider query result caching for frequently accessed data")

        analysis["optimization_opportunities"] = recommendations

        return analysis

    def get_query_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive query performance report"""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_unique_queries": len(self.query_stats),
                "total_slow_queries": len(self.slow_queries),
                "slow_query_threshold_seconds": self.slow_query_threshold
            },
            "top_slow_queries": [],
            "query_categories": {}
        }

        # Get top slow queries by average execution time
        query_performance = []
        for query_hash, stats in self.query_stats.items():
            if stats["total_executions"] > 0:
                query_performance.append({
                    "query_hash": query_hash,
                    "query_preview": stats["query"][:100] + "..." if len(stats["query"]) > 100 else stats["query"],
                    "avg_execution_time": stats["avg_time"],
                    "total_executions": stats["total_executions"],
                    "total_time": stats["total_time"],
                    "min_time": stats["min_time"],
                    "max_time": stats["max_time"]
                })

        # Sort by average execution time
        query_performance.sort(key=lambda x: x["avg_execution_time"], reverse=True)
        report["top_slow_queries"] = query_performance[:20]  # Top 20 slowest

        # Categorize queries
        select_queries = sum(1 for stats in self.query_stats.values() if stats["query"].strip().upper().startswith("SELECT"))
        insert_queries = sum(1 for stats in self.query_stats.values() if stats["query"].strip().upper().startswith("INSERT"))
        update_queries = sum(1 for stats in self.query_stats.values() if stats["query"].strip().upper().startswith("UPDATE"))
        delete_queries = sum(1 for stats in self.query_stats.values() if stats["query"].strip().upper().startswith("DELETE"))

        report["query_categories"] = {
            "SELECT": select_queries,
            "INSERT": insert_queries,
            "UPDATE": update_queries,
            "DELETE": delete_queries
        }

        return report

class MemoryOptimizer:
    """Memory optimization and garbage collection management"""

    def __init__(self):
        self.memory_stats: Dict[str, Any] = {}
        self.gc_thresholds = gc.get_threshold()
        self.memory_pressure_threshold = 0.8  # 80% of available memory

    async def monitor_memory_usage(self) -> Dict[str, Any]:
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        stats = {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "memory_percent": memory_percent,
            "system_memory": psutil.virtual_memory()._asdict(),
            "timestamp": datetime.utcnow()
        }

        self.memory_stats = stats

        # Check for memory pressure
        if memory_percent > self.memory_pressure_threshold * 100:
            await self._handle_memory_pressure()

        return stats

    async def _handle_memory_pressure(self) -> None:
        """Handle memory pressure situations"""
        logger.warning("Memory pressure detected, initiating optimization")

        # Force garbage collection
        collected = gc.collect()

        # Clear any caches if available
        # This would integrate with the CacheManager

        # Log memory optimization
        logger.info(f"Garbage collection completed: {collected} objects collected")

        # If still under pressure, additional measures
        process = psutil.Process()
        if process.memory_percent() > self.memory_pressure_threshold * 100:
            logger.warning("Memory pressure persists after garbage collection")
            # Could implement additional measures like cache clearing, object cleanup, etc.

    def optimize_object_references(self) -> Dict[str, Any]:
        """Optimize object references to reduce memory usage"""
        # Get object counts before optimization
        before_counts = self._get_object_counts()

        # Find and clean up weak references
        gc.collect()  # Clean up first

        # Additional cleanup strategies
        optimization_results = {
            "objects_before": before_counts,
            "garbage_collected": 0,
            "weakrefs_cleaned": 0,
            "caches_cleared": 0
        }

        # Get object counts after optimization
        after_counts = self._get_object_counts()
        optimization_results["objects_after"] = after_counts

        return optimization_results

    def _get_object_counts(self) -> Dict[str, int]:
        """Get counts of different object types"""
        objects = gc.get_objects()
        type_counts = {}

        for obj in objects[:1000]:  # Sample first 1000 objects
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        return dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    def set_gc_thresholds(self, gen0: int, gen1: int, gen2: int) -> None:
        """Set garbage collection thresholds"""
        self.gc_thresholds = (gen0, gen1, gen2)
        gc.set_threshold(gen0, gen1, gen2)
        logger.info(f"Updated GC thresholds: {gen0}, {gen1}, {gen2}")

    def get_memory_optimization_report(self) -> Dict[str, Any]:
        """Generate memory optimization report"""
        return {
            "current_memory_stats": self.memory_stats,
            "gc_thresholds": self.gc_thresholds,
            "gc_stats": gc.get_stats(),
            "object_counts": self._get_object_counts(),
            "memory_pressure_threshold": self.memory_pressure_threshold,
            "optimization_recommendations": self._get_memory_recommendations()
        }

    def _get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []

        if self.memory_stats:
            memory_percent = self.memory_stats.get("memory_percent", 0)

            if memory_percent > 80:
                recommendations.append("High memory usage detected - consider increasing instance memory or optimizing data structures")
            if memory_percent > 90:
                recommendations.append("Critical memory usage - implement memory limits and monitoring")

            # Check GC stats
            gc_stats = gc.get_stats()
            for i, gen_stats in enumerate(gc_stats):
                if gen_stats["collected"] > gen_stats["uncollectable"]:
                    recommendations.append(f"Generation {i} GC is working effectively")
                else:
                    recommendations.append(f"Review memory management for generation {i} objects")

        return recommendations

class PerformanceProfiler:
    """Code performance profiling and bottleneck identification"""

    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.active_profiles: Dict[str, Any] = {}

    def start_profiling(self, profile_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start performance profiling"""
        profile_id = f"profile_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self.active_profiles[profile_id] = {
            "name": profile_name,
            "start_time": time.perf_counter(),
            "metadata": metadata or {},
            "checkpoints": []
        }

        return profile_id

    def add_checkpoint(self, profile_id: str, checkpoint_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a performance checkpoint"""
        if profile_id not in self.active_profiles:
            return

        profile = self.active_profiles[profile_id]
        current_time = time.perf_counter()
        elapsed = current_time - profile["start_time"]

        checkpoint = {
            "name": checkpoint_name,
            "elapsed_time": elapsed,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }

        profile["checkpoints"].append(checkpoint)

    def stop_profiling(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Stop profiling and return results"""
        if profile_id not in self.active_profiles:
            return None

        profile = self.active_profiles[profile_id]
        end_time = time.perf_counter()
        total_time = end_time - profile["start_time"]

        results = {
            "profile_id": profile_id,
            "name": profile["name"],
            "total_time": total_time,
            "start_time": profile["start_time"],
            "end_time": end_time,
            "checkpoints": profile["checkpoints"],
            "metadata": profile["metadata"],
            "completed_at": datetime.utcnow()
        }

        # Store completed profile
        self.profiles[profile_id] = results

        # Remove from active profiles
        del self.active_profiles[profile_id]

        return results

    def analyze_performance_bottlenecks(self, profile_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance profile for bottlenecks"""
        checkpoints = profile_results.get("checkpoints", [])
        total_time = profile_results.get("total_time", 0)

        if not checkpoints:
            return {"bottlenecks": [], "recommendations": []}

        # Calculate time spent in each checkpoint
        checkpoint_times = []
        prev_time = 0

        for checkpoint in checkpoints:
            time_spent = checkpoint["elapsed_time"] - prev_time
            checkpoint_times.append({
                "name": checkpoint["name"],
                "time_spent": time_spent,
                "percentage": (time_spent / total_time) * 100 if total_time > 0 else 0,
                "metadata": checkpoint["metadata"]
            })
            prev_time = checkpoint["elapsed_time"]

        # Identify bottlenecks (top 3 time consumers)
        sorted_checkpoints = sorted(checkpoint_times, key=lambda x: x["time_spent"], reverse=True)
        bottlenecks = sorted_checkpoints[:3]

        # Generate recommendations
        recommendations = []
        for bottleneck in bottlenecks:
            if bottleneck["percentage"] > 50:
                recommendations.append(f"Critical bottleneck in '{bottleneck['name']}' - optimize this section")
            elif bottleneck["percentage"] > 20:
                recommendations.append(f"Significant time spent in '{bottleneck['name']}' - consider optimization")

        return {
            "bottlenecks": bottlenecks,
            "checkpoint_analysis": checkpoint_times,
            "recommendations": recommendations,
            "total_checkpoints": len(checkpoints)
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        completed_profiles = list(self.profiles.values())

        if not completed_profiles:
            return {"message": "No performance profiles available"}

        # Analyze all profiles
        total_profiles = len(completed_profiles)
        avg_total_time = sum(p["total_time"] for p in completed_profiles) / total_profiles

        # Find slowest operations
        all_checkpoints = []
        for profile in completed_profiles:
            for checkpoint in profile.get("checkpoints", []):
                all_checkpoints.append({
                    "profile_name": profile["name"],
                    "checkpoint_name": checkpoint["name"],
                    "time_spent": checkpoint["elapsed_time"] - (profile["checkpoints"][profile["checkpoints"].index(checkpoint) - 1]["elapsed_time"] if profile["checkpoints"].index(checkpoint) > 0 else 0)
                })

        # Sort checkpoints by time spent
        slowest_checkpoints = sorted(all_checkpoints, key=lambda x: x["time_spent"], reverse=True)[:10]

        return {
            "total_profiles": total_profiles,
            "average_total_time": avg_total_time,
            "slowest_operations": slowest_checkpoints,
            "active_profiles": len(self.active_profiles),
            "generated_at": datetime.utcnow().isoformat()
        }

def cached_result(cache_manager: CacheManager, cache_name: str, ttl_seconds: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try to get from cache first
            cached_result = cache_manager.get(cache_name, cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache the result
            cache_manager.put(cache_name, cache_key, result)

            return result

        return wrapper
    return decorator

def profile_performance(profiler: PerformanceProfiler, profile_name: str):
    """Decorator for performance profiling"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Start profiling
            profile_id = profiler.start_profiling(profile_name)

            try:
                # Execute function
                result = await func(*args, **kwargs)
                return result
            finally:
                # Stop profiling
                profiler.stop_profiling(profile_id)

        return wrapper
    return decorator

class PerformanceOptimizationEngine:
    """Comprehensive performance optimization engine"""

    def __init__(self):
        self.cache_manager = CacheManager()
        self.query_optimizer = QueryOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.performance_profiler = PerformanceProfiler()

        # Optimization settings
        self.optimization_enabled = True
        self.auto_optimization_interval = 3600  # 1 hour

        # Background optimization task
        self.optimization_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the optimization engine"""
        # Create default caches
        self.cache_manager.create_cache("api_responses", max_size=1000, strategy="ttl", ttl_seconds=300)
        self.cache_manager.create_cache("query_results", max_size=500, strategy="lru")
        self.cache_manager.create_cache("computation_results", max_size=200, strategy="lfu")

        # Start background optimization
        if self.optimization_enabled:
            self.optimization_task = asyncio.create_task(self._background_optimization())

        logger.info("Performance optimization engine initialized")

    async def _background_optimization(self):
        """Background optimization tasks"""
        while self.optimization_enabled:
            try:
                await asyncio.sleep(self.auto_optimization_interval)

                # Run periodic optimizations
                await self.optimize_memory_usage()
                await self.optimize_cache_performance()
                await self.generate_optimization_report()

            except Exception as e:
                logger.error(f"Background optimization error: {str(e)}")

    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        # Monitor current memory usage
        memory_stats = await self.memory_optimizer.monitor_memory_usage()

        # Run optimization if needed
        if memory_stats["memory_percent"] > 75:
            optimization_results = self.memory_optimizer.optimize_object_references()

            # Force garbage collection
            collected = gc.collect()

            optimization_results["garbage_collected"] = collected
            optimization_results["memory_before_mb"] = memory_stats["rss_mb"]

            # Check memory after optimization
            after_stats = await self.memory_optimizer.monitor_memory_usage()
            optimization_results["memory_after_mb"] = after_stats["rss_mb"]

            logger.info(f"Memory optimization completed: {optimization_results}")
            return optimization_results

        return {"message": "Memory usage within acceptable limits"}

    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """Optimize cache performance"""
        cache_stats = self.cache_manager.get_cache_stats()

        optimizations = {
            "caches_analyzed": cache_stats.get("total_caches", 0),
            "optimizations_applied": 0,
            "memory_reclaimed_mb": 0
        }

        # Analyze each cache
        for cache_name in self.cache_manager.caches.keys():
            cache_stats = self.cache_manager.get_cache_stats(cache_name)

            # Optimize based on hit rate
            hit_rate = cache_stats.get("hit_rate", 0)
            if hit_rate < 0.3:  # Low hit rate
                # Consider clearing or resizing cache
                if cache_stats["current_size"] > 100:
                    cleared = self.cache_manager.clear_cache(cache_name)
                    if cleared:
                        optimizations["optimizations_applied"] += 1

        return optimizations

    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "memory_stats": await self.memory_optimizer.monitor_memory_usage(),
            "cache_stats": self.cache_manager.get_cache_stats(),
            "query_performance": self.query_optimizer.get_query_performance_report(),
            "performance_profile": self.performance_profiler.get_performance_report(),
            "optimization_recommendations": []
        }

        # Generate recommendations
        recommendations = []

        # Memory recommendations
        memory_stats = report["memory_stats"]
        if memory_stats.get("memory_percent", 0) > 80:
            recommendations.append("High memory usage detected - consider scaling or optimization")

        # Cache recommendations
        cache_stats = report["cache_stats"]
        if cache_stats.get("total_caches", 0) > 0:
            avg_hit_rate = cache_stats.get("total_hits", 0) / max(1, cache_stats.get("total_hits", 0) + cache_stats.get("total_misses", 0))
            if avg_hit_rate < 0.5:
                recommendations.append("Low cache hit rates detected - review caching strategy")

        # Query recommendations
        query_report = report["query_performance"]
        if query_report.get("summary", {}).get("total_slow_queries", 0) > 0:
            recommendations.append("Slow queries detected - review database indexes and query optimization")

        report["optimization_recommendations"] = recommendations

        logger.info(f"Optimization report generated with {len(recommendations)} recommendations")
        return report

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            "optimization_enabled": self.optimization_enabled,
            "background_task_active": self.optimization_task is not None and not self.optimization_task.done(),
            "cache_stats": self.cache_manager.get_cache_stats(),
            "memory_stats": self.memory_optimizer.memory_stats,
            "active_profiles": len(self.performance_profiler.active_profiles),
            "completed_profiles": len(self.performance_profiler.profiles)
        }

    async def shutdown(self):
        """Shutdown the optimization engine"""
        self.optimization_enabled = False

        if self.optimization_task and not self.optimization_task.done():
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance optimization engine shut down")