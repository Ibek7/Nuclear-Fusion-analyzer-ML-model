"""
Query Optimization and Performance Monitoring for Nuclear Fusion Analysis.

This module provides:
- SQL query optimization and analysis
- Query execution plan analysis
- Performance monitoring and alerting
- Resource usage tracking
- Bottleneck identification
- Automatic performance tuning
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import threading
from collections import deque, defaultdict
import statistics

# Database query analysis
try:
    import sqlparse
    from sqlparse import sql, tokens
    HAS_SQLPARSE = True
except ImportError:
    HAS_SQLPARSE = False

# Performance monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of database queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"
    INDEX = "index"


class PerformanceLevel(Enum):
    """Performance severity levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    
    query_id: str
    query_text: str
    query_type: QueryType
    execution_time: float
    rows_examined: int = 0
    rows_returned: int = 0
    memory_used: int = 0
    cpu_time: float = 0.0
    io_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def performance_level(self) -> PerformanceLevel:
        """Determine performance level based on execution time."""
        if self.execution_time < 0.1:
            return PerformanceLevel.EXCELLENT
        elif self.execution_time < 0.5:
            return PerformanceLevel.GOOD
        elif self.execution_time < 2.0:
            return PerformanceLevel.FAIR
        elif self.execution_time < 10.0:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations == 0:
            return 0.0
        return self.cache_hits / total_cache_operations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_type": self.query_type.value,
            "execution_time": self.execution_time,
            "rows_examined": self.rows_examined,
            "rows_returned": self.rows_returned,
            "memory_used": self.memory_used,
            "cpu_time": self.cpu_time,
            "io_operations": self.io_operations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "performance_level": self.performance_level.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OptimizationSuggestion:
    """Query optimization suggestion."""
    
    suggestion_id: str
    query_id: str
    category: str
    severity: str
    description: str
    recommendation: str
    estimated_improvement: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "query_id": self.query_id,
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "recommendation": self.recommendation,
            "estimated_improvement": self.estimated_improvement
        }


class QueryAnalyzer:
    """SQL query analysis and optimization suggestions."""
    
    def __init__(self):
        """Initialize query analyzer."""
        self.optimization_rules = self._load_optimization_rules()
        
        logger.info("QueryAnalyzer initialized")
    
    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load query optimization rules."""
        return [
            {
                "name": "missing_where_clause",
                "pattern": r"SELECT.*FROM.*(?!WHERE)",
                "category": "Performance",
                "severity": "High",
                "description": "Query without WHERE clause may scan entire table",
                "recommendation": "Add WHERE clause to filter results"
            },
            {
                "name": "select_star",
                "pattern": r"SELECT\s+\*",
                "category": "Performance",
                "severity": "Medium",
                "description": "SELECT * retrieves all columns unnecessarily",
                "recommendation": "Specify only required columns"
            },
            {
                "name": "no_limit",
                "pattern": r"SELECT.*FROM.*(?!LIMIT)",
                "category": "Performance",
                "severity": "Medium",
                "description": "Query without LIMIT may return excessive rows",
                "recommendation": "Add LIMIT clause for large result sets"
            },
            {
                "name": "function_in_where",
                "pattern": r"WHERE.*\w+\(",
                "category": "Indexing",
                "severity": "High",
                "description": "Functions in WHERE clause prevent index usage",
                "recommendation": "Avoid functions on indexed columns in WHERE clause"
            },
            {
                "name": "like_leading_wildcard",
                "pattern": r"LIKE\s+['\"]%",
                "category": "Indexing",
                "severity": "High",
                "description": "LIKE with leading wildcard prevents index usage",
                "recommendation": "Avoid leading wildcards in LIKE patterns"
            },
            {
                "name": "or_in_where",
                "pattern": r"WHERE.*\sOR\s",
                "category": "Indexing",
                "severity": "Medium",
                "description": "OR conditions may prevent efficient index usage",
                "recommendation": "Consider using UNION instead of OR"
            },
            {
                "name": "subquery_in_select",
                "pattern": r"SELECT.*\(SELECT",
                "category": "Performance",
                "severity": "Medium",
                "description": "Correlated subqueries can be expensive",
                "recommendation": "Consider using JOINs instead of subqueries"
            },
            {
                "name": "distinct_without_reason",
                "pattern": r"SELECT\s+DISTINCT",
                "category": "Performance",
                "severity": "Low",
                "description": "DISTINCT adds overhead if not necessary",
                "recommendation": "Ensure DISTINCT is actually needed"
            }
        ]
    
    def analyze_query(self, query_text: str, query_id: str) -> List[OptimizationSuggestion]:
        """
        Analyze query and provide optimization suggestions.
        
        Args:
            query_text: SQL query text.
            query_id: Query identifier.
            
        Returns:
            List of optimization suggestions.
        """
        suggestions = []
        
        if not HAS_SQLPARSE:
            logger.warning("sqlparse not available for query analysis")
            return suggestions
        
        try:
            # Parse query
            parsed = sqlparse.parse(query_text)[0]
            query_upper = query_text.upper()
            
            # Apply optimization rules
            for rule in self.optimization_rules:
                import re
                if re.search(rule["pattern"], query_upper, re.IGNORECASE):
                    suggestion = OptimizationSuggestion(
                        suggestion_id=f"{query_id}_{rule['name']}",
                        query_id=query_id,
                        category=rule["category"],
                        severity=rule["severity"],
                        description=rule["description"],
                        recommendation=rule["recommendation"]
                    )
                    suggestions.append(suggestion)
            
            # Additional analysis based on parsed structure
            suggestions.extend(self._analyze_parsed_query(parsed, query_id))
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
        
        return suggestions
    
    def _analyze_parsed_query(self, parsed: sql.Statement, query_id: str) -> List[OptimizationSuggestion]:
        """Analyze parsed query structure."""
        suggestions = []
        
        try:
            # Check for complex joins
            join_count = str(parsed).upper().count('JOIN')
            if join_count > 5:
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"{query_id}_complex_joins",
                    query_id=query_id,
                    category="Performance",
                    severity="High",
                    description=f"Query has {join_count} joins which may be expensive",
                    recommendation="Consider denormalizing data or using materialized views"
                ))
            
            # Check for nested subqueries
            subquery_count = str(parsed).upper().count('SELECT') - 1
            if subquery_count > 2:
                suggestions.append(OptimizationSuggestion(
                    suggestion_id=f"{query_id}_nested_subqueries",
                    query_id=query_id,
                    category="Performance",
                    severity="Medium",
                    description=f"Query has {subquery_count} subqueries",
                    recommendation="Consider flattening subqueries into joins"
                ))
            
        except Exception as e:
            logger.error(f"Error in parsed query analysis: {e}")
        
        return suggestions
    
    def get_query_type(self, query_text: str) -> QueryType:
        """Determine query type from SQL text."""
        query_upper = query_text.strip().upper()
        
        if query_upper.startswith('SELECT'):
            return QueryType.SELECT
        elif query_upper.startswith('INSERT'):
            return QueryType.INSERT
        elif query_upper.startswith('UPDATE'):
            return QueryType.UPDATE
        elif query_upper.startswith('DELETE'):
            return QueryType.DELETE
        elif query_upper.startswith('CREATE'):
            return QueryType.CREATE
        elif query_upper.startswith('ALTER'):
            return QueryType.ALTER
        elif query_upper.startswith('DROP'):
            return QueryType.DROP
        elif 'INDEX' in query_upper:
            return QueryType.INDEX
        else:
            return QueryType.SELECT  # Default


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of historical metrics to keep.
        """
        self.max_history = max_history
        self.query_metrics: deque = deque(maxlen=max_history)
        self.performance_alerts: List[Dict[str, Any]] = []
        self.thresholds = self._get_default_thresholds()
        
        # Thread-safe operations
        self.lock = threading.RLock()
        
        # Aggregated statistics
        self.stats_by_type: defaultdict = defaultdict(list)
        self.hourly_stats: defaultdict = defaultdict(list)
        
        logger.info("PerformanceMonitor initialized")
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default performance thresholds."""
        return {
            "max_execution_time": 10.0,  # seconds
            "max_memory_usage": 100 * 1024 * 1024,  # 100MB
            "min_cache_hit_rate": 0.8,  # 80%
            "max_rows_examined_ratio": 10.0,  # examined/returned ratio
            "max_cpu_usage": 80.0,  # percentage
            "max_io_operations": 1000
        }
    
    def record_query_metrics(self, metrics: QueryMetrics):
        """Record query performance metrics."""
        with self.lock:
            self.query_metrics.append(metrics)
            
            # Update aggregated statistics
            self.stats_by_type[metrics.query_type].append(metrics)
            
            # Hourly aggregation
            hour_key = metrics.timestamp.replace(minute=0, second=0, microsecond=0)
            self.hourly_stats[hour_key].append(metrics)
            
            # Check for performance alerts
            self._check_performance_alerts(metrics)
        
        logger.debug(f"Recorded metrics for query {metrics.query_id}")
    
    def _check_performance_alerts(self, metrics: QueryMetrics):
        """Check if metrics trigger any performance alerts."""
        alerts = []
        
        # Execution time alert
        if metrics.execution_time > self.thresholds["max_execution_time"]:
            alerts.append({
                "type": "slow_query",
                "severity": "high" if metrics.execution_time > self.thresholds["max_execution_time"] * 2 else "medium",
                "message": f"Query {metrics.query_id} took {metrics.execution_time:.2f}s to execute",
                "query_id": metrics.query_id,
                "timestamp": metrics.timestamp.isoformat()
            })
        
        # Memory usage alert
        if metrics.memory_used > self.thresholds["max_memory_usage"]:
            alerts.append({
                "type": "high_memory",
                "severity": "high",
                "message": f"Query {metrics.query_id} used {metrics.memory_used / 1024 / 1024:.2f}MB memory",
                "query_id": metrics.query_id,
                "timestamp": metrics.timestamp.isoformat()
            })
        
        # Cache hit rate alert
        if metrics.cache_hit_rate < self.thresholds["min_cache_hit_rate"]:
            alerts.append({
                "type": "low_cache_hit_rate",
                "severity": "medium",
                "message": f"Query {metrics.query_id} has low cache hit rate: {metrics.cache_hit_rate:.2f}",
                "query_id": metrics.query_id,
                "timestamp": metrics.timestamp.isoformat()
            })
        
        # Row examination efficiency
        if metrics.rows_returned > 0:
            examined_ratio = metrics.rows_examined / metrics.rows_returned
            if examined_ratio > self.thresholds["max_rows_examined_ratio"]:
                alerts.append({
                    "type": "inefficient_scan",
                    "severity": "medium",
                    "message": f"Query {metrics.query_id} examined {examined_ratio:.1f}x more rows than returned",
                    "query_id": metrics.query_id,
                    "timestamp": metrics.timestamp.isoformat()
                })
        
        self.performance_alerts.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.performance_alerts = [
            alert for alert in self.performance_alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period."""
        with self.lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.query_metrics
                if m.timestamp > cutoff_time
            ]
            
            if not recent_metrics:
                return {"message": "No metrics available for specified period"}
            
            # Calculate summary statistics
            execution_times = [m.execution_time for m in recent_metrics]
            memory_usage = [m.memory_used for m in recent_metrics]
            cache_hit_rates = [m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate > 0]
            
            summary = {
                "period_hours": hours,
                "total_queries": len(recent_metrics),
                "execution_time": {
                    "avg": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "min": min(execution_times),
                    "max": max(execution_times),
                    "p95": sorted(execution_times)[int(len(execution_times) * 0.95)] if execution_times else 0
                },
                "memory_usage": {
                    "avg": statistics.mean(memory_usage) if memory_usage else 0,
                    "max": max(memory_usage) if memory_usage else 0
                },
                "cache_performance": {
                    "avg_hit_rate": statistics.mean(cache_hit_rates) if cache_hit_rates else 0
                },
                "query_types": defaultdict(int),
                "performance_levels": defaultdict(int),
                "recent_alerts": len([
                    a for a in self.performance_alerts
                    if datetime.fromisoformat(a["timestamp"]) > cutoff_time
                ])
            }
            
            # Query type breakdown
            for metrics in recent_metrics:
                summary["query_types"][metrics.query_type.value] += 1
                summary["performance_levels"][metrics.performance_level.value] += 1
            
            return summary
    
    def get_slow_queries(self, limit: int = 10, hours: int = 24) -> List[Dict[str, Any]]:
        """Get slowest queries in specified time period."""
        with self.lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.query_metrics
                if m.timestamp > cutoff_time
            ]
            
            # Sort by execution time and return top slowest
            slow_queries = sorted(recent_metrics, key=lambda m: m.execution_time, reverse=True)[:limit]
            
            return [m.to_dict() for m in slow_queries]
    
    def get_performance_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance alerts, optionally filtered by severity."""
        if severity:
            return [alert for alert in self.performance_alerts if alert["severity"] == severity]
        return self.performance_alerts.copy()
    
    def clear_alerts(self):
        """Clear all performance alerts."""
        with self.lock:
            self.performance_alerts.clear()
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """Update performance thresholds."""
        self.thresholds.update(thresholds)
        logger.info(f"Performance thresholds updated: {thresholds}")
    
    def get_hourly_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get hourly performance trends."""
        with self.lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            trends = {}
            for hour_key, metrics_list in self.hourly_stats.items():
                if hour_key > cutoff_time:
                    execution_times = [m.execution_time for m in metrics_list]
                    
                    trends[hour_key.isoformat()] = {
                        "query_count": len(metrics_list),
                        "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
                        "max_execution_time": max(execution_times) if execution_times else 0,
                        "avg_memory_usage": statistics.mean([m.memory_used for m in metrics_list]) if metrics_list else 0
                    }
            
            return trends


class QueryOptimizer:
    """
    Comprehensive query optimization and performance monitoring system.
    
    Provides query analysis, performance monitoring, and optimization suggestions.
    """
    
    def __init__(self):
        """Initialize query optimizer."""
        self.query_analyzer = QueryAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.optimization_cache: Dict[str, List[OptimizationSuggestion]] = {}
        
        logger.info("QueryOptimizer initialized")
    
    async def analyze_and_monitor_query(
        self,
        query_id: str,
        query_text: str,
        execution_time: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze query and record performance metrics.
        
        Args:
            query_id: Query identifier.
            query_text: SQL query text.
            execution_time: Query execution time.
            **kwargs: Additional metrics.
            
        Returns:
            Analysis results and optimization suggestions.
        """
        # Determine query type
        query_type = self.query_analyzer.get_query_type(query_text)
        
        # Create metrics object
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=query_text,
            query_type=query_type,
            execution_time=execution_time,
            rows_examined=kwargs.get('rows_examined', 0),
            rows_returned=kwargs.get('rows_returned', 0),
            memory_used=kwargs.get('memory_used', 0),
            cpu_time=kwargs.get('cpu_time', 0.0),
            io_operations=kwargs.get('io_operations', 0),
            cache_hits=kwargs.get('cache_hits', 0),
            cache_misses=kwargs.get('cache_misses', 0)
        )
        
        # Record metrics
        self.performance_monitor.record_query_metrics(metrics)
        
        # Get optimization suggestions (cache if available)
        if query_text in self.optimization_cache:
            suggestions = self.optimization_cache[query_text]
        else:
            suggestions = self.query_analyzer.analyze_query(query_text, query_id)
            self.optimization_cache[query_text] = suggestions
        
        return {
            "query_id": query_id,
            "metrics": metrics.to_dict(),
            "suggestions": [s.to_dict() for s in suggestions],
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_optimization_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        # Performance summary
        performance_summary = self.performance_monitor.get_performance_summary(hours)
        
        # Slow queries
        slow_queries = self.performance_monitor.get_slow_queries(limit=20, hours=hours)
        
        # Performance alerts
        alerts = self.performance_monitor.get_performance_alerts()
        
        # Optimization suggestions summary
        all_suggestions = []
        for suggestions_list in self.optimization_cache.values():
            all_suggestions.extend(suggestions_list)
        
        suggestion_categories = defaultdict(int)
        suggestion_severities = defaultdict(int)
        
        for suggestion in all_suggestions:
            suggestion_categories[suggestion.category] += 1
            suggestion_severities[suggestion.severity] += 1
        
        return {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_hours": hours,
            "performance_summary": performance_summary,
            "slow_queries": slow_queries,
            "alerts": alerts,
            "optimization_suggestions": {
                "total": len(all_suggestions),
                "by_category": dict(suggestion_categories),
                "by_severity": dict(suggestion_severities),
                "recent_suggestions": [s.to_dict() for s in all_suggestions[-10:]]
            },
            "recommendations": self._generate_recommendations(performance_summary, alerts, all_suggestions)
        }
    
    def _generate_recommendations(
        self,
        performance_summary: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        suggestions: List[OptimizationSuggestion]
    ) -> List[str]:
        """Generate high-level optimization recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        if "execution_time" in performance_summary:
            avg_time = performance_summary["execution_time"].get("avg", 0)
            if avg_time > 5.0:
                recommendations.append("Consider adding database indexes for frequently queried columns")
                recommendations.append("Review query patterns for optimization opportunities")
        
        # Alert-based recommendations
        alert_types = [alert["type"] for alert in alerts]
        if "slow_query" in alert_types:
            recommendations.append("Investigate and optimize slow-running queries")
        
        if "high_memory" in alert_types:
            recommendations.append("Optimize memory usage in database queries")
        
        if "low_cache_hit_rate" in alert_types:
            recommendations.append("Improve query caching strategies")
        
        # Suggestion-based recommendations
        high_severity_suggestions = [s for s in suggestions if s.severity == "High"]
        if len(high_severity_suggestions) > 5:
            recommendations.append("Address high-severity optimization suggestions")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Query performance is within acceptable ranges")
            recommendations.append("Continue monitoring for potential optimizations")
        
        return recommendations
    
    def clear_optimization_cache(self):
        """Clear optimization suggestion cache."""
        self.optimization_cache.clear()
        logger.info("Optimization cache cleared")
    
    def export_performance_data(self, hours: int = 24) -> Dict[str, Any]:
        """Export performance data for external analysis."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_metrics = [
            m.to_dict() for m in self.performance_monitor.query_metrics
            if m.timestamp > cutoff_time
        ]
        
        return {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_hours": hours,
            "total_queries": len(recent_metrics),
            "metrics": recent_metrics,
            "alerts": self.performance_monitor.get_performance_alerts(),
            "trends": self.performance_monitor.get_hourly_trends(hours)
        }


def create_query_optimizer() -> QueryOptimizer:
    """
    Create configured query optimizer.
    
    Returns:
        Configured query optimizer.
    """
    return QueryOptimizer()