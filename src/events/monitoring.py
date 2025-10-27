"""
Event Monitoring Dashboard.

This module provides:
- Real-time event monitoring
- Event metrics and analytics
- Event visualization and dashboards
- Event health monitoring
- Performance tracking
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import logging
from statistics import mean, median
import math

from . import EventType, Event, get_event_store, get_event_bus, get_message_broker, get_event_stream

logger = logging.getLogger(__name__)


@dataclass
class EventMetrics:
    """Event metrics for monitoring."""
    event_type: EventType
    count: int = 0
    rate_per_second: float = 0.0
    avg_processing_time: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    last_event_time: Optional[datetime] = None
    
    def update(self, processing_time: float, has_error: bool = False):
        """Update metrics with new data."""
        self.count += 1
        if has_error:
            self.error_count += 1
        self.error_rate = self.error_count / self.count if self.count > 0 else 0.0
        
        # Update average processing time
        if self.avg_processing_time == 0:
            self.avg_processing_time = processing_time
        else:
            self.avg_processing_time = (self.avg_processing_time + processing_time) / 2
        
        self.last_event_time = datetime.now()


@dataclass
class SystemHealth:
    """System health status."""
    overall_status: str = "healthy"  # healthy, warning, critical
    event_store_status: str = "healthy"
    event_bus_status: str = "healthy"
    message_broker_status: str = "healthy"
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0
    active_connections: int = 0
    uptime_seconds: int = 0
    last_check: datetime = field(default_factory=datetime.now)
    
    def get_overall_status(self) -> str:
        """Calculate overall status based on component statuses."""
        statuses = [
            self.event_store_status,
            self.event_bus_status,
            self.message_broker_status
        ]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        else:
            return "healthy"


class EventMonitor:
    """Event monitoring system."""
    
    def __init__(self, window_size: int = 300):  # 5 minutes
        """
        Initialize event monitor.
        
        Args:
            window_size: Time window for metrics in seconds.
        """
        self.window_size = window_size
        self.metrics: Dict[EventType, EventMetrics] = {}
        self.event_history: deque = deque(maxlen=1000)
        self.rate_windows: Dict[EventType, deque] = defaultdict(lambda: deque(maxlen=60))
        
        self.system_health = SystemHealth()
        self.start_time = datetime.now()
        
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info("EventMonitor initialized")
    
    def start(self):
        """Start monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Event monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Event monitoring stopped")
    
    def record_event(self, event: Event, processing_time: float = 0.0, has_error: bool = False):
        """
        Record event for monitoring.
        
        Args:
            event: Event to record.
            processing_time: Processing time in seconds.
            has_error: Whether the event processing had an error.
        """
        with self.lock:
            # Update metrics
            if event.event_type not in self.metrics:
                self.metrics[event.event_type] = EventMetrics(event.event_type)
            
            self.metrics[event.event_type].update(processing_time, has_error)
            
            # Add to history
            event_record = {
                "timestamp": datetime.now(),
                "event_type": event.event_type.value,
                "event_id": event.id,
                "processing_time": processing_time,
                "has_error": has_error,
                "source": event.source,
                "aggregate_id": event.aggregate_id
            }
            self.event_history.append(event_record)
            
            # Update rate windows
            current_minute = int(time.time() // 60)
            self.rate_windows[event.event_type].append(current_minute)
    
    def get_event_metrics(self) -> Dict[str, EventMetrics]:
        """Get current event metrics."""
        with self.lock:
            return {event_type.value: metrics for event_type, metrics in self.metrics.items()}
    
    def get_event_rates(self) -> Dict[str, float]:
        """Get current event rates per minute."""
        current_minute = int(time.time() // 60)
        rates = {}
        
        with self.lock:
            for event_type, rate_window in self.rate_windows.items():
                # Count events in the last minute
                recent_events = sum(1 for minute in rate_window if minute == current_minute - 1)
                rates[event_type.value] = recent_events
        
        return rates
    
    def get_top_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top events by volume.
        
        Args:
            limit: Number of top events to return.
            
        Returns:
            List of top events.
        """
        with self.lock:
            sorted_metrics = sorted(
                self.metrics.items(),
                key=lambda x: x[1].count,
                reverse=True
            )
            
            return [
                {
                    "event_type": event_type.value,
                    "count": metrics.count,
                    "rate": metrics.rate_per_second,
                    "avg_processing_time": metrics.avg_processing_time,
                    "error_rate": metrics.error_rate
                }
                for event_type, metrics in sorted_metrics[:limit]
            ]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        with self.lock:
            total_events = sum(metrics.count for metrics in self.metrics.values())
            total_errors = sum(metrics.error_count for metrics in self.metrics.values())
            
            error_by_type = {
                event_type.value: {
                    "count": metrics.error_count,
                    "rate": metrics.error_rate
                }
                for event_type, metrics in self.metrics.items()
                if metrics.error_count > 0
            }
            
            return {
                "total_events": total_events,
                "total_errors": total_errors,
                "overall_error_rate": total_errors / total_events if total_events > 0 else 0.0,
                "errors_by_type": error_by_type
            }
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent events.
        
        Args:
            limit: Number of recent events to return.
            
        Returns:
            List of recent events.
        """
        with self.lock:
            recent = list(self.event_history)[-limit:]
            return [
                {
                    "timestamp": record["timestamp"].isoformat(),
                    "event_type": record["event_type"],
                    "event_id": record["event_id"],
                    "processing_time": record["processing_time"],
                    "has_error": record["has_error"],
                    "source": record["source"],
                    "aggregate_id": record["aggregate_id"]
                }
                for record in reversed(recent)
            ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            processing_times = [
                record["processing_time"] 
                for record in self.event_history 
                if record["processing_time"] > 0
            ]
            
            if not processing_times:
                return {
                    "avg_processing_time": 0.0,
                    "median_processing_time": 0.0,
                    "min_processing_time": 0.0,
                    "max_processing_time": 0.0,
                    "p95_processing_time": 0.0,
                    "p99_processing_time": 0.0
                }
            
            processing_times.sort()
            count = len(processing_times)
            
            return {
                "avg_processing_time": mean(processing_times),
                "median_processing_time": median(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "p95_processing_time": processing_times[int(count * 0.95)] if count > 0 else 0.0,
                "p99_processing_time": processing_times[int(count * 0.99)] if count > 0 else 0.0
            }
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health."""
        self._update_system_health()
        return self.system_health
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "system_health": {
                "overall_status": self.system_health.overall_status,
                "event_store_status": self.system_health.event_store_status,
                "event_bus_status": self.system_health.event_bus_status,
                "message_broker_status": self.system_health.message_broker_status,
                "memory_usage_mb": self.system_health.memory_usage_mb,
                "cpu_usage_percent": self.system_health.cpu_usage_percent
            },
            "event_metrics": self.get_event_metrics(),
            "event_rates": self.get_event_rates(),
            "top_events": self.get_top_events(),
            "error_summary": self.get_error_summary(),
            "performance_stats": self.get_performance_stats(),
            "recent_events": self.get_recent_events(20)
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Update event rates
                self._update_event_rates()
                
                # Update system health
                self._update_system_health()
                
                # Record performance snapshot
                self._record_performance_snapshot()
                
                # Sleep for 10 seconds
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _update_event_rates(self):
        """Update event rates."""
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=self.window_size)
        
        with self.lock:
            for event_type, metrics in self.metrics.items():
                # Count events in the time window
                recent_events = [
                    record for record in self.event_history
                    if (record["event_type"] == event_type.value and
                        record["timestamp"] >= window_start)
                ]
                
                # Calculate rate per second
                if len(recent_events) > 0:
                    time_span = (current_time - recent_events[0]["timestamp"]).total_seconds()
                    metrics.rate_per_second = len(recent_events) / max(time_span, 1.0)
                else:
                    metrics.rate_per_second = 0.0
    
    def _update_system_health(self):
        """Update system health status."""
        try:
            # Get system resource usage
            import psutil
            
            process = psutil.Process()
            self.system_health.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.system_health.cpu_usage_percent = process.cpu_percent()
            
            # Check component health
            self._check_event_store_health()
            self._check_event_bus_health()
            self._check_message_broker_health()
            
            # Update uptime
            self.system_health.uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
            
            # Update overall status
            self.system_health.overall_status = self.system_health.get_overall_status()
            self.system_health.last_check = datetime.now()
            
        except ImportError:
            # psutil not available, use basic health check
            self.system_health.memory_usage_mb = 0.0
            self.system_health.cpu_usage_percent = 0.0
        except Exception as e:
            logger.error(f"Error updating system health: {e}")
    
    def _check_event_store_health(self):
        """Check event store health."""
        try:
            event_store = get_event_store()
            event_count = event_store.get_event_count()
            
            if event_count >= 0:
                self.system_health.event_store_status = "healthy"
            else:
                self.system_health.event_store_status = "warning"
                
        except Exception as e:
            logger.error(f"Event store health check failed: {e}")
            self.system_health.event_store_status = "critical"
    
    def _check_event_bus_health(self):
        """Check event bus health."""
        try:
            event_bus = get_event_bus()
            
            # Simple health check - if we can access the bus, it's healthy
            if event_bus.is_processing:
                self.system_health.event_bus_status = "healthy"
            else:
                self.system_health.event_bus_status = "warning"
                
        except Exception as e:
            logger.error(f"Event bus health check failed: {e}")
            self.system_health.event_bus_status = "critical"
    
    def _check_message_broker_health(self):
        """Check message broker health."""
        try:
            message_broker = get_message_broker()
            broker_stats = message_broker.get_broker_stats()
            
            if broker_stats and "queues" in broker_stats:
                self.system_health.message_broker_status = "healthy"
            else:
                self.system_health.message_broker_status = "warning"
                
        except Exception as e:
            logger.error(f"Message broker health check failed: {e}")
            self.system_health.message_broker_status = "critical"
    
    def _record_performance_snapshot(self):
        """Record performance snapshot."""
        snapshot = {
            "timestamp": datetime.now(),
            "event_count": sum(metrics.count for metrics in self.metrics.values()),
            "error_count": sum(metrics.error_count for metrics in self.metrics.values()),
            "avg_processing_time": mean([
                metrics.avg_processing_time for metrics in self.metrics.values()
                if metrics.avg_processing_time > 0
            ]) if self.metrics else 0.0,
            "memory_usage_mb": self.system_health.memory_usage_mb,
            "cpu_usage_percent": self.system_health.cpu_usage_percent
        }
        
        self.performance_history.append(snapshot)
        
        # Keep only recent history (last 24 hours with 10-second intervals)
        max_history = 24 * 60 * 6  # 8640 entries
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]


class AlertManager:
    """Alert manager for event monitoring."""
    
    def __init__(self, monitor: EventMonitor):
        """
        Initialize alert manager.
        
        Args:
            monitor: Event monitor instance.
        """
        self.monitor = monitor
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
        self.checking = False
        self.check_thread: Optional[threading.Thread] = None
        
        self._setup_default_rules()
        logger.info("AlertManager initialized")
    
    def start(self):
        """Start alert checking."""
        if not self.checking:
            self.checking = True
            self.check_thread = threading.Thread(target=self._alert_check_loop)
            self.check_thread.daemon = True
            self.check_thread.start()
            logger.info("Alert checking started")
    
    def stop(self):
        """Stop alert checking."""
        self.checking = False
        if self.check_thread and self.check_thread.is_alive():
            self.check_thread.join(timeout=5)
        logger.info("Alert checking stopped")
    
    def add_rule(self, rule_name: str, condition: str, severity: str = "warning", 
                message: str = "", cooldown_minutes: int = 5):
        """
        Add alert rule.
        
        Args:
            rule_name: Name of the rule.
            condition: Alert condition.
            severity: Alert severity (info, warning, critical).
            message: Alert message.
            cooldown_minutes: Cooldown period in minutes.
        """
        rule = {
            "name": rule_name,
            "condition": condition,
            "severity": severity,
            "message": message,
            "cooldown_minutes": cooldown_minutes,
            "last_fired": None
        }
        self.alert_rules.append(rule)
        logger.info(f"Alert rule added: {rule_name}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active_count = len(self.active_alerts)
        critical_count = sum(1 for alert in self.active_alerts.values() 
                           if alert["severity"] == "critical")
        warning_count = sum(1 for alert in self.active_alerts.values() 
                          if alert["severity"] == "warning")
        
        return {
            "total_active": active_count,
            "critical": critical_count,
            "warning": warning_count,
            "info": active_count - critical_count - warning_count
        }
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        # High error rate
        self.add_rule(
            "high_error_rate",
            "error_rate > 0.1",
            "critical",
            "High error rate detected: {error_rate:.2%}"
        )
        
        # System health critical
        self.add_rule(
            "system_health_critical",
            "system_status == 'critical'",
            "critical",
            "System health is critical"
        )
        
        # High memory usage
        self.add_rule(
            "high_memory_usage",
            "memory_usage_mb > 1000",
            "warning",
            "High memory usage: {memory_usage_mb:.1f} MB"
        )
        
        # Event processing slow
        self.add_rule(
            "slow_event_processing",
            "avg_processing_time > 5.0",
            "warning",
            "Slow event processing: {avg_processing_time:.2f}s average"
        )
    
    def _alert_check_loop(self):
        """Alert checking loop."""
        while self.checking:
            try:
                self._check_alerts()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}")
                time.sleep(30)
    
    def _check_alerts(self):
        """Check alert rules."""
        current_time = datetime.now()
        dashboard_data = self.monitor.get_dashboard_data()
        
        for rule in self.alert_rules:
            try:
                # Check cooldown
                if (rule["last_fired"] and 
                    current_time - rule["last_fired"] < timedelta(minutes=rule["cooldown_minutes"])):
                    continue
                
                # Evaluate condition
                if self._evaluate_condition(rule["condition"], dashboard_data):
                    self._fire_alert(rule, dashboard_data)
                else:
                    self._resolve_alert(rule["name"])
                
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _evaluate_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """
        Evaluate alert condition.
        
        Args:
            condition: Condition string.
            data: Dashboard data.
            
        Returns:
            True if condition is met.
        """
        # Simple condition evaluation
        error_summary = data.get("error_summary", {})
        system_health = data.get("system_health", {})
        performance_stats = data.get("performance_stats", {})
        
        # Replace placeholders with actual values
        context = {
            "error_rate": error_summary.get("overall_error_rate", 0.0),
            "system_status": system_health.get("overall_status", "healthy"),
            "memory_usage_mb": system_health.get("memory_usage_mb", 0.0),
            "cpu_usage_percent": system_health.get("cpu_usage_percent", 0.0),
            "avg_processing_time": performance_stats.get("avg_processing_time", 0.0)
        }
        
        # Simple evaluation (extend as needed)
        try:
            # Replace variables in condition
            for var, value in context.items():
                if isinstance(value, str):
                    condition = condition.replace(var, f"'{value}'")
                else:
                    condition = condition.replace(var, str(value))
            
            return eval(condition)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _fire_alert(self, rule: Dict[str, Any], data: Dict[str, Any]):
        """
        Fire alert.
        
        Args:
            rule: Alert rule.
            data: Dashboard data.
        """
        alert_id = rule["name"]
        
        if alert_id not in self.active_alerts:
            # Format message
            message = rule["message"]
            try:
                error_summary = data.get("error_summary", {})
                system_health = data.get("system_health", {})
                performance_stats = data.get("performance_stats", {})
                
                format_vars = {
                    "error_rate": error_summary.get("overall_error_rate", 0.0),
                    "memory_usage_mb": system_health.get("memory_usage_mb", 0.0),
                    "cpu_usage_percent": system_health.get("cpu_usage_percent", 0.0),
                    "avg_processing_time": performance_stats.get("avg_processing_time", 0.0)
                }
                
                message = message.format(**format_vars)
                
            except Exception:
                pass  # Use original message if formatting fails
            
            alert = {
                "id": alert_id,
                "rule_name": rule["name"],
                "severity": rule["severity"],
                "message": message,
                "fired_at": datetime.now(),
                "status": "active"
            }
            
            self.active_alerts[alert_id] = alert
            rule["last_fired"] = datetime.now()
            
            logger.warning(f"Alert fired: {rule['name']} - {message}")
    
    def _resolve_alert(self, rule_name: str):
        """
        Resolve alert.
        
        Args:
            rule_name: Name of the rule.
        """
        if rule_name in self.active_alerts:
            alert = self.active_alerts.pop(rule_name)
            alert["status"] = "resolved"
            alert["resolved_at"] = datetime.now()
            
            logger.info(f"Alert resolved: {rule_name}")


# Global monitor instance
_event_monitor = None
_alert_manager = None


def get_event_monitor() -> EventMonitor:
    """Get global event monitor."""
    global _event_monitor
    if _event_monitor is None:
        _event_monitor = EventMonitor()
    return _event_monitor


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(get_event_monitor())
    return _alert_manager


def start_monitoring():
    """Start event monitoring and alerting."""
    monitor = get_event_monitor()
    alert_manager = get_alert_manager()
    
    monitor.start()
    alert_manager.start()
    
    logger.info("Event monitoring and alerting started")


def stop_monitoring():
    """Stop event monitoring and alerting."""
    monitor = get_event_monitor()
    alert_manager = get_alert_manager()
    
    monitor.stop()
    alert_manager.stop()
    
    logger.info("Event monitoring and alerting stopped")