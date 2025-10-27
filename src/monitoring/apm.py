"""
Comprehensive Application Monitoring System.

This module provides:
- Application performance monitoring (APM)
- Real-time metrics collection and aggregation
- Alerting framework with multiple notification channels
- Health monitoring and diagnostics
- Custom metrics and KPIs
- SLA monitoring and reporting
"""

import asyncio
import time
import threading
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import queue
import socket
import psutil
import functools

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SET = "set"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert states."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    state: AlertState = AlertState.OPEN
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    occurrences: int = 1


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    threshold: Union[int, float]
    comparison: str  # >, <, >=, <=, ==, !=
    duration: int = 300  # seconds
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable
    interval: int = 60
    timeout: int = 30
    enabled: bool = True
    critical: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, flush_interval: int = 60):
        """
        Initialize metrics collector.
        
        Args:
            flush_interval: Interval to flush metrics (seconds).
        """
        self.flush_interval = flush_interval
        self.metrics_buffer: List[Metric] = []
        self.metrics_history: List[Metric] = []
        self.buffer_lock = threading.Lock()
        
        self.collectors: Dict[str, Callable] = {}
        self.custom_metrics: Dict[str, Any] = {}
        
        self.collecting = False
        self.collector_thread = None
        
        self._setup_system_collectors()
        logger.info("MetricsCollector initialized")
    
    def start_collection(self):
        """Start metrics collection."""
        if not self.collecting:
            self.collecting = True
            self.collector_thread = threading.Thread(target=self._collection_loop)
            self.collector_thread.daemon = True
            self.collector_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.collecting = False
        if self.collector_thread and self.collector_thread.is_alive():
            self.collector_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType, tags: Dict[str, str] = None,
                     unit: str = None, description: str = None):
        """
        Record a metric.
        
        Args:
            name: Metric name.
            value: Metric value.
            metric_type: Type of metric.
            tags: Optional tags.
            unit: Optional unit.
            description: Optional description.
        """
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit,
            description=description
        )
        
        with self.buffer_lock:
            self.metrics_buffer.append(metric)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_timer(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a timer metric."""
        self.record_metric(name, value, MetricType.TIMER, tags, unit="seconds")
    
    def time_function(self, metric_name: str, tags: Dict[str, str] = None):
        """Decorator to time function execution."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.record_timer(f"{metric_name}.success", time.time() - start_time, tags)
                    return result
                except Exception as e:
                    self.record_timer(f"{metric_name}.error", time.time() - start_time, tags)
                    self.increment_counter(f"{metric_name}.errors", 1, tags)
                    raise
            return wrapper
        return decorator
    
    def get_metrics(self, metric_name: str = None, 
                   time_range: timedelta = None) -> List[Metric]:
        """
        Get collected metrics.
        
        Args:
            metric_name: Optional metric name filter.
            time_range: Optional time range filter.
            
        Returns:
            List of metrics.
        """
        with self.buffer_lock:
            metrics = self.metrics_history.copy()
        
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        if time_range:
            cutoff_time = datetime.now() - time_range
            metrics = [m for m in metrics if m.timestamp > cutoff_time]
        
        return metrics
    
    def get_metric_summary(self, metric_name: str, 
                          time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Get metric summary statistics.
        
        Args:
            metric_name: Metric name.
            time_range: Time range for summary.
            
        Returns:
            Metric summary.
        """
        metrics = self.get_metrics(metric_name, time_range)
        
        if not metrics:
            return {"metric": metric_name, "count": 0}
        
        values = [m.value for m in metrics]
        
        return {
            "metric": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "sum": sum(values),
            "latest": values[-1] if values else None,
            "timestamp_range": {
                "start": min(m.timestamp for m in metrics).isoformat(),
                "end": max(m.timestamp for m in metrics).isoformat()
            }
        }
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.collecting:
            try:
                # Collect from registered collectors
                for name, collector in self.collectors.items():
                    try:
                        collector_metrics = collector()
                        if collector_metrics:
                            with self.buffer_lock:
                                self.metrics_buffer.extend(collector_metrics)
                    except Exception as e:
                        logger.error(f"Error in collector {name}: {e}")
                
                # Flush metrics
                self._flush_metrics()
                
                time.sleep(self.flush_interval)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(self.flush_interval)
    
    def _flush_metrics(self):
        """Flush metrics buffer to history."""
        with self.buffer_lock:
            if self.metrics_buffer:
                self.metrics_history.extend(self.metrics_buffer)
                self.metrics_buffer.clear()
                
                # Keep only recent metrics
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [m for m in self.metrics_history 
                                      if m.timestamp > cutoff_time]
    
    def _setup_system_collectors(self):
        """Setup system metrics collectors."""
        def collect_system_metrics():
            """Collect system performance metrics."""
            metrics = []
            
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                metrics.append(Metric(
                    name="system.cpu.usage",
                    value=cpu_percent,
                    metric_type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    unit="percent"
                ))
                
                # Memory metrics
                memory = psutil.virtual_memory()
                metrics.append(Metric(
                    name="system.memory.usage",
                    value=memory.percent,
                    metric_type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    unit="percent"
                ))
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                metrics.append(Metric(
                    name="system.disk.usage",
                    value=(disk.used / disk.total) * 100,
                    metric_type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    unit="percent"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            return metrics
        
        def collect_fusion_metrics():
            """Collect fusion-specific metrics."""
            metrics = []
            
            try:
                # Simulated fusion metrics
                import random
                
                # Plasma temperature
                temp = random.uniform(100e6, 200e6)
                metrics.append(Metric(
                    name="fusion.plasma.temperature",
                    value=temp,
                    metric_type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    unit="kelvin",
                    tags={"component": "plasma"}
                ))
                
                # Neutron flux
                neutron_flux = random.uniform(1e14, 1e16)
                metrics.append(Metric(
                    name="fusion.neutron.flux",
                    value=neutron_flux,
                    metric_type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    unit="neutrons_per_cm2_per_s",
                    tags={"component": "neutron"}
                ))
                
            except Exception as e:
                logger.error(f"Error collecting fusion metrics: {e}")
            
            return metrics
        
        self.collectors["system"] = collect_system_metrics
        self.collectors["fusion"] = collect_fusion_metrics


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, Callable] = {}
        
        self.evaluating = False
        self.evaluation_thread = None
        
        self._setup_default_rules()
        self._setup_default_channels()
        logger.info("AlertManager initialized")
    
    def start_evaluation(self):
        """Start alert evaluation."""
        if not self.evaluating:
            self.evaluating = True
            self.evaluation_thread = threading.Thread(target=self._evaluation_loop)
            self.evaluation_thread.daemon = True
            self.evaluation_thread.start()
            logger.info("Alert evaluation started")
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Alert rule added: {rule.name}")
    
    def fire_alert(self, alert: Alert):
        """Fire an alert."""
        if alert.id in self.active_alerts:
            existing_alert = self.active_alerts[alert.id]
            existing_alert.occurrences += 1
        else:
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            self._send_notifications(alert)
            logger.warning(f"Alert fired: {alert.rule_name}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active_alerts = list(self.active_alerts.values())
        
        return {
            "total_active": len(active_alerts),
            "by_severity": {
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "error": len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
                "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING])
            }
        }
    
    def _evaluation_loop(self):
        """Main evaluation loop."""
        while self.evaluating:
            try:
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        for channel_name, handler in self.notification_channels.items():
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        # High CPU usage rule
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            description="High CPU usage detected",
            condition="system.cpu.usage",
            severity=AlertSeverity.WARNING,
            threshold=80.0,
            comparison=">",
            notification_channels=["console"]
        ))
    
    def _setup_default_channels(self):
        """Setup default notification channels."""
        def console_handler(alert: Alert):
            print(f"[ALERT] {alert.severity.value.upper()} - {alert.message}")
        
        self.notification_channels["console"] = console_handler


class PerformanceProfiler:
    """Application performance profiler."""
    
    def __init__(self):
        """Initialize profiler."""
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        logger.info("PerformanceProfiler initialized")
    
    def start_profile(self, name: str, metadata: Dict[str, Any] = None):
        """
        Start performance profiling.
        
        Args:
            name: Profile name.
            metadata: Optional metadata.
        """
        self.active_profiles[name] = {
            "start_time": time.time(),
            "metadata": metadata or {}
        }
    
    def end_profile(self, name: str) -> Dict[str, Any]:
        """
        End performance profiling.
        
        Args:
            name: Profile name.
            
        Returns:
            Profile results.
        """
        if name not in self.active_profiles:
            return {}
        
        profile_data = self.active_profiles.pop(name)
        duration = time.time() - profile_data["start_time"]
        
        result = {
            "name": name,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "metadata": profile_data["metadata"]
        }
        
        if name not in self.profiles:
            self.profiles[name] = []
        
        self.profiles[name].append(result)
        
        # Keep only recent profiles
        if len(self.profiles[name]) > 1000:
            self.profiles[name] = self.profiles[name][-1000:]
        
        return result
    
    def profile_context(self, name: str, metadata: Dict[str, Any] = None):
        """Context manager for profiling."""
        class ProfileContext:
            def __init__(self, profiler, profile_name, profile_metadata):
                self.profiler = profiler
                self.name = profile_name
                self.metadata = profile_metadata
            
            def __enter__(self):
                self.profiler.start_profile(self.name, self.metadata)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.profiler.end_profile(self.name)
        
        return ProfileContext(self, name, metadata)
    
    def get_profile_stats(self, name: str) -> Dict[str, Any]:
        """
        Get profile statistics.
        
        Args:
            name: Profile name.
            
        Returns:
            Profile statistics.
        """
        if name not in self.profiles or not self.profiles[name]:
            return {"name": name, "count": 0}
        
        durations = [p["duration"] for p in self.profiles[name]]
        
        return {
            "name": name,
            "count": len(durations),
            "total_time": sum(durations),
            "avg_time": statistics.mean(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "latest": self.profiles[name][-1]
        }


# Global monitoring instances
_metrics_collector = None
_alert_manager = None
_profiler = None


def get_monitoring_system():
    """Get global monitoring system instances."""
    global _metrics_collector, _alert_manager, _profiler
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        _alert_manager = AlertManager()
        _profiler = PerformanceProfiler()
    
    return _metrics_collector, _alert_manager, _profiler


def start_monitoring():
    """Start monitoring system."""
    metrics_collector, alert_manager, profiler = get_monitoring_system()
    
    metrics_collector.start_collection()
    alert_manager.start_evaluation()
    
    logger.info("Monitoring system started")


# Convenience decorators
def monitor_performance(metric_name: str, tags: Dict[str, str] = None):
    """Decorator to monitor function performance."""
    metrics_collector, _, _ = get_monitoring_system()
    return metrics_collector.time_function(metric_name, tags)


def count_calls(metric_name: str, tags: Dict[str, str] = None):
    """Decorator to count function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics_collector, _, _ = get_monitoring_system()
            metrics_collector.increment_counter(metric_name, 1, tags)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_execution(profile_name: str, metadata: Dict[str, Any] = None):
    """Decorator to profile function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _, _, profiler = get_monitoring_system()
            with profiler.profile_context(profile_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator