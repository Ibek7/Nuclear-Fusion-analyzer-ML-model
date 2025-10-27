"""
Configuration Monitoring and Health Check System.

This module provides:
- Configuration health monitoring
- Performance impact tracking
- Configuration change notifications
- Automatic rollback capabilities
- Configuration drift detection
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
import time
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ChangeType(Enum):
    """Configuration change types."""
    VALUE_CHANGED = "value_changed"
    KEY_ADDED = "key_added"
    KEY_REMOVED = "key_removed"
    STRUCTURE_CHANGED = "structure_changed"


@dataclass
class HealthCheck:
    """Configuration health check definition."""
    name: str
    description: str
    check_function: Callable
    interval: int = 60  # seconds
    timeout: int = 30   # seconds
    enabled: bool = True
    critical: bool = False
    dependencies: List[str] = None
    
    def __post_init__(self):
        """Initialize health check."""
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class HealthReport:
    """Health check report."""
    check_name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize health report."""
        if self.details is None:
            self.details = {}


@dataclass
class ConfigurationChange:
    """Configuration change event."""
    timestamp: datetime
    change_type: ChangeType
    path: str
    old_value: Any = None
    new_value: Any = None
    source: str = "unknown"
    user: str = "system"


@dataclass
class PerformanceMetrics:
    """Configuration performance metrics."""
    timestamp: datetime
    config_load_time: float
    config_size: int
    memory_usage: float
    active_features: int
    error_count: int
    warning_count: int


class ConfigurationMonitor:
    """Monitors configuration health and performance."""
    
    def __init__(self, config_manager):
        """
        Initialize configuration monitor.
        
        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_reports: List[HealthReport] = []
        self.change_history: List[ConfigurationChange] = []
        self.performance_history: List[PerformanceMetrics] = []
        
        self.monitoring_enabled = True
        self.monitor_thread = None
        self.change_listeners: List[Callable] = []
        
        self._config_hash = None
        self._last_check_time = datetime.now()
        
        # Performance tracking
        self.performance_thresholds = {
            "config_load_time": 5.0,      # seconds
            "memory_usage": 1024 * 1024,  # bytes
            "error_rate": 0.05            # 5%
        }
        
        self._setup_default_health_checks()
        logger.info("ConfigurationMonitor initialized")
    
    def add_health_check(self, health_check: HealthCheck):
        """
        Add configuration health check.
        
        Args:
            health_check: Health check definition.
        """
        self.health_checks[health_check.name] = health_check
        logger.info(f"Health check added: {health_check.name}")
    
    def remove_health_check(self, name: str):
        """
        Remove health check.
        
        Args:
            name: Health check name.
        """
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Health check removed: {name}")
    
    def add_change_listener(self, listener: Callable):
        """
        Add configuration change listener.
        
        Args:
            listener: Change listener function.
        """
        self.change_listeners.append(listener)
    
    def start_monitoring(self):
        """Start configuration monitoring."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitoring_enabled = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Configuration monitoring started")
    
    def stop_monitoring(self):
        """Stop configuration monitoring."""
        self.monitoring_enabled = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Configuration monitoring stopped")
    
    def run_health_checks(self) -> List[HealthReport]:
        """
        Run all enabled health checks.
        
        Returns:
            List of health reports.
        """
        reports = []
        
        for check_name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            report = self._run_single_health_check(health_check)
            reports.append(report)
            
            # Store in history
            self.health_reports.append(report)
            
            # Keep only recent reports
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.health_reports = [r for r in self.health_reports if r.timestamp > cutoff_time]
        
        return reports
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get configuration health summary.
        
        Returns:
            Health summary data.
        """
        recent_reports = [r for r in self.health_reports 
                         if r.timestamp > datetime.now() - timedelta(minutes=10)]
        
        if not recent_reports:
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "total_checks": len(self.health_checks),
                "last_check": None,
                "issues": []
            }
        
        # Determine overall status
        statuses = [r.status for r in recent_reports]
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Collect issues
        issues = [r for r in recent_reports if r.status != HealthStatus.HEALTHY]
        
        return {
            "overall_status": overall_status.value,
            "total_checks": len(self.health_checks),
            "healthy_checks": len([r for r in recent_reports if r.status == HealthStatus.HEALTHY]),
            "warning_checks": len([r for r in recent_reports if r.status == HealthStatus.WARNING]),
            "critical_checks": len([r for r in recent_reports if r.status == HealthStatus.CRITICAL]),
            "last_check": max(r.timestamp for r in recent_reports).isoformat(),
            "issues": [{"check": r.check_name, "status": r.status.value, "message": r.message} 
                      for r in issues]
        }
    
    def detect_configuration_drift(self) -> List[ConfigurationChange]:
        """
        Detect configuration drift.
        
        Returns:
            List of detected changes.
        """
        changes = []
        
        try:
            # Get current configuration hash
            current_config = self.config_manager.get_all_settings()
            current_hash = self._calculate_config_hash(current_config)
            
            if self._config_hash is None:
                self._config_hash = current_hash
                return changes
            
            if current_hash != self._config_hash:
                # Configuration has changed
                change = ConfigurationChange(
                    timestamp=datetime.now(),
                    change_type=ChangeType.STRUCTURE_CHANGED,
                    path="root",
                    source="drift_detection"
                )
                changes.append(change)
                self.change_history.append(change)
                
                # Notify listeners
                for listener in self.change_listeners:
                    try:
                        listener(change)
                    except Exception as e:
                        logger.error(f"Error in change listener: {e}")
                
                self._config_hash = current_hash
        
        except Exception as e:
            logger.error(f"Error detecting configuration drift: {e}")
        
        return changes
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Collect current performance metrics.
        
        Returns:
            Performance metrics.
        """
        try:
            start_time = time.time()
            config = self.config_manager.get_all_settings()
            load_time = time.time() - start_time
            
            # Calculate metrics
            config_json = json.dumps(config)
            config_size = len(config_json.encode('utf-8'))
            
            # Memory usage (simplified)
            import sys
            memory_usage = sys.getsizeof(config)
            
            # Count active features
            feature_flags = config.get('feature_flags', {})
            active_features = sum(1 for flag in feature_flags.values() if flag)
            
            # Count recent errors/warnings
            recent_reports = [r for r in self.health_reports 
                             if r.timestamp > datetime.now() - timedelta(minutes=5)]
            error_count = len([r for r in recent_reports if r.status == HealthStatus.CRITICAL])
            warning_count = len([r for r in recent_reports if r.status == HealthStatus.WARNING])
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                config_load_time=load_time,
                config_size=config_size,
                memory_usage=memory_usage,
                active_features=active_features,
                error_count=error_count,
                warning_count=warning_count
            )
            
            # Store in history
            self.performance_history.append(metrics)
            
            # Keep only recent metrics
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.performance_history = [m for m in self.performance_history if m.timestamp > cutoff_time]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                config_load_time=0,
                config_size=0,
                memory_usage=0,
                active_features=0,
                error_count=1,
                warning_count=0
            )
    
    def check_performance_thresholds(self) -> List[str]:
        """
        Check if performance metrics exceed thresholds.
        
        Returns:
            List of threshold violations.
        """
        violations = []
        
        try:
            metrics = self.get_performance_metrics()
            
            if metrics.config_load_time > self.performance_thresholds["config_load_time"]:
                violations.append(f"Configuration load time too high: {metrics.config_load_time:.2f}s")
            
            if metrics.memory_usage > self.performance_thresholds["memory_usage"]:
                violations.append(f"Memory usage too high: {metrics.memory_usage} bytes")
            
            # Check error rate
            if len(self.performance_history) > 10:
                recent_metrics = self.performance_history[-10:]
                total_checks = sum(m.error_count + m.warning_count for m in recent_metrics)
                if total_checks > 0:
                    error_rate = sum(m.error_count for m in recent_metrics) / total_checks
                    if error_rate > self.performance_thresholds["error_rate"]:
                        violations.append(f"Error rate too high: {error_rate:.2%}")
        
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {e}")
            violations.append(f"Error checking thresholds: {e}")
        
        return violations
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring status.
        
        Returns:
            Monitoring status data.
        """
        health_summary = self.get_health_summary()
        performance_metrics = self.get_performance_metrics()
        performance_violations = self.check_performance_thresholds()
        
        recent_changes = [c for c in self.change_history 
                         if c.timestamp > datetime.now() - timedelta(hours=1)]
        
        return {
            "monitoring_enabled": self.monitoring_enabled,
            "last_check": self._last_check_time.isoformat(),
            "health": health_summary,
            "performance": asdict(performance_metrics),
            "performance_violations": performance_violations,
            "recent_changes": len(recent_changes),
            "total_health_checks": len(self.health_checks),
            "enabled_health_checks": len([c for c in self.health_checks.values() if c.enabled])
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Run health checks
                self.run_health_checks()
                
                # Detect configuration drift
                self.detect_configuration_drift()
                
                # Check performance
                self.check_performance_thresholds()
                
                self._last_check_time = datetime.now()
                
                # Sleep before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _run_single_health_check(self, health_check: HealthCheck) -> HealthReport:
        """
        Run single health check.
        
        Args:
            health_check: Health check to run.
            
        Returns:
            Health report.
        """
        start_time = time.time()
        
        try:
            # Run the check function
            result = health_check.check_function()
            duration = time.time() - start_time
            
            if isinstance(result, tuple):
                status, message, details = result
            elif isinstance(result, dict):
                status = result.get('status', HealthStatus.UNKNOWN)
                message = result.get('message', 'No message')
                details = result.get('details', {})
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                message = str(result)
                details = {}
            
            return HealthReport(
                check_name=health_check.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration=duration,
                details=details
            )
            
        except Exception as e:
            duration = time.time() - start_time
            status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
            
            return HealthReport(
                check_name=health_check.name,
                status=status,
                message=f"Health check failed: {e}",
                timestamp=datetime.now(),
                duration=duration,
                details={"error": str(e)}
            )
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Calculate configuration hash.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Configuration hash.
        """
        config_json = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        # Configuration accessibility check
        def check_config_access():
            try:
                config = self.config_manager.get_all_settings()
                return HealthStatus.HEALTHY, f"Configuration accessible ({len(config)} keys)", {}
            except Exception as e:
                return HealthStatus.CRITICAL, f"Cannot access configuration: {e}", {}
        
        self.add_health_check(HealthCheck(
            name="config_access",
            description="Check if configuration is accessible",
            check_function=check_config_access,
            interval=60,
            critical=True
        ))
        
        # Feature flags validation check
        def check_feature_flags():
            try:
                flags = self.config_manager.get_feature_flags()
                invalid_flags = [name for name, value in flags.items() if not isinstance(value, bool)]
                
                if invalid_flags:
                    return HealthStatus.WARNING, f"Invalid feature flags: {invalid_flags}", {}
                
                return HealthStatus.HEALTHY, f"All {len(flags)} feature flags valid", {}
            except Exception as e:
                return HealthStatus.WARNING, f"Error checking feature flags: {e}", {}
        
        self.add_health_check(HealthCheck(
            name="feature_flags",
            description="Validate feature flags",
            check_function=check_feature_flags,
            interval=300
        ))
        
        # Secrets availability check
        def check_secrets():
            try:
                # Check if required secrets are available
                required_secrets = ["database_password", "api_secret_key", "encryption_key"]
                missing_secrets = []
                
                for secret_name in required_secrets:
                    try:
                        self.config_manager.get_secret(secret_name)
                    except:
                        missing_secrets.append(secret_name)
                
                if missing_secrets:
                    return HealthStatus.WARNING, f"Missing secrets: {missing_secrets}", {}
                
                return HealthStatus.HEALTHY, "All required secrets available", {}
            except Exception as e:
                return HealthStatus.WARNING, f"Error checking secrets: {e}", {}
        
        self.add_health_check(HealthCheck(
            name="secrets",
            description="Check secrets availability",
            check_function=check_secrets,
            interval=600
        ))


class ConfigurationAlertManager:
    """Manages configuration-related alerts and notifications."""
    
    def __init__(self, monitor: ConfigurationMonitor):
        """
        Initialize alert manager.
        
        Args:
            monitor: Configuration monitor instance.
        """
        self.monitor = monitor
        self.alert_handlers: List[Callable] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_rules: List[Dict[str, Any]] = []
        
        self._setup_default_alert_rules()
        logger.info("ConfigurationAlertManager initialized")
    
    def add_alert_handler(self, handler: Callable):
        """
        Add alert handler.
        
        Args:
            handler: Alert handler function.
        """
        self.alert_handlers.append(handler)
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """
        Add alert rule.
        
        Args:
            rule: Alert rule definition.
        """
        self.alert_rules.append(rule)
    
    def check_alerts(self):
        """Check all alert rules and send notifications."""
        for rule in self.alert_rules:
            try:
                if self._evaluate_alert_rule(rule):
                    self._send_alert(rule)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.get('name', 'unknown')}: {e}")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        self.alert_rules = [
            {
                "name": "critical_health_check_failure",
                "condition": "critical_health_checks > 0",
                "message": "Critical health check failure detected",
                "severity": "critical"
            },
            {
                "name": "high_error_rate",
                "condition": "error_rate > 0.1",
                "message": "High configuration error rate detected",
                "severity": "warning"
            },
            {
                "name": "configuration_drift",
                "condition": "config_changes_last_hour > 5",
                "message": "High configuration change rate detected",
                "severity": "warning"
            }
        ]
    
    def _evaluate_alert_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Evaluate alert rule condition.
        
        Args:
            rule: Alert rule.
            
        Returns:
            True if alert should be triggered.
        """
        try:
            status = self.monitor.get_monitoring_status()
            
            # Simple condition evaluation
            condition = rule["condition"]
            
            if "critical_health_checks > 0" in condition:
                return status["health"]["critical_checks"] > 0
            
            if "error_rate > 0.1" in condition:
                violations = status.get("performance_violations", [])
                return any("Error rate too high" in v for v in violations)
            
            if "config_changes_last_hour > 5" in condition:
                return status["recent_changes"] > 5
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition {rule['condition']}: {e}")
            return False
    
    def _send_alert(self, rule: Dict[str, Any]):
        """
        Send alert notification.
        
        Args:
            rule: Alert rule that triggered.
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "rule": rule["name"],
            "message": rule["message"],
            "severity": rule["severity"],
            "details": self.monitor.get_monitoring_status()
        }
        
        # Store in history
        self.alert_history.append(alert)
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Configuration alert: {rule['name']} - {rule['message']}")


# Default alert handlers
def console_alert_handler(alert: Dict[str, Any]):
    """Console alert handler."""
    print(f"[ALERT] {alert['timestamp']} - {alert['rule']}: {alert['message']}")


def file_alert_handler(filename: str = "config_alerts.log"):
    """File alert handler factory."""
    def handler(alert: Dict[str, Any]):
        with open(filename, 'a') as f:
            f.write(f"{alert['timestamp']} - {alert['rule']}: {alert['message']}\n")
    return handler