"""
SLA Monitoring and Reporting System.

This module provides:
- Service Level Agreement (SLA) definition and monitoring
- SLA violation detection and alerting
- Availability and performance tracking
- SLA reporting and analytics
- Uptime monitoring
- Response time tracking
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import threading

import logging
logger = logging.getLogger(__name__)


class SLAMetricType(Enum):
    """SLA metric types."""
    AVAILABILITY = "availability"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    UPTIME = "uptime"


class SLAStatus(Enum):
    """SLA status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class SLATarget:
    """SLA target definition."""
    name: str
    description: str
    metric_type: SLAMetricType
    target_value: float
    comparison: str  # >, <, >=, <=
    measurement_window: int  # seconds
    evaluation_frequency: int = 60  # seconds
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLAMeasurement:
    """SLA measurement data point."""
    target_name: str
    timestamp: datetime
    measured_value: float
    target_value: float
    is_compliant: bool
    measurement_window: int
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLAViolation:
    """SLA violation record."""
    id: str
    target_name: str
    violation_start: datetime
    violation_end: Optional[datetime]
    duration: Optional[float]
    measured_value: float
    target_value: float
    severity: str = "medium"
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLAReport:
    """SLA compliance report."""
    target_name: str
    report_period: timedelta
    report_start: datetime
    report_end: datetime
    total_measurements: int
    compliant_measurements: int
    compliance_percentage: float
    violations: List[SLAViolation]
    average_value: float
    min_value: float
    max_value: float
    p95_value: float
    p99_value: float


class SLAMonitor:
    """Service Level Agreement monitoring system."""
    
    def __init__(self):
        """Initialize SLA monitor."""
        self.sla_targets: Dict[str, SLATarget] = {}
        self.measurements: List[SLAMeasurement] = []
        self.violations: List[SLAViolation] = []
        self.active_violations: Dict[str, SLAViolation] = {}
        
        self.monitoring_enabled = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.measurement_handlers: List[Callable[[SLAMeasurement], None]] = []
        self.violation_handlers: List[Callable[[SLAViolation], None]] = []
        
        # Data retention settings
        self.max_measurements = 100000
        self.max_violations = 10000
        self.measurement_retention_days = 30
        
        self._setup_default_targets()
        logger.info("SLAMonitor initialized")
    
    def add_target(self, target: SLATarget):
        """
        Add SLA target for monitoring.
        
        Args:
            target: SLA target definition.
        """
        self.sla_targets[target.name] = target
        logger.info(f"SLA target added: {target.name}")
    
    def remove_target(self, target_name: str):
        """
        Remove SLA target.
        
        Args:
            target_name: Name of target to remove.
        """
        if target_name in self.sla_targets:
            del self.sla_targets[target_name]
            logger.info(f"SLA target removed: {target_name}")
    
    def add_measurement_handler(self, handler: Callable[[SLAMeasurement], None]):
        """
        Add measurement handler.
        
        Args:
            handler: Handler function for measurements.
        """
        self.measurement_handlers.append(handler)
    
    def add_violation_handler(self, handler: Callable[[SLAViolation], None]):
        """
        Add violation handler.
        
        Args:
            handler: Handler function for violations.
        """
        self.violation_handlers.append(handler)
    
    def start_monitoring(self):
        """Start SLA monitoring."""
        if not self.monitoring_enabled:
            self.monitoring_enabled = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("SLA monitoring started")
    
    def stop_monitoring(self):
        """Stop SLA monitoring."""
        self.monitoring_enabled = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("SLA monitoring stopped")
    
    def record_measurement(self, target_name: str, value: float, 
                          timestamp: datetime = None, tags: Dict[str, str] = None):
        """
        Record SLA measurement.
        
        Args:
            target_name: Name of SLA target.
            value: Measured value.
            timestamp: Measurement timestamp.
            tags: Additional tags.
        """
        if target_name not in self.sla_targets:
            logger.warning(f"Unknown SLA target: {target_name}")
            return
        
        target = self.sla_targets[target_name]
        if not target.enabled:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check compliance
        is_compliant = self._evaluate_compliance(value, target)
        
        measurement = SLAMeasurement(
            target_name=target_name,
            timestamp=timestamp,
            measured_value=value,
            target_value=target.target_value,
            is_compliant=is_compliant,
            measurement_window=target.measurement_window,
            tags=tags or {}
        )
        
        self.measurements.append(measurement)
        
        # Check for violations
        if not is_compliant:
            self._handle_violation(target, measurement)
        else:
            self._check_violation_recovery(target_name)
        
        # Call measurement handlers
        for handler in self.measurement_handlers:
            try:
                handler(measurement)
            except Exception as e:
                logger.error(f"Error in measurement handler: {e}")
        
        # Cleanup old measurements
        self._cleanup_measurements()
    
    def get_compliance_status(self, target_name: str, 
                            time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Get compliance status for a target.
        
        Args:
            target_name: Name of SLA target.
            time_range: Time range for analysis.
            
        Returns:
            Compliance status data.
        """
        if target_name not in self.sla_targets:
            return {"error": f"Unknown SLA target: {target_name}"}
        
        target = self.sla_targets[target_name]
        cutoff_time = datetime.now() - time_range
        
        # Get recent measurements
        recent_measurements = [
            m for m in self.measurements
            if m.target_name == target_name and m.timestamp > cutoff_time
        ]
        
        if not recent_measurements:
            return {
                "target_name": target_name,
                "status": SLAStatus.UNKNOWN.value,
                "compliance_percentage": 0,
                "total_measurements": 0,
                "time_range_hours": time_range.total_seconds() / 3600
            }
        
        # Calculate compliance
        compliant_count = sum(1 for m in recent_measurements if m.is_compliant)
        compliance_percentage = (compliant_count / len(recent_measurements)) * 100
        
        # Determine status
        if compliance_percentage >= 99.9:
            status = SLAStatus.HEALTHY
        elif compliance_percentage >= 99.0:
            status = SLAStatus.WARNING
        else:
            status = SLAStatus.VIOLATED
        
        # Get recent violations
        recent_violations = [
            v for v in self.violations
            if v.target_name == target_name and v.violation_start > cutoff_time
        ]
        
        # Calculate statistics
        values = [m.measured_value for m in recent_measurements]
        
        return {
            "target_name": target_name,
            "status": status.value,
            "compliance_percentage": compliance_percentage,
            "total_measurements": len(recent_measurements),
            "compliant_measurements": compliant_count,
            "violations": len(recent_violations),
            "active_violation": target_name in self.active_violations,
            "target_value": target.target_value,
            "comparison": target.comparison,
            "current_value": values[-1] if values else None,
            "average_value": statistics.mean(values) if values else None,
            "min_value": min(values) if values else None,
            "max_value": max(values) if values else None,
            "time_range_hours": time_range.total_seconds() / 3600
        }
    
    def get_all_compliance_status(self, time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Get compliance status for all targets.
        
        Args:
            time_range: Time range for analysis.
            
        Returns:
            All compliance status data.
        """
        overall_status = {}
        target_statuses = {}
        
        for target_name in self.sla_targets:
            target_statuses[target_name] = self.get_compliance_status(target_name, time_range)
        
        # Calculate overall metrics
        total_targets = len(self.sla_targets)
        healthy_targets = sum(1 for s in target_statuses.values() 
                             if s.get("status") == SLAStatus.HEALTHY.value)
        violated_targets = sum(1 for s in target_statuses.values() 
                              if s.get("status") == SLAStatus.VIOLATED.value)
        
        return {
            "overall": {
                "total_targets": total_targets,
                "healthy_targets": healthy_targets,
                "violated_targets": violated_targets,
                "active_violations": len(self.active_violations),
                "health_percentage": (healthy_targets / total_targets * 100) if total_targets > 0 else 0
            },
            "targets": target_statuses,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_sla_report(self, target_name: str, 
                          report_period: timedelta = timedelta(days=7)) -> SLAReport:
        """
        Generate SLA compliance report.
        
        Args:
            target_name: Name of SLA target.
            report_period: Report period.
            
        Returns:
            SLA compliance report.
        """
        report_end = datetime.now()
        report_start = report_end - report_period
        
        # Get measurements for the period
        period_measurements = [
            m for m in self.measurements
            if m.target_name == target_name and report_start <= m.timestamp <= report_end
        ]
        
        # Get violations for the period
        period_violations = [
            v for v in self.violations
            if v.target_name == target_name and report_start <= v.violation_start <= report_end
        ]
        
        if not period_measurements:
            return SLAReport(
                target_name=target_name,
                report_period=report_period,
                report_start=report_start,
                report_end=report_end,
                total_measurements=0,
                compliant_measurements=0,
                compliance_percentage=0,
                violations=period_violations,
                average_value=0,
                min_value=0,
                max_value=0,
                p95_value=0,
                p99_value=0
            )
        
        # Calculate statistics
        compliant_count = sum(1 for m in period_measurements if m.is_compliant)
        compliance_percentage = (compliant_count / len(period_measurements)) * 100
        
        values = [m.measured_value for m in period_measurements]
        sorted_values = sorted(values)
        
        p95_index = int(len(sorted_values) * 0.95)
        p99_index = int(len(sorted_values) * 0.99)
        
        return SLAReport(
            target_name=target_name,
            report_period=report_period,
            report_start=report_start,
            report_end=report_end,
            total_measurements=len(period_measurements),
            compliant_measurements=compliant_count,
            compliance_percentage=compliance_percentage,
            violations=period_violations,
            average_value=statistics.mean(values),
            min_value=min(values),
            max_value=max(values),
            p95_value=sorted_values[p95_index] if p95_index < len(sorted_values) else max(values),
            p99_value=sorted_values[p99_index] if p99_index < len(sorted_values) else max(values)
        )
    
    def acknowledge_violation(self, violation_id: str, acknowledged_by: str):
        """
        Acknowledge SLA violation.
        
        Args:
            violation_id: Violation ID.
            acknowledged_by: User acknowledging the violation.
        """
        # Find violation in history
        for violation in self.violations:
            if violation.id == violation_id:
                violation.acknowledged = True
                violation.acknowledged_by = acknowledged_by
                violation.acknowledged_at = datetime.now()
                logger.info(f"SLA violation acknowledged: {violation_id} by {acknowledged_by}")
                return
        
        # Check active violations
        for violation in self.active_violations.values():
            if violation.id == violation_id:
                violation.acknowledged = True
                violation.acknowledged_by = acknowledged_by
                violation.acknowledged_at = datetime.now()
                logger.info(f"Active SLA violation acknowledged: {violation_id} by {acknowledged_by}")
                return
        
        logger.warning(f"SLA violation not found: {violation_id}")
    
    def get_violation_summary(self, time_range: timedelta = timedelta(days=1)) -> Dict[str, Any]:
        """
        Get violation summary.
        
        Args:
            time_range: Time range for analysis.
            
        Returns:
            Violation summary.
        """
        cutoff_time = datetime.now() - time_range
        
        recent_violations = [
            v for v in self.violations
            if v.violation_start > cutoff_time
        ]
        
        # Group by target
        violations_by_target = {}
        for violation in recent_violations:
            target = violation.target_name
            if target not in violations_by_target:
                violations_by_target[target] = []
            violations_by_target[target].append(violation)
        
        # Calculate total downtime
        total_downtime = sum(
            v.duration for v in recent_violations
            if v.duration is not None
        )
        
        return {
            "total_violations": len(recent_violations),
            "active_violations": len(self.active_violations),
            "violations_by_target": {
                target: len(violations) 
                for target, violations in violations_by_target.items()
            },
            "total_downtime_seconds": total_downtime,
            "unacknowledged_violations": len([
                v for v in recent_violations if not v.acknowledged
            ]),
            "time_range_hours": time_range.total_seconds() / 3600
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # This would typically collect metrics from external sources
                # For now, we'll just clean up old data
                self._cleanup_measurements()
                self._cleanup_violations()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in SLA monitoring loop: {e}")
                time.sleep(60)
    
    def _evaluate_compliance(self, value: float, target: SLATarget) -> bool:
        """
        Evaluate if a value meets SLA compliance.
        
        Args:
            value: Measured value.
            target: SLA target.
            
        Returns:
            True if compliant.
        """
        if target.comparison == ">":
            return value > target.target_value
        elif target.comparison == "<":
            return value < target.target_value
        elif target.comparison == ">=":
            return value >= target.target_value
        elif target.comparison == "<=":
            return value <= target.target_value
        elif target.comparison == "==":
            return value == target.target_value
        else:
            logger.warning(f"Unknown comparison operator: {target.comparison}")
            return False
    
    def _handle_violation(self, target: SLATarget, measurement: SLAMeasurement):
        """
        Handle SLA violation.
        
        Args:
            target: SLA target.
            measurement: Non-compliant measurement.
        """
        target_name = target.name
        
        # Check if there's already an active violation
        if target_name in self.active_violations:
            # Update existing violation
            violation = self.active_violations[target_name]
            violation.measured_value = measurement.measured_value
        else:
            # Create new violation
            violation_id = f"sla_violation_{target_name}_{int(time.time())}"
            violation = SLAViolation(
                id=violation_id,
                target_name=target_name,
                violation_start=measurement.timestamp,
                measured_value=measurement.measured_value,
                target_value=measurement.target_value,
                tags=measurement.tags
            )
            
            self.active_violations[target_name] = violation
            
            # Call violation handlers
            for handler in self.violation_handlers:
                try:
                    handler(violation)
                except Exception as e:
                    logger.error(f"Error in violation handler: {e}")
            
            logger.warning(f"SLA violation started: {target_name}")
    
    def _check_violation_recovery(self, target_name: str):
        """
        Check if a violation has been resolved.
        
        Args:
            target_name: Name of SLA target.
        """
        if target_name in self.active_violations:
            violation = self.active_violations[target_name]
            violation.violation_end = datetime.now()
            violation.duration = (violation.violation_end - violation.violation_start).total_seconds()
            
            # Move to violation history
            self.violations.append(violation)
            del self.active_violations[target_name]
            
            logger.info(f"SLA violation resolved: {target_name} after {violation.duration:.2f}s")
    
    def _cleanup_measurements(self):
        """Clean up old measurements."""
        if len(self.measurements) > self.max_measurements:
            self.measurements = self.measurements[-self.max_measurements:]
        
        # Remove measurements older than retention period
        cutoff_date = datetime.now() - timedelta(days=self.measurement_retention_days)
        self.measurements = [m for m in self.measurements if m.timestamp > cutoff_date]
    
    def _cleanup_violations(self):
        """Clean up old violations."""
        if len(self.violations) > self.max_violations:
            self.violations = self.violations[-self.max_violations:]
    
    def _setup_default_targets(self):
        """Setup default SLA targets."""
        default_targets = [
            SLATarget(
                name="api_response_time",
                description="API response time should be under 500ms",
                metric_type=SLAMetricType.RESPONSE_TIME,
                target_value=500.0,  # milliseconds
                comparison="<",
                measurement_window=300,  # 5 minutes
                tags={"service": "api"}
            ),
            SLATarget(
                name="system_availability",
                description="System availability should be above 99.9%",
                metric_type=SLAMetricType.AVAILABILITY,
                target_value=99.9,  # percentage
                comparison=">=",
                measurement_window=3600,  # 1 hour
                tags={"service": "system"}
            ),
            SLATarget(
                name="error_rate",
                description="Error rate should be below 1%",
                metric_type=SLAMetricType.ERROR_RATE,
                target_value=1.0,  # percentage
                comparison="<",
                measurement_window=600,  # 10 minutes
                tags={"service": "api"}
            ),
            SLATarget(
                name="fusion_analysis_time",
                description="Fusion analysis should complete within 30 seconds",
                metric_type=SLAMetricType.RESPONSE_TIME,
                target_value=30.0,  # seconds
                comparison="<",
                measurement_window=600,
                tags={"service": "fusion_analysis"}
            )
        ]
        
        for target in default_targets:
            self.add_target(target)


class SLAReporter:
    """SLA reporting and analytics."""
    
    def __init__(self, sla_monitor: SLAMonitor):
        """
        Initialize SLA reporter.
        
        Args:
            sla_monitor: SLA monitor instance.
        """
        self.sla_monitor = sla_monitor
        logger.info("SLAReporter initialized")
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for SLA dashboard.
        
        Returns:
            Dashboard data.
        """
        # Get current status
        current_status = self.sla_monitor.get_all_compliance_status(timedelta(hours=24))
        
        # Get violation summary
        violation_summary = self.sla_monitor.get_violation_summary(timedelta(days=7))
        
        # Generate trend data
        trend_data = {}
        for target_name in self.sla_monitor.sla_targets:
            hourly_compliance = []
            for hours_ago in range(24, 0, -1):
                start_time = timedelta(hours=hours_ago)
                end_time = timedelta(hours=hours_ago-1)
                status = self.sla_monitor.get_compliance_status(target_name, end_time)
                hourly_compliance.append({
                    "hour": hours_ago,
                    "compliance": status.get("compliance_percentage", 0)
                })
            trend_data[target_name] = hourly_compliance
        
        return {
            "current_status": current_status,
            "violation_summary": violation_summary,
            "trend_data": trend_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_weekly_report(self) -> str:
        """
        Generate weekly SLA report.
        
        Returns:
            Formatted report string.
        """
        report_lines = [
            "Nuclear Fusion Analyzer - Weekly SLA Report",
            "=" * 50,
            f"Report Period: {datetime.now() - timedelta(days=7)} to {datetime.now()}",
            ""
        ]
        
        # Overall summary
        overall_status = self.sla_monitor.get_all_compliance_status(timedelta(days=7))
        report_lines.extend([
            "OVERALL SLA HEALTH",
            "-" * 20,
            f"Total SLA Targets: {overall_status['overall']['total_targets']}",
            f"Healthy Targets: {overall_status['overall']['healthy_targets']}",
            f"Violated Targets: {overall_status['overall']['violated_targets']}",
            f"Health Percentage: {overall_status['overall']['health_percentage']:.1f}%",
            ""
        ])
        
        # Individual target reports
        report_lines.append("INDIVIDUAL TARGET PERFORMANCE")
        report_lines.append("-" * 30)
        
        for target_name in self.sla_monitor.sla_targets:
            target_report = self.sla_monitor.generate_sla_report(target_name, timedelta(days=7))
            
            report_lines.extend([
                f"{target_name}:",
                f"  Compliance: {target_report.compliance_percentage:.2f}%",
                f"  Measurements: {target_report.total_measurements}",
                f"  Violations: {len(target_report.violations)}",
                f"  Average Value: {target_report.average_value:.2f}",
                f"  P95: {target_report.p95_value:.2f}",
                f"  P99: {target_report.p99_value:.2f}",
                ""
            ])
        
        # Violation summary
        violation_summary = self.sla_monitor.get_violation_summary(timedelta(days=7))
        report_lines.extend([
            "VIOLATION SUMMARY",
            "-" * 20,
            f"Total Violations: {violation_summary['total_violations']}",
            f"Total Downtime: {violation_summary['total_downtime_seconds']:.2f} seconds",
            f"Unacknowledged: {violation_summary['unacknowledged_violations']}",
            ""
        ])
        
        return "\n".join(report_lines)


# Global SLA monitor instance
_sla_monitor = None
_sla_reporter = None


def get_sla_monitor() -> SLAMonitor:
    """Get global SLA monitor."""
    global _sla_monitor
    if _sla_monitor is None:
        _sla_monitor = SLAMonitor()
    return _sla_monitor


def get_sla_reporter() -> SLAReporter:
    """Get SLA reporter."""
    global _sla_reporter
    if _sla_reporter is None:
        _sla_reporter = SLAReporter(get_sla_monitor())
    return _sla_reporter


# Convenience functions
def record_api_response_time(response_time_ms: float):
    """Record API response time for SLA monitoring."""
    get_sla_monitor().record_measurement("api_response_time", response_time_ms)


def record_system_availability(availability_percentage: float):
    """Record system availability for SLA monitoring."""
    get_sla_monitor().record_measurement("system_availability", availability_percentage)


def record_error_rate(error_percentage: float):
    """Record error rate for SLA monitoring."""
    get_sla_monitor().record_measurement("error_rate", error_percentage)


def record_fusion_analysis_time(analysis_time_seconds: float):
    """Record fusion analysis time for SLA monitoring."""
    get_sla_monitor().record_measurement("fusion_analysis_time", analysis_time_seconds)