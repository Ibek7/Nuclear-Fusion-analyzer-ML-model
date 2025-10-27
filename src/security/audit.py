"""
Security audit logging and compliance monitoring.

This module provides comprehensive security event logging,
audit trails, and compliance monitoring capabilities.
"""

import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import uuid
from pathlib import Path
import gzip
import threading
from queue import Queue, Empty
import os

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Security audit event types."""
    
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    
    # Session events
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SESSION_REVOKED = "session_revoked"
    SESSION_HIJACK_ATTEMPT = "session_hijack_attempt"
    
    # API events
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_USED = "api_key_used"
    API_RATE_LIMIT_EXCEEDED = "api_rate_limit_exceeded"
    
    # Data events
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    MALICIOUS_REQUEST = "malicious_request"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Compliance events
    GDPR_REQUEST = "gdpr_request"
    DATA_RETENTION_POLICY = "data_retention_policy"
    COMPLIANCE_VIOLATION = "compliance_violation"


class SeverityLevel(Enum):
    """Audit event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Security audit event."""
    
    event_id: str
    event_type: AuditEventType
    severity: SeverityLevel
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    session_id: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    risk_score: Optional[float] = None
    geo_location: Optional[str] = None
    device_info: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Generate hash for integrity
        self.integrity_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate integrity hash for the event."""
        # Create a canonical representation
        data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'action': self.action,
            'outcome': self.outcome
        }
        
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary."""
        data['event_type'] = AuditEventType(data['event_type'])
        data['severity'] = SeverityLevel(data['severity'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AuditConfig:
    """Audit logging configuration."""
    
    # Storage settings
    log_directory: str = "audit_logs"
    max_file_size_mb: int = 100
    max_files: int = 10
    compress_old_files: bool = True
    
    # Security settings
    encrypt_logs: bool = True
    enable_integrity_checking: bool = True
    enable_real_time_alerts: bool = True
    
    # Retention settings
    retention_days: int = 365
    archive_after_days: int = 90
    
    # Performance settings
    async_logging: bool = True
    batch_size: int = 100
    flush_interval_seconds: int = 30
    
    # Monitoring settings
    enable_metrics: bool = True
    alert_on_critical: bool = True
    alert_on_high_risk: bool = True
    risk_threshold: float = 0.8


class AuditLogger:
    """
    Security audit logger.
    
    Provides secure, tamper-resistant audit logging.
    """
    
    def __init__(self, config: AuditConfig):
        """
        Initialize audit logger.
        
        Args:
            config: Audit configuration.
        """
        self.config = config
        self.log_directory = Path(config.log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Event queue for async logging
        self.event_queue: Queue = Queue()
        self.logging_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Current log file
        self.current_log_file: Optional[Path] = None
        self.current_file_size = 0
        
        # Event handlers
        self.event_handlers: List[Callable[[AuditEvent], None]] = []
        
        logger.info("AuditLogger initialized")
    
    def start(self):
        """Start audit logging service."""
        if self.config.async_logging and not self.running:
            self.running = True
            self.logging_thread = threading.Thread(target=self._logging_worker, daemon=True)
            self.logging_thread.start()
        
        logger.info("AuditLogger started")
    
    def stop(self):
        """Stop audit logging service."""
        if self.running:
            self.running = False
            
            # Signal thread to stop
            self.event_queue.put(None)
            
            if self.logging_thread:
                self.logging_thread.join(timeout=5)
        
        logger.info("AuditLogger stopped")
    
    def add_event_handler(self, handler: Callable[[AuditEvent], None]):
        """
        Add event handler for real-time processing.
        
        Args:
            handler: Event handler function.
        """
        self.event_handlers.append(handler)
    
    def log_event(self, event: AuditEvent):
        """
        Log security audit event.
        
        Args:
            event: Audit event to log.
        """
        if self.config.async_logging:
            self.event_queue.put(event)
        else:
            self._write_event(event)
        
        # Call event handlers
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def _logging_worker(self):
        """Background logging worker thread."""
        batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Get event with timeout
                event = self.event_queue.get(timeout=1)
                
                if event is None:  # Stop signal
                    break
                
                batch.append(event)
                
                # Flush batch if size or time threshold reached
                if (len(batch) >= self.config.batch_size or
                    time.time() - last_flush >= self.config.flush_interval_seconds):
                    
                    self._write_batch(batch)
                    batch.clear()
                    last_flush = time.time()
                
            except Empty:
                # Flush any pending events
                if batch:
                    self._write_batch(batch)
                    batch.clear()
                    last_flush = time.time()
            
            except Exception as e:
                logger.error(f"Logging worker error: {e}")
        
        # Final flush
        if batch:
            self._write_batch(batch)
    
    def _write_event(self, event: AuditEvent):
        """Write single event to log."""
        self._write_batch([event])
    
    def _write_batch(self, events: List[AuditEvent]):
        """Write batch of events to log."""
        if not events:
            return
        
        try:
            # Ensure log file is available
            log_file = self._get_current_log_file()
            
            # Write events
            with open(log_file, 'a', encoding='utf-8') as f:
                for event in events:
                    event_json = json.dumps(event.to_dict())
                    f.write(event_json + '\n')
                    self.current_file_size += len(event_json) + 1
            
            # Check if rotation is needed
            self._check_rotation()
            
        except Exception as e:
            logger.error(f"Failed to write audit events: {e}")
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path."""
        if self.current_log_file is None or not self.current_log_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_log_file = self.log_directory / f"audit_{timestamp}.log"
            self.current_file_size = 0
        
        return self.current_log_file
    
    def _check_rotation(self):
        """Check if log rotation is needed."""
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        
        if self.current_file_size >= max_size_bytes:
            self._rotate_logs()
    
    def _rotate_logs(self):
        """Rotate log files."""
        # Compress current file if configured
        if self.config.compress_old_files and self.current_log_file:
            compressed_path = self.current_log_file.with_suffix('.log.gz')
            
            with open(self.current_log_file, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            self.current_log_file.unlink()
        
        # Create new log file
        self.current_log_file = None
        self.current_file_size = 0
        
        # Clean up old files
        self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """Clean up old log files."""
        log_files = list(self.log_directory.glob("audit_*.log*"))
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove files exceeding max count
        for old_file in log_files[self.config.max_files:]:
            try:
                old_file.unlink()
                logger.info(f"Removed old audit log: {old_file}")
            except Exception as e:
                logger.error(f"Failed to remove old log {old_file}: {e}")


class RiskAssessment:
    """
    Security risk assessment for audit events.
    
    Calculates risk scores based on event patterns and context.
    """
    
    def __init__(self):
        """Initialize risk assessment."""
        self.risk_weights = {
            AuditEventType.LOGIN_FAILURE: 0.3,
            AuditEventType.ACCOUNT_LOCKED: 0.7,
            AuditEventType.ACCESS_DENIED: 0.4,
            AuditEventType.SESSION_HIJACK_ATTEMPT: 0.9,
            AuditEventType.SECURITY_VIOLATION: 0.8,
            AuditEventType.INTRUSION_ATTEMPT: 0.9,
            AuditEventType.MALICIOUS_REQUEST: 0.7,
            AuditEventType.API_RATE_LIMIT_EXCEEDED: 0.5,
        }
        
        # Track event patterns for anomaly detection
        self.user_patterns: Dict[str, List[AuditEvent]] = {}
        self.ip_patterns: Dict[str, List[AuditEvent]] = {}
        
        logger.info("RiskAssessment initialized")
    
    def assess_risk(self, event: AuditEvent) -> float:
        """
        Assess risk score for audit event.
        
        Args:
            event: Audit event to assess.
            
        Returns:
            Risk score between 0 and 1.
        """
        base_risk = self.risk_weights.get(event.event_type, 0.1)
        
        # Adjust for severity
        severity_multiplier = {
            SeverityLevel.LOW: 0.5,
            SeverityLevel.MEDIUM: 1.0,
            SeverityLevel.HIGH: 1.5,
            SeverityLevel.CRITICAL: 2.0
        }
        
        risk_score = base_risk * severity_multiplier[event.severity]
        
        # Pattern-based adjustments
        risk_score += self._assess_pattern_risk(event)
        
        # Time-based adjustments
        risk_score += self._assess_time_risk(event)
        
        # Geographic adjustments
        risk_score += self._assess_geo_risk(event)
        
        return min(risk_score, 1.0)
    
    def _assess_pattern_risk(self, event: AuditEvent) -> float:
        """Assess risk based on behavioral patterns."""
        additional_risk = 0.0
        
        # Check for repeated failures
        if event.user_id:
            user_events = self.user_patterns.get(event.user_id, [])
            
            # Count recent failures
            recent_failures = sum(
                1 for e in user_events[-10:]
                if e.outcome == 'failure' and
                (event.timestamp - e.timestamp).total_seconds() < 3600
            )
            
            if recent_failures >= 3:
                additional_risk += 0.3
            elif recent_failures >= 5:
                additional_risk += 0.5
        
        # Check for IP-based patterns
        ip_events = self.ip_patterns.get(event.ip_address, [])
        
        # High frequency from same IP
        recent_events = [
            e for e in ip_events
            if (event.timestamp - e.timestamp).total_seconds() < 300
        ]
        
        if len(recent_events) > 50:
            additional_risk += 0.4
        
        return additional_risk
    
    def _assess_time_risk(self, event: AuditEvent) -> float:
        """Assess risk based on timing patterns."""
        # Off-hours access (configurable)
        hour = event.timestamp.hour
        
        if hour < 6 or hour > 22:  # Outside business hours
            return 0.2
        
        return 0.0
    
    def _assess_geo_risk(self, event: AuditEvent) -> float:
        """Assess risk based on geographic patterns."""
        # Would integrate with GeoIP service
        # For now, return 0
        return 0.0
    
    def update_patterns(self, event: AuditEvent):
        """Update behavioral patterns."""
        if event.user_id:
            if event.user_id not in self.user_patterns:
                self.user_patterns[event.user_id] = []
            
            self.user_patterns[event.user_id].append(event)
            
            # Keep only recent events
            cutoff = event.timestamp - timedelta(days=7)
            self.user_patterns[event.user_id] = [
                e for e in self.user_patterns[event.user_id]
                if e.timestamp > cutoff
            ]
        
        # Update IP patterns
        if event.ip_address not in self.ip_patterns:
            self.ip_patterns[event.ip_address] = []
        
        self.ip_patterns[event.ip_address].append(event)
        
        # Keep only recent events
        cutoff = event.timestamp - timedelta(hours=24)
        self.ip_patterns[event.ip_address] = [
            e for e in self.ip_patterns[event.ip_address]
            if e.timestamp > cutoff
        ]


class ComplianceMonitor:
    """
    Compliance monitoring for security standards.
    
    Monitors adherence to GDPR, SOC2, ISO27001, etc.
    """
    
    def __init__(self):
        """Initialize compliance monitor."""
        self.compliance_rules: Dict[str, Callable[[AuditEvent], bool]] = {}
        self.violations: List[Dict[str, Any]] = []
        
        self._setup_default_rules()
        
        logger.info("ComplianceMonitor initialized")
    
    def _setup_default_rules(self):
        """Setup default compliance rules."""
        # GDPR rules
        self.compliance_rules['gdpr_data_access_logged'] = self._check_data_access_logged
        self.compliance_rules['gdpr_consent_tracked'] = self._check_consent_tracked
        
        # SOC2 rules
        self.compliance_rules['soc2_access_control'] = self._check_access_control
        self.compliance_rules['soc2_monitoring'] = self._check_monitoring
        
        # General security rules
        self.compliance_rules['failed_login_monitoring'] = self._check_failed_login_monitoring
        self.compliance_rules['privileged_access_monitoring'] = self._check_privileged_access
    
    def check_compliance(self, event: AuditEvent) -> List[str]:
        """
        Check event against compliance rules.
        
        Args:
            event: Audit event to check.
            
        Returns:
            List of violated rules.
        """
        violations = []
        
        for rule_name, rule_func in self.compliance_rules.items():
            try:
                if not rule_func(event):
                    violations.append(rule_name)
                    
                    # Record violation
                    self.violations.append({
                        'rule': rule_name,
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'details': event.details
                    })
                    
            except Exception as e:
                logger.error(f"Compliance rule {rule_name} error: {e}")
        
        return violations
    
    def _check_data_access_logged(self, event: AuditEvent) -> bool:
        """Check if data access is properly logged."""
        if event.event_type == AuditEventType.DATA_ACCESS:
            return all(key in event.details for key in ['resource', 'action', 'user_id'])
        return True
    
    def _check_consent_tracked(self, event: AuditEvent) -> bool:
        """Check if user consent is tracked."""
        # Implementation would check for consent tracking
        return True
    
    def _check_access_control(self, event: AuditEvent) -> bool:
        """Check access control compliance."""
        if event.event_type in [AuditEventType.ACCESS_GRANTED, AuditEventType.ACCESS_DENIED]:
            return 'authorization_check' in event.details
        return True
    
    def _check_monitoring(self, event: AuditEvent) -> bool:
        """Check monitoring compliance."""
        # All events should have proper monitoring
        return event.user_id is not None or event.ip_address is not None
    
    def _check_failed_login_monitoring(self, event: AuditEvent) -> bool:
        """Check failed login monitoring."""
        if event.event_type == AuditEventType.LOGIN_FAILURE:
            return 'attempt_count' in event.details
        return True
    
    def _check_privileged_access(self, event: AuditEvent) -> bool:
        """Check privileged access monitoring."""
        if 'admin' in event.details.get('roles', []):
            return event.event_type != AuditEventType.ACCESS_GRANTED
        return True


def create_audit_system(config: AuditConfig) -> Dict[str, Any]:
    """
    Create comprehensive audit system.
    
    Args:
        config: Audit configuration.
        
    Returns:
        Dictionary with audit components.
    """
    audit_logger = AuditLogger(config)
    risk_assessment = RiskAssessment()
    compliance_monitor = ComplianceMonitor()
    
    def enhanced_event_handler(event: AuditEvent):
        """Enhanced event handler with risk and compliance."""
        # Assess risk
        event.risk_score = risk_assessment.assess_risk(event)
        risk_assessment.update_patterns(event)
        
        # Check compliance
        violations = compliance_monitor.check_compliance(event)
        if violations:
            event.details['compliance_violations'] = violations
        
        # Alert on high risk or critical events
        if (event.risk_score and event.risk_score > config.risk_threshold) or \
           event.severity == SeverityLevel.CRITICAL:
            logger.warning(f"High-risk audit event: {event.event_type.value} "
                         f"(risk: {event.risk_score}, severity: {event.severity.value})")
    
    # Add enhanced handler
    audit_logger.add_event_handler(enhanced_event_handler)
    
    return {
        'logger': audit_logger,
        'risk_assessment': risk_assessment,
        'compliance_monitor': compliance_monitor,
        'config': config
    }


def create_audit_event(
    event_type: AuditEventType,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    session_id: Optional[str] = None,
    ip_address: str = "unknown",
    user_agent: Optional[str] = None,
    resource: Optional[str] = None,
    action: str = "unknown",
    outcome: str = "success",
    severity: SeverityLevel = SeverityLevel.MEDIUM,
    **details
) -> AuditEvent:
    """
    Create audit event with common parameters.
    
    Args:
        event_type: Type of audit event.
        user_id: User identifier.
        username: Username.
        session_id: Session identifier.
        ip_address: Client IP address.
        user_agent: Client user agent.
        resource: Resource being accessed.
        action: Action being performed.
        outcome: Outcome of the action.
        severity: Event severity level.
        **details: Additional event details.
        
    Returns:
        Created audit event.
    """
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        severity=severity,
        timestamp=datetime.now(timezone.utc),
        user_id=user_id,
        username=username,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        resource=resource,
        action=action,
        outcome=outcome,
        details=details
    )