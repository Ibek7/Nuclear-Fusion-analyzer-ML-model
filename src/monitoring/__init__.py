"""
Comprehensive Monitoring and Alerting System for Nuclear Fusion Analysis.

This module provides enterprise-grade monitoring capabilities including:
- Application Performance Monitoring (APM)
- Real-time metrics collection and aggregation
- Alerting framework with multiple notification channels
- Distributed tracing and structured logging
- SLA monitoring and compliance reporting
- Health monitoring and diagnostics
- Performance profiling and bottleneck identification
- Custom metrics and KPIs
- Dashboard integration and visualization
"""

from .dashboard import (
    FusionDashboard,
    RealTimeDataSimulator,
    DashboardData,
    create_fusion_dashboard
)

from .apm import (
    MetricsCollector,
    AlertManager,
    PerformanceProfiler,
    MetricType,
    AlertSeverity,
    AlertState,
    Metric,
    Alert,
    AlertRule,
    get_monitoring_system,
    start_monitoring,
    monitor_performance,
    count_calls,
    profile_execution
)

from .tracing import (
    DistributedTracer,
    StructuredLogger,
    LogAggregator,
    Span,
    Trace,
    TraceLevel,
    SpanType,
    get_tracer,
    get_logger,
    get_log_aggregator,
    trace_function,
    log_function_calls
)

from .sla import (
    SLAMonitor,
    SLAReporter,
    SLATarget,
    SLAMeasurement,
    SLAViolation,
    SLAReport,
    SLAMetricType,
    SLAStatus,
    get_sla_monitor,
    get_sla_reporter,
    record_api_response_time,
    record_system_availability,
    record_error_rate,
    record_fusion_analysis_time
)

__all__ = [
    # Dashboard components
    'FusionDashboard',
    'RealTimeDataSimulator', 
    'DashboardData',
    'create_fusion_dashboard',
    
    # APM components
    'MetricsCollector',
    'AlertManager',
    'PerformanceProfiler',
    'MetricType',
    'AlertSeverity',
    'AlertState',
    'Metric',
    'Alert',
    'AlertRule',
    'get_monitoring_system',
    'start_monitoring',
    'monitor_performance',
    'count_calls',
    'profile_execution',
    
    # Tracing components
    'DistributedTracer',
    'StructuredLogger',
    'LogAggregator',
    'Span',
    'Trace',
    'TraceLevel',
    'SpanType',
    'get_tracer',
    'get_logger',
    'get_log_aggregator',
    'trace_function',
    'log_function_calls',
    
    # SLA components
    'SLAMonitor',
    'SLAReporter',
    'SLATarget',
    'SLAMeasurement',
    'SLAViolation',
    'SLAReport',
    'SLAMetricType',
    'SLAStatus',
    'get_sla_monitor',
    'get_sla_reporter',
    'record_api_response_time',
    'record_system_availability',
    'record_error_rate',
    'record_fusion_analysis_time'
]

__version__ = "2.0.0"