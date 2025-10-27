"""
Distributed Tracing and Advanced Logging System.

This module provides:
- Distributed tracing with correlation IDs
- Structured logging with JSON formatting
- Log aggregation and analysis
- Request/response tracing
- Performance bottleneck identification
- Error tracking and aggregation
"""

import time
import json
import uuid
import threading
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import functools

logger = logging.getLogger(__name__)


class TraceLevel(Enum):
    """Trace levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SpanType(Enum):
    """Span types."""
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    CACHE_OPERATION = "cache_operation"
    EXTERNAL_SERVICE = "external_service"
    COMPUTATION = "computation"
    ML_INFERENCE = "ml_inference"
    FUSION_ANALYSIS = "fusion_analysis"


@dataclass
class Span:
    """Distributed tracing span."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    span_type: SpanType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None
    
    def finish(self):
        """Finish the span."""
        if self.end_time is None:
            self.end_time = datetime.now()
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    def log(self, message: str, level: TraceLevel = TraceLevel.INFO, **kwargs):
        """Add log entry to span."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def set_tag(self, key: str, value: Any):
        """Set span tag."""
        self.tags[key] = value
    
    def set_error(self, error: Exception):
        """Set span error."""
        self.status = "error"
        self.error = str(error)
        self.tags["error"] = True
        self.tags["error.type"] = type(error).__name__
        self.tags["error.message"] = str(error)
        self.log(f"Error: {error}", TraceLevel.ERROR, traceback=traceback.format_exc())


@dataclass
class Trace:
    """Distributed trace containing multiple spans."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def add_span(self, span: Span):
        """Add span to trace."""
        self.spans.append(span)
    
    def finish(self):
        """Finish the trace."""
        if self.end_time is None:
            self.end_time = datetime.now()
            self.duration = (self.end_time - self.start_time).total_seconds()
        
        # Ensure all spans are finished
        for span in self.spans:
            if span.end_time is None:
                span.finish()
    
    def get_root_span(self) -> Optional[Span]:
        """Get root span (span with no parent)."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None
    
    def get_span_tree(self) -> Dict[str, List[Span]]:
        """Get spans organized as a tree."""
        tree = {}
        for span in self.spans:
            parent_id = span.parent_span_id or "root"
            if parent_id not in tree:
                tree[parent_id] = []
            tree[parent_id].append(span)
        return tree


class DistributedTracer:
    """Distributed tracing system."""
    
    def __init__(self):
        """Initialize distributed tracer."""
        self.active_traces: Dict[str, Trace] = {}
        self.completed_traces: List[Trace] = []
        self.local_storage = threading.local()
        
        self.max_trace_retention = 1000
        self.trace_handlers: List[Callable[[Trace], None]] = []
        
        logger.info("DistributedTracer initialized")
    
    def start_trace(self, operation_name: str, trace_id: str = None) -> Trace:
        """
        Start a new trace.
        
        Args:
            operation_name: Name of the operation.
            trace_id: Optional trace ID.
            
        Returns:
            New trace instance.
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        trace = Trace(trace_id=trace_id)
        trace.tags["operation"] = operation_name
        
        # Create root span
        root_span = self.start_span(
            operation_name=operation_name,
            span_type=SpanType.HTTP_REQUEST,
            trace_id=trace_id
        )
        
        self.active_traces[trace_id] = trace
        self._set_current_trace(trace)
        
        logger.debug(f"Started trace: {trace_id}")
        return trace
    
    def start_span(self, operation_name: str, span_type: SpanType,
                   trace_id: str = None, parent_span_id: str = None) -> Span:
        """
        Start a new span.
        
        Args:
            operation_name: Name of the operation.
            span_type: Type of span.
            trace_id: Trace ID.
            parent_span_id: Parent span ID.
            
        Returns:
            New span instance.
        """
        # Get current trace context
        current_trace = self._get_current_trace()
        if current_trace and trace_id is None:
            trace_id = current_trace.trace_id
        
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            span_type=span_type,
            start_time=datetime.now()
        )
        
        # Add to trace
        if trace_id in self.active_traces:
            self.active_traces[trace_id].add_span(span)
        else:
            # Create new trace if it doesn't exist
            trace = Trace(trace_id=trace_id)
            trace.add_span(span)
            self.active_traces[trace_id] = trace
        
        self._set_current_span(span)
        
        logger.debug(f"Started span: {span_id} in trace: {trace_id}")
        return span
    
    def finish_span(self, span: Span):
        """
        Finish a span.
        
        Args:
            span: Span to finish.
        """
        span.finish()
        logger.debug(f"Finished span: {span.span_id}")
    
    def finish_trace(self, trace: Trace):
        """
        Finish a trace.
        
        Args:
            trace: Trace to finish.
        """
        trace.finish()
        
        # Move to completed traces
        if trace.trace_id in self.active_traces:
            del self.active_traces[trace.trace_id]
        
        self.completed_traces.append(trace)
        
        # Limit retention
        if len(self.completed_traces) > self.max_trace_retention:
            self.completed_traces = self.completed_traces[-self.max_trace_retention:]
        
        # Call handlers
        for handler in self.trace_handlers:
            try:
                handler(trace)
            except Exception as e:
                logger.error(f"Error in trace handler: {e}")
        
        logger.debug(f"Finished trace: {trace.trace_id}")
    
    def add_trace_handler(self, handler: Callable[[Trace], None]):
        """
        Add trace completion handler.
        
        Args:
            handler: Handler function.
        """
        self.trace_handlers.append(handler)
    
    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return getattr(self.local_storage, 'current_span', None)
    
    def get_current_trace(self) -> Optional[Trace]:
        """Get current active trace."""
        return self._get_current_trace()
    
    def trace_function(self, operation_name: str = None, 
                      span_type: SpanType = SpanType.COMPUTATION,
                      tags: Dict[str, Any] = None):
        """
        Decorator to trace function execution.
        
        Args:
            operation_name: Operation name (defaults to function name).
            span_type: Type of span.
            tags: Additional tags.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                span = self.start_span(op_name, span_type)
                
                if tags:
                    for key, value in tags.items():
                        span.set_tag(key, value)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_tag("success", True)
                    return result
                except Exception as e:
                    span.set_error(e)
                    raise
                finally:
                    self.finish_span(span)
            
            return wrapper
        return decorator
    
    @contextmanager
    def span_context(self, operation_name: str, span_type: SpanType,
                    tags: Dict[str, Any] = None):
        """
        Context manager for span creation.
        
        Args:
            operation_name: Operation name.
            span_type: Type of span.
            tags: Additional tags.
        """
        span = self.start_span(operation_name, span_type)
        
        if tags:
            for key, value in tags.items():
                span.set_tag(key, value)
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)
    
    def get_trace_analytics(self, time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Get trace analytics.
        
        Args:
            time_range: Time range for analysis.
            
        Returns:
            Analytics data.
        """
        cutoff_time = datetime.now() - time_range
        recent_traces = [t for t in self.completed_traces if t.start_time > cutoff_time]
        
        if not recent_traces:
            return {"total_traces": 0, "time_range": time_range.total_seconds()}
        
        # Calculate statistics
        durations = [t.duration for t in recent_traces if t.duration]
        error_traces = [t for t in recent_traces if any(s.status == "error" for s in t.spans)]
        
        span_types = {}
        for trace in recent_traces:
            for span in trace.spans:
                span_type = span.span_type.value
                if span_type not in span_types:
                    span_types[span_type] = {"count": 0, "total_duration": 0, "errors": 0}
                
                span_types[span_type]["count"] += 1
                if span.duration:
                    span_types[span_type]["total_duration"] += span.duration
                if span.status == "error":
                    span_types[span_type]["errors"] += 1
        
        # Calculate averages
        for span_type_data in span_types.values():
            if span_type_data["count"] > 0:
                span_type_data["avg_duration"] = span_type_data["total_duration"] / span_type_data["count"]
                span_type_data["error_rate"] = span_type_data["errors"] / span_type_data["count"]
        
        return {
            "total_traces": len(recent_traces),
            "error_traces": len(error_traces),
            "error_rate": len(error_traces) / len(recent_traces),
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "span_types": span_types,
            "time_range_seconds": time_range.total_seconds()
        }
    
    def _get_current_trace(self) -> Optional[Trace]:
        """Get current trace from thread local storage."""
        return getattr(self.local_storage, 'current_trace', None)
    
    def _set_current_trace(self, trace: Trace):
        """Set current trace in thread local storage."""
        self.local_storage.current_trace = trace
    
    def _set_current_span(self, span: Span):
        """Set current span in thread local storage."""
        self.local_storage.current_span = span


class StructuredLogger:
    """Structured logging with JSON formatting."""
    
    def __init__(self, name: str = "fusion_analyzer"):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name.
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Configure JSON formatter
        self._setup_json_formatter()
        
        # Integration with tracing
        self.tracer: Optional[DistributedTracer] = None
        
        logger.info("StructuredLogger initialized")
    
    def set_tracer(self, tracer: DistributedTracer):
        """
        Set distributed tracer for correlation.
        
        Args:
            tracer: Distributed tracer instance.
        """
        self.tracer = tracer
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with structured data."""
        if error:
            kwargs.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            })
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method."""
        # Add trace correlation if available
        if self.tracer:
            current_span = self.tracer.get_current_span()
            current_trace = self.tracer.get_current_trace()
            
            if current_trace:
                kwargs["trace_id"] = current_trace.trace_id
            
            if current_span:
                kwargs["span_id"] = current_span.span_id
                # Also add to span logs
                current_span.log(message, self._level_to_trace_level(level), **kwargs)
        
        # Add timestamp and other metadata
        kwargs.update({
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            "thread": threading.current_thread().name
        })
        
        # Create structured log entry
        log_data = {"message": message, **kwargs}
        self.logger.log(level, json.dumps(log_data))
    
    def _level_to_trace_level(self, level: int) -> TraceLevel:
        """Convert logging level to trace level."""
        if level >= logging.CRITICAL:
            return TraceLevel.CRITICAL
        elif level >= logging.ERROR:
            return TraceLevel.ERROR
        elif level >= logging.WARNING:
            return TraceLevel.WARNING
        elif level >= logging.DEBUG:
            return TraceLevel.DEBUG
        else:
            return TraceLevel.INFO
    
    def _setup_json_formatter(self):
        """Setup JSON formatter for the logger."""
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create console handler with JSON formatter
        console_handler = logging.StreamHandler()
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                try:
                    # Try to parse as JSON (for structured logs)
                    return record.getMessage()
                except:
                    # Fallback to standard formatting
                    return json.dumps({
                        "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno
                    })
        
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)


class LogAggregator:
    """Log aggregation and analysis system."""
    
    def __init__(self):
        """Initialize log aggregator."""
        self.log_entries: List[Dict[str, Any]] = []
        self.error_patterns: Dict[str, int] = {}
        self.performance_metrics: List[Dict[str, Any]] = []
        
        self.max_entries = 10000
        logger.info("LogAggregator initialized")
    
    def add_log_entry(self, entry: Dict[str, Any]):
        """
        Add log entry for aggregation.
        
        Args:
            entry: Log entry data.
        """
        self.log_entries.append(entry)
        
        # Analyze for patterns
        self._analyze_entry(entry)
        
        # Limit retention
        if len(self.log_entries) > self.max_entries:
            self.log_entries = self.log_entries[-self.max_entries:]
    
    def get_error_summary(self, time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Get error summary.
        
        Args:
            time_range: Time range for analysis.
            
        Returns:
            Error summary.
        """
        cutoff_time = datetime.now() - time_range
        
        recent_errors = []
        for entry in self.log_entries:
            try:
                entry_time = datetime.fromisoformat(entry.get("timestamp", ""))
                if entry_time > cutoff_time and entry.get("level") == "ERROR":
                    recent_errors.append(entry)
            except:
                continue
        
        # Count error types
        error_types = {}
        for error in recent_errors:
            error_type = error.get("error_type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "error_types": error_types,
            "time_range_hours": time_range.total_seconds() / 3600,
            "recent_errors": recent_errors[-10:]  # Last 10 errors
        }
    
    def get_performance_trends(self, time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Get performance trends.
        
        Args:
            time_range: Time range for analysis.
            
        Returns:
            Performance trends.
        """
        cutoff_time = datetime.now() - time_range
        
        recent_metrics = []
        for metric in self.performance_metrics:
            try:
                metric_time = datetime.fromisoformat(metric.get("timestamp", ""))
                if metric_time > cutoff_time:
                    recent_metrics.append(metric)
            except:
                continue
        
        if not recent_metrics:
            return {"total_metrics": 0}
        
        # Calculate averages
        avg_response_time = sum(m.get("response_time", 0) for m in recent_metrics) / len(recent_metrics)
        avg_cpu_usage = sum(m.get("cpu_usage", 0) for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.get("memory_usage", 0) for m in recent_metrics) / len(recent_metrics)
        
        return {
            "total_metrics": len(recent_metrics),
            "avg_response_time": avg_response_time,
            "avg_cpu_usage": avg_cpu_usage,
            "avg_memory_usage": avg_memory_usage,
            "time_range_hours": time_range.total_seconds() / 3600
        }
    
    def _analyze_entry(self, entry: Dict[str, Any]):
        """
        Analyze log entry for patterns.
        
        Args:
            entry: Log entry to analyze.
        """
        # Track error patterns
        if entry.get("level") == "ERROR":
            error_type = entry.get("error_type", "Unknown")
            self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
        
        # Extract performance metrics
        if "response_time" in entry or "cpu_usage" in entry or "memory_usage" in entry:
            metric = {
                "timestamp": entry.get("timestamp"),
                "response_time": entry.get("response_time"),
                "cpu_usage": entry.get("cpu_usage"),
                "memory_usage": entry.get("memory_usage")
            }
            self.performance_metrics.append(metric)
            
            # Limit retention
            if len(self.performance_metrics) > 1000:
                self.performance_metrics = self.performance_metrics[-1000:]


# Global instances
_tracer = None
_structured_logger = None
_log_aggregator = None


def get_tracer() -> DistributedTracer:
    """Get global distributed tracer."""
    global _tracer
    if _tracer is None:
        _tracer = DistributedTracer()
    return _tracer


def get_logger(name: str = "fusion_analyzer") -> StructuredLogger:
    """Get structured logger."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger(name)
        _structured_logger.set_tracer(get_tracer())
    return _structured_logger


def get_log_aggregator() -> LogAggregator:
    """Get log aggregator."""
    global _log_aggregator
    if _log_aggregator is None:
        _log_aggregator = LogAggregator()
    return _log_aggregator


# Convenience decorators
def trace_function(operation_name: str = None, span_type: SpanType = SpanType.COMPUTATION):
    """Decorator to trace function execution."""
    return get_tracer().trace_function(operation_name, span_type)


def log_function_calls(logger_name: str = "fusion_analyzer"):
    """Decorator to log function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            logger.info(f"Calling function: {func.__name__}", 
                       function=func.__name__, 
                       module=func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Function completed: {func.__name__}", 
                           function=func.__name__, 
                           duration=duration,
                           success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Function failed: {func.__name__}", 
                           error=e,
                           function=func.__name__, 
                           duration=duration,
                           success=False)
                raise
        return wrapper
    return decorator