"""
Performance monitoring and profiling utilities for the Nuclear Fusion Analyzer.

This module provides tools for monitoring model performance, API response times,
memory usage, and system resource utilization during fusion analysis workflows.
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from dataclasses import dataclass, field
from contextlib import contextmanager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    peak_memory_mb: float = 0.0
    gpu_memory_mb: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    operation_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ModelPerformanceMetrics:
    """Container for ML model performance metrics."""
    
    model_name: str = ""
    training_time: float = 0.0
    prediction_time: float = 0.0
    model_size_mb: float = 0.0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    dataset_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for fusion analysis operations.
    
    Tracks execution time, memory usage, CPU utilization, and other performance
    metrics for various operations including model training, prediction, and
    data processing.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the performance monitor.
        
        Args:
            max_history: Maximum number of metrics records to keep in memory.
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.model_metrics: Dict[str, List[ModelPerformanceMetrics]] = defaultdict(list)
        self.api_metrics: deque = deque(maxlen=max_history)
        self.system_metrics: deque = deque(maxlen=max_history)
        
        # Real-time monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Performance thresholds
        self.thresholds = {
            'execution_time_warning': 5.0,  # seconds
            'execution_time_critical': 30.0,  # seconds
            'memory_usage_warning': 1024.0,  # MB
            'memory_usage_critical': 4096.0,  # MB
            'cpu_usage_warning': 80.0,  # percent
            'cpu_usage_critical': 95.0,  # percent
        }
        
        logger.info(f"PerformanceMonitor initialized with max_history={max_history}")
    
    @contextmanager
    def monitor_operation(self, operation_name: str, parameters: Optional[Dict] = None):
        """
        Context manager for monitoring operation performance.
        
        Args:
            operation_name: Name of the operation being monitored.
            parameters: Optional parameters passed to the operation.
            
        Yields:
            PerformanceMetrics: Object to store additional metrics during execution.
            
        Example:
            >>> monitor = PerformanceMonitor()
            >>> with monitor.monitor_operation("model_training", {"model": "rf"}) as metrics:
            ...     # Training code here
            ...     model.fit(X, y)
            ...     metrics.dataset_size = len(X)
        """
        if parameters is None:
            parameters = {}
            
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            parameters=parameters.copy(),
            timestamp=datetime.now()
        )
        
        peak_memory = start_memory
        
        try:
            yield metrics
            
            # Final measurements
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            end_cpu = psutil.cpu_percent()
            
            # Update metrics
            metrics.execution_time = end_time - start_time
            metrics.memory_usage_mb = end_memory - start_memory
            metrics.cpu_usage_percent = (start_cpu + end_cpu) / 2
            metrics.peak_memory_mb = max(peak_memory, end_memory)
            
            # Check for GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Check thresholds and log warnings
            self._check_thresholds(metrics)
            
            logger.info(
                f"Operation '{operation_name}' completed: "
                f"time={metrics.execution_time:.3f}s, "
                f"memory={metrics.memory_usage_mb:.1f}MB, "
                f"cpu={metrics.cpu_usage_percent:.1f}%"
            )
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            metrics.execution_time = time.time() - start_time
            self.metrics_history.append(metrics)
            logger.error(f"Operation '{operation_name}' failed: {e}")
            raise
    
    def monitor_model_performance(
        self, 
        model_name: str,
        training_time: float,
        prediction_time: float,
        accuracy_metrics: Dict[str, float],
        **kwargs
    ) -> ModelPerformanceMetrics:
        """
        Record model performance metrics.
        
        Args:
            model_name: Name of the ML model.
            training_time: Time taken to train the model (seconds).
            prediction_time: Time taken for prediction (seconds).
            accuracy_metrics: Dictionary of accuracy metrics (R², MAE, etc.).
            **kwargs: Additional metrics (model_size_mb, feature_importance, etc.).
            
        Returns:
            ModelPerformanceMetrics: The recorded metrics.
        """
        metrics = ModelPerformanceMetrics(
            model_name=model_name,
            training_time=training_time,
            prediction_time=prediction_time,
            accuracy_metrics=accuracy_metrics.copy(),
            timestamp=datetime.now()
        )
        
        # Add optional metrics
        if 'model_size_mb' in kwargs:
            metrics.model_size_mb = kwargs['model_size_mb']
        if 'feature_importance' in kwargs:
            metrics.feature_importance = kwargs['feature_importance'].copy()
        if 'hyperparameters' in kwargs:
            metrics.hyperparameters = kwargs['hyperparameters'].copy()
        if 'dataset_size' in kwargs:
            metrics.dataset_size = kwargs['dataset_size']
            
        self.model_metrics[model_name].append(metrics)
        
        logger.info(
            f"Model '{model_name}' performance recorded: "
            f"train_time={training_time:.3f}s, "
            f"pred_time={prediction_time:.6f}s, "
            f"R²={accuracy_metrics.get('r2', 'N/A')}"
        )
        
        return metrics
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed warning or critical thresholds."""
        warnings = []
        
        # Execution time checks
        if metrics.execution_time > self.thresholds['execution_time_critical']:
            warnings.append(f"CRITICAL: Execution time {metrics.execution_time:.1f}s exceeds critical threshold")
        elif metrics.execution_time > self.thresholds['execution_time_warning']:
            warnings.append(f"WARNING: Execution time {metrics.execution_time:.1f}s exceeds warning threshold")
        
        # Memory usage checks
        if abs(metrics.memory_usage_mb) > self.thresholds['memory_usage_critical']:
            warnings.append(f"CRITICAL: Memory usage {metrics.memory_usage_mb:.1f}MB exceeds critical threshold")
        elif abs(metrics.memory_usage_mb) > self.thresholds['memory_usage_warning']:
            warnings.append(f"WARNING: Memory usage {metrics.memory_usage_mb:.1f}MB exceeds warning threshold")
        
        # CPU usage checks
        if metrics.cpu_usage_percent > self.thresholds['cpu_usage_critical']:
            warnings.append(f"CRITICAL: CPU usage {metrics.cpu_usage_percent:.1f}% exceeds critical threshold")
        elif metrics.cpu_usage_percent > self.thresholds['cpu_usage_warning']:
            warnings.append(f"WARNING: CPU usage {metrics.cpu_usage_percent:.1f}% exceeds warning threshold")
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
    
    def start_real_time_monitoring(self, interval: float = 1.0):
        """
        Start real-time system monitoring in background thread.
        
        Args:
            interval: Monitoring interval in seconds.
        """
        if self._monitoring_active:
            logger.warning("Real-time monitoring already active")
            return
        
        self._monitoring_active = True
        self._stop_event.clear()
        
        def monitor_loop():
            """Background monitoring loop."""
            while not self._stop_event.wait(interval):
                try:
                    # Collect system metrics
                    memory = psutil.virtual_memory()
                    cpu_percent = psutil.cpu_percent()
                    
                    system_metric = {
                        'timestamp': datetime.now(),
                        'memory_total_mb': memory.total / 1024 / 1024,
                        'memory_used_mb': memory.used / 1024 / 1024,
                        'memory_percent': memory.percent,
                        'cpu_percent': cpu_percent,
                        'cpu_count': psutil.cpu_count(),
                    }
                    
                    # Add disk usage
                    disk = psutil.disk_usage('/')
                    system_metric.update({
                        'disk_total_gb': disk.total / 1024 / 1024 / 1024,
                        'disk_used_gb': disk.used / 1024 / 1024 / 1024,
                        'disk_percent': (disk.used / disk.total) * 100
                    })
                    
                    self.system_metrics.append(system_metric)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"Real-time monitoring started with {interval}s interval")
    
    def stop_real_time_monitoring(self):
        """Stop real-time system monitoring."""
        if not self._monitoring_active:
            return
        
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self._monitoring_active = False
        logger.info("Real-time monitoring stopped")
    
    def get_performance_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            last_n: Number of recent operations to include (None for all).
            
        Returns:
            Dictionary containing performance summary statistics.
        """
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)
        if last_n:
            recent_metrics = recent_metrics[-last_n:]
        
        # Extract successful operations only
        successful_metrics = [m for m in recent_metrics if m.success]
        
        if not successful_metrics:
            return {"error": "No successful operations found"}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in successful_metrics]
        memory_usage = [m.memory_usage_mb for m in successful_metrics]
        cpu_usage = [m.cpu_usage_percent for m in successful_metrics]
        
        summary = {
            'total_operations': len(recent_metrics),
            'successful_operations': len(successful_metrics),
            'success_rate': len(successful_metrics) / len(recent_metrics),
            'execution_time': {
                'mean': np.mean(execution_times),
                'median': np.median(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times),
                'p95': np.percentile(execution_times, 95),
            },
            'memory_usage_mb': {
                'mean': np.mean(memory_usage),
                'median': np.median(memory_usage),
                'std': np.std(memory_usage),
                'min': np.min(memory_usage),
                'max': np.max(memory_usage),
            },
            'cpu_usage_percent': {
                'mean': np.mean(cpu_usage),
                'median': np.median(cpu_usage),
                'std': np.std(cpu_usage),
                'min': np.min(cpu_usage),
                'max': np.max(cpu_usage),
            },
            'time_range': {
                'start': min(m.timestamp for m in recent_metrics),
                'end': max(m.timestamp for m in recent_metrics),
            }
        }
        
        # Add operation breakdown
        operation_counts = defaultdict(int)
        operation_times = defaultdict(list)
        
        for metric in successful_metrics:
            operation_counts[metric.operation_name] += 1
            operation_times[metric.operation_name].append(metric.execution_time)
        
        summary['operations'] = {
            name: {
                'count': operation_counts[name],
                'mean_time': np.mean(operation_times[name]),
                'total_time': np.sum(operation_times[name]),
            }
            for name in operation_counts
        }
        
        return summary
    
    def get_model_performance_comparison(self) -> Dict[str, Any]:
        """
        Compare performance across different models.
        
        Returns:
            Dictionary containing model performance comparison.
        """
        if not self.model_metrics:
            return {"error": "No model performance data available"}
        
        comparison = {}
        
        for model_name, metrics_list in self.model_metrics.items():
            if not metrics_list:
                continue
            
            latest_metric = metrics_list[-1]
            all_training_times = [m.training_time for m in metrics_list]
            all_prediction_times = [m.prediction_time for m in metrics_list]
            
            comparison[model_name] = {
                'latest_metrics': {
                    'training_time': latest_metric.training_time,
                    'prediction_time': latest_metric.prediction_time,
                    'model_size_mb': latest_metric.model_size_mb,
                    'accuracy_metrics': latest_metric.accuracy_metrics,
                    'dataset_size': latest_metric.dataset_size,
                },
                'historical_performance': {
                    'training_time': {
                        'mean': np.mean(all_training_times),
                        'std': np.std(all_training_times),
                        'min': np.min(all_training_times),
                        'max': np.max(all_training_times),
                    },
                    'prediction_time': {
                        'mean': np.mean(all_prediction_times),
                        'std': np.std(all_prediction_times),
                        'min': np.min(all_prediction_times),
                        'max': np.max(all_prediction_times),
                    },
                    'training_runs': len(metrics_list),
                }
            }
        
        return comparison
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """
        Export performance metrics to file.
        
        Args:
            filepath: Path to save the metrics file.
            format: Export format ('json', 'csv', or 'excel').
        """
        if format.lower() == 'json':
            data = {
                'performance_metrics': [
                    {
                        'operation_name': m.operation_name,
                        'execution_time': m.execution_time,
                        'memory_usage_mb': m.memory_usage_mb,
                        'cpu_usage_percent': m.cpu_usage_percent,
                        'peak_memory_mb': m.peak_memory_mb,
                        'timestamp': m.timestamp.isoformat(),
                        'success': m.success,
                        'parameters': m.parameters,
                    }
                    for m in self.metrics_history
                ],
                'model_metrics': {
                    model_name: [
                        {
                            'training_time': m.training_time,
                            'prediction_time': m.prediction_time,
                            'model_size_mb': m.model_size_mb,
                            'accuracy_metrics': m.accuracy_metrics,
                            'timestamp': m.timestamp.isoformat(),
                            'dataset_size': m.dataset_size,
                        }
                        for m in metrics_list
                    ]
                    for model_name, metrics_list in self.model_metrics.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format.lower() == 'csv':
            # Export performance metrics to CSV
            if self.metrics_history:
                df_data = []
                for m in self.metrics_history:
                    row = {
                        'operation_name': m.operation_name,
                        'execution_time': m.execution_time,
                        'memory_usage_mb': m.memory_usage_mb,
                        'cpu_usage_percent': m.cpu_usage_percent,
                        'peak_memory_mb': m.peak_memory_mb,
                        'timestamp': m.timestamp,
                        'success': m.success,
                    }
                    # Add parameters as separate columns
                    for key, value in m.parameters.items():
                        row[f'param_{key}'] = value
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(filepath, index=False)
        
        logger.info(f"Metrics exported to {filepath} in {format} format")
    
    def reset_metrics(self):
        """Clear all stored metrics."""
        self.metrics_history.clear()
        self.model_metrics.clear()
        self.api_metrics.clear()
        self.system_metrics.clear()
        logger.info("All performance metrics cleared")


def performance_monitor(operation_name: str, monitor_instance: Optional[PerformanceMonitor] = None):
    """
    Decorator for monitoring function performance.
    
    Args:
        operation_name: Name of the operation being monitored.
        monitor_instance: PerformanceMonitor instance to use (creates new if None).
        
    Example:
        >>> @performance_monitor("data_processing")
        ... def process_data(data):
        ...     return data.processed()
    """
    if monitor_instance is None:
        monitor_instance = PerformanceMonitor()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitor_instance.monitor_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global performance monitor instance
global_monitor = PerformanceMonitor()