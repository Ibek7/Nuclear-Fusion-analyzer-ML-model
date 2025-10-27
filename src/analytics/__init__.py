"""
Advanced Analytics and Reporting Engine for Nuclear Fusion Analysis.

This module provides:
- Business intelligence dashboard
- Custom report generation
- Real-time analytics processing
- Statistical analysis and insights
- Performance metrics and KPIs
- Data visualization and charts
- Report scheduling and delivery
- Interactive analytics API
"""

import asyncio
import json
import uuid
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timezone, timedelta
import logging
import math
import statistics

# Data processing
import numpy as np
import pandas as pd

# Visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Statistical analysis
try:
    from scipy import stats
    from scipy.signal import find_peaks
    import sklearn.cluster as cluster
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Business intelligence
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    PERCENTAGE = "percentage"
    RATE = "rate"


class ChartType(Enum):
    """Types of charts."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    AREA = "area"
    BUBBLE = "bubble"
    CANDLESTICK = "candlestick"


class ReportFormat(Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"


@dataclass
class Metric:
    """Analytics metric definition."""
    
    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "description": self.description,
            "unit": self.unit
        }


@dataclass
class KPI:
    """Key Performance Indicator."""
    
    name: str
    current_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    improvement_direction: str = "higher"  # "higher" or "lower"
    description: str = ""
    
    @property
    def performance_percentage(self) -> float:
        """Calculate performance as percentage of target."""
        if self.target_value == 0:
            return 0.0
        return (self.current_value / self.target_value) * 100
    
    @property
    def status(self) -> str:
        """Get KPI status."""
        if self.improvement_direction == "higher":
            if self.current_value >= self.target_value:
                return "excellent"
            elif self.current_value >= self.threshold_warning:
                return "good"
            elif self.current_value >= self.threshold_critical:
                return "warning"
            else:
                return "critical"
        else:  # lower is better
            if self.current_value <= self.target_value:
                return "excellent"
            elif self.current_value <= self.threshold_warning:
                return "good"
            elif self.current_value <= self.threshold_critical:
                return "warning"
            else:
                return "critical"


@dataclass
class Report:
    """Analytics report definition."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: List[Metric] = field(default_factory=list)
    kpis: List[KPI] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    format: ReportFormat = ReportFormat.HTML
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "metrics": [m.to_dict() for m in self.metrics],
            "kpis": [asdict(kpi) for kpi in self.kpis],
            "charts": self.charts,
            "data": self.data,
            "format": self.format.value
        }


class DataProcessor:
    """Data processing utilities for analytics."""
    
    @staticmethod
    def calculate_statistics(data: List[float]) -> Dict[str, float]:
        """Calculate basic statistics."""
        if not data:
            return {}
        
        return {
            "count": len(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "mode": statistics.mode(data) if len(set(data)) < len(data) else None,
            "std": statistics.stdev(data) if len(data) > 1 else 0,
            "var": statistics.variance(data) if len(data) > 1 else 0,
            "min": min(data),
            "max": max(data),
            "range": max(data) - min(data),
            "q1": np.percentile(data, 25),
            "q3": np.percentile(data, 75),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25)
        }
    
    @staticmethod
    def detect_anomalies(data: List[float], method: str = "zscore", threshold: float = 3.0) -> List[int]:
        """Detect anomalies in data."""
        if not data or len(data) < 3:
            return []
        
        data_array = np.array(data)
        anomalies = []
        
        if method == "zscore":
            z_scores = np.abs(stats.zscore(data_array))
            anomalies = np.where(z_scores > threshold)[0].tolist()
        
        elif method == "iqr":
            q1, q3 = np.percentile(data_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomalies = np.where((data_array < lower_bound) | (data_array > upper_bound))[0].tolist()
        
        elif method == "isolation_forest" and HAS_SCIPY:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data_array.reshape(-1, 1))
            anomalies = np.where(outliers == -1)[0].tolist()
        
        return anomalies
    
    @staticmethod
    def calculate_correlation(x: List[float], y: List[float]) -> Dict[str, float]:
        """Calculate correlation between two variables."""
        if len(x) != len(y) or len(x) < 2:
            return {}
        
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Calculate R-squared
        r_squared = correlation ** 2
        
        return {
            "correlation": correlation,
            "r_squared": r_squared,
            "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
        }
    
    @staticmethod
    def calculate_trend(data: List[float], timestamps: Optional[List[datetime]] = None) -> Dict[str, Any]:
        """Calculate trend in time series data."""
        if len(data) < 2:
            return {}
        
        if timestamps is None:
            x = list(range(len(data)))
        else:
            x = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
            "direction": trend_direction,
            "strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
        }


class ChartGenerator:
    """Chart generation utilities."""
    
    @staticmethod
    def create_line_chart(
        data: Dict[str, List[float]],
        title: str = "Line Chart",
        x_title: str = "X",
        y_title: str = "Y"
    ) -> Dict[str, Any]:
        """Create line chart."""
        if not HAS_PLOTLY:
            return {"error": "Plotly not available"}
        
        fig = go.Figure()
        
        for name, values in data.items():
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines+markers',
                name=name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            showlegend=True,
            template="plotly_white"
        )
        
        return {
            "type": "line",
            "config": fig.to_dict(),
            "html": fig.to_html()
        }
    
    @staticmethod
    def create_bar_chart(
        categories: List[str],
        values: List[float],
        title: str = "Bar Chart"
    ) -> Dict[str, Any]:
        """Create bar chart."""
        if not HAS_PLOTLY:
            return {"error": "Plotly not available"}
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color='rgb(55, 83, 109)')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Categories",
            yaxis_title="Values",
            template="plotly_white"
        )
        
        return {
            "type": "bar",
            "config": fig.to_dict(),
            "html": fig.to_html()
        }
    
    @staticmethod
    def create_heatmap(
        data: List[List[float]],
        x_labels: List[str],
        y_labels: List[str],
        title: str = "Heatmap"
    ) -> Dict[str, Any]:
        """Create heatmap."""
        if not HAS_PLOTLY:
            return {"error": "Plotly not available"}
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_white"
        )
        
        return {
            "type": "heatmap",
            "config": fig.to_dict(),
            "html": fig.to_html()
        }
    
    @staticmethod
    def create_pie_chart(
        labels: List[str],
        values: List[float],
        title: str = "Pie Chart"
    ) -> Dict[str, Any]:
        """Create pie chart."""
        if not HAS_PLOTLY:
            return {"error": "Plotly not available"}
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig.update_layout(
            title=title,
            template="plotly_white"
        )
        
        return {
            "type": "pie",
            "config": fig.to_dict(),
            "html": fig.to_html()
        }


class MetricsCollector:
    """Metrics collection and aggregation."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, List[Metric]] = {}
        self.kpis: Dict[str, KPI] = {}
        
        logger.info("MetricsCollector initialized")
    
    def add_metric(self, metric: Metric):
        """Add metric to collection."""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        
        self.metrics[metric.name].append(metric)
        
        # Keep only last 1000 metrics per name
        if len(self.metrics[metric.name]) > 1000:
            self.metrics[metric.name] = self.metrics[metric.name][-1000:]
    
    def add_kpi(self, kpi: KPI):
        """Add or update KPI."""
        self.kpis[kpi.name] = kpi
    
    def get_metric_history(self, name: str, hours: int = 24) -> List[Metric]:
        """Get metric history for specified time period."""
        if name not in self.metrics:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def calculate_metric_statistics(self, name: str, hours: int = 24) -> Dict[str, Any]:
        """Calculate statistics for metric."""
        history = self.get_metric_history(name, hours)
        if not history:
            return {}
        
        values = [m.value for m in history]
        stats = DataProcessor.calculate_statistics(values)
        
        # Add trend analysis
        timestamps = [m.timestamp for m in history]
        trend = DataProcessor.calculate_trend(values, timestamps)
        stats["trend"] = trend
        
        return stats
    
    def get_all_kpis(self) -> List[KPI]:
        """Get all KPIs."""
        return list(self.kpis.values())
    
    def get_kpi_summary(self) -> Dict[str, Any]:
        """Get KPI summary."""
        kpis = self.get_all_kpis()
        if not kpis:
            return {}
        
        status_counts = {}
        for kpi in kpis:
            status = kpi.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_performance = sum(kpi.performance_percentage for kpi in kpis) / len(kpis)
        
        return {
            "total_kpis": len(kpis),
            "status_breakdown": status_counts,
            "average_performance": total_performance,
            "kpis": [asdict(kpi) for kpi in kpis]
        }


class ReportGenerator:
    """Report generation engine."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize report generator.
        
        Args:
            metrics_collector: Metrics collector instance.
        """
        self.metrics_collector = metrics_collector
        self.chart_generator = ChartGenerator()
        
        logger.info("ReportGenerator initialized")
    
    def generate_performance_report(self) -> Report:
        """Generate performance report."""
        report = Report(
            title="Nuclear Fusion Analysis Performance Report",
            description="Comprehensive performance analysis of the fusion analysis system"
        )
        
        # Collect KPIs
        report.kpis = self.metrics_collector.get_all_kpis()
        
        # Generate performance charts
        kpi_names = [kpi.name for kpi in report.kpis]
        kpi_values = [kpi.performance_percentage for kpi in report.kpis]
        
        if kpi_names and kpi_values:
            performance_chart = self.chart_generator.create_bar_chart(
                kpi_names,
                kpi_values,
                "KPI Performance (%)"
            )
            report.charts.append(performance_chart)
        
        # Add summary data
        report.data["kpi_summary"] = self.metrics_collector.get_kpi_summary()
        
        return report
    
    def generate_trend_report(self, metric_names: List[str], hours: int = 24) -> Report:
        """Generate trend analysis report."""
        report = Report(
            title="Trend Analysis Report",
            description=f"Trend analysis for the last {hours} hours"
        )
        
        # Collect metrics and generate charts
        trend_data = {}
        for metric_name in metric_names:
            history = self.metrics_collector.get_metric_history(metric_name, hours)
            if history:
                values = [m.value for m in history]
                trend_data[metric_name] = values
                
                # Add statistics
                stats = self.metrics_collector.calculate_metric_statistics(metric_name, hours)
                report.data[f"{metric_name}_stats"] = stats
        
        if trend_data:
            trend_chart = self.chart_generator.create_line_chart(
                trend_data,
                f"Metric Trends - Last {hours} Hours",
                "Time",
                "Value"
            )
            report.charts.append(trend_chart)
        
        return report
    
    def generate_anomaly_report(self, metric_names: List[str], hours: int = 24) -> Report:
        """Generate anomaly detection report."""
        report = Report(
            title="Anomaly Detection Report",
            description=f"Anomaly analysis for the last {hours} hours"
        )
        
        anomaly_summary = {}
        
        for metric_name in metric_names:
            history = self.metrics_collector.get_metric_history(metric_name, hours)
            if not history:
                continue
            
            values = [m.value for m in history]
            timestamps = [m.timestamp for m in history]
            
            # Detect anomalies
            anomaly_indices = DataProcessor.detect_anomalies(values, method="zscore")
            
            if anomaly_indices:
                anomaly_times = [timestamps[i] for i in anomaly_indices]
                anomaly_values = [values[i] for i in anomaly_indices]
                
                anomaly_summary[metric_name] = {
                    "count": len(anomaly_indices),
                    "timestamps": [t.isoformat() for t in anomaly_times],
                    "values": anomaly_values,
                    "severity": "high" if len(anomaly_indices) > len(values) * 0.1 else "medium" if len(anomaly_indices) > len(values) * 0.05 else "low"
                }
        
        report.data["anomalies"] = anomaly_summary
        
        return report
    
    def export_report(self, report: Report, filepath: str) -> bool:
        """Export report to file."""
        try:
            if report.format == ReportFormat.JSON:
                with open(filepath, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2)
            
            elif report.format == ReportFormat.HTML:
                html_content = self._generate_html_report(report)
                with open(filepath, 'w') as f:
                    f.write(html_content)
            
            elif report.format == ReportFormat.CSV:
                # Export metrics as CSV
                df = pd.DataFrame([m.to_dict() for m in report.metrics])
                df.to_csv(filepath, index=False)
            
            logger.info(f"Report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return False
    
    def _generate_html_report(self, report: Report) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .kpi {{ display: inline-block; margin: 10px; padding: 15px; border-radius: 5px; }}
                .excellent {{ background-color: #d4edda; }}
                .good {{ background-color: #d1ecf1; }}
                .warning {{ background-color: #fff3cd; }}
                .critical {{ background-color: #f8d7da; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p>{report.description}</p>
                <p>Generated: {report.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <h2>Key Performance Indicators</h2>
            <div class="kpis">
        """
        
        for kpi in report.kpis:
            html += f"""
                <div class="kpi {kpi.status}">
                    <h3>{kpi.name}</h3>
                    <p>Current: {kpi.current_value:.2f}</p>
                    <p>Target: {kpi.target_value:.2f}</p>
                    <p>Performance: {kpi.performance_percentage:.1f}%</p>
                    <p>Status: {kpi.status.title()}</p>
                </div>
            """
        
        html += """
            </div>
            
            <h2>Charts</h2>
        """
        
        for chart in report.charts:
            if "html" in chart:
                html += f'<div class="chart">{chart["html"]}</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html


class AnalyticsEngine:
    """
    Comprehensive analytics and reporting engine.
    
    Provides business intelligence, metrics collection, and report generation.
    """
    
    def __init__(self):
        """Initialize analytics engine."""
        self.metrics_collector = MetricsCollector()
        self.report_generator = ReportGenerator(self.metrics_collector)
        self.running = False
        
        # Default KPIs for fusion analysis
        self._setup_default_kpis()
        
        logger.info("AnalyticsEngine initialized")
    
    def _setup_default_kpis(self):
        """Setup default KPIs for fusion analysis."""
        default_kpis = [
            KPI(
                name="Plasma Temperature",
                current_value=0.0,
                target_value=100.0,
                threshold_warning=80.0,
                threshold_critical=60.0,
                improvement_direction="higher",
                description="Average plasma temperature in millions of Kelvin"
            ),
            KPI(
                name="Confinement Time",
                current_value=0.0,
                target_value=10.0,
                threshold_warning=8.0,
                threshold_critical=5.0,
                improvement_direction="higher",
                description="Energy confinement time in seconds"
            ),
            KPI(
                name="Energy Gain",
                current_value=0.0,
                target_value=1.0,
                threshold_warning=0.8,
                threshold_critical=0.5,
                improvement_direction="higher",
                description="Energy gain factor (Q value)"
            ),
            KPI(
                name="Prediction Accuracy",
                current_value=0.0,
                target_value=95.0,
                threshold_warning=90.0,
                threshold_critical=85.0,
                improvement_direction="higher",
                description="ML model prediction accuracy percentage"
            ),
            KPI(
                name="System Uptime",
                current_value=99.9,
                target_value=99.9,
                threshold_warning=99.5,
                threshold_critical=99.0,
                improvement_direction="higher",
                description="System uptime percentage"
            )
        ]
        
        for kpi in default_kpis:
            self.metrics_collector.add_kpi(kpi)
    
    async def start(self):
        """Start analytics engine."""
        self.running = True
        logger.info("AnalyticsEngine started")
    
    async def stop(self):
        """Stop analytics engine."""
        self.running = False
        logger.info("AnalyticsEngine stopped")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
        unit: str = ""
    ):
        """Record a metric."""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels or {},
            description=description,
            unit=unit
        )
        
        self.metrics_collector.add_metric(metric)
    
    def update_kpi(
        self,
        name: str,
        current_value: float
    ):
        """Update KPI current value."""
        if name in self.metrics_collector.kpis:
            self.metrics_collector.kpis[name].current_value = current_value
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for analytics dashboard."""
        return {
            "kpis": self.metrics_collector.get_kpi_summary(),
            "metrics_summary": {
                name: self.metrics_collector.calculate_metric_statistics(name)
                for name in self.metrics_collector.metrics.keys()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def create_custom_report(
        self,
        title: str,
        description: str,
        metric_names: List[str],
        include_trends: bool = True,
        include_anomalies: bool = True,
        hours: int = 24
    ) -> Report:
        """Create custom analytics report."""
        report = Report(title=title, description=description)
        
        # Add requested metrics
        for metric_name in metric_names:
            history = self.metrics_collector.get_metric_history(metric_name, hours)
            report.metrics.extend(history)
        
        # Add KPIs
        report.kpis = self.metrics_collector.get_all_kpis()
        
        # Generate trend analysis
        if include_trends:
            trend_data = {}
            for metric_name in metric_names:
                history = self.metrics_collector.get_metric_history(metric_name, hours)
                if history:
                    trend_data[metric_name] = [m.value for m in history]
            
            if trend_data:
                trend_chart = self.report_generator.chart_generator.create_line_chart(
                    trend_data,
                    "Metric Trends",
                    "Time",
                    "Value"
                )
                report.charts.append(trend_chart)
        
        # Generate anomaly analysis
        if include_anomalies:
            anomaly_summary = {}
            for metric_name in metric_names:
                history = self.metrics_collector.get_metric_history(metric_name, hours)
                if history:
                    values = [m.value for m in history]
                    anomalies = DataProcessor.detect_anomalies(values)
                    if anomalies:
                        anomaly_summary[metric_name] = len(anomalies)
            
            report.data["anomaly_counts"] = anomaly_summary
        
        return report


def create_analytics_engine() -> AnalyticsEngine:
    """
    Create configured analytics engine.
    
    Returns:
        Configured analytics engine.
    """
    return AnalyticsEngine()