"""
Interactive Analytics Dashboard for Nuclear Fusion Analysis.

This module provides:
- Real-time analytics dashboard
- Interactive visualizations
- Live metrics monitoring
- Custom widget system
- Responsive layout
- Export capabilities
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
import logging

# Dashboard framework
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    HAS_DASHBOARD_DEPS = True
except ImportError:
    HAS_DASHBOARD_DEPS = False

# Real-time updates
try:
    import websockets
    import json
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from . import AnalyticsEngine, Metric, KPI, MetricType

logger = logging.getLogger(__name__)


class DashboardWidget:
    """Base class for dashboard widgets."""
    
    def __init__(self, title: str, widget_id: str):
        """
        Initialize dashboard widget.
        
        Args:
            title: Widget title.
            widget_id: Unique widget identifier.
        """
        self.title = title
        self.widget_id = widget_id
        self.data: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
    
    def update_data(self, data: Dict[str, Any]):
        """Update widget data."""
        self.data = data
    
    def render(self) -> Any:
        """Render widget. Override in subclasses."""
        raise NotImplementedError


class MetricWidget(DashboardWidget):
    """Widget for displaying single metrics."""
    
    def __init__(self, title: str, widget_id: str, metric_name: str):
        """
        Initialize metric widget.
        
        Args:
            title: Widget title.
            widget_id: Widget ID.
            metric_name: Name of metric to display.
        """
        super().__init__(title, widget_id)
        self.metric_name = metric_name
    
    def render(self):
        """Render metric widget."""
        if not HAS_DASHBOARD_DEPS:
            return "Dashboard dependencies not available"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "current_value" in self.data:
                st.metric(
                    label=self.title,
                    value=f"{self.data['current_value']:.2f}",
                    delta=f"{self.data.get('change', 0):.2f}"
                )
        
        with col2:
            if "target_value" in self.data:
                st.metric(
                    label="Target",
                    value=f"{self.data['target_value']:.2f}"
                )
        
        with col3:
            if "performance_percentage" in self.data:
                performance = self.data['performance_percentage']
                color = "normal"
                if performance >= 100:
                    color = "normal"  # Green
                elif performance >= 80:
                    color = "normal"  # Yellow would be shown differently
                else:
                    color = "inverse"  # Red
                
                st.metric(
                    label="Performance",
                    value=f"{performance:.1f}%"
                )


class ChartWidget(DashboardWidget):
    """Widget for displaying charts."""
    
    def __init__(self, title: str, widget_id: str, chart_type: str = "line"):
        """
        Initialize chart widget.
        
        Args:
            title: Widget title.
            widget_id: Widget ID.
            chart_type: Type of chart (line, bar, pie, etc.).
        """
        super().__init__(title, widget_id)
        self.chart_type = chart_type
    
    def render(self):
        """Render chart widget."""
        if not HAS_DASHBOARD_DEPS:
            return "Dashboard dependencies not available"
        
        if "chart_data" not in self.data:
            st.write("No data available")
            return
        
        chart_data = self.data["chart_data"]
        
        if self.chart_type == "line":
            fig = go.Figure()
            for series_name, values in chart_data.items():
                fig.add_trace(go.Scatter(
                    y=values,
                    mode='lines+markers',
                    name=series_name
                ))
            
            fig.update_layout(
                title=self.title,
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif self.chart_type == "bar":
            if isinstance(chart_data, dict) and len(chart_data) == 2:
                keys = list(chart_data.keys())
                x_data = chart_data[keys[0]]
                y_data = chart_data[keys[1]]
                
                fig = go.Figure(data=[
                    go.Bar(x=x_data, y=y_data)
                ])
                
                fig.update_layout(
                    title=self.title,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif self.chart_type == "pie":
            if "labels" in chart_data and "values" in chart_data:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=chart_data["labels"],
                        values=chart_data["values"]
                    )
                ])
                
                fig.update_layout(
                    title=self.title,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)


class KPIWidget(DashboardWidget):
    """Widget for displaying KPIs."""
    
    def render(self):
        """Render KPI widget."""
        if not HAS_DASHBOARD_DEPS:
            return "Dashboard dependencies not available"
        
        if "kpis" not in self.data:
            st.write("No KPI data available")
            return
        
        kpis = self.data["kpis"]
        
        # Display KPIs in grid
        cols = st.columns(min(len(kpis), 4))
        
        for i, kpi in enumerate(kpis):
            with cols[i % len(cols)]:
                # Color based on status
                if kpi.get("status") == "excellent":
                    status_color = "#28a745"  # Green
                elif kpi.get("status") == "good":
                    status_color = "#17a2b8"  # Blue
                elif kpi.get("status") == "warning":
                    status_color = "#ffc107"  # Yellow
                else:
                    status_color = "#dc3545"  # Red
                
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid {status_color}; margin: 5px 0;">
                    <h4>{kpi.get('name', 'Unknown KPI')}</h4>
                    <p><strong>Current:</strong> {kpi.get('current_value', 0):.2f}</p>
                    <p><strong>Target:</strong> {kpi.get('target_value', 0):.2f}</p>
                    <p><strong>Performance:</strong> {kpi.get('performance_percentage', 0):.1f}%</p>
                    <p><strong>Status:</strong> {kpi.get('status', 'unknown').title()}</p>
                </div>
                """, unsafe_allow_html=True)


class TableWidget(DashboardWidget):
    """Widget for displaying data tables."""
    
    def render(self):
        """Render table widget."""
        if not HAS_DASHBOARD_DEPS:
            return "Dashboard dependencies not available"
        
        if "table_data" not in self.data:
            st.write("No table data available")
            return
        
        df = pd.DataFrame(self.data["table_data"])
        st.dataframe(df, use_container_width=True)


class AnalyticsDashboard:
    """
    Interactive analytics dashboard.
    
    Provides real-time monitoring and visualization of fusion analysis metrics.
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        """
        Initialize analytics dashboard.
        
        Args:
            analytics_engine: Analytics engine instance.
        """
        if not HAS_DASHBOARD_DEPS:
            raise RuntimeError("Dashboard dependencies not available")
        
        self.analytics_engine = analytics_engine
        self.widgets: Dict[str, DashboardWidget] = {}
        self.layout: List[List[str]] = []  # Widget layout grid
        
        # Setup default widgets
        self._setup_default_widgets()
        
        logger.info("AnalyticsDashboard initialized")
    
    def _setup_default_widgets(self):
        """Setup default dashboard widgets."""
        # KPI overview widget
        self.add_widget(KPIWidget("Key Performance Indicators", "kpi_overview"))
        
        # Metric trend charts
        self.add_widget(ChartWidget("Plasma Temperature Trend", "temp_trend", "line"))
        self.add_widget(ChartWidget("Energy Confinement", "confinement_chart", "line"))
        self.add_widget(ChartWidget("Prediction Accuracy", "accuracy_chart", "line"))
        
        # System metrics
        self.add_widget(MetricWidget("System Uptime", "uptime_metric", "system_uptime"))
        self.add_widget(MetricWidget("Processing Speed", "speed_metric", "processing_speed"))
        
        # Recent data table
        self.add_widget(TableWidget("Recent Metrics", "recent_metrics"))
        
        # Default layout (2x3 grid)
        self.layout = [
            ["kpi_overview", "kpi_overview"],
            ["temp_trend", "confinement_chart"],
            ["accuracy_chart", "uptime_metric"],
            ["recent_metrics", "recent_metrics"]
        ]
    
    def add_widget(self, widget: DashboardWidget):
        """Add widget to dashboard."""
        self.widgets[widget.widget_id] = widget
    
    def remove_widget(self, widget_id: str):
        """Remove widget from dashboard."""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
    
    def update_widget_data(self, widget_id: str, data: Dict[str, Any]):
        """Update data for specific widget."""
        if widget_id in self.widgets:
            self.widgets[widget_id].update_data(data)
    
    def render(self):
        """Render complete dashboard."""
        if not HAS_DASHBOARD_DEPS:
            st.error("Dashboard dependencies not available")
            return
        
        # Dashboard header
        st.set_page_config(
            page_title="Nuclear Fusion Analytics Dashboard",
            page_icon="⚛️",
            layout="wide"
        )
        
        st.title("⚛️ Nuclear Fusion Analytics Dashboard")
        st.markdown("Real-time monitoring and analysis of fusion plasma parameters")
        
        # Refresh controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 Refresh Data"):
                self._refresh_data()
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        with col3:
            st.write(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Render widgets in layout
        for row in self.layout:
            cols = st.columns(len(row))
            
            for i, widget_id in enumerate(row):
                if widget_id in self.widgets:
                    with cols[i]:
                        try:
                            self.widgets[widget_id].render()
                        except Exception as e:
                            st.error(f"Error rendering widget {widget_id}: {e}")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(5)
            st.experimental_rerun()
    
    def _refresh_data(self):
        """Refresh all widget data."""
        try:
            # Get dashboard data from analytics engine
            dashboard_data = self.analytics_engine.generate_dashboard_data()
            
            # Update KPI widget
            if "kpi_overview" in self.widgets:
                kpi_data = dashboard_data.get("kpis", {})
                self.update_widget_data("kpi_overview", {"kpis": kpi_data.get("kpis", [])})
            
            # Update metric widgets with recent data
            for widget_id, widget in self.widgets.items():
                if isinstance(widget, MetricWidget):
                    metric_history = self.analytics_engine.metrics_collector.get_metric_history(
                        widget.metric_name, hours=1
                    )
                    
                    if metric_history:
                        current_value = metric_history[-1].value if metric_history else 0
                        previous_value = metric_history[-2].value if len(metric_history) > 1 else current_value
                        change = current_value - previous_value
                        
                        self.update_widget_data(widget_id, {
                            "current_value": current_value,
                            "change": change,
                            "target_value": 100.0,  # Default target
                            "performance_percentage": (current_value / 100.0) * 100
                        })
                
                elif isinstance(widget, ChartWidget):
                    # Update chart widgets with trend data
                    if "temp_trend" in widget_id:
                        temp_history = self.analytics_engine.metrics_collector.get_metric_history(
                            "plasma_temperature", hours=24
                        )
                        if temp_history:
                            values = [m.value for m in temp_history[-50:]]  # Last 50 points
                            self.update_widget_data(widget_id, {
                                "chart_data": {"Temperature": values}
                            })
                    
                    elif "confinement_chart" in widget_id:
                        conf_history = self.analytics_engine.metrics_collector.get_metric_history(
                            "confinement_time", hours=24
                        )
                        if conf_history:
                            values = [m.value for m in conf_history[-50:]]
                            self.update_widget_data(widget_id, {
                                "chart_data": {"Confinement Time": values}
                            })
                    
                    elif "accuracy_chart" in widget_id:
                        acc_history = self.analytics_engine.metrics_collector.get_metric_history(
                            "prediction_accuracy", hours=24
                        )
                        if acc_history:
                            values = [m.value for m in acc_history[-50:]]
                            self.update_widget_data(widget_id, {
                                "chart_data": {"Accuracy": values}
                            })
            
            # Update table widget with recent metrics
            if "recent_metrics" in self.widgets:
                all_metrics = []
                for metric_list in self.analytics_engine.metrics_collector.metrics.values():
                    all_metrics.extend(metric_list[-10:])  # Last 10 of each type
                
                # Sort by timestamp
                all_metrics.sort(key=lambda m: m.timestamp, reverse=True)
                
                table_data = []
                for metric in all_metrics[:20]:  # Top 20 most recent
                    table_data.append({
                        "Metric": metric.name,
                        "Value": f"{metric.value:.2f}",
                        "Timestamp": metric.timestamp.strftime("%H:%M:%S"),
                        "Unit": metric.unit
                    })
                
                self.update_widget_data("recent_metrics", {"table_data": table_data})
            
            logger.info("Dashboard data refreshed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing dashboard data: {e}")


def create_dashboard_app(analytics_engine: AnalyticsEngine):
    """
    Create Streamlit dashboard application.
    
    Args:
        analytics_engine: Analytics engine instance.
        
    Returns:
        Dashboard function to run.
    """
    if not HAS_DASHBOARD_DEPS:
        raise RuntimeError("Dashboard dependencies not available")
    
    dashboard = AnalyticsDashboard(analytics_engine)
    
    def run_dashboard():
        """Run the dashboard."""
        dashboard.render()
    
    return run_dashboard


def run_dashboard_server(analytics_engine: AnalyticsEngine, port: int = 8501):
    """
    Run dashboard server.
    
    Args:
        analytics_engine: Analytics engine instance.
        port: Port to run server on.
    """
    if not HAS_DASHBOARD_DEPS:
        raise RuntimeError("Dashboard dependencies not available")
    
    import subprocess
    import sys
    import os
    
    # Create temporary dashboard script
    dashboard_script = f"""
import sys
sys.path.append('{os.getcwd()}')

from src.analytics import create_analytics_engine
from src.analytics.dashboard import create_dashboard_app

# Create analytics engine (in real app, this would be injected)
analytics_engine = create_analytics_engine()

# Create and run dashboard
dashboard_app = create_dashboard_app(analytics_engine)
dashboard_app()
"""
    
    # Write script to temporary file
    script_path = "temp_dashboard.py"
    with open(script_path, 'w') as f:
        f.write(dashboard_script)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", script_path,
            "--server.port", str(port),
            "--server.address", "localhost"
        ])
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)