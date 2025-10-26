"""
Real-time monitoring dashboard for nuclear fusion analysis.

This module provides a real-time web dashboard for monitoring fusion
reactor parameters, model predictions, and system performance.
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass, asdict

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import dash_bootstrap_components as dbc
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

import pandas as pd
import numpy as np
from collections import deque, defaultdict
import queue
import websocket
import socket

from src.data.generator import FusionDataGenerator
from src.models.fusion_predictor import FusionPredictor
from src.models.anomaly_detector import FusionAnomalyDetector
from src.utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class DashboardData:
    """Container for dashboard data."""
    
    timestamp: datetime
    plasma_params: Dict[str, float]
    predictions: Dict[str, float]
    anomaly_scores: Dict[str, float]
    system_metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]


class RealTimeDataSimulator:
    """
    Simulates real-time fusion reactor data for dashboard demonstration.
    
    In a real implementation, this would connect to actual reactor sensors
    and control systems.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize the data simulator.
        
        Args:
            update_interval: Time between data updates (seconds).
        """
        self.update_interval = update_interval
        self.generator = FusionDataGenerator(random_state=int(time.time()))
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._data_queue = queue.Queue(maxsize=1000)
        
        # Base parameters for simulation
        self.base_params = {
            'magnetic_field': 5.3,
            'plasma_current': 15.0,
            'electron_density': 1.0e20,
            'ion_temperature': 20.0,
            'electron_temperature': 15.0,
            'neutral_beam_power': 50.0,
            'rf_heating_power': 30.0
        }
        
        # Simulation state
        self.time_step = 0
        self.scenario = 'normal'  # 'normal', 'disruption', 'shutdown'
        
        logger.info(f"RealTimeDataSimulator initialized with {update_interval}s interval")
    
    def start_simulation(self):
        """Start the real-time data simulation."""
        if self.is_running:
            logger.warning("Simulation already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()
        
        logger.info("Real-time data simulation started")
    
    def stop_simulation(self):
        """Stop the real-time data simulation."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info("Real-time data simulation stopped")
    
    def get_latest_data(self) -> Optional[DashboardData]:
        """Get the latest data from the simulation."""
        try:
            return self._data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def set_scenario(self, scenario: str):
        """Set the simulation scenario."""
        self.scenario = scenario
        logger.info(f"Simulation scenario changed to: {scenario}")
    
    def _simulation_loop(self):
        """Main simulation loop running in background thread."""
        while self.is_running:
            try:
                # Generate new data point
                data = self._generate_data_point()
                
                # Add to queue
                if not self._data_queue.full():
                    self._data_queue.put(data)
                else:
                    # Remove oldest if queue is full
                    try:
                        self._data_queue.get_nowait()
                        self._data_queue.put(data)
                    except queue.Empty:
                        pass
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(self.update_interval)
    
    def _generate_data_point(self) -> DashboardData:
        """Generate a single data point."""
        self.time_step += 1
        current_time = datetime.now()
        
        # Modify parameters based on scenario
        params = self.base_params.copy()
        
        if self.scenario == 'normal':
            # Normal operation with small variations
            for key in params:
                noise_factor = 0.05  # 5% noise
                params[key] *= (1 + np.random.normal(0, noise_factor))
        
        elif self.scenario == 'disruption':
            # Simulate plasma disruption
            if self.time_step % 50 < 5:  # 5 time steps of disruption every 50
                params['plasma_current'] *= 0.1  # Current collapse
                params['ion_temperature'] *= 0.3  # Temperature drop
                params['electron_temperature'] *= 0.3
        
        elif self.scenario == 'shutdown':
            # Simulate controlled shutdown
            decay_factor = 0.99
            for key in ['plasma_current', 'ion_temperature', 'electron_temperature']:
                params[key] *= decay_factor
        
        # Ensure physical bounds
        params = self._apply_physics_bounds(params)
        
        # Generate derived parameters
        derived_params = self.generator._calculate_derived_parameters(params)
        all_params = {**params, **derived_params}
        
        # Simulate predictions (in real system, these would come from ML models)
        predictions = {
            'q_factor': all_params.get('q_factor', 1.0),
            'confinement_time': all_params.get('confinement_time', 1.0),
            'beta': all_params.get('beta', 0.02)
        }
        
        # Simulate anomaly scores
        anomaly_scores = {
            'isolation_forest': np.random.uniform(0, 1),
            'svm': np.random.uniform(0, 1),
            'disruption_probability': np.random.uniform(0, 0.1) if self.scenario == 'normal' else np.random.uniform(0.7, 1.0)
        }
        
        # Simulate system metrics
        system_metrics = {
            'cpu_usage': np.random.uniform(20, 80),
            'memory_usage': np.random.uniform(40, 90),
            'gpu_usage': np.random.uniform(10, 70),
            'model_latency': np.random.uniform(0.001, 0.01)
        }
        
        # Generate alerts
        alerts = []
        if predictions['q_factor'] < 0.5:
            alerts.append({
                'level': 'warning',
                'message': 'Q factor below 0.5',
                'timestamp': current_time.isoformat()
            })
        
        if anomaly_scores['disruption_probability'] > 0.7:
            alerts.append({
                'level': 'critical',
                'message': 'High disruption probability detected',
                'timestamp': current_time.isoformat()
            })
        
        return DashboardData(
            timestamp=current_time,
            plasma_params=all_params,
            predictions=predictions,
            anomaly_scores=anomaly_scores,
            system_metrics=system_metrics,
            alerts=alerts
        )
    
    def _apply_physics_bounds(self, params: Dict[str, float]) -> Dict[str, float]:
        """Apply physical bounds to parameters."""
        bounds = {
            'magnetic_field': (0.1, 20.0),
            'plasma_current': (0.1, 50.0),
            'electron_density': (1e18, 5e21),
            'ion_temperature': (0.1, 200.0),
            'electron_temperature': (0.1, 200.0),
            'neutral_beam_power': (0.0, 500.0),
            'rf_heating_power': (0.0, 200.0)
        }
        
        for param, value in params.items():
            if param in bounds:
                min_val, max_val = bounds[param]
                params[param] = np.clip(value, min_val, max_val)
        
        return params


class FusionDashboard:
    """
    Real-time fusion monitoring dashboard.
    
    Provides web-based visualization of fusion reactor parameters,
    model predictions, and system health.
    """
    
    def __init__(self, framework: str = 'streamlit'):
        """
        Initialize the dashboard.
        
        Args:
            framework: Dashboard framework ('streamlit' or 'dash').
        """
        self.framework = framework
        self.simulator = RealTimeDataSimulator(update_interval=0.5)
        self.data_history = deque(maxlen=1000)
        self.performance_monitor = PerformanceMonitor()
        
        # Dashboard state
        self.is_running = False
        self.alerts = deque(maxlen=100)
        
        logger.info(f"FusionDashboard initialized with {framework} framework")
    
    def start_dashboard(self, host: str = 'localhost', port: int = 8501):
        """Start the dashboard server."""
        if self.framework == 'streamlit' and HAS_STREAMLIT:
            self._start_streamlit_dashboard()
        elif self.framework == 'dash' and HAS_DASH:
            self._start_dash_dashboard(host, port)
        else:
            raise ValueError(f"Framework {self.framework} not available or not installed")
    
    def _start_streamlit_dashboard(self):
        """Start Streamlit-based dashboard."""
        logger.info("Starting Streamlit dashboard")
        
        # Start data simulation
        self.simulator.start_simulation()
        
        # Streamlit app configuration
        st.set_page_config(
            page_title="Nuclear Fusion Monitor",
            page_icon="⚛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main dashboard layout
        self._render_streamlit_dashboard()
    
    def _render_streamlit_dashboard(self):
        """Render the Streamlit dashboard."""
        # Header
        st.title("⚛️ Nuclear Fusion Real-Time Monitor")
        st.markdown("---")
        
        # Sidebar controls
        st.sidebar.header("Control Panel")
        
        # Scenario selection
        scenario = st.sidebar.selectbox(
            "Simulation Scenario",
            ['normal', 'disruption', 'shutdown'],
            index=0
        )
        self.simulator.set_scenario(scenario)
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        
        if auto_refresh:
            time.sleep(1)
            st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("Refresh Data"):
            st.rerun()
        
        # Get latest data
        latest_data = self.simulator.get_latest_data()
        if latest_data:
            self.data_history.append(latest_data)
        
        if not self.data_history:
            st.warning("No data available. Please wait for data generation...")
            return
        
        # Current status
        current_data = self.data_history[-1]
        
        # Status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            q_factor = current_data.predictions['q_factor']
            st.metric(
                "Q Factor",
                f"{q_factor:.3f}",
                delta=None,
                delta_color="normal"
            )
        
        with col2:
            disruption_prob = current_data.anomaly_scores['disruption_probability']
            st.metric(
                "Disruption Risk",
                f"{disruption_prob:.1%}",
                delta=None,
                delta_color="inverse"
            )
        
        with col3:
            plasma_current = current_data.plasma_params['plasma_current']
            st.metric(
                "Plasma Current",
                f"{plasma_current:.1f} MA",
                delta=None
            )
        
        with col4:
            system_status = "🟢 Normal" if disruption_prob < 0.3 else "🟡 Warning" if disruption_prob < 0.7 else "🔴 Critical"
            st.metric(
                "System Status",
                system_status,
                delta=None
            )
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time Data", "🎯 Predictions", "⚠️ Anomalies", "🖥️ System"])
        
        with tab1:
            self._render_realtime_plots()
        
        with tab2:
            self._render_prediction_plots()
        
        with tab3:
            self._render_anomaly_plots()
        
        with tab4:
            self._render_system_plots()
        
        # Alerts section
        if current_data.alerts:
            st.markdown("---")
            st.subheader("🚨 Active Alerts")
            for alert in current_data.alerts:
                if alert['level'] == 'critical':
                    st.error(f"**{alert['level'].upper()}**: {alert['message']}")
                elif alert['level'] == 'warning':
                    st.warning(f"**{alert['level'].upper()}**: {alert['message']}")
                else:
                    st.info(f"**{alert['level'].upper()}**: {alert['message']}")
    
    def _render_realtime_plots(self):
        """Render real-time parameter plots."""
        if len(self.data_history) < 2:
            st.info("Collecting data...")
            return
        
        # Prepare time series data
        df_data = []
        for data in list(self.data_history)[-50:]:  # Last 50 points
            row = {
                'timestamp': data.timestamp,
                **data.plasma_params
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Plot key parameters
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Q Factor', 'Plasma Current', 'Ion Temperature', 'Magnetic Field'],
            vertical_spacing=0.1
        )
        
        # Q Factor
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['q_factor'], name='Q Factor', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Plasma Current
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['plasma_current'], name='Plasma Current', line=dict(color='green')),
            row=1, col=2
        )
        
        # Ion Temperature
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['ion_temperature'], name='Ion Temperature', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Magnetic Field
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['magnetic_field'], name='Magnetic Field', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Real-Time Plasma Parameters")
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Q Factor", row=1, col=1)
        fig.update_yaxes(title_text="Current (MA)", row=1, col=2)
        fig.update_yaxes(title_text="Temperature (keV)", row=2, col=1)
        fig.update_yaxes(title_text="Field (T)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_prediction_plots(self):
        """Render prediction visualization."""
        if len(self.data_history) < 2:
            st.info("Collecting data...")
            return
        
        # Current predictions
        current_data = self.data_history[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Q Factor gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current_data.predictions['q_factor'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Q Factor"},
                delta = {'reference': 1.0},
                gauge = {
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 5], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Beta gauge
            fig_beta = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_data.predictions['beta'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Beta"},
                gauge = {
                    'axis': {'range': [None, 0.1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.05], 'color': "lightgray"},
                        {'range': [0.05, 0.1], 'color': "yellow"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.08
                    }
                }
            ))
            fig_beta.update_layout(height=300)
            st.plotly_chart(fig_beta, use_container_width=True)
    
    def _render_anomaly_plots(self):
        """Render anomaly detection visualization."""
        if len(self.data_history) < 2:
            st.info("Collecting data...")
            return
        
        # Prepare anomaly score time series
        df_anomaly = []
        for data in list(self.data_history)[-50:]:
            row = {
                'timestamp': data.timestamp,
                **data.anomaly_scores
            }
            df_anomaly.append(row)
        
        df = pd.DataFrame(df_anomaly)
        
        # Anomaly scores plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['disruption_probability'],
            name='Disruption Probability',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['isolation_forest'],
            name='Isolation Forest',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['svm'],
            name='One-Class SVM',
            line=dict(color='green')
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        
        fig.update_layout(
            title="Anomaly Detection Scores",
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_system_plots(self):
        """Render system performance visualization."""
        if len(self.data_history) < 2:
            st.info("Collecting data...")
            return
        
        # Current system metrics
        current_data = self.data_history[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("CPU Usage", current_data.system_metrics['cpu_usage'], "%"),
            ("Memory Usage", current_data.system_metrics['memory_usage'], "%"),
            ("GPU Usage", current_data.system_metrics['gpu_usage'], "%"),
            ("Model Latency", current_data.system_metrics['model_latency'] * 1000, "ms")
        ]
        
        for i, (name, value, unit) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                if unit == "%":
                    color = "red" if value > 90 else "orange" if value > 70 else "green"
                else:
                    color = "red" if value > 10 else "orange" if value > 5 else "green"
                
                st.metric(name, f"{value:.1f} {unit}")
    
    def stop_dashboard(self):
        """Stop the dashboard and simulation."""
        self.simulator.stop_simulation()
        self.is_running = False
        logger.info("Dashboard stopped")


def create_fusion_dashboard(framework: str = 'streamlit') -> FusionDashboard:
    """
    Create and configure a fusion monitoring dashboard.
    
    Args:
        framework: Dashboard framework to use.
        
    Returns:
        Configured FusionDashboard instance.
    """
    return FusionDashboard(framework=framework)


# Streamlit app entry point
if HAS_STREAMLIT and __name__ == "__main__":
    dashboard = create_fusion_dashboard('streamlit')
    dashboard._render_streamlit_dashboard()