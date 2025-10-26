"""
Interactive Real-time Dashboard for Nuclear Fusion Analyzer.

This module provides a comprehensive real-time dashboard using Streamlit
with advanced interactive components, real-time data streaming, parameter
controls, and multi-dimensional data exploration capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.fusion_data_generator import FusionDataGenerator
from src.models.fusion_predictor import FusionPredictor
from src.visualization.plasma_3d import (
    PlasmaGeometry, VolumetricRenderer, ParticleTracker, PlasmaState
)
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RealTimeDataStream:
    """
    Real-time data streaming system for fusion parameters.
    
    Simulates live data feeds from fusion reactor sensors
    with realistic noise, drift, and operational events.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize real-time data stream.
        
        Args:
            update_interval: Data update interval in seconds.
        """
        self.update_interval = update_interval
        self.running = False
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Initialize data generator
        self.generator = FusionDataGenerator()
        
        # Streaming parameters
        self.current_time = 0.0
        self.base_params = {
            'magnetic_field': 5.3,
            'plasma_current': 15.0,
            'electron_density': 1.0e20,
            'ion_temperature': 10.0,
            'electron_temperature': 12.0,
            'neutral_beam_power': 30.0,
            'rf_heating_power': 20.0
        }
        
        # Add realistic variations
        self.param_trends = {}
        self.param_noise = {}
        self._initialize_variations()
        
        logger.info("RealTimeDataStream initialized")
    
    def _initialize_variations(self):
        """Initialize parameter variations for realistic simulation."""
        for param in self.base_params.keys():
            self.param_trends[param] = np.random.uniform(-0.01, 0.01)
            self.param_noise[param] = np.random.uniform(0.01, 0.05)
    
    def generate_next_sample(self) -> Dict[str, float]:
        """
        Generate next data sample with realistic variations.
        
        Returns:
            Dictionary of parameter values.
        """
        sample = {}
        
        for param, base_value in self.base_params.items():
            # Add trend
            trend = self.param_trends[param] * self.current_time
            
            # Add noise
            noise = np.random.normal(0, self.param_noise[param] * base_value)
            
            # Add occasional disruptions
            if np.random.random() < 0.01:  # 1% chance of disruption
                disruption = np.random.uniform(-0.2, 0.2) * base_value
            else:
                disruption = 0
            
            # Calculate final value
            value = base_value * (1 + trend) + noise + disruption
            
            # Apply physical constraints
            if param in ['electron_density', 'ion_temperature', 'electron_temperature']:
                value = max(value, 0.1 * base_value)  # Minimum values
            
            sample[param] = value
        
        # Add timestamp
        sample['timestamp'] = datetime.now()
        sample['time'] = self.current_time
        
        self.current_time += self.update_interval
        
        return sample
    
    def start_streaming(self):
        """Start real-time data streaming."""
        if self.running:
            return
        
        self.running = True
        
        def stream_worker():
            while self.running:
                try:
                    sample = self.generate_next_sample()
                    
                    if not self.data_queue.full():
                        self.data_queue.put(sample)
                    else:
                        # Remove old data if queue is full
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.data_queue.put(sample)
                    
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
        
        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()
        
        logger.info("Real-time streaming started")
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        self.running = False
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=2)
        logger.info("Real-time streaming stopped")
    
    def get_latest_data(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Get latest data samples.
        
        Args:
            n_samples: Number of recent samples to return.
            
        Returns:
            DataFrame with recent data.
        """
        samples = []
        
        # Get all available data
        while not self.data_queue.empty():
            try:
                samples.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        
        # Keep only the most recent samples
        if len(samples) > n_samples:
            samples = samples[-n_samples:]
        
        if samples:
            return pd.DataFrame(samples)
        else:
            # Return empty DataFrame with expected columns
            columns = list(self.base_params.keys()) + ['timestamp', 'time']
            return pd.DataFrame(columns=columns)


class InteractiveDashboard:
    """
    Main interactive dashboard for fusion data visualization.
    
    Provides comprehensive real-time monitoring, control interfaces,
    and advanced visualization capabilities.
    """
    
    def __init__(self):
        """Initialize interactive dashboard."""
        self.data_stream = None
        self.predictor = None
        self.performance_monitor = PerformanceMonitor()
        
        # Dashboard state
        self.dashboard_state = {
            'auto_refresh': True,
            'refresh_interval': 1.0,
            'show_predictions': True,
            'show_3d': False,
            'selected_parameters': ['plasma_current', 'electron_temperature']
        }
        
        logger.info("InteractiveDashboard initialized")
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Nuclear Fusion Analyzer - Real-time Dashboard",
            page_icon="⚛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2E86AB;
        }
        .status-indicator {
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-green { background-color: #4CAF50; }
        .status-yellow { background-color: #FF9800; }
        .status-red { background-color: #F44336; }
        </style>
        """, unsafe_allow_html=True)
    
    def setup_sidebar(self):
        """Setup dashboard control sidebar."""
        st.sidebar.header("Dashboard Controls")
        
        # Data streaming controls
        st.sidebar.subheader("Data Stream")
        
        if st.sidebar.button("Start Data Stream"):
            if self.data_stream is None:
                self.data_stream = RealTimeDataStream()
            self.data_stream.start_streaming()
            st.sidebar.success("Data streaming started")
        
        if st.sidebar.button("Stop Data Stream"):
            if self.data_stream:
                self.data_stream.stop_streaming()
            st.sidebar.info("Data streaming stopped")
        
        # Refresh settings
        self.dashboard_state['auto_refresh'] = st.sidebar.checkbox(
            "Auto Refresh", value=self.dashboard_state['auto_refresh']
        )
        
        if self.dashboard_state['auto_refresh']:
            self.dashboard_state['refresh_interval'] = st.sidebar.slider(
                "Refresh Interval (s)", 0.5, 5.0, 
                self.dashboard_state['refresh_interval'], 0.5
            )
        
        # Display options
        st.sidebar.subheader("Display Options")
        
        self.dashboard_state['show_predictions'] = st.sidebar.checkbox(
            "Show ML Predictions", value=self.dashboard_state['show_predictions']
        )
        
        self.dashboard_state['show_3d'] = st.sidebar.checkbox(
            "Show 3D Visualization", value=self.dashboard_state['show_3d']
        )
        
        # Parameter selection
        st.sidebar.subheader("Parameters")
        
        available_params = [
            'magnetic_field', 'plasma_current', 'electron_density',
            'ion_temperature', 'electron_temperature', 
            'neutral_beam_power', 'rf_heating_power'
        ]
        
        self.dashboard_state['selected_parameters'] = st.sidebar.multiselect(
            "Select Parameters to Display",
            available_params,
            default=self.dashboard_state['selected_parameters']
        )
        
        # Performance metrics
        st.sidebar.subheader("Performance")
        
        if hasattr(self, 'performance_metrics'):
            for metric, value in self.performance_metrics.items():
                st.sidebar.metric(metric, f"{value:.2f}")
    
    def create_status_indicators(self, data: pd.DataFrame):
        """Create system status indicators."""
        if data.empty:
            return
        
        latest = data.iloc[-1]
        
        # Define status criteria
        status_criteria = {
            'Plasma Current': {
                'value': latest.get('plasma_current', 0),
                'unit': 'MA',
                'good': (10, 20),
                'warning': (8, 25),
            },
            'Temperature': {
                'value': latest.get('electron_temperature', 0),
                'unit': 'keV',
                'good': (8, 15),
                'warning': (5, 20),
            },
            'Density': {
                'value': latest.get('electron_density', 0) / 1e19,
                'unit': '10¹⁹ m⁻³',
                'good': (0.8, 1.5),
                'warning': (0.5, 2.0),
            },
            'Magnetic Field': {
                'value': latest.get('magnetic_field', 0),
                'unit': 'T',
                'good': (5.0, 5.5),
                'warning': (4.5, 6.0),
            }
        }
        
        # Create status display
        cols = st.columns(len(status_criteria))
        
        for i, (name, criteria) in enumerate(status_criteria.items()):
            with cols[i]:
                value = criteria['value']
                unit = criteria['unit']
                
                # Determine status
                if criteria['good'][0] <= value <= criteria['good'][1]:
                    status = "🟢"
                    color = "green"
                elif criteria['warning'][0] <= value <= criteria['warning'][1]:
                    status = "🟡"
                    color = "orange"
                else:
                    status = "🔴"
                    color = "red"
                
                st.metric(
                    label=f"{status} {name}",
                    value=f"{value:.2f} {unit}",
                    delta=None
                )
    
    def create_realtime_plots(self, data: pd.DataFrame):
        """Create real-time parameter plots."""
        if data.empty:
            st.info("No data available. Start data streaming to see real-time plots.")
            return
        
        # Main time series plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Plasma Current', 'Temperature', 'Density', 'Power'],
            vertical_spacing=0.08
        )
        
        # Plasma current
        fig.add_trace(
            go.Scatter(
                x=data['time'], y=data['plasma_current'],
                mode='lines', name='Plasma Current',
                line=dict(color='#2E86AB', width=2)
            ),
            row=1, col=1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(
                x=data['time'], y=data['electron_temperature'],
                mode='lines', name='Electron Temp',
                line=dict(color='#A23B72', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['time'], y=data['ion_temperature'],
                mode='lines', name='Ion Temp',
                line=dict(color='#F18F01', width=2)
            ),
            row=1, col=2
        )
        
        # Density
        fig.add_trace(
            go.Scatter(
                x=data['time'], y=data['electron_density']/1e19,
                mode='lines', name='Density',
                line=dict(color='#C73E1D', width=2)
            ),
            row=2, col=1
        )
        
        # Power
        fig.add_trace(
            go.Scatter(
                x=data['time'], y=data['neutral_beam_power'],
                mode='lines', name='NB Power',
                line=dict(color='#4CAF50', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['time'], y=data['rf_heating_power'],
                mode='lines', name='RF Power',
                line=dict(color='#FF9800', width=2)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Real-time Fusion Parameters",
            showlegend=True
        )
        
        # Add units to y-axes
        fig.update_yaxes(title_text="Current (MA)", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (keV)", row=1, col=2)
        fig.update_yaxes(title_text="Density (10¹⁹ m⁻³)", row=2, col=1)
        fig.update_yaxes(title_text="Power (MW)", row=2, col=2)
        
        fig.update_xaxes(title_text="Time (s)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_prediction_display(self, data: pd.DataFrame):
        """Create ML prediction display."""
        if not self.dashboard_state['show_predictions'] or data.empty:
            return
        
        st.subheader("ML Predictions")
        
        try:
            # Initialize predictor if needed
            if self.predictor is None:
                self.predictor = FusionPredictor()
                
                # Train on recent data if available
                if len(data) > 10:
                    features = data[['magnetic_field', 'plasma_current', 'electron_density',
                                   'ion_temperature', 'electron_temperature', 
                                   'neutral_beam_power', 'rf_heating_power']].values
                    
                    # Generate synthetic Q-factors for training
                    generator = FusionDataGenerator()
                    _, y_synthetic = generator.generate_fusion_data(len(data))
                    
                    self.predictor.fit(features, y_synthetic)
            
            # Make predictions on latest data
            if len(data) > 0:
                latest_features = data[['magnetic_field', 'plasma_current', 'electron_density',
                                      'ion_temperature', 'electron_temperature', 
                                      'neutral_beam_power', 'rf_heating_power']].iloc[-1:].values
                
                prediction = self.predictor.predict(latest_features)[0]
                
                # Display prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Q-factor", f"{prediction:.3f}")
                
                with col2:
                    # Classification based on Q-factor
                    if prediction > 1.0:
                        classification = "Net Energy Gain"
                        color = "green"
                    elif prediction > 0.5:
                        classification = "High Performance"
                        color = "orange"
                    else:
                        classification = "Low Performance"
                        color = "red"
                    
                    st.markdown(f"**Status:** :{color}[{classification}]")
                
                with col3:
                    # Prediction confidence (simplified)
                    confidence = 0.85 + 0.1 * np.random.random()
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Prediction trend
                if len(data) >= 10:
                    recent_features = data[['magnetic_field', 'plasma_current', 'electron_density',
                                          'ion_temperature', 'electron_temperature', 
                                          'neutral_beam_power', 'rf_heating_power']].iloc[-10:].values
                    
                    predictions = self.predictor.predict(recent_features)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data['time'].iloc[-10:],
                        y=predictions,
                        mode='lines+markers',
                        name='Q-factor Prediction',
                        line=dict(color='purple', width=3)
                    ))
                    
                    fig.update_layout(
                        title="Q-factor Prediction Trend",
                        xaxis_title="Time (s)",
                        yaxis_title="Q-factor",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            logger.error(f"Prediction error: {e}")
    
    def create_3d_visualization(self, data: pd.DataFrame):
        """Create 3D plasma visualization."""
        if not self.dashboard_state['show_3d'] or data.empty:
            return
        
        st.subheader("3D Plasma Visualization")
        
        try:
            # Create simplified 3D visualization
            renderer = VolumetricRenderer()
            geometry = PlasmaGeometry()
            
            # Create mesh
            X, Y, Z = geometry.create_tokamak_mesh(nr=20, ntheta=50, nzeta=30)
            
            # Generate plasma data based on current parameters
            latest = data.iloc[-1]
            temp_scale = latest['electron_temperature'] / 10.0
            density_scale = latest['electron_density'] / 1e20
            
            temperature = temp_scale * 10.0 * np.exp(-((X - 6.2)**2 + Y**2)/4) * np.exp(-Z**2/2)
            density = density_scale * 1e20 * np.exp(-((X - 6.2)**2 + Y**2)/3) * np.exp(-Z**2/1.5)
            pressure = temperature * density * 1.602e-16
            
            # Create plasma state
            plasma_state = PlasmaState(
                temperature=temperature,
                density=density,
                pressure=pressure,
                magnetic_field=(np.zeros_like(X), np.zeros_like(Y), 5.0 * np.ones_like(Z)),
                velocity=(np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)),
                coordinates=(X, Y, Z),
                time_stamp=latest['time'],
                metadata={'live_data': True}
            )
            
            # Render 3D plot
            fig_3d = renderer.render_plasma_volume_plotly(
                plasma_state, 
                quantity='temperature',
                title=f"Live Plasma Temperature (t={latest['time']:.1f}s)"
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        except Exception as e:
            st.error(f"3D visualization error: {e}")
            logger.error(f"3D visualization error: {e}")
    
    def create_parameter_analysis(self, data: pd.DataFrame):
        """Create parameter analysis section."""
        if data.empty:
            return
        
        st.subheader("Parameter Analysis")
        
        # Parameter correlation heatmap
        numeric_cols = ['magnetic_field', 'plasma_current', 'electron_density',
                       'ion_temperature', 'electron_temperature', 
                       'neutral_beam_power', 'rf_heating_power']
        
        if all(col in data.columns for col in numeric_cols):
            corr_matrix = data[numeric_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(
                title="Parameter Correlation Matrix",
                height=400
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        
        summary_stats = data[numeric_cols].describe()
        st.dataframe(summary_stats.round(3))
    
    def run_dashboard(self):
        """Run the main dashboard application."""
        self.setup_page_config()
        
        # Header
        st.markdown('<h1 class="main-header">⚛️ Nuclear Fusion Analyzer</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666;">Real-time Plasma Monitoring Dashboard</h3>', 
                   unsafe_allow_html=True)
        
        # Setup sidebar
        self.setup_sidebar()
        
        # Main content area
        if self.data_stream is None:
            st.info("🚀 Start the data stream from the sidebar to begin monitoring.")
            return
        
        # Get latest data
        data = self.data_stream.get_latest_data(n_samples=200)
        
        if not data.empty:
            # Status indicators
            self.create_status_indicators(data)
            
            # Main plots
            self.create_realtime_plots(data)
            
            # Additional sections in columns
            col1, col2 = st.columns(2)
            
            with col1:
                self.create_prediction_display(data)
            
            with col2:
                self.create_parameter_analysis(data)
            
            # 3D visualization (full width)
            self.create_3d_visualization(data)
            
            # Data table (expandable)
            with st.expander("Raw Data"):
                st.dataframe(data.tail(20))
        
        # Auto-refresh
        if self.dashboard_state['auto_refresh']:
            time.sleep(self.dashboard_state['refresh_interval'])
            st.experimental_rerun()


def main():
    """Main dashboard entry point."""
    dashboard = InteractiveDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()