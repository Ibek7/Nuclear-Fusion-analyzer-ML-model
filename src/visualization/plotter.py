"""
Visualization utilities for nuclear fusion data analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not available. Basic plotting disabled.")
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Interactive plotting disabled.")
    PLOTLY_AVAILABLE = False


class FusionPlotter:
    """
    Comprehensive plotting utilities for nuclear fusion data visualization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the fusion plotter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_style = self.config.get('theme', 'plotly_white')
        self.default_width = self.config.get('width', 800)
        self.default_height = self.config.get('height', 600)
        
        # Set default styles
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
            sns.set_palette("husl")
    
    def plot_plasma_parameters(self, data: pd.DataFrame, 
                             save_path: str = None) -> Optional['go.Figure']:
        """
        Plot plasma parameters over time or samples.
        
        Args:
            data: Fusion data with plasma parameters
            save_path: Path to save the plot
            
        Returns:
            Plotly figure if available
        """
        if not PLOTLY_AVAILABLE:
            return self._plot_plasma_parameters_matplotlib(data, save_path)
        
        # Key plasma parameters to plot
        plasma_params = [
            'plasma_temperature', 'plasma_density', 'magnetic_field',
            'pressure', 'confinement_time', 'beta_plasma', 'safety_factor'
        ]
        
        # Filter available parameters
        available_params = [param for param in plasma_params if param in data.columns]
        
        if not available_params:
            print("No plasma parameters found in data")
            return None
        
        # Create subplots
        n_params = len(available_params)
        cols = 3
        rows = (n_params + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=available_params,
            specs=[[{"secondary_y": False}] * cols for _ in range(rows)]
        )
        
        # Add traces for each parameter
        for i, param in enumerate(available_params):
            row = i // cols + 1
            col = i % cols + 1
            
            # Determine x-axis (time if available, otherwise sample index)
            x_data = data.get('timestamp', data.index)
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=data[param],
                    mode='lines',
                    name=param,
                    line=dict(width=2)
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title="Plasma Parameters",
            height=self.default_height * (rows / 2),
            width=self.default_width,
            template=self.default_style,
            showlegend=False
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_fusion_efficiency(self, data: pd.DataFrame, 
                             save_path: str = None) -> Optional['go.Figure']:
        """
        Plot fusion efficiency metrics and Q factor.
        
        Args:
            data: Fusion data with efficiency metrics
            save_path: Path to save the plot
            
        Returns:
            Plotly figure if available
        """
        if not PLOTLY_AVAILABLE:
            return self._plot_fusion_efficiency_matplotlib(data, save_path)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Q Factor vs Time", "Fusion Power vs Heating Power",
                "Lawson Criterion", "Energy Confinement Time"
            ]
        )
        
        x_data = data.get('timestamp', data.index)
        
        # Q Factor
        if 'q_factor' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=data['q_factor'],
                    mode='lines', name='Q Factor',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add breakeven line
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                         annotation_text="Breakeven", row=1, col=1)
        
        # Fusion Power vs Heating Power
        if all(col in data.columns for col in ['fusion_power', 'total_heating_power']):
            fig.add_trace(
                go.Scatter(
                    x=data['total_heating_power'], y=data['fusion_power'],
                    mode='markers', name='Power Relation',
                    marker=dict(color='green', size=6)
                ),
                row=1, col=2
            )
            
            # Add breakeven line
            max_power = max(data['total_heating_power'].max(), data['fusion_power'].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_power], y=[0, max_power],
                    mode='lines', name='Breakeven',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
        
        # Lawson Criterion
        if 'lawson_criterion' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=data['lawson_criterion'],
                    mode='lines', name='Lawson Criterion',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
            # Add threshold line
            fig.add_hline(y=1e21, line_dash="dash", line_color="red",
                         annotation_text="Ignition Threshold", row=2, col=1)
        
        # Energy Confinement Time
        if 'energy_confinement_time' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=data['energy_confinement_time'],
                    mode='lines', name='Energy Confinement',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Fusion Efficiency Metrics",
            height=self.default_height,
            width=self.default_width,
            template=self.default_style,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                              save_path: str = None) -> Optional['go.Figure']:
        """
        Plot correlation matrix of fusion parameters.
        
        Args:
            data: Fusion data
            save_path: Path to save the plot
            
        Returns:
            Plotly figure if available
        """
        if not PLOTLY_AVAILABLE:
            return self._plot_correlation_matrix_matplotlib(data, save_path)
        
        # Select numerical columns
        numerical_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Parameter Correlation Matrix",
            height=self.default_height,
            width=self.default_width,
            template=self.default_style
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_anomaly_detection(self, data: pd.DataFrame, 
                             anomaly_column: str = 'is_anomaly',
                             save_path: str = None) -> Optional['go.Figure']:
        """
        Plot anomaly detection results.
        
        Args:
            data: Data with anomaly predictions
            anomaly_column: Column containing anomaly labels
            save_path: Path to save the plot
            
        Returns:
            Plotly figure if available
        """
        if not PLOTLY_AVAILABLE:
            return self._plot_anomaly_detection_matplotlib(data, anomaly_column, save_path)
        
        if anomaly_column not in data.columns:
            print(f"Anomaly column '{anomaly_column}' not found")
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Anomalies Over Time", "Q Factor Anomalies",
                "Plasma Stability", "Disruption Risk"
            ]
        )
        
        x_data = data.get('timestamp', data.index)
        normal_mask = ~data[anomaly_column]
        anomaly_mask = data[anomaly_column]
        
        # Anomalies over time
        fig.add_trace(
            go.Scatter(
                x=x_data[normal_mask], y=[0] * sum(normal_mask),
                mode='markers', name='Normal',
                marker=dict(color='green', size=8, symbol='circle')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_data[anomaly_mask], y=[1] * sum(anomaly_mask),
                mode='markers', name='Anomaly',
                marker=dict(color='red', size=8, symbol='x')
            ),
            row=1, col=1
        )
        
        # Q Factor with anomalies
        if 'q_factor' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data[normal_mask], y=data.loc[normal_mask, 'q_factor'],
                    mode='markers', name='Normal Q',
                    marker=dict(color='blue', size=6)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_data[anomaly_mask], y=data.loc[anomaly_mask, 'q_factor'],
                    mode='markers', name='Anomaly Q',
                    marker=dict(color='red', size=8, symbol='x')
                ),
                row=1, col=2
            )
        
        # Plasma stability
        if 'plasma_stability' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=data['plasma_stability'],
                    mode='lines', name='Stability',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            # Mark anomalous regions
            for idx in data[anomaly_mask].index:
                fig.add_vline(x=x_data[idx], line_color="red", line_width=1,
                             opacity=0.5, row=2, col=1)
        
        # Disruption risk
        if 'disruption_probability' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data, y=data['disruption_probability'],
                    mode='lines', name='Disruption Risk',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=2
            )
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="High Risk", row=2, col=2)
        
        fig.update_layout(
            title="Anomaly Detection Results",
            height=self.default_height,
            width=self.default_width,
            template=self.default_style,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_model_performance(self, model_results: Dict[str, Any], 
                             save_path: str = None) -> Optional['go.Figure']:
        """
        Plot model performance comparison.
        
        Args:
            model_results: Results from model training
            save_path: Path to save the plot
            
        Returns:
            Plotly figure if available
        """
        if not PLOTLY_AVAILABLE:
            return self._plot_model_performance_matplotlib(model_results, save_path)
        
        # Extract metrics
        models = []
        train_r2 = []
        val_r2 = []
        train_rmse = []
        val_rmse = []
        
        for model_name, results in model_results.items():
            if isinstance(results, dict) and 'train_r2' in results:
                models.append(model_name)
                train_r2.append(results.get('train_r2', 0))
                val_r2.append(results.get('val_r2', 0))
                train_rmse.append(results.get('train_rmse', 0))
                val_rmse.append(results.get('val_rmse', 0))
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["R² Score", "RMSE"]
        )
        
        # R² scores
        fig.add_trace(
            go.Bar(
                x=models, y=train_r2,
                name='Training R²',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=models, y=val_r2,
                name='Validation R²',
                marker_color='darkblue'
            ),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(
                x=models, y=train_rmse,
                name='Training RMSE',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=models, y=val_rmse,
                name='Validation RMSE',
                marker_color='darkred'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=self.default_height,
            width=self.default_width,
            template=self.default_style,
            barmode='group'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_3d_plasma_state(self, data: pd.DataFrame, 
                           save_path: str = None) -> Optional['go.Figure']:
        """
        Create 3D visualization of plasma state space.
        
        Args:
            data: Fusion data
            save_path: Path to save the plot
            
        Returns:
            Plotly figure if available
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        # Check for required columns
        required_cols = ['plasma_temperature', 'plasma_density', 'magnetic_field']
        if not all(col in data.columns for col in required_cols):
            print("Required columns for 3D plot not available")
            return None
        
        # Color by Q factor if available
        color_col = 'q_factor' if 'q_factor' in data.columns else None
        
        fig = go.Figure(data=go.Scatter3d(
            x=data['plasma_temperature'],
            y=data['plasma_density'],
            z=data['magnetic_field'],
            mode='markers',
            marker=dict(
                size=5,
                color=data[color_col] if color_col else 'blue',
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col or 'Value')
            ),
            text=[f"Sample {i}" for i in data.index],
            hovertemplate="<b>%{text}</b><br>" +
                         "Temperature: %{x:.2e}<br>" +
                         "Density: %{y:.2e}<br>" +
                         "Magnetic Field: %{z:.2f}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title="3D Plasma State Space",
            scene=dict(
                xaxis_title="Plasma Temperature (K)",
                yaxis_title="Plasma Density (m⁻³)",
                zaxis_title="Magnetic Field (T)"
            ),
            height=self.default_height,
            width=self.default_width,
            template=self.default_style
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    # Matplotlib fallback methods
    def _plot_plasma_parameters_matplotlib(self, data: pd.DataFrame, 
                                          save_path: str = None):
        """Matplotlib fallback for plasma parameters."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        plasma_params = ['plasma_temperature', 'plasma_density', 'magnetic_field']
        available_params = [p for p in plasma_params if p in data.columns]
        
        if not available_params:
            return None
        
        fig, axes = plt.subplots(len(available_params), 1, figsize=(10, 6))
        if len(available_params) == 1:
            axes = [axes]
        
        for i, param in enumerate(available_params):
            axes[i].plot(data.index, data[param])
            axes[i].set_title(param)
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_fusion_efficiency_matplotlib(self, data: pd.DataFrame, 
                                          save_path: str = None):
        """Matplotlib fallback for fusion efficiency."""
        if not MATPLOTLIB_AVAILABLE or 'q_factor' not in data.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['q_factor'], label='Q Factor')
        ax.axhline(y=1.0, color='r', linestyle='--', label='Breakeven')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Q Factor')
        ax.set_title('Fusion Efficiency')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_correlation_matrix_matplotlib(self, data: pd.DataFrame, 
                                           save_path: str = None):
        """Matplotlib fallback for correlation matrix."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        numerical_data = data.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax)
        ax.set_title('Parameter Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_anomaly_detection_matplotlib(self, data: pd.DataFrame, 
                                          anomaly_column: str,
                                          save_path: str = None):
        """Matplotlib fallback for anomaly detection."""
        if not MATPLOTLIB_AVAILABLE or anomaly_column not in data.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        normal_mask = ~data[anomaly_column]
        anomaly_mask = data[anomaly_column]
        
        ax.scatter(data.index[normal_mask], [0] * sum(normal_mask), 
                  c='green', label='Normal', alpha=0.6)
        ax.scatter(data.index[anomaly_mask], [1] * sum(anomaly_mask), 
                  c='red', label='Anomaly', alpha=0.8)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Anomaly Status')
        ax.set_title('Anomaly Detection Results')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_model_performance_matplotlib(self, model_results: Dict[str, Any], 
                                          save_path: str = None):
        """Matplotlib fallback for model performance."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models = []
        val_r2 = []
        
        for model_name, results in model_results.items():
            if isinstance(results, dict) and 'val_r2' in results:
                models.append(model_name)
                val_r2.append(results.get('val_r2', 0))
        
        if not models:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(models, val_r2, color='skyblue')
        ax.set_xlabel('Model')
        ax.set_ylabel('Validation R² Score')
        ax.set_title('Model Performance Comparison')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
        
        return fig