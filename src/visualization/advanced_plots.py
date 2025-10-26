"""
Advanced visualization components for nuclear fusion analysis.

This module provides comprehensive visualization capabilities including
3D plasma visualizations, real-time monitoring dashboards, interactive
parameter exploration, and publication-ready scientific plots.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns

try:
    # 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False

try:
    # Interactive plotting
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    # Advanced visualization
    import bokeh.plotting as bk
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256, Plasma256
    from bokeh.layouts import gridplot, column, row
    HAS_BOKEH = True
except ImportError:
    HAS_BOKEH = False

try:
    # Scientific visualization
    import mayavi.mlab as mlab
    HAS_MAYAVI = True
except ImportError:
    HAS_MAYAVI = False

from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for plot styling and appearance."""
    
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = 'default'
    color_palette: str = 'viridis'
    font_size: int = 12
    title_size: int = 16
    label_size: int = 14
    legend_size: int = 12
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.8
    grid: bool = True
    tight_layout: bool = True


class FusionVisualizationSuite:
    """
    Comprehensive visualization suite for nuclear fusion data.
    
    Provides static plots, interactive visualizations, 3D representations,
    and real-time monitoring displays for fusion reactor analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the visualization suite.
        
        Args:
            config: Configuration dictionary for visualizations.
        """
        self.config = config or {}
        self.plot_config = PlotConfig(**self.config.get('plot_config', {}))
        
        # Set matplotlib style
        plt.style.use(self.plot_config.style)
        sns.set_palette(self.plot_config.color_palette)
        
        # Create color maps for fusion visualizations
        self.fusion_colormaps = self._create_fusion_colormaps()
        
        logger.info("FusionVisualizationSuite initialized")
    
    def _create_fusion_colormaps(self) -> Dict[str, Any]:
        """Create custom colormaps for fusion visualizations."""
        colormaps = {}
        
        # Plasma temperature colormap (blue to red)
        plasma_colors = ['#000080', '#0066CC', '#00CCFF', '#66FF66', '#FFFF00', '#FF6600', '#FF0000']
        colormaps['plasma_temp'] = ListedColormap(plasma_colors)
        
        # Magnetic field colormap (purple to yellow)
        magnetic_colors = ['#4B0082', '#8B00FF', '#00BFFF', '#00FF7F', '#FFFF00']
        colormaps['magnetic'] = ListedColormap(magnetic_colors)
        
        # Density colormap (black to white)
        density_colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC', '#FFFFFF']
        colormaps['density'] = ListedColormap(density_colors)
        
        return colormaps
    
    def plot_plasma_parameters_time_series(self, 
                                           data: pd.DataFrame,
                                           parameters: Optional[List[str]] = None,
                                           interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot time series of plasma parameters.
        
        Args:
            data: DataFrame with time series data.
            parameters: List of parameters to plot.
            interactive: Whether to create interactive plot.
            
        Returns:
            Matplotlib or Plotly figure.
        """
        if parameters is None:
            parameters = ['plasma_current', 'magnetic_field', 'ion_temperature', 'electron_density']
        
        # Filter available parameters
        available_params = [p for p in parameters if p in data.columns]
        
        if not available_params:
            raise ValueError("No specified parameters found in data")
        
        if interactive and HAS_PLOTLY:
            return self._plot_interactive_time_series(data, available_params)
        else:
            return self._plot_static_time_series(data, available_params)
    
    def _plot_interactive_time_series(self, data: pd.DataFrame, parameters: List[str]) -> go.Figure:
        """Create interactive time series plot with Plotly."""
        n_params = len(parameters)
        rows = (n_params + 1) // 2
        
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=parameters,
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, param in enumerate(parameters):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[param],
                    name=param,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>{param}</b><br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
                ),
                row=row,
                col=col
            )
        
        fig.update_layout(
            title=dict(
                text="Fusion Plasma Parameters Time Series",
                x=0.5,
                font=dict(size=self.plot_config.title_size)
            ),
            showlegend=False,
            height=300 * rows,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Time")
        
        return fig
    
    def _plot_static_time_series(self, data: pd.DataFrame, parameters: List[str]) -> plt.Figure:
        """Create static time series plot with Matplotlib."""
        n_params = len(parameters)
        rows = (n_params + 1) // 2
        cols = 2
        
        fig, axes = plt.subplots(
            rows, cols, 
            figsize=(self.plot_config.figsize[0], self.plot_config.figsize[1] * rows / 2),
            dpi=self.plot_config.dpi
        )
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(parameters):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            ax.plot(data.index, data[param], linewidth=self.plot_config.line_width, alpha=self.plot_config.alpha)
            ax.set_title(param, fontsize=self.plot_config.title_size)
            ax.set_xlabel('Time', fontsize=self.plot_config.label_size)
            ax.set_ylabel('Value', fontsize=self.plot_config.label_size)
            ax.grid(self.plot_config.grid, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(parameters), rows * cols):
            row = i // 2
            col = i % 2
            fig.delaxes(axes[row, col])
        
        if self.plot_config.tight_layout:
            plt.tight_layout()
        
        return fig
    
    def plot_plasma_cross_section(self, 
                                  temperature: np.ndarray,
                                  density: np.ndarray,
                                  magnetic_field: np.ndarray,
                                  r_grid: np.ndarray,
                                  z_grid: np.ndarray,
                                  interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot 2D cross-section of plasma parameters.
        
        Args:
            temperature: 2D temperature array.
            density: 2D density array.
            magnetic_field: 2D magnetic field array.
            r_grid: Radial coordinate grid.
            z_grid: Vertical coordinate grid.
            interactive: Whether to create interactive plot.
            
        Returns:
            Matplotlib or Plotly figure.
        """
        if interactive and HAS_PLOTLY:
            return self._plot_interactive_cross_section(
                temperature, density, magnetic_field, r_grid, z_grid
            )
        else:
            return self._plot_static_cross_section(
                temperature, density, magnetic_field, r_grid, z_grid
            )
    
    def _plot_interactive_cross_section(self, 
                                        temperature: np.ndarray,
                                        density: np.ndarray,
                                        magnetic_field: np.ndarray,
                                        r_grid: np.ndarray,
                                        z_grid: np.ndarray) -> go.Figure:
        """Create interactive plasma cross-section with Plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Temperature (keV)', 'Density (m⁻³)', 'Magnetic Field (T)', 'Flux Surfaces'],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Temperature plot
        fig.add_trace(
            go.Heatmap(
                z=temperature,
                x=r_grid[0, :],
                y=z_grid[:, 0],
                colorscale='Plasma',
                name='Temperature',
                hovertemplate="R: %{x:.2f}<br>Z: %{y:.2f}<br>T: %{z:.2f} keV<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Density plot
        fig.add_trace(
            go.Heatmap(
                z=density,
                x=r_grid[0, :],
                y=z_grid[:, 0],
                colorscale='Viridis',
                name='Density',
                hovertemplate="R: %{x:.2f}<br>Z: %{y:.2f}<br>n: %{z:.2e} m⁻³<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Magnetic field plot
        fig.add_trace(
            go.Heatmap(
                z=magnetic_field,
                x=r_grid[0, :],
                y=z_grid[:, 0],
                colorscale='Electric',
                name='Magnetic Field',
                hovertemplate="R: %{x:.2f}<br>Z: %{y:.2f}<br>B: %{z:.2f} T<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Flux surfaces (contour plot)
        fig.add_trace(
            go.Contour(
                z=magnetic_field,
                x=r_grid[0, :],
                y=z_grid[:, 0],
                colorscale='Blues',
                name='Flux Surfaces',
                contours=dict(showlabels=True),
                hovertemplate="R: %{x:.2f}<br>Z: %{y:.2f}<br>Ψ: %{z:.2f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text="Plasma Cross-Section Analysis",
                x=0.5,
                font=dict(size=self.plot_config.title_size)
            ),
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _plot_static_cross_section(self, 
                                   temperature: np.ndarray,
                                   density: np.ndarray,
                                   magnetic_field: np.ndarray,
                                   r_grid: np.ndarray,
                                   z_grid: np.ndarray) -> plt.Figure:
        """Create static plasma cross-section with Matplotlib."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.plot_config.dpi)
        
        # Temperature plot
        im1 = axes[0, 0].contourf(r_grid, z_grid, temperature, levels=20, cmap=self.fusion_colormaps['plasma_temp'])
        axes[0, 0].set_title('Temperature (keV)', fontsize=self.plot_config.title_size)
        axes[0, 0].set_xlabel('R (m)', fontsize=self.plot_config.label_size)
        axes[0, 0].set_ylabel('Z (m)', fontsize=self.plot_config.label_size)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Density plot
        im2 = axes[0, 1].contourf(r_grid, z_grid, density, levels=20, cmap='viridis')
        axes[0, 1].set_title('Density (m⁻³)', fontsize=self.plot_config.title_size)
        axes[0, 1].set_xlabel('R (m)', fontsize=self.plot_config.label_size)
        axes[0, 1].set_ylabel('Z (m)', fontsize=self.plot_config.label_size)
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Magnetic field plot
        im3 = axes[1, 0].contourf(r_grid, z_grid, magnetic_field, levels=20, cmap=self.fusion_colormaps['magnetic'])
        axes[1, 0].set_title('Magnetic Field (T)', fontsize=self.plot_config.title_size)
        axes[1, 0].set_xlabel('R (m)', fontsize=self.plot_config.label_size)
        axes[1, 0].set_ylabel('Z (m)', fontsize=self.plot_config.label_size)
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Flux surfaces (contour lines)
        cs = axes[1, 1].contour(r_grid, z_grid, magnetic_field, levels=10, colors='blue', linewidths=2)
        axes[1, 1].clabel(cs, inline=True, fontsize=10)
        axes[1, 1].set_title('Flux Surfaces', fontsize=self.plot_config.title_size)
        axes[1, 1].set_xlabel('R (m)', fontsize=self.plot_config.label_size)
        axes[1, 1].set_ylabel('Z (m)', fontsize=self.plot_config.label_size)
        axes[1, 1].set_aspect('equal')
        
        if self.plot_config.tight_layout:
            plt.tight_layout()
        
        return fig
    
    def plot_3d_plasma_volume(self, 
                              temperature: np.ndarray,
                              r_grid: np.ndarray,
                              theta_grid: np.ndarray,
                              z_grid: np.ndarray,
                              use_mayavi: bool = False) -> Union[plt.Figure, Any]:
        """
        Create 3D visualization of plasma volume.
        
        Args:
            temperature: 3D temperature array.
            r_grid: Radial coordinate grid.
            theta_grid: Toroidal coordinate grid.
            z_grid: Vertical coordinate grid.
            use_mayavi: Whether to use Mayavi for 3D rendering.
            
        Returns:
            3D visualization figure.
        """
        if use_mayavi and HAS_MAYAVI:
            return self._plot_mayavi_3d(temperature, r_grid, theta_grid, z_grid)
        else:
            return self._plot_matplotlib_3d(temperature, r_grid, theta_grid, z_grid)
    
    def _plot_matplotlib_3d(self, 
                            temperature: np.ndarray,
                            r_grid: np.ndarray,
                            theta_grid: np.ndarray,
                            z_grid: np.ndarray) -> plt.Figure:
        """Create 3D plot with Matplotlib."""
        if not HAS_3D:
            raise ImportError("3D plotting requires mpl_toolkits.mplot3d")
        
        fig = plt.figure(figsize=(12, 10), dpi=self.plot_config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert to Cartesian coordinates
        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)
        Z = z_grid
        
        # Create isosurface
        threshold = np.percentile(temperature, 75)  # Show hottest 25%
        
        # Use volume rendering approach with scatter plot
        mask = temperature > threshold
        x_hot = X[mask]
        y_hot = Y[mask]
        z_hot = Z[mask]
        temp_hot = temperature[mask]
        
        scatter = ax.scatter(x_hot, y_hot, z_hot, c=temp_hot, 
                           cmap=self.fusion_colormaps['plasma_temp'],
                           s=50, alpha=0.6)
        
        ax.set_xlabel('X (m)', fontsize=self.plot_config.label_size)
        ax.set_ylabel('Y (m)', fontsize=self.plot_config.label_size)
        ax.set_zlabel('Z (m)', fontsize=self.plot_config.label_size)
        ax.set_title('3D Plasma Temperature Distribution', fontsize=self.plot_config.title_size)
        
        plt.colorbar(scatter, ax=ax, shrink=0.8, label='Temperature (keV)')
        
        return fig
    
    def _plot_mayavi_3d(self, 
                        temperature: np.ndarray,
                        r_grid: np.ndarray,
                        theta_grid: np.ndarray,
                        z_grid: np.ndarray):
        """Create advanced 3D visualization with Mayavi."""
        # Convert to Cartesian coordinates
        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)
        Z = z_grid
        
        # Clear previous scene
        mlab.clf()
        
        # Create volume rendering
        src = mlab.pipeline.scalar_field(X, Y, Z, temperature)
        
        # Add volume visualization
        vol = mlab.pipeline.volume(src, vmin=temperature.min(), vmax=temperature.max())
        
        # Add isosurface
        threshold = np.percentile(temperature, 80)
        iso = mlab.pipeline.iso_surface(src, contours=[threshold], opacity=0.3)
        
        # Set colormap
        vol.module_manager.scalar_lut_manager.lut.table = self._mayavi_colormap('plasma')
        iso.module_manager.scalar_lut_manager.lut.table = self._mayavi_colormap('plasma')
        
        # Add colorbar
        mlab.colorbar(title='Temperature (keV)', orientation='vertical')
        
        # Set scene properties
        mlab.title('3D Plasma Temperature Volume')
        mlab.axes(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        
        return mlab.gcf()
    
    def _mayavi_colormap(self, name: str) -> np.ndarray:
        """Convert matplotlib colormap to Mayavi format."""
        if name == 'plasma':
            # Create plasma-like colormap
            colors = np.array([
                [0, 0, 128, 255],      # Dark blue
                [0, 102, 204, 255],    # Blue
                [0, 204, 255, 255],    # Light blue
                [102, 255, 102, 255],  # Green
                [255, 255, 0, 255],    # Yellow
                [255, 102, 0, 255],    # Orange
                [255, 0, 0, 255]       # Red
            ])
        else:
            # Default to viridis-like
            colors = np.array([
                [68, 1, 84, 255],      # Purple
                [59, 82, 139, 255],    # Blue
                [33, 145, 140, 255],   # Teal
                [94, 201, 98, 255],    # Green
                [253, 231, 37, 255]    # Yellow
            ])
        
        # Interpolate to 256 colors
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(colors))
        x_new = np.linspace(0, 1, 256)
        
        interp_colors = np.zeros((256, 4))
        for i in range(4):
            f = interp1d(x_old, colors[:, i], kind='linear')
            interp_colors[:, i] = f(x_new)
        
        return interp_colors.astype(np.uint8)
    
    def plot_parameter_correlation_matrix(self, 
                                          data: pd.DataFrame,
                                          method: str = 'pearson',
                                          interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot correlation matrix of fusion parameters.
        
        Args:
            data: DataFrame with fusion parameters.
            method: Correlation method ('pearson', 'spearman', 'kendall').
            interactive: Whether to create interactive plot.
            
        Returns:
            Correlation matrix visualization.
        """
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method)
        
        if interactive and HAS_PLOTLY:
            return self._plot_interactive_correlation(corr_matrix)
        else:
            return self._plot_static_correlation(corr_matrix)
    
    def _plot_interactive_correlation(self, corr_matrix: pd.DataFrame) -> go.Figure:
        """Create interactive correlation matrix with Plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(
                text="Parameter Correlation Matrix",
                x=0.5,
                font=dict(size=self.plot_config.title_size)
            ),
            xaxis_title="Parameters",
            yaxis_title="Parameters",
            width=800,
            height=800
        )
        
        return fig
    
    def _plot_static_correlation(self, corr_matrix: pd.DataFrame) -> plt.Figure:
        """Create static correlation matrix with Matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.plot_config.dpi)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Parameter Correlation Matrix', fontsize=self.plot_config.title_size)
        
        if self.plot_config.tight_layout:
            plt.tight_layout()
        
        return fig
    
    def create_parameter_dashboard(self, data: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive parameter monitoring dashboard.
        
        Args:
            data: DataFrame with fusion parameters and timestamps.
            
        Returns:
            Interactive dashboard figure.
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for dashboard creation")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Plasma Current', 'Q Factor', 'Beta',
                'Ion Temperature', 'Electron Density', 'Magnetic Field',
                'Fusion Power', 'Confinement Time', 'Energy Content'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        parameters = [
            'plasma_current', 'q_factor', 'beta',
            'ion_temperature', 'electron_density', 'magnetic_field',
            'fusion_power', 'confinement_time', 'energy_content'
        ]
        
        colors = px.colors.qualitative.Set1
        
        for i, param in enumerate(parameters):
            if param not in data.columns:
                continue
                
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            # Add time series
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[param],
                    name=param,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>{param}</b><br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
                ),
                row=row,
                col=col
            )
            
            # Add mean line
            mean_val = data[param].mean()
            fig.add_hline(
                y=mean_val,
                line_dash="dash",
                line_color="gray",
                row=row,
                col=col,
                annotation_text=f"Mean: {mean_val:.2f}"
            )
        
        fig.update_layout(
            title=dict(
                text="Nuclear Fusion Parameter Dashboard",
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=False,
            height=900,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Time")
        
        return fig
    
    def save_plot(self, fig: Union[plt.Figure, go.Figure], 
                  filename: str, 
                  format: str = 'png',
                  **kwargs):
        """
        Save plot to file.
        
        Args:
            fig: Figure to save.
            filename: Output filename.
            format: Output format.
            **kwargs: Additional arguments for saving.
        """
        if isinstance(fig, plt.Figure):
            fig.savefig(filename, format=format, dpi=self.plot_config.dpi, 
                       bbox_inches='tight', **kwargs)
        elif HAS_PLOTLY and isinstance(fig, go.Figure):
            if format.lower() == 'html':
                fig.write_html(filename, **kwargs)
            else:
                fig.write_image(filename, format=format, **kwargs)
        else:
            raise ValueError(f"Unsupported figure type: {type(fig)}")
        
        logger.info(f"Plot saved to {filename}")


def create_visualization_suite(config_path: Optional[str] = None) -> FusionVisualizationSuite:
    """
    Create visualization suite with configuration.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configured FusionVisualizationSuite.
    """
    config = {}
    if config_path:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config().get('visualization', {})
    
    return FusionVisualizationSuite(config)