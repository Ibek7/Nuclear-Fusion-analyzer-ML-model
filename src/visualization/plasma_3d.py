"""
3D Plasma Visualization Engine for Nuclear Fusion Analyzer.

This module provides state-of-the-art 3D visualization capabilities for plasma physics,
including volumetric rendering, particle tracking, magnetic field line visualization,
and immersive VR-ready outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

try:
    import vtk
    from vtk.util import numpy_support
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    from mayavi import mlab
    HAS_MAYAVI = True
except ImportError:
    HAS_MAYAVI = False

logger = logging.getLogger(__name__)


@dataclass
class PlasmaState:
    """Container for plasma state information."""
    
    temperature: np.ndarray
    density: np.ndarray
    pressure: np.ndarray
    magnetic_field: Tuple[np.ndarray, np.ndarray, np.ndarray]
    velocity: Tuple[np.ndarray, np.ndarray, np.ndarray]
    coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]
    time_stamp: float
    metadata: Dict[str, Any]


class PlasmaGeometry:
    """
    Plasma geometry and mesh generation for tokamak configurations.
    
    Creates realistic 3D geometries for different fusion reactor designs
    including ITER-like tokamaks, stellarators, and spherical tokamaks.
    """
    
    def __init__(self, reactor_type: str = 'tokamak'):
        """
        Initialize plasma geometry.
        
        Args:
            reactor_type: Type of reactor ('tokamak', 'stellarator', 'spherical_tokamak').
        """
        self.reactor_type = reactor_type
        self.R0 = 6.2  # Major radius (m) - ITER-like
        self.a = 2.0   # Minor radius (m)
        self.kappa = 1.7  # Elongation
        self.delta = 0.33  # Triangularity
        
        logger.info(f"PlasmaGeometry initialized for {reactor_type}")
    
    def create_tokamak_mesh(self, 
                           nr: int = 50, 
                           ntheta: int = 100, 
                           nzeta: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3D mesh for tokamak geometry.
        
        Args:
            nr: Number of radial points.
            ntheta: Number of poloidal points.
            nzeta: Number of toroidal points.
            
        Returns:
            Tuple of (R, theta, zeta) coordinate arrays.
        """
        # Radial coordinate (normalized)
        rho = np.linspace(0, 1, nr)
        
        # Poloidal angle
        theta = np.linspace(0, 2*np.pi, ntheta)
        
        # Toroidal angle
        zeta = np.linspace(0, 2*np.pi, nzeta)
        
        # Create 3D mesh
        rho_3d, theta_3d, zeta_3d = np.meshgrid(rho, theta, zeta, indexing='ij')
        
        # Calculate physical coordinates
        r_minor = rho_3d * self.a
        
        # Apply shaping (elongation and triangularity)
        Z = r_minor * np.sin(theta_3d + self.delta * np.sin(theta_3d)) * self.kappa
        R = self.R0 + r_minor * np.cos(theta_3d + self.delta * np.sin(theta_3d))
        
        # Convert to Cartesian coordinates
        X = R * np.cos(zeta_3d)
        Y = R * np.sin(zeta_3d)
        
        return X, Y, Z
    
    def create_magnetic_surfaces(self, n_surfaces: int = 10) -> List[np.ndarray]:
        """
        Create nested magnetic flux surfaces.
        
        Args:
            n_surfaces: Number of flux surfaces.
            
        Returns:
            List of surface coordinate arrays.
        """
        surfaces = []
        
        for i in range(n_surfaces):
            rho = (i + 1) / n_surfaces
            
            # Higher resolution for surface visualization
            theta = np.linspace(0, 2*np.pi, 200)
            zeta = np.linspace(0, 2*np.pi, 100)
            
            theta_2d, zeta_2d = np.meshgrid(theta, zeta)
            
            # Surface coordinates
            r_minor = rho * self.a
            Z = r_minor * np.sin(theta_2d + self.delta * np.sin(theta_2d)) * self.kappa
            R = self.R0 + r_minor * np.cos(theta_2d + self.delta * np.sin(theta_2d))
            
            X = R * np.cos(zeta_2d)
            Y = R * np.sin(zeta_2d)
            
            surfaces.append((X, Y, Z))
        
        return surfaces


class VolumetricRenderer:
    """
    Advanced volumetric rendering for 3D plasma visualization.
    
    Provides high-quality volume rendering with opacity mapping,
    lighting effects, and scientific color maps.
    """
    
    def __init__(self, backend: str = 'plotly'):
        """
        Initialize volumetric renderer.
        
        Args:
            backend: Rendering backend ('plotly', 'pyvista', 'mayavi').
        """
        self.backend = backend
        self._validate_backend()
        
        # Rendering parameters
        self.opacity_scale = 0.1
        self.color_scale = 'plasma'
        self.lighting_enabled = True
        
        logger.info(f"VolumetricRenderer initialized with {backend}")
    
    def _validate_backend(self):
        """Validate rendering backend availability."""
        if self.backend == 'pyvista' and not HAS_PYVISTA:
            logger.warning("PyVista not available, falling back to plotly")
            self.backend = 'plotly'
        elif self.backend == 'mayavi' and not HAS_MAYAVI:
            logger.warning("Mayavi not available, falling back to plotly")
            self.backend = 'plotly'
    
    def render_plasma_volume_plotly(self, 
                                   plasma_state: PlasmaState,
                                   quantity: str = 'temperature',
                                   title: str = "Plasma Volume Rendering") -> go.Figure:
        """
        Render plasma volume using Plotly.
        
        Args:
            plasma_state: Plasma state data.
            quantity: Quantity to visualize.
            title: Plot title.
            
        Returns:
            Plotly figure object.
        """
        # Get coordinates and data
        X, Y, Z = plasma_state.coordinates
        
        if quantity == 'temperature':
            values = plasma_state.temperature
            colorbar_title = "Temperature (keV)"
        elif quantity == 'density':
            values = plasma_state.density
            colorbar_title = "Density (10¹⁹ m⁻³)"
        elif quantity == 'pressure':
            values = plasma_state.pressure
            colorbar_title = "Pressure (Pa)"
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
        
        # Create volume plot
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(), 
            z=Z.flatten(),
            value=values.flatten(),
            isomin=np.percentile(values, 10),
            isomax=np.percentile(values, 90),
            opacity=self.opacity_scale,
            surface_count=20,
            colorscale=self.color_scale,
            colorbar=dict(title=colorbar_title, thickness=20),
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        # Add magnetic flux surfaces
        geometry = PlasmaGeometry()
        surfaces = geometry.create_magnetic_surfaces(n_surfaces=5)
        
        for i, (surf_X, surf_Y, surf_Z) in enumerate(surfaces):
            # Subsample for visualization
            step = 5
            fig.add_trace(go.Surface(
                x=surf_X[::step, ::step],
                y=surf_Y[::step, ::step],
                z=surf_Z[::step, ::step],
                opacity=0.1,
                colorscale='Greys',
                showscale=False,
                name=f'Flux Surface {i+1}'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def render_magnetic_field_lines(self, 
                                   plasma_state: PlasmaState,
                                   n_lines: int = 20,
                                   title: str = "Magnetic Field Lines") -> go.Figure:
        """
        Render magnetic field lines using streamlines.
        
        Args:
            plasma_state: Plasma state data.
            n_lines: Number of field lines.
            title: Plot title.
            
        Returns:
            Plotly figure with field lines.
        """
        X, Y, Z = plasma_state.coordinates
        Bx, By, Bz = plasma_state.magnetic_field
        
        # Create streamlines (simplified approach)
        fig = go.Figure()
        
        # Select seed points for field lines
        geometry = PlasmaGeometry()
        seed_points = []
        
        for i in range(n_lines):
            theta = 2 * np.pi * i / n_lines
            r = 0.5 * geometry.a
            R = geometry.R0 + r * np.cos(theta)
            x_seed = R * np.cos(0)
            y_seed = R * np.sin(0)
            z_seed = r * np.sin(theta)
            seed_points.append([x_seed, y_seed, z_seed])
        
        # Integrate field lines (simplified)
        for seed in seed_points:
            # Simple field line integration
            line_points = [seed]
            current_point = np.array(seed)
            
            for step in range(100):
                # Find nearest grid point
                distances = np.sqrt((X.flatten() - current_point[0])**2 + 
                                  (Y.flatten() - current_point[1])**2 + 
                                  (Z.flatten() - current_point[2])**2)
                nearest_idx = np.argmin(distances)
                
                # Get field at nearest point
                b_field = np.array([Bx.flatten()[nearest_idx], 
                                   By.flatten()[nearest_idx], 
                                   Bz.flatten()[nearest_idx]])
                
                # Normalize and step
                if np.linalg.norm(b_field) > 0:
                    b_field = b_field / np.linalg.norm(b_field)
                    current_point = current_point + 0.1 * b_field
                    line_points.append(current_point.copy())
                else:
                    break
            
            # Convert to array
            line_array = np.array(line_points)
            
            # Add field line to plot
            fig.add_trace(go.Scatter3d(
                x=line_array[:, 0],
                y=line_array[:, 1],
                z=line_array[:, 2],
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=2, y=2, z=1)
                ),
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        return fig


class ParticleTracker:
    """
    Particle trajectory visualization for fusion plasma.
    
    Tracks and visualizes charged particle motion in 3D magnetic fields,
    including orbit types, confinement analysis, and loss regions.
    """
    
    def __init__(self):
        """Initialize particle tracker."""
        self.particles = []
        self.trajectories = []
        
        # Physical constants
        self.q_e = 1.602e-19  # Elementary charge
        self.m_p = 1.673e-27  # Proton mass
        self.m_e = 9.109e-31  # Electron mass
        
        logger.info("ParticleTracker initialized")
    
    def add_particle(self, 
                    position: np.ndarray,
                    velocity: np.ndarray,
                    charge: float,
                    mass: float,
                    species: str = 'ion'):
        """
        Add particle to tracking system.
        
        Args:
            position: Initial position [x, y, z].
            velocity: Initial velocity [vx, vy, vz].
            charge: Particle charge (in units of e).
            mass: Particle mass (kg).
            species: Particle species label.
        """
        particle = {
            'position': np.array(position),
            'velocity': np.array(velocity),
            'charge': charge,
            'mass': mass,
            'species': species,
            'trajectory': [np.array(position)]
        }
        
        self.particles.append(particle)
        logger.info(f"Added {species} particle at {position}")
    
    def integrate_orbit(self, 
                       particle: Dict,
                       plasma_state: PlasmaState,
                       dt: float = 1e-8,
                       n_steps: int = 10000) -> np.ndarray:
        """
        Integrate particle orbit in magnetic field.
        
        Args:
            particle: Particle dictionary.
            plasma_state: Plasma state with magnetic field.
            dt: Time step (s).
            n_steps: Number of integration steps.
            
        Returns:
            Trajectory array.
        """
        trajectory = []
        pos = particle['position'].copy()
        vel = particle['velocity'].copy()
        
        X, Y, Z = plasma_state.coordinates
        Bx, By, Bz = plasma_state.magnetic_field
        
        for step in range(n_steps):
            # Interpolate magnetic field at particle position
            # (Simplified - would use proper interpolation in practice)
            distances = np.sqrt((X.flatten() - pos[0])**2 + 
                              (Y.flatten() - pos[1])**2 + 
                              (Z.flatten() - pos[2])**2)
            nearest_idx = np.argmin(distances)
            
            B_field = np.array([Bx.flatten()[nearest_idx],
                               By.flatten()[nearest_idx],
                               Bz.flatten()[nearest_idx]])
            
            # Lorentz force: F = q(v × B)
            force = particle['charge'] * self.q_e * np.cross(vel, B_field)
            
            # Acceleration: a = F/m
            accel = force / particle['mass']
            
            # Update velocity and position (leapfrog integration)
            vel = vel + accel * dt
            pos = pos + vel * dt
            
            trajectory.append(pos.copy())
            
            # Check if particle is lost (simple boundary check)
            r = np.sqrt(pos[0]**2 + pos[1]**2)
            if r > 8.0 or abs(pos[2]) > 3.0:  # Simple loss criterion
                break
        
        return np.array(trajectory)
    
    def visualize_trajectories(self, 
                              plasma_state: PlasmaState,
                              title: str = "Particle Trajectories") -> go.Figure:
        """
        Visualize all particle trajectories.
        
        Args:
            plasma_state: Plasma state data.
            title: Plot title.
            
        Returns:
            Plotly figure with trajectories.
        """
        fig = go.Figure()
        
        # Color map for different species
        colors = {'ion': 'red', 'electron': 'blue', 'alpha': 'green'}
        
        for i, particle in enumerate(self.particles):
            # Integrate orbit
            trajectory = self.integrate_orbit(particle, plasma_state)
            
            # Add trajectory to plot
            color = colors.get(particle['species'], 'black')
            
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines',
                line=dict(color=color, width=4),
                name=f"{particle['species']} {i+1}"
            ))
            
            # Add starting point
            fig.add_trace(go.Scatter3d(
                x=[trajectory[0, 0]],
                y=[trajectory[0, 1]],
                z=[trajectory[0, 2]],
                mode='markers',
                marker=dict(color=color, size=8, symbol='circle'),
                showlegend=False
            ))
        
        # Add plasma boundary
        geometry = PlasmaGeometry()
        X, Y, Z = geometry.create_tokamak_mesh(nr=2, ntheta=50, nzeta=50)
        
        # Outer boundary
        fig.add_trace(go.Surface(
            x=X[-1, :, :],
            y=Y[-1, :, :],
            z=Z[-1, :, :],
            opacity=0.1,
            colorscale='Greys',
            showscale=False,
            name='Plasma Boundary'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=2, y=2, z=1.5)
                ),
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        return fig


class ImmersiveVisualization:
    """
    Immersive and VR-ready visualization components.
    
    Creates VR-compatible outputs and immersive visualization experiences
    for fusion plasma data exploration.
    """
    
    def __init__(self):
        """Initialize immersive visualization."""
        self.vr_enabled = False
        self.stereo_enabled = False
        
        logger.info("ImmersiveVisualization initialized")
    
    def create_vr_plasma_scene(self, 
                              plasma_state: PlasmaState,
                              output_path: str = "./vr_scene.html") -> str:
        """
        Create VR-ready plasma visualization scene.
        
        Args:
            plasma_state: Plasma state data.
            output_path: Output path for VR scene.
            
        Returns:
            Path to created VR scene.
        """
        # Create VR-compatible HTML with A-Frame
        vr_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VR Plasma Visualization</title>
            <script src="https://aframe.io/releases/1.2.0/aframe.min.js"></script>
            <script src="https://cdn.jsdelivr.net/gh/donmccurdy/aframe-extras@v6.1.1/dist/aframe-extras.min.js"></script>
        </head>
        <body>
            <a-scene embedded style="height: 600px; width: 100%;" 
                     background="color: #000016"
                     vr-mode-ui="enabled: true">
                
                <!-- VR Controls -->
                <a-entity id="cameraRig" position="0 1.6 3">
                    <a-camera look-controls wasd-controls></a-camera>
                    <a-entity laser-controls="hand: right" raycaster="objects: .clickable"></a-entity>
                    <a-entity laser-controls="hand: left" raycaster="objects: .clickable"></a-entity>
                </a-entity>
                
                <!-- Lighting -->
                <a-light type="ambient" color="#404040"></a-light>
                <a-light type="directional" position="1 1 1" color="#ffffff"></a-light>
                
                <!-- Plasma Volume (simplified representation) -->
                <a-torus position="0 0 0" 
                         color="#ff6b35" 
                         radius="3" 
                         radius-tubular="1.5"
                         opacity="0.3"
                         material="transparent: true; shader: standard;">
                    <a-animation attribute="rotation" 
                                 to="0 360 0" 
                                 dur="20000" 
                                 repeat="indefinite">
                    </a-animation>
                </a-torus>
                
                <!-- Magnetic Field Lines -->
                <a-entity id="fieldLines">
                    <!-- Field lines would be generated dynamically -->
                </a-entity>
                
                <!-- Information Panel -->
                <a-plane position="2 2 -1" 
                         width="1.5" 
                         height="1" 
                         color="#1a1a1a" 
                         text="value: Plasma Temperature: {np.mean(plasma_state.temperature):.1f} keV\\nDensity: {np.mean(plasma_state.density):.2e} m^-3\\nTime: {plasma_state.time_stamp:.3f} s; 
                               color: white; 
                               align: center; 
                               wrapCount: 30">
                </a-plane>
                
                <!-- Interactive Controls -->
                <a-box position="-2 1 -1" 
                       width="0.3" 
                       height="0.3" 
                       depth="0.3" 
                       color="#4a90e2" 
                       class="clickable"
                       text="value: RESET; color: white; align: center">
                </a-box>
                
                <!-- Floor Grid -->
                <a-plane position="0 -2 0" 
                         rotation="-90 0 0" 
                         width="10" 
                         height="10" 
                         color="#1a1a1a" 
                         material="wireframe: true; transparent: true; opacity: 0.2">
                </a-plane>
                
            </a-scene>
            
            <script>
                // VR interaction scripts
                AFRAME.registerComponent('clickable', {{
                    init: function() {{
                        this.el.addEventListener('click', function(evt) {{
                            console.log('Clicked:', evt.target);
                            // Add interaction logic here
                        }});
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(vr_html)
        
        logger.info(f"VR scene created: {output_path}")
        return output_path
    
    def create_stereo_view(self, 
                          fig: go.Figure,
                          eye_separation: float = 0.1) -> Tuple[go.Figure, go.Figure]:
        """
        Create stereo pair for 3D viewing.
        
        Args:
            fig: Original Plotly figure.
            eye_separation: Distance between stereo cameras.
            
        Returns:
            Tuple of (left_eye, right_eye) figures.
        """
        # Create left eye view
        left_fig = go.Figure(fig)
        left_camera = left_fig.layout.scene.camera
        left_camera.eye.x = left_camera.eye.x - eye_separation
        left_fig.update_layout(title=fig.layout.title.text + " (Left Eye)")
        
        # Create right eye view
        right_fig = go.Figure(fig)
        right_camera = right_fig.layout.scene.camera
        right_camera.eye.x = right_camera.eye.x + eye_separation
        right_fig.update_layout(title=fig.layout.title.text + " (Right Eye)")
        
        return left_fig, right_fig


def create_advanced_plasma_visualization_suite(data: pd.DataFrame,
                                             output_dir: str = "./advanced_visualizations") -> Dict[str, str]:
    """
    Create comprehensive advanced visualization suite.
    
    Args:
        data: Input fusion data.
        output_dir: Output directory.
        
    Returns:
        Dictionary of created visualization files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    created_files = {}
    
    try:
        # Create plasma geometry
        geometry = PlasmaGeometry()
        X, Y, Z = geometry.create_tokamak_mesh()
        
        # Generate synthetic 3D plasma data
        temperature = 10.0 * np.exp(-((X - 6.2)**2 + Y**2)/4) * np.exp(-Z**2/2)
        density = 1e20 * np.exp(-((X - 6.2)**2 + Y**2)/3) * np.exp(-Z**2/1.5)
        pressure = temperature * density * 1.602e-16  # Convert to Pascals
        
        # Simple magnetic field
        Bx = 0.1 * np.ones_like(X)
        By = 0.1 * np.ones_like(Y)
        Bz = 5.0 * 6.2 / np.sqrt(X**2 + Y**2)  # Toroidal field
        
        # Create plasma state
        plasma_state = PlasmaState(
            temperature=temperature,
            density=density,
            pressure=pressure,
            magnetic_field=(Bx, By, Bz),
            velocity=(np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)),
            coordinates=(X, Y, Z),
            time_stamp=0.0,
            metadata={'reactor_type': 'tokamak'}
        )
        
        # Initialize visualization components
        volumetric_renderer = VolumetricRenderer()
        particle_tracker = ParticleTracker()
        immersive_viz = ImmersiveVisualization()
        
        # Create volumetric renderings
        temp_fig = volumetric_renderer.render_plasma_volume_plotly(
            plasma_state, quantity='temperature', title="3D Plasma Temperature"
        )
        temp_path = output_path / "plasma_temperature_3d.html"
        temp_fig.write_html(str(temp_path))
        created_files['plasma_temperature_3d'] = str(temp_path)
        
        density_fig = volumetric_renderer.render_plasma_volume_plotly(
            plasma_state, quantity='density', title="3D Plasma Density"
        )
        density_path = output_path / "plasma_density_3d.html"
        density_fig.write_html(str(density_path))
        created_files['plasma_density_3d'] = str(density_path)
        
        # Create magnetic field visualization
        field_fig = volumetric_renderer.render_magnetic_field_lines(
            plasma_state, title="Magnetic Field Lines"
        )
        field_path = output_path / "magnetic_field_lines.html"
        field_fig.write_html(str(field_path))
        created_files['magnetic_field_lines'] = str(field_path)
        
        # Add particles and create trajectory visualization
        # Add some test particles
        particle_tracker.add_particle(
            position=[7.0, 0.0, 0.0],
            velocity=[1e5, 2e5, 0.0],
            charge=1.0,
            mass=particle_tracker.m_p,
            species='ion'
        )
        
        particle_tracker.add_particle(
            position=[6.5, 1.0, 0.5],
            velocity=[-1e6, 1e6, 0.5e6],
            charge=-1.0,
            mass=particle_tracker.m_e,
            species='electron'
        )
        
        trajectory_fig = particle_tracker.visualize_trajectories(
            plasma_state, title="Particle Trajectories"
        )
        trajectory_path = output_path / "particle_trajectories.html"
        trajectory_fig.write_html(str(trajectory_path))
        created_files['particle_trajectories'] = str(trajectory_path)
        
        # Create VR scene
        vr_path = output_path / "vr_plasma_scene.html"
        immersive_viz.create_vr_plasma_scene(plasma_state, str(vr_path))
        created_files['vr_scene'] = str(vr_path)
        
        # Create stereo views
        left_fig, right_fig = immersive_viz.create_stereo_view(temp_fig)
        
        left_path = output_path / "temperature_left_eye.html"
        right_path = output_path / "temperature_right_eye.html"
        
        left_fig.write_html(str(left_path))
        right_fig.write_html(str(right_path))
        
        created_files['stereo_left'] = str(left_path)
        created_files['stereo_right'] = str(right_path)
        
        logger.info(f"Created {len(created_files)} advanced visualizations")
        
    except Exception as e:
        logger.error(f"Error creating advanced visualizations: {e}")
    
    return created_files