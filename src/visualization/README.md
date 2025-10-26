# Advanced 3D Visualization Suite

This directory contains state-of-the-art 3D visualization capabilities for nuclear fusion plasma analysis.

## Components

### 1. 3D Plasma Visualization Engine (`plasma_3d.py`)

**PlasmaGeometry**
- Creates realistic 3D tokamak, stellarator, and spherical tokamak geometries
- Generates nested magnetic flux surfaces with proper shaping parameters
- Supports ITER-like configurations with elongation and triangularity

**VolumetricRenderer** 
- Advanced volumetric rendering for 3D plasma parameters
- High-quality opacity mapping and scientific color schemes
- Support for multiple backends (Plotly, PyVista, Mayavi)
- Magnetic field line visualization with streamline integration

**ParticleTracker**
- Charged particle trajectory visualization in 3D magnetic fields
- Orbit classification (trapped, passing, lost particles)
- Lorentz force integration with realistic physics
- Multi-species particle tracking (ions, electrons, alpha particles)

**ImmersiveVisualization**
- VR-ready plasma visualization scenes using A-Frame
- Stereo rendering for 3D viewing experiences
- Interactive VR controls and information panels
- WebXR compatibility for modern VR headsets

### 2. Real-time Interactive Dashboard (`realtime_dashboard.py`)

**RealTimeDataStream**
- Simulates live fusion reactor data feeds
- Realistic parameter variations with trends, noise, and disruptions
- Configurable update intervals and data buffering
- Physics-based constraints and operational event simulation

**InteractiveDashboard**
- Comprehensive Streamlit-based monitoring interface
- Real-time parameter visualization with multiple plot types
- ML prediction integration with confidence metrics
- Status indicators with operational regime classification

**Advanced Features**
- Multi-dimensional parameter correlation analysis
- 3D plasma state visualization integrated with live data
- Configurable refresh rates and display options
- Performance monitoring and system diagnostics

### 3. Dashboard Launcher (`launch_dashboard.py`)

**Automated Setup**
- Dependency checking and validation
- Streamlit configuration with custom themes
- Error handling and graceful shutdown
- Command-line interface with configuration options

## Usage Examples

### Basic 3D Visualization

```python
from src.visualization.plasma_3d import (
    PlasmaGeometry, VolumetricRenderer, PlasmaState
)

# Create tokamak geometry
geometry = PlasmaGeometry(reactor_type='tokamak')
X, Y, Z = geometry.create_tokamak_mesh()

# Generate plasma data
temperature = 10.0 * np.exp(-((X - 6.2)**2 + Y**2)/4) * np.exp(-Z**2/2)
density = 1e20 * np.exp(-((X - 6.2)**2 + Y**2)/3) * np.exp(-Z**2/1.5)

# Create plasma state
plasma_state = PlasmaState(
    temperature=temperature,
    density=density,
    coordinates=(X, Y, Z),
    # ... other parameters
)

# Render volumetric visualization
renderer = VolumetricRenderer()
fig = renderer.render_plasma_volume_plotly(plasma_state, quantity='temperature')
fig.show()
```

### Particle Trajectory Analysis

```python
from src.visualization.plasma_3d import ParticleTracker

# Initialize particle tracker
tracker = ParticleTracker()

# Add test particles
tracker.add_particle(
    position=[7.0, 0.0, 0.0],
    velocity=[1e5, 2e5, 0.0],
    charge=1.0,
    mass=tracker.m_p,
    species='ion'
)

# Visualize trajectories
fig = tracker.visualize_trajectories(plasma_state)
fig.show()
```

### Real-time Dashboard

```python
# Launch dashboard
python scripts/launch_dashboard.py --port 8501 --debug

# Or programmatically
from src.visualization.realtime_dashboard import InteractiveDashboard

dashboard = InteractiveDashboard()
dashboard.run_dashboard()
```

### VR Visualization

```python
from src.visualization.plasma_3d import ImmersiveVisualization

# Create VR scene
vr_viz = ImmersiveVisualization()
vr_scene_path = vr_viz.create_vr_plasma_scene(plasma_state, "vr_scene.html")

# View in VR-capable browser
```

## Advanced Features

### Multi-Backend Rendering

The visualization engine supports multiple rendering backends:

- **Plotly**: Web-based interactive visualization with excellent browser compatibility
- **PyVista**: High-performance scientific visualization with advanced rendering capabilities  
- **Mayavi**: Scientific data visualization with 3D scene management
- **VTK**: Low-level visualization toolkit for custom rendering pipelines

### Scientific Color Maps

Custom color maps optimized for fusion plasma visualization:

- `plasma_temperature()`: Optimized for temperature gradients
- `magnetic_field()`: Diverging colormap for field visualization
- `density_profile()`: Perceptually uniform for density distributions
- `fusion_power()`: High-contrast for power deposition patterns

### Real-time Performance

Optimized for real-time visualization with:

- Efficient data buffering and streaming
- Adaptive LOD (Level of Detail) rendering
- GPU-accelerated volume rendering where available
- Configurable refresh rates and update intervals

### Physics Integration

Accurate physics modeling including:

- Realistic tokamak geometry with shaping parameters
- Magnetic field line integration using field-following algorithms
- Particle orbit classification (banana, potato, stochastic)
- MHD equilibrium-consistent field structures

## Configuration

### Dashboard Settings

```python
dashboard_config = {
    'auto_refresh': True,
    'refresh_interval': 1.0,  # seconds
    'max_data_points': 1000,
    'default_parameters': [
        'plasma_current',
        'electron_temperature',
        'electron_density'
    ]
}
```

### Visualization Settings

```python
viz_config = {
    'backend': 'plotly',  # 'plotly', 'pyvista', 'mayavi'
    'opacity_scale': 0.1,
    'color_scale': 'plasma',
    'lighting_enabled': True,
    'vr_enabled': False
}
```

### Streaming Settings

```python
stream_config = {
    'update_interval': 1.0,  # seconds
    'buffer_size': 1000,     # max samples
    'noise_level': 0.05,     # relative noise
    'disruption_probability': 0.01  # per update
}
```

## Performance Optimization

### Memory Management

- Automatic data buffer management with configurable limits
- Efficient numpy array operations for large datasets
- Garbage collection optimization for long-running sessions

### Rendering Performance

- Adaptive mesh resolution based on viewing distance
- LOD rendering for complex geometries
- GPU acceleration where available
- Efficient update mechanisms for real-time data

### Network Optimization

- Compressed data transmission for remote dashboards
- Delta encoding for incremental updates  
- Efficient WebSocket communication for real-time streaming

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
python scripts/launch_dashboard.py

# Access at http://localhost:8501
```

### Production Deployment

```bash
# Using Docker
docker build -t fusion-dashboard .
docker run -p 8501:8501 fusion-dashboard

# Using systemd service
sudo systemctl start fusion-dashboard
```

### Cloud Deployment

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fusion-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fusion-dashboard
  template:
    metadata:
      labels:
        app: fusion-dashboard
    spec:
      containers:
      - name: dashboard
        image: fusion-dashboard:latest
        ports:
        - containerPort: 8501
```

## Troubleshooting

### Common Issues

**VTK/Mayavi Installation**
```bash
# On macOS
brew install vtk
pip install mayavi

# On Ubuntu
sudo apt-get install vtk9-dev
pip install mayavi
```

**WebGL Issues**
- Enable hardware acceleration in browser
- Update graphics drivers
- Use compatible WebGL context

**Performance Issues**
- Reduce mesh resolution for large datasets
- Disable real-time updates for complex visualizations
- Use appropriate backend for hardware capabilities

### Debug Mode

```bash
# Launch with debug logging
python scripts/launch_dashboard.py --debug

# Enable Streamlit debug mode
export STREAMLIT_LOGGER_LEVEL=debug
```

This advanced visualization suite provides comprehensive tools for exploring and understanding nuclear fusion plasma physics through state-of-the-art 3D visualization and real-time monitoring capabilities.