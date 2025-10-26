"""
Real-time monitoring components for nuclear fusion analysis.

This module provides real-time monitoring capabilities including
dashboards, data simulation, and performance tracking for fusion
reactor systems.
"""

from .dashboard import (
    FusionDashboard,
    RealTimeDataSimulator,
    DashboardData,
    create_fusion_dashboard
)

__all__ = [
    'FusionDashboard',
    'RealTimeDataSimulator', 
    'DashboardData',
    'create_fusion_dashboard'
]

__version__ = "1.0.0"