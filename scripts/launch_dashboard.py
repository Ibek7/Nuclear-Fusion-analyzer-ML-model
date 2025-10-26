"""
Launch script for the Nuclear Fusion Analyzer Real-time Dashboard.

This script provides a convenient way to start the Streamlit dashboard
with proper configuration and error handling.

Usage:
    python launch_dashboard.py
    python launch_dashboard.py --port 8501 --debug
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True


def launch_dashboard(port: int = 8501, debug: bool = False, host: str = 'localhost'):
    """
    Launch the Streamlit dashboard.
    
    Args:
        port: Port number for the dashboard.
        debug: Enable debug mode.
        host: Host address.
    """
    if not check_dependencies():
        sys.exit(1)
    
    # Get the path to the dashboard script
    dashboard_script = Path(__file__).parent / "realtime_dashboard.py"
    
    if not dashboard_script.exists():
        logger.error(f"Dashboard script not found: {dashboard_script}")
        sys.exit(1)
    
    # Prepare Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_script),
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true" if not debug else "false",
        "--server.runOnSave", "true" if debug else "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    if debug:
        cmd.extend(["--logger.level", "debug"])
    
    logger.info(f"Starting Nuclear Fusion Analyzer Dashboard on {host}:{port}")
    logger.info("Press Ctrl+C to stop the dashboard")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['STREAMLIT_THEME_BASE'] = 'light'
        env['STREAMLIT_THEME_PRIMARY_COLOR'] = '#2E86AB'
        env['STREAMLIT_THEME_BACKGROUND_COLOR'] = '#FFFFFF'
        env['STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR'] = '#F0F2F6'
        env['STREAMLIT_THEME_TEXT_COLOR'] = '#262730'
        
        # Launch Streamlit
        subprocess.run(cmd, env=env, check=True)
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start dashboard: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    """Main entry point for dashboard launcher."""
    parser = argparse.ArgumentParser(
        description="Launch Nuclear Fusion Analyzer Real-time Dashboard"
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port number for the dashboard (default: 8501)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address for the dashboard (default: localhost)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with auto-reload'
    )
    
    args = parser.parse_args()
    
    launch_dashboard(port=args.port, debug=args.debug, host=args.host)


if __name__ == "__main__":
    main()