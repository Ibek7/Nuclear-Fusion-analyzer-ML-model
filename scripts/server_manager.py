#!/usr/bin/env python3
"""
Production server management script for Nuclear Fusion Analyzer.

This script provides comprehensive server management capabilities including
server startup, monitoring, scaling, health checks, graceful shutdown,
and production deployment utilities.

Usage:
    python server_manager.py start --port 8000 --workers 4
    python server_manager.py status
    python server_manager.py scale --workers 8
    python server_manager.py stop --graceful
"""

import argparse
import sys
import os
import signal
import time
import json
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import threading
import queue

# Add src to path for imports  
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.config_manager import ConfigManager
from src.utils.performance_monitor import PerformanceMonitor

logger = setup_logger(__name__)


class ServerProcess:
    """Container for server process information."""
    
    def __init__(self, pid: int, port: int, worker_id: str, started_at: str):
        self.pid = pid
        self.port = port
        self.worker_id = worker_id
        self.started_at = started_at
        self.process = psutil.Process(pid)
    
    def is_running(self) -> bool:
        """Check if process is still running."""
        try:
            return self.process.is_running()
        except psutil.NoSuchProcess:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get process statistics."""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            
            return {
                'pid': self.pid,
                'port': self.port,
                'worker_id': self.worker_id,
                'started_at': self.started_at,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_info.rss / 1024 / 1024,
                'status': 'running' if self.is_running() else 'stopped'
            }
        except psutil.NoSuchProcess:
            return {
                'pid': self.pid,
                'port': self.port,
                'worker_id': self.worker_id,
                'started_at': self.started_at,
                'status': 'stopped'
            }


class ServerManager:
    """
    Production server manager for Nuclear Fusion Analyzer API.
    
    Manages multiple worker processes, health monitoring, scaling,
    and graceful shutdown capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize server manager.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        self.processes: Dict[str, ServerProcess] = {}
        self.monitor_thread = None
        self.stop_monitoring = False
        
        self.state_file = Path("./server_state.json")
        self.pid_dir = Path("./pids")
        self.pid_dir.mkdir(exist_ok=True)
        
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("ServerManager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load server configuration."""
        default_config = {
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'worker_class': 'uvicorn.workers.UvicornWorker',
                'max_requests': 1000,
                'max_requests_jitter': 100,
                'timeout': 30,
                'keepalive': 5,
                'preload_app': True
            },
            'monitoring': {
                'health_check_interval': 30,
                'metrics_interval': 60,
                'restart_on_failure': True,
                'max_memory_mb': 1024,
                'max_cpu_percent': 80
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': './logs/server.log'
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            config_manager = ConfigManager(self.config_path)
            user_config = config_manager.get_config()
            
            # Merge configurations
            for section, values in user_config.items():
                if section in default_config:
                    default_config[section].update(values)
                else:
                    default_config[section] = values
        
        return default_config
    
    def start(self, 
              port: Optional[int] = None,
              workers: Optional[int] = None,
              host: Optional[str] = None) -> Dict[str, Any]:
        """
        Start server with specified configuration.
        
        Args:
            port: Server port.
            workers: Number of worker processes.
            host: Server host.
            
        Returns:
            Startup result.
        """
        # Override config with command line arguments
        server_config = self.config['server'].copy()
        if port:
            server_config['port'] = port
        if workers:
            server_config['workers'] = workers
        if host:
            server_config['host'] = host
        
        # Check if server is already running
        if self._is_server_running():
            raise RuntimeError("Server is already running")
        
        logger.info(f"Starting server with {server_config['workers']} workers on {server_config['host']}:{server_config['port']}")
        
        # Start worker processes
        started_processes = []
        base_port = server_config['port']
        
        for i in range(server_config['workers']):
            worker_port = base_port + i
            worker_id = f"worker_{i}"
            
            process = self._start_worker(
                worker_id=worker_id,
                port=worker_port,
                host=server_config['host'],
                config=server_config
            )
            
            if process:
                started_processes.append(process)
                self.processes[worker_id] = process
            else:
                logger.error(f"Failed to start worker {worker_id}")
        
        if not started_processes:
            raise RuntimeError("Failed to start any worker processes")
        
        # Save server state
        self._save_state()
        
        # Start monitoring
        self._start_monitoring()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        result = {
            'status': 'started',
            'workers': len(started_processes),
            'ports': [p.port for p in started_processes],
            'started_at': datetime.now().isoformat()
        }
        
        logger.info(f"Server started successfully: {result}")
        return result
    
    def stop(self, graceful: bool = True, timeout: int = 30) -> Dict[str, Any]:
        """
        Stop server processes.
        
        Args:
            graceful: Whether to perform graceful shutdown.
            timeout: Shutdown timeout in seconds.
            
        Returns:
            Shutdown result.
        """
        logger.info(f"Stopping server (graceful={graceful}, timeout={timeout}s)")
        
        if not self.processes:
            logger.info("No running processes found")
            return {'status': 'already_stopped'}
        
        # Stop monitoring
        self.stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        stopped_processes = []
        failed_processes = []
        
        for worker_id, process in self.processes.items():
            try:
                if process.is_running():
                    if graceful:
                        # Send SIGTERM for graceful shutdown
                        process.process.terminate()
                        
                        # Wait for graceful shutdown
                        try:
                            process.process.wait(timeout=timeout)
                            stopped_processes.append(worker_id)
                            logger.info(f"Worker {worker_id} stopped gracefully")
                        except psutil.TimeoutExpired:
                            # Force kill if graceful shutdown fails
                            logger.warning(f"Worker {worker_id} did not stop gracefully, forcing shutdown")
                            process.process.kill()
                            stopped_processes.append(worker_id)
                    else:
                        # Force kill
                        process.process.kill()
                        stopped_processes.append(worker_id)
                        logger.info(f"Worker {worker_id} killed")
                else:
                    stopped_processes.append(worker_id)
                    logger.info(f"Worker {worker_id} was already stopped")
                    
            except Exception as e:
                logger.error(f"Error stopping worker {worker_id}: {e}")
                failed_processes.append(worker_id)
        
        # Clean up
        self.processes.clear()
        self._cleanup_state()
        
        result = {
            'status': 'stopped',
            'stopped_workers': stopped_processes,
            'failed_workers': failed_processes,
            'stopped_at': datetime.now().isoformat()
        }
        
        logger.info(f"Server stopped: {result}")
        return result
    
    def status(self) -> Dict[str, Any]:
        """
        Get server status.
        
        Returns:
            Server status information.
        """
        if not self.processes:
            return {
                'status': 'stopped',
                'workers': 0,
                'processes': []
            }
        
        process_stats = []
        running_count = 0
        
        for worker_id, process in self.processes.items():
            stats = process.get_stats()
            process_stats.append(stats)
            
            if stats['status'] == 'running':
                running_count += 1
        
        # Calculate aggregate statistics
        total_memory = sum(p.get('memory_mb', 0) for p in process_stats)
        avg_cpu = sum(p.get('cpu_percent', 0) for p in process_stats) / len(process_stats) if process_stats else 0
        
        return {
            'status': 'running' if running_count > 0 else 'stopped',
            'workers': {
                'total': len(self.processes),
                'running': running_count,
                'stopped': len(self.processes) - running_count
            },
            'resources': {
                'total_memory_mb': total_memory,
                'average_cpu_percent': avg_cpu
            },
            'processes': process_stats
        }
    
    def scale(self, workers: int) -> Dict[str, Any]:
        """
        Scale number of worker processes.
        
        Args:
            workers: Target number of workers.
            
        Returns:
            Scaling result.
        """
        current_workers = len(self.processes)
        
        if workers == current_workers:
            return {'status': 'no_change', 'workers': current_workers}
        
        logger.info(f"Scaling from {current_workers} to {workers} workers")
        
        if workers > current_workers:
            # Scale up
            return self._scale_up(workers - current_workers)
        else:
            # Scale down
            return self._scale_down(current_workers - workers)
    
    def restart(self, graceful: bool = True) -> Dict[str, Any]:
        """
        Restart server.
        
        Args:
            graceful: Whether to perform graceful restart.
            
        Returns:
            Restart result.
        """
        logger.info("Restarting server")
        
        # Save current configuration
        current_config = {
            'port': self.config['server']['port'],
            'workers': len(self.processes),
            'host': self.config['server']['host']
        }
        
        # Stop server
        stop_result = self.stop(graceful=graceful)
        
        # Wait a moment
        time.sleep(2)
        
        # Start server with same configuration
        start_result = self.start(**current_config)
        
        return {
            'status': 'restarted',
            'stop_result': stop_result,
            'start_result': start_result,
            'restarted_at': datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health check results.
        """
        results = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }
        
        # Check process health
        for worker_id, process in self.processes.items():
            check_result = {
                'name': f'worker_{worker_id}',
                'status': 'healthy',
                'details': {}
            }
            
            try:
                stats = process.get_stats()
                
                # Check if process is running
                if stats['status'] != 'running':
                    check_result['status'] = 'unhealthy'
                    check_result['details']['error'] = 'Process not running'
                
                # Check memory usage
                memory_limit = self.config['monitoring']['max_memory_mb']
                if stats.get('memory_mb', 0) > memory_limit:
                    check_result['status'] = 'warning'
                    check_result['details']['memory_warning'] = f"Memory usage {stats['memory_mb']:.1f}MB exceeds limit {memory_limit}MB"
                
                # Check CPU usage
                cpu_limit = self.config['monitoring']['max_cpu_percent']
                if stats.get('cpu_percent', 0) > cpu_limit:
                    check_result['status'] = 'warning'
                    check_result['details']['cpu_warning'] = f"CPU usage {stats['cpu_percent']:.1f}% exceeds limit {cpu_limit}%"
                
                check_result['details']['stats'] = stats
                
            except Exception as e:
                check_result['status'] = 'unhealthy'
                check_result['details']['error'] = str(e)
            
            results['checks'].append(check_result)
            
            # Update overall status
            if check_result['status'] == 'unhealthy':
                results['overall_status'] = 'unhealthy'
            elif check_result['status'] == 'warning' and results['overall_status'] == 'healthy':
                results['overall_status'] = 'warning'
        
        return results
    
    def _start_worker(self, 
                      worker_id: str,
                      port: int,
                      host: str,
                      config: Dict[str, Any]) -> Optional[ServerProcess]:
        """Start a single worker process."""
        try:
            # Create command
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api.fusion_api:app",
                "--host", host,
                "--port", str(port),
                "--workers", "1",
                "--log-level", "info"
            ]
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Check if process started successfully
            if process.poll() is not None:
                # Process died
                stdout, stderr = process.communicate()
                logger.error(f"Worker {worker_id} failed to start: {stderr.decode()}")
                return None
            
            # Create ServerProcess object
            server_process = ServerProcess(
                pid=process.pid,
                port=port,
                worker_id=worker_id,
                started_at=datetime.now().isoformat()
            )
            
            # Save PID file
            pid_file = self.pid_dir / f"{worker_id}.pid"
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
            
            logger.info(f"Worker {worker_id} started on port {port} (PID: {process.pid})")
            return server_process
            
        except Exception as e:
            logger.error(f"Failed to start worker {worker_id}: {e}")
            return None
    
    def _scale_up(self, additional_workers: int) -> Dict[str, Any]:
        """Scale up by adding workers."""
        base_port = max(p.port for p in self.processes.values()) + 1
        host = self.config['server']['host']
        
        started_workers = []
        
        for i in range(additional_workers):
            worker_port = base_port + i
            worker_id = f"worker_{len(self.processes) + i}"
            
            process = self._start_worker(
                worker_id=worker_id,
                port=worker_port,
                host=host,
                config=self.config['server']
            )
            
            if process:
                self.processes[worker_id] = process
                started_workers.append(worker_id)
        
        self._save_state()
        
        return {
            'status': 'scaled_up',
            'added_workers': len(started_workers),
            'total_workers': len(self.processes),
            'new_workers': started_workers
        }
    
    def _scale_down(self, workers_to_remove: int) -> Dict[str, Any]:
        """Scale down by removing workers."""
        # Select workers to remove (LIFO - last in, first out)
        workers_to_stop = list(self.processes.keys())[-workers_to_remove:]
        
        stopped_workers = []
        
        for worker_id in workers_to_stop:
            process = self.processes[worker_id]
            
            try:
                if process.is_running():
                    process.process.terminate()
                    process.process.wait(timeout=10)
                
                # Clean up PID file
                pid_file = self.pid_dir / f"{worker_id}.pid"
                if pid_file.exists():
                    pid_file.unlink()
                
                del self.processes[worker_id]
                stopped_workers.append(worker_id)
                
                logger.info(f"Worker {worker_id} stopped for scaling")
                
            except Exception as e:
                logger.error(f"Error stopping worker {worker_id}: {e}")
        
        self._save_state()
        
        return {
            'status': 'scaled_down',
            'removed_workers': len(stopped_workers),
            'total_workers': len(self.processes),
            'stopped_workers': stopped_workers
        }
    
    def _is_server_running(self) -> bool:
        """Check if server is already running."""
        return len(self.processes) > 0 and any(p.is_running() for p in self.processes.values())
    
    def _start_monitoring(self):
        """Start monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        interval = self.config['monitoring']['health_check_interval']
        
        while not self.stop_monitoring:
            try:
                # Perform health checks
                health_result = self.health_check()
                
                # Log health status
                if health_result['overall_status'] != 'healthy':
                    logger.warning(f"Health check failed: {health_result}")
                
                # Check for failed processes
                failed_workers = []
                for worker_id, process in list(self.processes.items()):
                    if not process.is_running():
                        failed_workers.append(worker_id)
                
                # Restart failed workers if configured
                if failed_workers and self.config['monitoring']['restart_on_failure']:
                    logger.warning(f"Restarting failed workers: {failed_workers}")
                    self._restart_failed_workers(failed_workers)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _restart_failed_workers(self, failed_workers: List[str]):
        """Restart failed worker processes."""
        for worker_id in failed_workers:
            if worker_id in self.processes:
                old_process = self.processes[worker_id]
                
                # Start new worker
                new_process = self._start_worker(
                    worker_id=worker_id,
                    port=old_process.port,
                    host=self.config['server']['host'],
                    config=self.config['server']
                )
                
                if new_process:
                    self.processes[worker_id] = new_process
                    logger.info(f"Worker {worker_id} restarted successfully")
                else:
                    logger.error(f"Failed to restart worker {worker_id}")
                    del self.processes[worker_id]
    
    def _save_state(self):
        """Save server state to file."""
        state = {
            'processes': {
                worker_id: {
                    'pid': process.pid,
                    'port': process.port,
                    'worker_id': process.worker_id,
                    'started_at': process.started_at
                }
                for worker_id, process in self.processes.items()
            },
            'config': self.config,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _cleanup_state(self):
        """Clean up state files."""
        if self.state_file.exists():
            self.state_file.unlink()
        
        # Clean up PID files
        for pid_file in self.pid_dir.glob("*.pid"):
            pid_file.unlink()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, performing graceful shutdown")
            self.stop(graceful=True)
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Nuclear Fusion Analyzer Server Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start server')
    start_parser.add_argument('--port', type=int, help='Server port')
    start_parser.add_argument('--workers', type=int, help='Number of workers')
    start_parser.add_argument('--host', help='Server host')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop server')
    stop_parser.add_argument('--graceful', action='store_true', default=True, help='Graceful shutdown')
    stop_parser.add_argument('--timeout', type=int, default=30, help='Shutdown timeout')
    
    # Status command
    subparsers.add_parser('status', help='Get server status')
    
    # Scale command
    scale_parser = subparsers.add_parser('scale', help='Scale workers')
    scale_parser.add_argument('--workers', type=int, required=True, help='Target number of workers')
    
    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart server')
    restart_parser.add_argument('--graceful', action='store_true', default=True, help='Graceful restart')
    
    # Health command
    subparsers.add_parser('health', help='Perform health check')
    
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize server manager
    manager = ServerManager(args.config)
    
    try:
        if args.command == 'start':
            result = manager.start(
                port=args.port,
                workers=args.workers,
                host=args.host
            )
            print(json.dumps(result, indent=2))
        
        elif args.command == 'stop':
            result = manager.stop(
                graceful=args.graceful,
                timeout=args.timeout
            )
            print(json.dumps(result, indent=2))
        
        elif args.command == 'status':
            result = manager.status()
            print(json.dumps(result, indent=2))
        
        elif args.command == 'scale':
            result = manager.scale(args.workers)
            print(json.dumps(result, indent=2))
        
        elif args.command == 'restart':
            result = manager.restart(graceful=args.graceful)
            print(json.dumps(result, indent=2))
        
        elif args.command == 'health':
            result = manager.health_check()
            print(json.dumps(result, indent=2))
            
            # Exit with error code if unhealthy
            if result['overall_status'] == 'unhealthy':
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()