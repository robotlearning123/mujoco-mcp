#!/opt/miniconda3/bin/mjpython
"""
Enhanced MuJoCo Viewer Server with Improved Reliability and Performance
- Connection pooling and management
- Advanced error handling and recovery
- Performance monitoring and optimization
- Memory leak prevention
- Graceful shutdown handling
"""

import time
import json
import socket
import threading
import logging
import os
import gc
import signal
from typing import Dict, Any, List
from dataclasses import dataclass
from collections import deque
import uuid
import psutil

import mujoco
import mujoco.viewer
import numpy as np
import contextlib
import builtins

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mujoco_viewer_server.log')
    ]
)
logger = logging.getLogger("enhanced_mujoco_viewer_server")


@dataclass
class ConnectionStats:
    """Connection statistics"""
    created_time: float
    requests_handled: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_activity: float = 0.0
    errors: int = 0


@dataclass
class PerformanceStats:
    """Performance statistics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    models_loaded: int = 0
    requests_per_second: float = 0.0
    avg_response_time: float = 0.0


class ModelViewer:
    """Enhanced viewer manager for a single model"""

    def __init__(self, model_id: str, model_source: str):
        self.model_id = model_id
        self.model = None
        self.data = None
        self.viewer = None
        self.simulation_running = False
        self.created_time = time.time()
        self.last_access = time.time()
        self.access_count = 0
        self.viewer_lock = threading.RLock()

        try:
            # Load model - supports file path or XML string
            if os.path.exists(model_source):
                self.model = mujoco.MjModel.from_xml_path(model_source)
                logger.info(f"Loaded model from file: {model_source}")
            else:
                self.model = mujoco.MjModel.from_xml_string(model_source)
                logger.info("Loaded model from XML string")

            self.data = mujoco.MjData(self.model)

            # Launch passive viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=True
            )

            # Start simulation thread
            self.simulation_running = True
            self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.sim_thread.start()

            logger.info(f"ModelViewer created for {model_id}")

        except Exception as e:
            logger.exception(f"Failed to create ModelViewer for {model_id}: {e}")
            self.cleanup()
            raise

    def _simulation_loop(self):
        """Simulation loop for the model"""
        try:
            while self.simulation_running and self.viewer.is_running():
                with self.viewer_lock:
                    if self.model and self.data:
                        mujoco.mj_step(self.model, self.data)
                        with self.viewer.lock():
                            self.viewer.sync()

                time.sleep(0.01)  # 100Hz simulation

        except Exception as e:
            logger.exception(f"Simulation loop error for {self.model_id}: {e}")
        finally:
            logger.info(f"Simulation loop ended for {self.model_id}")

    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        self.last_access = time.time()
        self.access_count += 1

        try:
            with self.viewer_lock:
                if self.data is None:
                    return {"error": "No data available"}

                return {
                    "time": self.data.time,
                    "qpos": self.data.qpos.copy(),
                    "qvel": self.data.qvel.copy(),
                    "qacc": self.data.qacc.copy() if hasattr(self.data, 'qacc') else [],
                    "ctrl": self.data.ctrl.copy() if hasattr(self.data, 'ctrl') else [],
                    "xpos": self.data.xpos.copy() if hasattr(self.data, 'xpos') else []
                }
        except Exception as e:
            logger.exception(f"Error getting state for {self.model_id}: {e}")
            return {"error": str(e)}

    def set_joint_positions(self, positions: List[float]) -> bool:
        """Set joint positions"""
        self.last_access = time.time()
        self.access_count += 1

        try:
            with self.viewer_lock:
                if self.data is None:
                    return False

                # Ensure positions array is correct size
                n_positions = min(len(positions), len(self.data.qpos))
                self.data.qpos[:n_positions] = positions[:n_positions]

                # Reset velocities
                self.data.qvel[:] = 0

                # Forward kinematics
                mujoco.mj_forward(self.model, self.data)

                return True

        except Exception as e:
            logger.exception(f"Error setting joint positions for {self.model_id}: {e}")
            return False

    def reset(self) -> bool:
        """Reset simulation to initial state"""
        self.last_access = time.time()
        self.access_count += 1

        try:
            with self.viewer_lock:
                if self.model and self.data:
                    mujoco.mj_resetData(self.model, self.data)
                    mujoco.mj_forward(self.model, self.data)
                    return True
                return False

        except Exception as e:
            logger.exception(f"Error resetting {self.model_id}: {e}")
            return False

    def cleanup(self):
        """Clean up resources"""
        try:
            self.simulation_running = False

            if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
                self.sim_thread.join(timeout=1.0)

            if self.viewer:
                with contextlib.suppress(builtins.BaseException):
                    self.viewer.close()
                self.viewer = None

            # Clear references to allow garbage collection
            self.model = None
            self.data = None

            logger.info(f"Cleaned up ModelViewer for {self.model_id}")

        except Exception as e:
            logger.exception(f"Error during cleanup of {self.model_id}: {e}")

    def is_stale(self, timeout: float = 300.0) -> bool:
        """Check if model viewer is stale (unused for too long)"""
        return (time.time() - self.last_access) > timeout

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            "model_id": self.model_id,
            "created_time": self.created_time,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "simulation_running": self.simulation_running,
            "viewer_running": self.viewer.is_running() if self.viewer else False,
            "thread_alive": hasattr(self, 'sim_thread') and self.sim_thread.is_alive()
        }


class ConnectionManager:
    """Manages client connections with connection pooling"""

    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.connections: Dict[str, ConnectionStats] = {}
        self.connection_lock = threading.RLock()
        self.request_times = deque(maxlen=1000)  # Track request timing for rate calculation

    def register_connection(self, conn_id: str) -> bool:
        """Register a new connection"""
        with self.connection_lock:
            if len(self.connections) >= self.max_connections:
                logger.warning(f"Maximum connections ({self.max_connections}) reached")
                return False

            self.connections[conn_id] = ConnectionStats(
                created_time=time.time(),
                last_activity=time.time()
            )
            logger.info(f"Registered connection {conn_id}")
            return True

    def update_connection_activity(self, conn_id: str, bytes_sent: int = 0, bytes_received: int = 0, error: bool = False):
        """Update connection activity"""
        with self.connection_lock:
            if conn_id in self.connections:
                stats = self.connections[conn_id]
                stats.last_activity = time.time()
                stats.requests_handled += 1
                stats.bytes_sent += bytes_sent
                stats.bytes_received += bytes_received
                if error:
                    stats.errors += 1

                # Track request timing
                self.request_times.append(time.time())

    def unregister_connection(self, conn_id: str):
        """Unregister a connection"""
        with self.connection_lock:
            if conn_id in self.connections:
                del self.connections[conn_id]
                logger.info(f"Unregistered connection {conn_id}")

    def cleanup_stale_connections(self, timeout: float = 300.0):
        """Clean up stale connections"""
        current_time = time.time()
        stale_connections = []

        with self.connection_lock:
            for conn_id, stats in self.connections.items():
                if (current_time - stats.last_activity) > timeout:
                    stale_connections.append(conn_id)

        for conn_id in stale_connections:
            self.unregister_connection(conn_id)
            logger.info(f"Cleaned up stale connection {conn_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self.connection_lock:
            total_requests = sum(stats.requests_handled for stats in self.connections.values())
            total_errors = sum(stats.errors for stats in self.connections.values())

            # Calculate requests per second
            current_time = time.time()
            recent_requests = [t for t in self.request_times if (current_time - t) < 60.0]
            rps = len(recent_requests) / 60.0 if recent_requests else 0.0

            return {
                "active_connections": len(self.connections),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": total_errors / max(total_requests, 1),
                "requests_per_second": rps
            }


class PerformanceMonitor:
    """Monitors server performance"""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = True
        self.stats_history = deque(maxlen=100)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Performance monitoring loop"""
        while self.monitoring:
            try:
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)

                stats = PerformanceStats(
                    cpu_usage=cpu_percent,
                    memory_usage=memory_mb
                )

                self.stats_history.append(stats)

                # Log performance warnings
                if cpu_percent > 80:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                if memory_mb > 1000:  # 1GB
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")

            except Exception as e:
                logger.exception(f"Performance monitoring error: {e}")

            time.sleep(10.0)  # Monitor every 10 seconds

    def get_current_stats(self) -> PerformanceStats:
        """Get current performance statistics"""
        if self.stats_history:
            return self.stats_history[-1]
        return PerformanceStats()

    def get_average_stats(self, window: int = 10) -> PerformanceStats:
        """Get average performance statistics"""
        if not self.stats_history:
            return PerformanceStats()

        recent_stats = list(self.stats_history)[-window:]

        return PerformanceStats(
            cpu_usage=np.mean([s.cpu_usage for s in recent_stats]),
            memory_usage=np.mean([s.memory_usage for s in recent_stats])
        )

    def stop(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)


class EnhancedMuJoCoViewerServer:
    """Enhanced MuJoCo Viewer Server with improved reliability and performance"""

    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.running = False
        self.socket = None

        # Model management
        self.models: Dict[str, ModelViewer] = {}
        self.models_lock = threading.RLock()

        # Connection management
        self.connection_manager = ConnectionManager()
        self.client_threads: List[threading.Thread] = []
        self.threads_lock = threading.Lock()

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Cleanup management
        self.cleanup_thread = None
        self.last_cleanup = time.time()

        # Graceful shutdown
        self.shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Enhanced MuJoCo Viewer Server initialized on {host}:{port}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()

    def start(self):
        """Start the server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(1.0)  # Non-blocking accept with timeout

            self.socket.bind((self.host, self.port))
            self.socket.listen(10)

            self.running = True
            logger.info(f"Enhanced MuJoCo Viewer Server listening on {self.host}:{self.port}")

            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()

            # Main server loop
            while self.running and not self.shutdown_event.is_set():
                try:
                    client_socket, address = self.socket.accept()

                    # Check connection limits
                    conn_id = str(uuid.uuid4())
                    if self.connection_manager.register_connection(conn_id):
                        client_thread = threading.Thread(
                            target=self._handle_client,
                            args=(client_socket, address, conn_id),
                            daemon=True
                        )

                        with self.threads_lock:
                            # Clean up finished threads
                            self.client_threads = [t for t in self.client_threads if t.is_alive()]
                            self.client_threads.append(client_thread)

                        client_thread.start()
                        logger.info(f"Accepted connection from {address} (ID: {conn_id})")
                    else:
                        client_socket.close()
                        logger.warning(f"Rejected connection from {address} - too many connections")

                except TimeoutError:
                    continue  # Normal timeout, check shutdown event
                except Exception as e:
                    if self.running:
                        logger.exception(f"Error accepting connections: {e}")

        except Exception as e:
            logger.exception(f"Failed to start server: {e}")
        finally:
            self.shutdown()

    def _cleanup_loop(self):
        """Periodic cleanup of stale resources"""
        while self.running and not self.shutdown_event.is_set():
            try:
                current_time = time.time()

                # Run cleanup every 60 seconds
                if (current_time - self.last_cleanup) > 60.0:
                    self._cleanup_stale_resources()
                    self.last_cleanup = current_time

                time.sleep(10.0)

            except Exception as e:
                logger.exception(f"Cleanup loop error: {e}")

    def _cleanup_stale_resources(self):
        """Clean up stale models and connections"""
        logger.info("Running stale resource cleanup...")

        # Clean up stale connections
        self.connection_manager.cleanup_stale_connections()

        # Clean up stale models
        stale_models = []
        with self.models_lock:
            for model_id, model_viewer in self.models.items():
                if model_viewer.is_stale():
                    stale_models.append(model_id)

        for model_id in stale_models:
            self._remove_model(model_id)
            logger.info(f"Cleaned up stale model {model_id}")

        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            logger.info(f"Garbage collected {collected} objects")

    def _handle_client(self, client_socket: socket.socket, address: tuple, conn_id: str):
        """Handle client connection with enhanced error handling"""
        try:
            client_socket.settimeout(30.0)  # 30 second timeout

            while self.running and not self.shutdown_event.is_set():
                try:
                    # Receive data with size limit
                    data = client_socket.recv(65536)  # 64KB buffer
                    if not data:
                        break

                    self.connection_manager.update_connection_activity(
                        conn_id, bytes_received=len(data)
                    )

                    # Process command
                    response_start = time.time()
                    response = self._process_command(data.decode('utf-8'))
                    response_time = time.time() - response_start

                    # Send response
                    response_data = json.dumps(response).encode('utf-8')
                    client_socket.sendall(response_data)

                    self.connection_manager.update_connection_activity(
                        conn_id, bytes_sent=len(response_data)
                    )

                    # Log slow requests
                    if response_time > 1.0:
                        logger.warning(f"Slow request from {address}: {response_time:.2f}s")

                except TimeoutError:
                    logger.warning(f"Client {address} timed out")
                    break
                except ConnectionResetError:
                    logger.info(f"Client {address} disconnected")
                    break
                except Exception as e:
                    logger.exception(f"Error handling client {address}: {e}")
                    self.connection_manager.update_connection_activity(conn_id, error=True)
                    break

        finally:
            with contextlib.suppress(builtins.BaseException):
                client_socket.close()
            self.connection_manager.unregister_connection(conn_id)
            logger.info(f"Client {address} connection closed")

    def _process_command(self, command_str: str) -> Dict[str, Any]:
        """Process command with enhanced error handling"""
        try:
            command = json.loads(command_str)
            cmd_type = command.get("type")

            if cmd_type == "ping":
                return self._handle_ping()
            elif cmd_type == "load_model":
                return self._handle_load_model(command)
            elif cmd_type == "get_state":
                return self._handle_get_state(command)
            elif cmd_type == "set_joint_positions":
                return self._handle_set_joint_positions(command)
            elif cmd_type == "reset":
                return self._handle_reset(command)
            elif cmd_type == "close_model":
                return self._handle_close_model(command)
            elif cmd_type == "list_models":
                return self._handle_list_models()
            elif cmd_type == "get_diagnostics":
                return self._handle_get_diagnostics()
            elif cmd_type == "shutdown_server":
                return self._handle_shutdown()
            else:
                return {"success": False, "error": f"Unknown command: {cmd_type}"}

        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON decode error: {e}"}
        except Exception as e:
            logger.exception(f"Command processing error: {e}")
            return {"success": False, "error": f"Internal error: {e}"}

    def _handle_ping(self) -> Dict[str, Any]:
        """Handle ping command with server stats"""
        conn_stats = self.connection_manager.get_stats()
        perf_stats = self.performance_monitor.get_current_stats()

        return {
            "success": True,
            "pong": True,
            "server_info": {
                "version": "0.8.2-enhanced",
                "models_loaded": len(self.models),
                "active_connections": conn_stats["active_connections"],
                "requests_per_second": conn_stats["requests_per_second"],
                "cpu_usage": perf_stats.cpu_usage,
                "memory_usage": perf_stats.memory_usage,
                "uptime": time.time() - self.performance_monitor.process.create_time()
            }
        }

    def _handle_load_model(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle load model command"""
        try:
            model_id = command.get("model_id", str(uuid.uuid4()))
            model_xml = command.get("model_xml")

            if not model_xml:
                return {"success": False, "error": "No model_xml provided"}

            # Remove existing model if it exists
            if model_id in self.models:
                self._remove_model(model_id)

            # Create new model viewer
            with self.models_lock:
                model_viewer = ModelViewer(model_id, model_xml)
                self.models[model_id] = model_viewer

            return {
                "success": True,
                "model_id": model_id,
                "message": f"Model {model_id} loaded successfully",
                "model_info": {
                    "nq": model_viewer.model.nq,
                    "nv": model_viewer.model.nv,
                    "nbody": model_viewer.model.nbody,
                    "ngeom": model_viewer.model.ngeom
                }
            }

        except Exception as e:
            logger.exception(f"Load model error: {e}")
            return {"success": False, "error": str(e)}

    def _handle_get_state(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get state command"""
        model_id = command.get("model_id")

        with self.models_lock:
            if model_id not in self.models:
                return {"success": False, "error": f"Model {model_id} not found"}

            state = self.models[model_id].get_state()

            if "error" in state:
                return {"success": False, "error": state["error"]}

            return {"success": True, "state": state}

    def _handle_set_joint_positions(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set joint positions command"""
        model_id = command.get("model_id")
        positions = command.get("positions", [])

        with self.models_lock:
            if model_id not in self.models:
                return {"success": False, "error": f"Model {model_id} not found"}

            success = self.models[model_id].set_joint_positions(positions)

            if success:
                return {"success": True, "message": "Joint positions set successfully"}
            else:
                return {"success": False, "error": "Failed to set joint positions"}

    def _handle_reset(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reset command"""
        model_id = command.get("model_id")

        with self.models_lock:
            if model_id not in self.models:
                return {"success": False, "error": f"Model {model_id} not found"}

            success = self.models[model_id].reset()

            if success:
                return {"success": True, "message": "Model reset successfully"}
            else:
                return {"success": False, "error": "Failed to reset model"}

    def _handle_close_model(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle close model command"""
        model_id = command.get("model_id")

        if self._remove_model(model_id):
            return {"success": True, "message": f"Model {model_id} closed successfully"}
        else:
            return {"success": False, "error": f"Model {model_id} not found"}

    def _handle_list_models(self) -> Dict[str, Any]:
        """Handle list models command"""
        with self.models_lock:
            models_info = {}
            for model_id, model_viewer in self.models.items():
                models_info[model_id] = model_viewer.get_stats()

        return {"success": True, "models": models_info}

    def _handle_get_diagnostics(self) -> Dict[str, Any]:
        """Handle get diagnostics command"""
        conn_stats = self.connection_manager.get_stats()
        perf_stats = self.performance_monitor.get_average_stats()

        with self.models_lock:
            model_stats = {
                model_id: viewer.get_stats()
                for model_id, viewer in self.models.items()
            }

        return {
            "success": True,
            "diagnostics": {
                "server": {
                    "running": self.running,
                    "host": self.host,
                    "port": self.port,
                    "uptime": time.time() - self.performance_monitor.process.create_time()
                },
                "performance": {
                    "cpu_usage": perf_stats.cpu_usage,
                    "memory_usage": perf_stats.memory_usage,
                    "requests_per_second": conn_stats["requests_per_second"]
                },
                "connections": conn_stats,
                "models": model_stats
            }
        }

    def _handle_shutdown(self) -> Dict[str, Any]:
        """Handle shutdown command"""
        logger.info("Shutdown command received")
        threading.Thread(target=self.shutdown, daemon=True).start()
        return {"success": True, "message": "Server shutting down"}

    def _remove_model(self, model_id: str) -> bool:
        """Remove a model and clean up resources"""
        with self.models_lock:
            if model_id in self.models:
                model_viewer = self.models[model_id]
                model_viewer.cleanup()
                del self.models[model_id]
                logger.info(f"Removed model {model_id}")
                return True
            return False

    def shutdown(self):
        """Graceful shutdown of the server"""
        if not self.running:
            return

        logger.info("Starting graceful shutdown...")
        self.running = False
        self.shutdown_event.set()

        # Close server socket
        if self.socket:
            with contextlib.suppress(builtins.BaseException):
                self.socket.close()

        # Wait for client threads to finish
        with self.threads_lock:
            for thread in self.client_threads:
                if thread.is_alive():
                    thread.join(timeout=2.0)

        # Clean up all models
        with self.models_lock:
            for model_id in list(self.models.keys()):
                self._remove_model(model_id)

        # Stop performance monitoring
        self.performance_monitor.stop()

        logger.info("Server shutdown complete")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced MuJoCo Viewer Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8888, help='Port to bind to')
    parser.add_argument('--log-level', default='INFO', help='Logging level')

    args = parser.parse_args()

    # Set logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger().setLevel(numeric_level)

    # Create and start server
    server = EnhancedMuJoCoViewerServer(args.host, args.port)

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
