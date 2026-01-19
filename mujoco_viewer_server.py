#!/opt/miniconda3/bin/mjpython
"""
MuJoCo Viewer Server - Enhanced Version
Supports concurrent multi-model management
Uses official mujoco.viewer.launch_passive() API
Communicates with MCP server via Socket

Fixed issues:
1. Support for multiple concurrent connections
2. Increased receive buffer size
3. Improved error handling and timeout management
4. Support for independent management of multiple models
"""

import time
import json
import socket
import threading
import logging
import sys
import os
from typing import Dict, Any
import uuid

import mujoco
import mujoco.viewer
import contextlib
import builtins

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mujoco_viewer_server")


class ModelViewer:
    """Viewer manager for a single model"""

    def __init__(self, model_id: str, model_source: str):
        self.model_id = model_id
        self.model = None
        self.data = None
        self.viewer = None
        self.simulation_running = False
        self.created_time = time.time()

        # Load model - supports file path or XML string
        # Paths are resolved relative to the XML file's directory
        try:
            if os.path.exists(model_source):
                logger.info(f"Loading model {model_id} from file: {model_source}")
                self.model = mujoco.MjModel.from_xml_path(model_source)
            else:
                logger.info(f"Loading model {model_id} from XML string")
                self.model = mujoco.MjModel.from_xml_string(model_source)
        except FileNotFoundError as e:
            logger.error(f"Model file not found for {model_id}: {model_source}")
            raise RuntimeError(f"Failed to load model {model_id}: file not found at {model_source}") from e
        except Exception as e:
            logger.error(f"Failed to load MuJoCo model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e

        # Create simulation data
        try:
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            logger.error(f"Failed to create MjData for model {model_id}: {e}")
            raise RuntimeError(f"Failed to initialize simulation data for {model_id}: {e}") from e

        # Start viewer
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as e:
            logger.error(f"Failed to launch viewer for model {model_id}: {e}")
            raise RuntimeError(f"Failed to start viewer for {model_id}: {e}") from e

        # Start simulation loop
        self.simulation_running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        logger.info(f"Created ModelViewer for {model_id}")

    def _simulation_loop(self):
        """Simulation loop"""
        while self.simulation_running and self.viewer and self.viewer.is_running():
            with self.viewer.lock():
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.002)  # ~500Hz

    def get_state(self) -> Dict[str, Any]:
        """Get state"""
        with self.viewer.lock():
            return {
                "time": self.data.time,
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist(),
            }

    def set_joint_positions(self, positions: list) -> bool:
        """Set joint positions"""
        with self.viewer.lock():
            for i, pos in enumerate(positions[: self.model.nq]):
                self.data.qpos[i] = pos
            mujoco.mj_forward(self.model, self.data)
        return True

    def reset(self):
        """Reset simulation"""
        with self.viewer.lock():
            mujoco.mj_resetData(self.model, self.data)

    def close(self):
        """Close viewer"""
        self.simulation_running = False
        if self.viewer:
            try:
                # Force close the viewer window
                if hasattr(self.viewer, "close"):
                    self.viewer.close()
                elif hasattr(self.viewer, "_window") and self.viewer._window:
                    # For older MuJoCo versions, try to close the window directly
                    try:
                        self.viewer._window.close()
                    except (AttributeError, RuntimeError) as e:
                        logger.debug(f"Failed to close viewer window for {self.model_id}: {e}")

                # Wait for simulation thread to finish
                if hasattr(self, "sim_thread") and self.sim_thread.is_alive():
                    self.sim_thread.join(timeout=2.0)
                    if self.sim_thread.is_alive():
                        logger.warning(f"Simulation thread for {self.model_id} did not terminate within timeout")
            except KeyboardInterrupt:
                # Never suppress user interrupts
                raise
            except (AttributeError, RuntimeError, OSError) as e:
                # Expected errors during cleanup
                logger.warning(f"Error closing viewer for {self.model_id}: {e}")
            except Exception as e:
                # Unexpected errors should be logged as errors
                logger.error(f"Unexpected error closing viewer for {self.model_id}: {e}")
            finally:
                self.viewer = None
        logger.info(f"Closed ModelViewer for {self.model_id}")


class MuJoCoViewerServer:
    """Single Viewer MuJoCo Server - supports model replacement"""

    def __init__(self, port: int = 8888):
        self.port = port
        self.running = False
        self.socket_server = None

        # Single model manager - only supports one active viewer
        self.current_viewer: ModelViewer | None = None
        self.current_model_id: str | None = None
        self.viewer_lock = threading.Lock()

        # Client management
        self.client_threads = []

        # Command handlers
        self._command_handlers = {
            "load_model": self._handle_load_model,
            "start_viewer": self._handle_start_viewer,
            "get_state": self._handle_get_state,
            "set_joint_positions": self._handle_set_joint_positions,
            "reset": self._handle_reset,
            "close_model": self._handle_close_model,
            "replace_model": self._handle_replace_model,
            "list_models": self._handle_list_models,
            "ping": self._handle_ping,
            "get_diagnostics": self._handle_get_diagnostics,
            "capture_render": self._handle_capture_render,
            "close_viewer": self._handle_close_viewer,
            "shutdown_server": self._handle_shutdown_server,
        }

    def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command - Single Viewer mode"""
        cmd_type = command.get("type")

        try:
            handler = self._command_handlers.get(cmd_type)
            if handler:
                return handler(command)
            logger.warning(f"Unknown command type received: {cmd_type}")
            return {"success": False, "error": f"Unknown command: {cmd_type}"}

        except (KeyError, ValueError, TypeError) as e:
            # Expected parameter validation errors
            logger.warning(f"Invalid parameters for command {cmd_type}: {e}")
            return {"success": False, "error": f"Invalid parameters: {e}"}
        except RuntimeError as e:
            # Expected runtime errors (model loading failures, etc.)
            logger.error(f"Runtime error handling command {cmd_type}: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            # Unexpected errors - these indicate bugs
            logger.exception(f"Unexpected error handling command {cmd_type}: {e}")
            return {"success": False, "error": f"Internal server error: {str(e)}"}

    def _check_viewer_available(self, model_id: str | None) -> Dict[str, Any] | None:
        """Check if viewer is available for the given model. Returns error dict or None if OK."""
        if not self.current_viewer or (model_id and self.current_model_id != model_id):
            return {
                "success": False,
                "error": f"Model {model_id} not found or no active viewer",
            }
        return None

    def _handle_load_model(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Load a new model, replacing any existing one."""
        model_id = command.get("model_id", str(uuid.uuid4()))
        model_source = command.get("model_xml")

        with self.viewer_lock:
            if self.current_viewer:
                logger.info(f"Closing existing viewer for {self.current_model_id}")
                self.current_viewer.close()
                time.sleep(2.0)

            logger.info(f"Creating new viewer for model {model_id}")
            self.current_viewer = ModelViewer(model_id, model_source)
            self.current_model_id = model_id

        return {
            "success": True,
            "model_id": model_id,
            "model_info": {
                "nq": self.current_viewer.model.nq,
                "nv": self.current_viewer.model.nv,
                "nbody": self.current_viewer.model.nbody,
            },
        }

    def _handle_start_viewer(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Compatible with old version - viewer already started with load_model."""
        return {"success": True, "message": "Viewer already started"}

    def _handle_get_state(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Get current simulation state."""
        model_id = command.get("model_id")
        error = self._check_viewer_available(model_id)
        if error:
            return error

        state = self.current_viewer.get_state()
        return {"success": True, **state}

    def _handle_set_joint_positions(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Set joint positions."""
        model_id = command.get("model_id")
        positions = command.get("positions", [])

        error = self._check_viewer_available(model_id)
        if error:
            return error

        self.current_viewer.set_joint_positions(positions)
        return {"success": True, "positions_set": positions}

    def _handle_reset(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Reset simulation."""
        model_id = command.get("model_id")
        error = self._check_viewer_available(model_id)
        if error:
            return error

        self.current_viewer.reset()
        return {"success": True}

    def _handle_close_model(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Close the current model."""
        model_id = command.get("model_id")
        with self.viewer_lock:
            if self.current_viewer and (not model_id or self.current_model_id == model_id):
                logger.info(f"Closing current model {self.current_model_id}")
                self.current_viewer.close()
                self.current_viewer = None
                self.current_model_id = None
        return {"success": True, "message": f"Model {model_id} closed successfully"}

    def _handle_replace_model(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Replace current model with a new one."""
        model_id = command.get("model_id", str(uuid.uuid4()))
        model_source = command.get("model_xml")

        with self.viewer_lock:
            if self.current_viewer:
                logger.info(f"Replacing existing model {self.current_model_id} with {model_id}")
                self.current_viewer.close()
                time.sleep(2.0)

            self.current_viewer = ModelViewer(model_id, model_source)
            self.current_model_id = model_id

        return {
            "success": True,
            "model_id": model_id,
            "message": f"Model {model_id} replaced successfully",
            "model_info": {
                "nq": self.current_viewer.model.nq,
                "nv": self.current_viewer.model.nv,
                "nbody": self.current_viewer.model.nbody,
            },
        }

    def _handle_list_models(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """List all loaded models."""
        models_info = {}
        with self.viewer_lock:
            if self.current_viewer and self.current_model_id:
                models_info[self.current_model_id] = {
                    "created_time": self.current_viewer.created_time,
                    "viewer_running": self.current_viewer.viewer
                    and self.current_viewer.viewer.is_running(),
                }
        return {"success": True, "models": models_info}

    def _handle_ping(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Ping the server."""
        with self.viewer_lock:
            models_count = 1 if self.current_viewer else 0
            current_model = self.current_model_id
        return {
            "success": True,
            "pong": True,
            "models_count": models_count,
            "current_model": current_model,
            "server_running": self.running,
            "server_info": {
                "version": "0.7.4",
                "mode": "single_viewer",
                "port": self.port,
                "active_threads": len(self.client_threads),
            },
        }

    def _handle_get_diagnostics(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Get diagnostic information."""
        model_id = command.get("model_id")
        models_count = 1 if self.current_viewer else 0
        diagnostics = {
            "success": True,
            "server_status": {
                "running": self.running,
                "mode": "single_viewer",
                "models_count": models_count,
                "current_model": self.current_model_id,
                "active_connections": len(self.client_threads),
                "port": self.port,
            },
            "models": {},
        }

        with self.viewer_lock:
            if self.current_viewer and self.current_model_id:
                diagnostics["models"][self.current_model_id] = {
                    "created_time": self.current_viewer.created_time,
                    "viewer_running": self.current_viewer.viewer
                    and self.current_viewer.viewer.is_running(),
                    "simulation_running": self.current_viewer.simulation_running,
                    "thread_alive": hasattr(self.current_viewer, "sim_thread")
                    and self.current_viewer.sim_thread.is_alive(),
                }

        if model_id and self.current_model_id == model_id:
            diagnostics["requested_model"] = diagnostics["models"][model_id]

        return diagnostics

    def _handle_capture_render(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Capture current rendered image."""
        model_id = command.get("model_id")
        width = command.get("width", 640)
        height = command.get("height", 480)

        error = self._check_viewer_available(model_id)
        if error:
            return error

        try:
            renderer = mujoco.Renderer(self.current_viewer.model, height, width)
            renderer.update_scene(self.current_viewer.data)
            pixels = renderer.render()

            import base64
            from PIL import Image
            import io

            image = Image.fromarray(pixels)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            return {
                "success": True,
                "image_data": img_base64,
                "width": width,
                "height": height,
                "format": "png",
            }

        except Exception as e:
            logger.exception(f"Failed to capture render: {e}")
            return {"success": False, "error": str(e)}

    def _handle_close_viewer(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Completely close viewer GUI window."""
        with self.viewer_lock:
            if not self.current_viewer:
                return {"success": True, "message": "No viewer is currently open"}

            logger.info(f"Closing viewer GUI for model {self.current_model_id}")
            self.current_viewer.close()
            self.current_viewer = None
            self.current_model_id = None
            return {"success": True, "message": "Viewer GUI closed successfully"}

    def _handle_shutdown_server(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Completely shutdown server."""
        logger.info("Shutdown command received")
        self.running = False
        with self.viewer_lock:
            if self.current_viewer:
                self.current_viewer.close()
                self.current_viewer = None
                self.current_model_id = None
        return {"success": True, "message": "Server shutdown initiated"}

    def handle_client(self, client_socket: socket.socket, address):
        """Handle single client connection - in separate thread"""
        logger.info(f"Client connected from {address}")

        # Set larger receive buffer
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        try:
            while self.running:
                # Receive data - support large messages
                data = b""
                while True:
                    chunk = client_socket.recv(8192)
                    if not chunk:
                        if data:
                            break
                        else:
                            # Connection closed
                            return
                    data += chunk

                    # Check if complete JSON received
                    try:
                        json.loads(data.decode("utf-8"))
                        break
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Continue receiving partial JSON
                        if len(data) > 1024 * 1024:  # 1MB limit
                            logger.exception(f"Message too large: {len(data)} bytes from {address}")
                            raise ValueError(f"Message exceeds 1MB limit: {len(data)} bytes")
                        continue

                # Parse command
                command = json.loads(data.decode("utf-8"))
                logger.debug(f"Received command: {command.get('type', 'unknown')}")

                # Process command
                response = self.handle_command(command)

                # Send response
                response_json = json.dumps(response) + "\n"
                client_socket.send(response_json.encode("utf-8"))

        except Exception as e:
            logger.exception(f"Error handling client {address}: {e}")
            try:
                error_response = {"success": False, "error": str(e)}
                client_socket.send(json.dumps(error_response).encode("utf-8"))
            except (OSError, BrokenPipeError) as send_error:
                logger.exception(f"Failed to send error response to {address}: {send_error}")
                # Client likely disconnected, safe to ignore
            except Exception as send_error:
                logger.exception(f"Unexpected error sending error response to {address}")
        finally:
            client_socket.close()
            logger.info(f"Client {address} disconnected")

    def start_socket_server(self):
        """Start Socket server - supports multiple connections"""
        # Check if port is available
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind(("localhost", self.port))
            test_socket.close()
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.exception(
                    f"Port {self.port} is already in use. Please choose a different "
                    f"port or kill the existing process."
                )
                raise
            else:
                logger.exception(f"Failed to bind to port {self.port}: {e}")
                raise

        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.socket_server.bind(("localhost", self.port))
            self.socket_server.listen(10)  # Support multiple connections
            logger.info(f"MuJoCo Viewer Server listening on port {self.port}")

            while self.running:
                try:
                    client_socket, address = self.socket_server.accept()

                    # Create separate thread for each client
                    client_thread = threading.Thread(
                        target=self.handle_client, args=(client_socket, address), daemon=True
                    )
                    client_thread.start()
                    self.client_threads.append(client_thread)

                except Exception as e:
                    if self.running:
                        logger.exception(f"Error accepting connection: {e}")
        except Exception as e:
            logger.exception(f"Failed to start socket server: {e}")
            raise

    def start(self):
        """Start server"""
        self.running = True
        logger.info("Starting Enhanced MuJoCo Viewer Server...")

        try:
            self.start_socket_server()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop server"""
        logger.info("Stopping MuJoCo Viewer Server...")
        self.running = False

        # Close current viewer
        with self.viewer_lock:
            if self.current_viewer:
                logger.info(f"Closing current viewer for {self.current_model_id}")
                self.current_viewer.close()
                self.current_viewer = None
                self.current_model_id = None

        # Close socket
        if self.socket_server:
            with contextlib.suppress(builtins.BaseException):
                self.socket_server.close()

        logger.info("Server stopped")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced MuJoCo Viewer Server")
    parser.add_argument("--port", type=int, default=8888, help="Socket server port")
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum number of port binding retries"
    )
    args = parser.parse_args()

    # Try different ports if the default one is in use
    for retry in range(args.max_retries):
        try:
            port = args.port + retry if retry > 0 else args.port
            server = MuJoCoViewerServer(port=port)
            server.start()
            break
        except OSError as e:
            if e.errno == 48 and retry < args.max_retries - 1:  # Address already in use
                print(f"Port {port} is in use, trying port {port + 1}...")
                continue
            else:
                print(f"Failed to start server: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
