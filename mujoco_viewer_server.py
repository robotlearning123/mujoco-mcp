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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        if os.path.exists(model_source):
            # If it's a file path, use from_xml_path to load (so relative paths are resolved correctly)
            self.model = mujoco.MjModel.from_xml_path(model_source)
        else:
            # Otherwise assume it's an XML string
            self.model = mujoco.MjModel.from_xml_string(model_source)

        self.data = mujoco.MjData(self.model)

        # Start viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

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
                "ctrl": self.data.ctrl.tolist()
            }

    def set_joint_positions(self, positions: list) -> bool:
        """Set joint positions"""
        with self.viewer.lock():
            for i, pos in enumerate(positions[:self.model.nq]):
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
                if hasattr(self.viewer, 'close'):
                    self.viewer.close()
                elif hasattr(self.viewer, '_window') and self.viewer._window:
                    # For older MuJoCo versions, try to close the window directly
                    with contextlib.suppress(builtins.BaseException):
                        self.viewer._window.close()
                # Wait for simulation thread to finish
                if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
                    self.sim_thread.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error closing viewer for {self.model_id}: {e}")
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

    def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command - Single Viewer mode"""
        cmd_type = command.get("type")

        try:
            if cmd_type == "load_model":
                model_id = command.get("model_id", str(uuid.uuid4()))
                model_source = command.get("model_xml")  # Can be XML string or file path

                with self.viewer_lock:
                    # If there's an existing viewer, close it
                    if self.current_viewer:
                        logger.info(f"Closing existing viewer for {self.current_model_id}")
                        self.current_viewer.close()
                        time.sleep(2.0)  # Give time for viewer to close completely

                    # Create new viewer
                    logger.info(f"Creating new viewer for model {model_id}")
                    self.current_viewer = ModelViewer(model_id, model_source)
                    self.current_model_id = model_id

                return {
                    "success": True,
                    "model_id": model_id,
                    "model_info": {
                        "nq": self.current_viewer.model.nq,
                        "nv": self.current_viewer.model.nv,
                        "nbody": self.current_viewer.model.nbody
                    }
                }

            elif cmd_type == "start_viewer":
                # Compatible with old version, but viewer is already started when load_model
                return {"success": True, "message": "Viewer already started"}

            elif cmd_type == "get_state":
                model_id = command.get("model_id")
                if not self.current_viewer or (model_id and self.current_model_id != model_id):
                    return {"success": False, "error": f"Model {model_id} not found or no active viewer"}

                state = self.current_viewer.get_state()
                return {"success": True, **state}

            elif cmd_type == "set_joint_positions":
                model_id = command.get("model_id")
                positions = command.get("positions", [])

                if not self.current_viewer or (model_id and self.current_model_id != model_id):
                    return {"success": False, "error": f"Model {model_id} not found or no active viewer"}

                self.current_viewer.set_joint_positions(positions)
                return {"success": True, "positions_set": positions}

            elif cmd_type == "reset":
                model_id = command.get("model_id")
                if not self.current_viewer or (model_id and self.current_model_id != model_id):
                    return {"success": False, "error": f"Model {model_id} not found or no active viewer"}

                self.current_viewer.reset()
                return {"success": True}

            elif cmd_type == "close_model":
                model_id = command.get("model_id")
                with self.viewer_lock:
                    if self.current_viewer and (not model_id or self.current_model_id == model_id):
                        logger.info(f"Closing current model {self.current_model_id}")
                        self.current_viewer.close()
                        self.current_viewer = None
                        self.current_model_id = None
                return {"success": True, "message": f"Model {model_id} closed successfully"}

            elif cmd_type == "replace_model":
                model_id = command.get("model_id", str(uuid.uuid4()))
                model_source = command.get("model_xml")  # Can be XML string or file path

                with self.viewer_lock:
                    # Close existing viewer if it exists
                    if self.current_viewer:
                        logger.info(f"Replacing existing model {self.current_model_id} with {model_id}")
                        self.current_viewer.close()
                        time.sleep(2.0)  # Give time for viewer to close completely

                    # Create new viewer
                    self.current_viewer = ModelViewer(model_id, model_source)
                    self.current_model_id = model_id

                return {
                    "success": True,
                    "model_id": model_id,
                    "message": f"Model {model_id} replaced successfully",
                    "model_info": {
                        "nq": self.current_viewer.model.nq,
                        "nv": self.current_viewer.model.nv,
                        "nbody": self.current_viewer.model.nbody
                    }
                }

            elif cmd_type == "list_models":
                models_info = {}
                with self.viewer_lock:
                    if self.current_viewer and self.current_model_id:
                        models_info[self.current_model_id] = {
                            "created_time": self.current_viewer.created_time,
                            "viewer_running": self.current_viewer.viewer and self.current_viewer.viewer.is_running()
                        }
                return {"success": True, "models": models_info}

            elif cmd_type == "ping":
                models_count = 1 if self.current_viewer else 0
                return {
                    "success": True,
                    "pong": True,
                    "models_count": models_count,
                    "current_model": self.current_model_id,
                    "server_running": self.running,
                    "server_info": {
                        "version": "0.7.4",
                        "mode": "single_viewer",
                        "port": self.port,
                        "active_threads": len(self.client_threads)
                    }
                }

            elif cmd_type == "get_diagnostics":
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
                        "port": self.port
                    },
                    "models": {}
                }

                with self.viewer_lock:
                    if self.current_viewer and self.current_model_id:
                        diagnostics["models"][self.current_model_id] = {
                            "created_time": self.current_viewer.created_time,
                            "viewer_running": self.current_viewer.viewer and self.current_viewer.viewer.is_running(),
                            "simulation_running": self.current_viewer.simulation_running,
                            "thread_alive": hasattr(self.current_viewer, 'sim_thread') and self.current_viewer.sim_thread.is_alive()
                        }

                if model_id and self.current_model_id == model_id:
                    diagnostics["requested_model"] = diagnostics["models"][model_id]

                return diagnostics

            elif cmd_type == "capture_render":
                """Capture current rendered image"""
                model_id = command.get("model_id")
                width = command.get("width", 640)
                height = command.get("height", 480)

                if not self.current_viewer or (model_id and self.current_model_id != model_id):
                    return {"success": False, "error": f"Model {model_id} not found or no active viewer"}

                try:
                    # Create renderer
                    renderer = mujoco.Renderer(self.current_viewer.model, height, width)

                    # Update scene
                    renderer.update_scene(self.current_viewer.data)

                    # Render image
                    pixels = renderer.render()

                    # Convert to base64
                    import base64
                    from PIL import Image
                    import io

                    # Create PIL image
                    image = Image.fromarray(pixels)

                    # Save to byte stream
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_data = img_buffer.getvalue()

                    # Convert to base64
                    img_base64 = base64.b64encode(img_data).decode('utf-8')

                    return {
                        "success": True,
                        "image_data": img_base64,
                        "width": width,
                        "height": height,
                        "format": "png"
                    }

                except Exception as e:
                    logger.exception(f"Failed to capture render: {e}")
                    return {"success": False, "error": str(e)}

            elif cmd_type == "close_viewer":
                """Completely close viewer GUI window"""
                with self.viewer_lock:
                    if self.current_viewer:
                        logger.info(f"Closing viewer GUI for model {self.current_model_id}")
                        self.current_viewer.close()
                        self.current_viewer = None
                        self.current_model_id = None
                        return {"success": True, "message": "Viewer GUI closed successfully"}
                    else:
                        return {"success": True, "message": "No viewer is currently open"}

            elif cmd_type == "shutdown_server":
                """Completely shutdown server"""
                logger.info("Shutdown command received")
                self.running = False
                with self.viewer_lock:
                    if self.current_viewer:
                        self.current_viewer.close()
                        self.current_viewer = None
                        self.current_model_id = None
                return {"success": True, "message": "Server shutdown initiated"}

            else:
                return {"success": False, "error": f"Unknown command: {cmd_type}"}

        except Exception as e:
            logger.exception(f"Error handling command {cmd_type}: {e}")
            return {"success": False, "error": str(e)}

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
                        json.loads(data.decode('utf-8'))
                        break
                    except:
                        # Continue receiving
                        if len(data) > 1024 * 1024:  # 1MB limit
                            raise ValueError("Message too large")
                        continue

                # Parse command
                command = json.loads(data.decode('utf-8'))
                logger.debug(f"Received command: {command.get('type', 'unknown')}")

                # Process command
                response = self.handle_command(command)

                # Send response
                response_json = json.dumps(response) + '\n'
                client_socket.send(response_json.encode('utf-8'))

        except Exception as e:
            logger.exception(f"Error handling client {address}: {e}")
            try:
                error_response = {"success": False, "error": str(e)}
                client_socket.send(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()
            logger.info(f"Client {address} disconnected")

    def start_socket_server(self):
        """Start Socket server - supports multiple connections"""
        # Check if port is available
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.bind(('localhost', self.port))
            test_socket.close()
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.exception(f"Port {self.port} is already in use. Please choose a different port or kill the existing process.")
                raise
            else:
                logger.exception(f"Failed to bind to port {self.port}: {e}")
                raise

        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.socket_server.bind(('localhost', self.port))
            self.socket_server.listen(10)  # Support multiple connections
            logger.info(f"MuJoCo Viewer Server listening on port {self.port}")

            while self.running:
                try:
                    client_socket, address = self.socket_server.accept()

                    # Create separate thread for each client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
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
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of port binding retries")
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
