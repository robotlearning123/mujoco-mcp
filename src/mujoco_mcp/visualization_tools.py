#!/usr/bin/env python3
"""
Real-time Visualization Tools for MuJoCo MCP
Advanced plotting, monitoring, and analysis tools for robot simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path

from .viewer_client import MuJoCoViewerClient


@dataclass
class PlotConfig:
    """Configuration for real-time plots"""
    title: str = "Real-time Plot"
    xlabel: str = "Time (s)"
    ylabel: str = "Value"
    line_style: str = "-"
    color: str = "blue"
    alpha: float = 1.0
    line_width: float = 1.0
    max_points: int = 1000
    update_interval: int = 50  # milliseconds


@dataclass
class VisualizationData:
    """Data structure for visualization"""
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimePlotter:
    """Real-time plotting for simulation data"""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        self.data = VisualizationData()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.lines = []
        self.running = False
        self.animation = None
        
        # Setup plot
        self.ax.set_title(config.title)
        self.ax.set_xlabel(config.xlabel)
        self.ax.set_ylabel(config.ylabel)
        self.ax.grid(True, alpha=0.3)
        
    def add_data_source(self, label: str, color: str = None):
        """Add a new data source to plot"""
        if color is None:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            color = colors[len(self.lines) % len(colors)]
        
        line, = self.ax.plot([], [], 
                           label=label,
                           color=color,
                           linewidth=self.config.line_width,
                           alpha=self.config.alpha)
        self.lines.append(line)
        self.data.labels.append(label)
        
        # Add empty data arrays for this source
        if not hasattr(self.data, 'multi_values'):
            self.data.multi_values = []
        self.data.multi_values.append(deque(maxlen=self.config.max_points))
        
        self.ax.legend()
        return len(self.lines) - 1
    
    def update_data(self, timestamp: float, values: List[float]):
        """Update data for all sources"""
        self.data.timestamps.append(timestamp)
        
        if not hasattr(self.data, 'multi_values'):
            self.data.multi_values = [deque(maxlen=self.config.max_points) for _ in range(len(values))]
        
        for i, value in enumerate(values):
            if i < len(self.data.multi_values):
                self.data.multi_values[i].append(value)
    
    def _update_plot(self, frame):
        """Update plot animation frame"""
        if not self.data.timestamps:
            return self.lines
        
        times = list(self.data.timestamps)
        
        for i, line in enumerate(self.lines):
            if i < len(self.data.multi_values) and self.data.multi_values[i]:
                values = list(self.data.multi_values[i])
                # Ensure times and values have same length
                min_len = min(len(times), len(values))
                line.set_data(times[-min_len:], values[-min_len:])
        
        # Auto-scale axes
        if times:
            self.ax.set_xlim(max(0, times[-1] - 30), times[-1] + 1)  # Show last 30 seconds
            
            all_values = []
            for value_deque in getattr(self.data, 'multi_values', []):
                all_values.extend(list(value_deque))
            
            if all_values:
                y_min, y_max = min(all_values), max(all_values)
                margin = (y_max - y_min) * 0.1
                self.ax.set_ylim(y_min - margin, y_max + margin)
        
        return self.lines
    
    def start(self):
        """Start real-time plotting"""
        if not self.running:
            self.running = True
            self.animation = animation.FuncAnimation(
                self.fig, self._update_plot,
                interval=self.config.update_interval,
                blit=True
            )
            plt.show()
    
    def stop(self):
        """Stop real-time plotting"""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
    
    def save_plot(self, filename: str):
        """Save current plot to file"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')


class InteractiveVisualizer:
    """Interactive visualization with Plotly"""
    
    def __init__(self):
        self.data_sources = {}
        self.fig = None
        self.update_thread = None
        self.running = False
        
    def create_dashboard(self, title: str = "MuJoCo MCP Dashboard") -> go.Figure:
        """Create interactive dashboard"""
        
        # Create subplots
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Joint Positions", "Joint Velocities",
                "End-Effector Position", "Control Signals",
                "Forces/Torques", "Performance Metrics"
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        self.fig.update_layout(
            title=title,
            showlegend=True,
            height=800,
            updatemenus=[
                {
                    "buttons": [
                        {"label": "Play", "method": "animate", "args": [None]},
                        {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}}]}
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ]
        )
        
        return self.fig
    
    def add_data_stream(self, name: str, data_type: str, subplot_row: int, subplot_col: int):
        """Add data stream to dashboard"""
        self.data_sources[name] = {
            "type": data_type,
            "row": subplot_row,
            "col": subplot_col,
            "data": deque(maxlen=1000),
            "timestamps": deque(maxlen=1000)
        }
    
    def update_data_stream(self, name: str, timestamp: float, value: Any):
        """Update data stream"""
        if name in self.data_sources:
            source = self.data_sources[name]
            source["timestamps"].append(timestamp)
            source["data"].append(value)
    
    def update_dashboard(self):
        """Update dashboard with latest data"""
        if not self.fig:
            return
        
        with self.fig.batch_update():
            for name, source in self.data_sources.items():
                if source["data"]:
                    x_data = list(source["timestamps"])
                    y_data = list(source["data"])
                    
                    # Check if trace exists
                    trace_exists = False
                    for trace in self.fig.data:
                        if trace.name == name:
                            trace_exists = True
                            trace.x = x_data
                            trace.y = y_data
                            break
                    
                    if not trace_exists:
                        self.fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=y_data,
                                mode='lines',
                                name=name,
                                line=dict(width=2)
                            ),
                            row=source["row"],
                            col=source["col"]
                        )
    
    def show_dashboard(self):
        """Show interactive dashboard"""
        if self.fig:
            self.fig.show()
    
    def save_dashboard(self, filename: str):
        """Save dashboard to HTML file"""
        if self.fig:
            pio.write_html(self.fig, filename, auto_open=False)


class RobotStateMonitor:
    """Monitor and visualize robot state in real-time"""
    
    def __init__(self, viewer_client: MuJoCoViewerClient):
        self.viewer_client = viewer_client
        self.monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        
        # Visualization components
        self.joint_plotter = None
        self.force_plotter = None
        self.trajectory_plotter = None
        self.dashboard = InteractiveVisualizer()
        
        # Data storage
        self.state_history = deque(maxlen=10000)
        self.start_time = None
        
    def start_monitoring(self, model_id: str, update_rate: float = 50.0):
        """Start monitoring robot state"""
        if not self.monitoring:
            self.monitoring = True
            self.start_time = time.time()
            
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(model_id, update_rate),
                daemon=True
            )
            self.monitor_thread.start()
            
            # Setup visualizations
            self._setup_visualizations()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitoring_loop(self, model_id: str, update_rate: float):
        """Main monitoring loop"""
        dt = 1.0 / update_rate
        
        while self.monitoring:
            try:
                # Get current state
                response = self.viewer_client.send_command({
                    "type": "get_state",
                    "model_id": model_id
                })
                
                if response.get("success"):
                    state = response.get("state", {})
                    timestamp = time.time() - self.start_time
                    
                    # Store state
                    state_entry = {
                        "timestamp": timestamp,
                        "state": state
                    }
                    self.state_history.append(state_entry)
                    self.data_queue.put(state_entry)
                    
                    # Update visualizations
                    self._update_visualizations(state_entry)
                
                time.sleep(dt)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _setup_visualizations(self):
        """Setup visualization components"""
        # Joint position plotter
        joint_config = PlotConfig(
            title="Joint Positions",
            ylabel="Position (rad)",
            max_points=1000
        )
        self.joint_plotter = RealTimePlotter(joint_config)
        
        # Force/torque plotter
        force_config = PlotConfig(
            title="Forces and Torques",
            ylabel="Force/Torque (N/Nm)",
            max_points=1000
        )
        self.force_plotter = RealTimePlotter(force_config)
        
        # Setup dashboard
        self.dashboard.create_dashboard("Robot State Monitor")
        self.dashboard.add_data_stream("joint_pos", "position", 1, 1)
        self.dashboard.add_data_stream("joint_vel", "velocity", 1, 2)
        self.dashboard.add_data_stream("end_effector", "position", 2, 1)
    
    def _update_visualizations(self, state_entry: Dict[str, Any]):
        """Update all visualizations with new state"""
        timestamp = state_entry["timestamp"]
        state = state_entry["state"]
        
        # Extract data
        qpos = np.array(state.get("qpos", []))
        qvel = np.array(state.get("qvel", []))
        
        # Update joint position plotter
        if self.joint_plotter and len(qpos) > 0:
            # Add data sources if not already done
            if not self.joint_plotter.lines:
                for i in range(len(qpos)):
                    self.joint_plotter.add_data_source(f"Joint {i+1}")
            
            self.joint_plotter.update_data(timestamp, qpos.tolist())
        
        # Update dashboard
        if len(qpos) > 0:
            self.dashboard.update_data_stream("joint_pos", timestamp, np.mean(qpos))
        if len(qvel) > 0:
            self.dashboard.update_data_stream("joint_vel", timestamp, np.mean(qvel))
    
    def show_joint_positions(self):
        """Show joint position plot"""
        if self.joint_plotter:
            self.joint_plotter.start()
    
    def show_dashboard(self):
        """Show interactive dashboard"""
        self.dashboard.show_dashboard()
    
    def export_data(self, filename: str):
        """Export monitoring data to file"""
        data = {
            "metadata": {
                "start_time": self.start_time,
                "total_samples": len(self.state_history),
                "export_time": time.time()
            },
            "states": list(self.state_history)
        }
        
        filepath = Path(filename)
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif filepath.suffix == '.npz':
            # Export as numpy arrays
            timestamps = [entry["timestamp"] for entry in self.state_history]
            qpos_data = []
            qvel_data = []
            
            for entry in self.state_history:
                state = entry["state"]
                qpos_data.append(state.get("qpos", []))
                qvel_data.append(state.get("qvel", []))
            
            np.savez(filepath,
                    timestamps=np.array(timestamps),
                    qpos=np.array(qpos_data),
                    qvel=np.array(qvel_data))
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics from monitoring data"""
        if not self.state_history:
            return {}
        
        # Extract time series data
        timestamps = [entry["timestamp"] for entry in self.state_history]
        qpos_data = []
        qvel_data = []
        
        for entry in self.state_history:
            state = entry["state"]
            qpos_data.append(state.get("qpos", []))
            qvel_data.append(state.get("qvel", []))
        
        qpos_array = np.array(qpos_data)
        qvel_array = np.array(qvel_data)
        
        analysis = {
            "duration": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            "sample_rate": len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0,
            "joint_statistics": {},
            "motion_metrics": {}
        }
        
        # Joint statistics
        if qpos_array.size > 0:
            analysis["joint_statistics"] = {
                "position_mean": np.mean(qpos_array, axis=0).tolist(),
                "position_std": np.std(qpos_array, axis=0).tolist(),
                "position_range": (np.max(qpos_array, axis=0) - np.min(qpos_array, axis=0)).tolist()
            }
        
        if qvel_array.size > 0:
            analysis["joint_statistics"].update({
                "velocity_mean": np.mean(qvel_array, axis=0).tolist(),
                "velocity_std": np.std(qvel_array, axis=0).tolist(),
                "velocity_max": np.max(np.abs(qvel_array), axis=0).tolist()
            })
        
        # Motion smoothness metrics
        if qvel_array.size > 0:
            # Calculate jerk (derivative of acceleration)
            jerk = np.diff(qvel_array, axis=0)
            analysis["motion_metrics"] = {
                "smoothness_score": 1.0 / (1.0 + np.mean(np.abs(jerk))),
                "max_jerk": np.max(np.abs(jerk)).tolist() if jerk.size > 0 else 0
            }
        
        return analysis


class TrajectoryVisualizer:
    """Visualize robot trajectories in 3D"""
    
    def __init__(self):
        self.trajectories = {}
        self.fig = None
        
    def add_trajectory(self, name: str, positions: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """Add trajectory for visualization"""
        if timestamps is None:
            timestamps = np.arange(len(positions))
        
        self.trajectories[name] = {
            "positions": positions,
            "timestamps": timestamps
        }
    
    def create_3d_plot(self) -> go.Figure:
        """Create 3D trajectory plot"""
        self.fig = go.Figure()
        
        for name, traj in self.trajectories.items():
            positions = traj["positions"]
            
            if positions.shape[1] >= 3:
                self.fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines+markers',
                    name=name,
                    line=dict(width=4),
                    marker=dict(size=3)
                ))
        
        self.fig.update_layout(
            title="3D Robot Trajectories",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='cube'
            ),
            showlegend=True
        )
        
        return self.fig
    
    def show_plot(self):
        """Show 3D trajectory plot"""
        if self.fig:
            self.fig.show()
    
    def animate_trajectory(self, name: str, speed: float = 1.0):
        """Create animated trajectory visualization"""
        if name not in self.trajectories:
            return
        
        traj = self.trajectories[name]
        positions = traj["positions"]
        
        if positions.shape[1] < 3:
            return
        
        # Create frames for animation
        frames = []
        for i in range(len(positions)):
            frame_data = go.Scatter3d(
                x=positions[:i+1, 0],
                y=positions[:i+1, 1],
                z=positions[:i+1, 2],
                mode='lines+markers',
                name=name,
                line=dict(width=4),
                marker=dict(size=3)
            )
            frames.append(go.Frame(data=[frame_data]))
        
        fig = go.Figure(
            data=[go.Scatter3d(x=[], y=[], z=[], mode='lines+markers')],
            frames=frames
        )
        
        fig.update_layout(
            title=f"Animated Trajectory: {name}",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='cube'
            ),
            updatemenus=[{
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 50}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}}]}
                ],
                "direction": "left",
                "type": "buttons"
            }]
        )
        
        fig.show()


# Factory functions for common visualization setups
def create_robot_monitor(robot_type: str, model_id: str) -> RobotStateMonitor:
    """Create robot state monitor for specific robot type"""
    client = MuJoCoViewerClient()
    monitor = RobotStateMonitor(client)
    
    if client.connect():
        monitor.start_monitoring(model_id)
    
    return monitor


def create_performance_dashboard(models: List[str]) -> InteractiveVisualizer:
    """Create performance dashboard for multiple models"""
    dashboard = InteractiveVisualizer()
    dashboard.create_dashboard("Multi-Robot Performance Dashboard")
    
    for i, model in enumerate(models):
        row = (i // 2) + 1
        col = (i % 2) + 1
        dashboard.add_data_stream(f"{model}_pos", "position", row, col)
        dashboard.add_data_stream(f"{model}_vel", "velocity", row, col)
    
    return dashboard


def analyze_trajectory_file(filename: str) -> Dict[str, Any]:
    """Analyze trajectory data from exported file"""
    filepath = Path(filename)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            states = data.get("states", [])
    elif filepath.suffix == '.npz':
        data = np.load(filepath)
        timestamps = data['timestamps']
        qpos = data['qpos']
        qvel = data['qvel']
        
        states = []
        for i in range(len(timestamps)):
            states.append({
                "timestamp": timestamps[i],
                "state": {"qpos": qpos[i].tolist(), "qvel": qvel[i].tolist()}
            })
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Create temporary monitor for analysis
    monitor = RobotStateMonitor(None)
    monitor.state_history = deque(states)
    
    return monitor.analyze_performance()


# Example usage
if __name__ == "__main__":
    # Example: Create and show robot monitor
    client = MuJoCoViewerClient()
    if client.connect():
        monitor = RobotStateMonitor(client)
        monitor.start_monitoring("test_model")
        monitor.show_joint_positions()
        
        # Let it run for a while
        time.sleep(10)
        
        # Export data and analyze
        monitor.export_data("robot_data.json")
        analysis = monitor.analyze_performance()
        print("Performance Analysis:", analysis)
        
        monitor.stop_monitoring()