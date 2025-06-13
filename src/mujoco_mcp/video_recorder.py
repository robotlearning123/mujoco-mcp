"""
Video recording utility for MuJoCo MCP
Records simulation by capturing frames at regular intervals
"""

import os
import time
import base64
import subprocess
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile
import shutil

class VideoRecorder:
    """Records MuJoCo simulation as video"""
    
    def __init__(self, server, model_id: str, fps: int = 30):
        """
        Initialize video recorder
        
        Args:
            server: MuJoCo remote server instance
            model_id: Model to record
            fps: Frames per second for output video
        """
        self.server = server
        self.model_id = model_id
        self.fps = fps
        self.frames = []
        self.temp_dir = None
        self.recording = False
        self.frame_count = 0
        
    def start_recording(self):
        """Start recording frames"""
        if self.recording:
            return {"success": False, "error": "Already recording"}
            
        # Create temporary directory for frames
        self.temp_dir = tempfile.mkdtemp(prefix="mujoco_recording_")
        self.frames = []
        self.frame_count = 0
        self.recording = True
        
        return {"success": True, "message": f"Recording started in {self.temp_dir}"}
    
    def capture_frame(self) -> bool:
        """Capture a single frame"""
        if not self.recording:
            return False
            
        # Capture render
        result = self.server.call_tool("capture_render", {
            "model_id": self.model_id,
            "width": 640,
            "height": 480
        })
        
        if result.get("success"):
            image_data = result.get("image_data")
            if image_data:
                # Save frame to temp directory
                frame_path = os.path.join(self.temp_dir, f"frame_{self.frame_count:06d}.png")
                with open(frame_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                
                self.frames.append(frame_path)
                self.frame_count += 1
                return True
                
        return False
    
    def stop_recording(self, output_path: str = "simulation.mp4") -> Dict[str, Any]:
        """Stop recording and create video"""
        if not self.recording:
            return {"success": False, "error": "Not recording"}
            
        self.recording = False
        
        if len(self.frames) == 0:
            return {"success": False, "error": "No frames captured"}
        
        # Check if ffmpeg is available
        if not self._check_ffmpeg():
            # Fallback: create animated GIF using PIL
            return self._create_gif(output_path.replace('.mp4', '.gif'))
        
        # Create video using ffmpeg
        return self._create_video_ffmpeg(output_path)
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except:
            return False
    
    def _create_video_ffmpeg(self, output_path: str) -> Dict[str, Any]:
        """Create video using ffmpeg"""
        try:
            # FFmpeg command to create video from frames
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-framerate', str(self.fps),
                '-i', os.path.join(self.temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',  # Quality (lower = better)
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Clean up temp directory
                shutil.rmtree(self.temp_dir)
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "frame_count": self.frame_count,
                    "duration": self.frame_count / self.fps,
                    "format": "mp4"
                }
            else:
                return {
                    "success": False,
                    "error": f"FFmpeg failed: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create video: {str(e)}"
            }
    
    def _create_gif(self, output_path: str) -> Dict[str, Any]:
        """Create animated GIF as fallback"""
        try:
            from PIL import Image
            
            # Load all frames
            images = []
            for frame_path in self.frames:
                images.append(Image.open(frame_path))
            
            # Save as animated GIF
            if images:
                images[0].save(
                    output_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=1000 // self.fps,  # Duration in milliseconds
                    loop=0
                )
                
                # Clean up temp directory
                shutil.rmtree(self.temp_dir)
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "frame_count": self.frame_count,
                    "duration": self.frame_count / self.fps,
                    "format": "gif"
                }
            else:
                return {
                    "success": False,
                    "error": "No images to create GIF"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create GIF: {str(e)}"
            }
    
    def record_simulation(self, duration: float, step_size: int = 1, 
                         output_path: str = "simulation.mp4") -> Dict[str, Any]:
        """
        Record a complete simulation
        
        Args:
            duration: Duration in seconds
            step_size: Simulation steps between frames
            output_path: Output video path
        """
        # Start recording
        start_result = self.start_recording()
        if not start_result.get("success"):
            return start_result
        
        # Calculate frame timing
        frame_interval = 1.0 / self.fps
        start_time = time.time()
        last_frame_time = start_time
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Capture frame if enough time has passed
                if current_time - last_frame_time >= frame_interval:
                    if self.capture_frame():
                        last_frame_time = current_time
                    else:
                        print(f"Warning: Failed to capture frame {self.frame_count}")
                
                # Step simulation
                self.server.call_tool("step_simulation", {
                    "model_id": self.model_id,
                    "steps": step_size
                })
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
            
            # Stop recording and create video
            return self.stop_recording(output_path)
            
        except Exception as e:
            self.recording = False
            return {
                "success": False,
                "error": f"Recording failed: {str(e)}"
            }


def create_video_from_simulation(server, model_id: str, duration: float = 5.0,
                               output_path: str = "simulation.mp4") -> Dict[str, Any]:
    """
    Convenience function to record a video
    
    Args:
        server: MuJoCo remote server instance
        model_id: Model to record
        duration: Recording duration in seconds
        output_path: Output video path
    """
    recorder = VideoRecorder(server, model_id)
    return recorder.record_simulation(duration, output_path=output_path)