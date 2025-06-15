import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

class TelemetryLogger:
    """Real-time telemetry logging for GolfBot debugging and optimization"""
    
    def __init__(self, session_name: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Create session directory
        if session_name is None:
            session_name = datetime.now().strftime("golfbot_%Y%m%d_%H%M%S")
        
        self.session_dir = Path("logs") / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.telemetry_file = self.session_dir / "telemetry.jsonl"
        self.summary_file = self.session_dir / "session_summary.json"
        
        # Data buffers
        self.telemetry_buffer = []
        self.buffer_lock = threading.Lock()
        self.session_start = time.time()
        
        # Performance tracking
        self.frame_count = 0
        self.total_balls_detected = 0
        self.total_collections = 0
        self.errors_logged = 0
        
        self.logger.info(f"Telemetry session started: {session_name}")
    
    def log_frame_data(self, 
                      ball_positions: List[Tuple[int, int]] = None,
                      robot_position: Tuple[float, float, float] = None,
                      servo_states: Dict[str, int] = None,
                      motor_speeds: Dict[str, float] = None,
                      camera_fps: float = None,
                      power_levels: Dict[str, float] = None,
                      action: str = None,
                      errors: List[str] = None,
                      extra_data: Dict[str, Any] = None):
        """Log a complete frame of telemetry data"""
        
        timestamp = datetime.now().isoformat()
        
        # Build telemetry record
        record = {
            "timestamp": timestamp,
            "frame_number": self.frame_count,
            "ball_positions": ball_positions or [],
            "robot_position": robot_position,
            "servo_states": servo_states or {},
            "motor_speeds": motor_speeds or {},
            "camera_fps": camera_fps,
            "power_levels": power_levels or {},
            "action": action,
            "errors": errors or []
        }
        
        # Add any extra data
        if extra_data:
            record.update(extra_data)
        
        # Update statistics
        self.frame_count += 1
        if ball_positions:
            self.total_balls_detected += len(ball_positions)
        if errors:
            self.errors_logged += len(errors)
        
        # Thread-safe buffer append
        with self.buffer_lock:
            self.telemetry_buffer.append(record)
        
        # Write to file periodically
        if len(self.telemetry_buffer) >= 10:
            self._flush_buffer()
    
    def log_ball_detection(self, balls, orange_ball=None, goals=None):
        """Log vision detection results"""
        ball_positions = [(ball.center[0], ball.center[1]) for ball in balls] if balls else []
        
        extra_data = {
            "ball_count": len(balls) if balls else 0,
            "orange_ball_detected": orange_ball is not None,
            "goals_detected": len(goals) if goals else 0,
            "ball_details": []
        }
        
        # Add detailed ball info
        if balls:
            for i, ball in enumerate(balls):
                extra_data["ball_details"].append({
                    "id": i,
                    "center": ball.center,
                    "radius": ball.radius,
                    "area": ball.area,
                    "confidence": ball.confidence,
                    "distance_from_center": ball.distance_from_center
                })
        
        if orange_ball:
            extra_data["orange_ball_details"] = {
                "center": orange_ball.center,
                "radius": orange_ball.radius,
                "area": orange_ball.area,
                "confidence": orange_ball.confidence,
                "distance_from_center": orange_ball.distance_from_center
            }
        
        self.log_frame_data(
            ball_positions=ball_positions,
            action="vision_detection",
            extra_data=extra_data
        )
    
    def log_hardware_state(self, hardware):
        """Log current hardware state"""
        # Get servo angles directly from PCA9685 servos
        servo_angles = hardware.get_servo_angles()
        servo_states = {
            "s1": int(servo_angles.get("servo1", 90)),
            "s2": int(servo_angles.get("servo2", 90)),
            "s3": int(servo_angles.get("servo3", 90))
        }
        
        motor_speeds = {
            "in1": hardware.motor_in1.value if hardware.motor_in1.is_active else 0,
            "in2": hardware.motor_in2.value if hardware.motor_in2.is_active else 0,
            "in3": hardware.motor_in3.value if hardware.motor_in3.is_active else 0,
            "in4": hardware.motor_in4.value if hardware.motor_in4.is_active else 0,
            "current_speed": hardware.current_speed
        }
        
        extra_data = {
            "collected_balls": hardware.get_ball_count(),
            "has_balls": hardware.has_balls()
        }
        
        self.log_frame_data(
            servo_states=servo_states,
            motor_speeds=motor_speeds,
            action="hardware_state",
            extra_data=extra_data
        )
    
    def log_state_transition(self, old_state, new_state, reason=""):
        """Log robot state changes"""
        self.log_frame_data(
            action=f"state_change_{old_state.value}_to_{new_state.value}",
            extra_data={
                "old_state": old_state.value,
                "new_state": new_state.value,
                "reason": reason
            }
        )
    
    def log_collection_attempt(self, success: bool, ball_type: str = "regular"):
        """Log ball collection attempts"""
        if success:
            self.total_collections += 1
        
        self.log_frame_data(
            action=f"collection_{ball_type}_{'success' if success else 'failed'}",
            extra_data={
                "collection_success": success,
                "ball_type": ball_type,
                "total_collections": self.total_collections
            }
        )
    
    def log_delivery_attempt(self, balls_delivered: int, goal_type: str):
        """Log ball delivery attempts"""
        points = balls_delivered * (150 if goal_type == "A" else 100)
        
        self.log_frame_data(
            action=f"delivery_goal_{goal_type}",
            extra_data={
                "balls_delivered": balls_delivered,
                "goal_type": goal_type,
                "points_earned": points
            }
        )
    
    def log_error(self, error_msg: str, error_type: str = "general"):
        """Log error events"""
        self.log_frame_data(
            action=f"error_{error_type}",
            errors=[error_msg],
            extra_data={"error_type": error_type}
        )
    
    def log_performance_metrics(self, fps: float = None, processing_time: float = None):
        """Log performance metrics"""
        self.log_frame_data(
            camera_fps=fps,
            action="performance_metrics",
            extra_data={
                "processing_time_ms": processing_time * 1000 if processing_time else None,
                "frames_processed": self.frame_count
            }
        )
    
    def _flush_buffer(self):
        """Write buffer to file"""
        try:
            with self.buffer_lock:
                if not self.telemetry_buffer:
                    return
                
                # Append to JSONL file (one JSON object per line)
                with open(self.telemetry_file, 'a') as f:
                    for record in self.telemetry_buffer:
                        f.write(json.dumps(record) + '\n')
                
                self.telemetry_buffer.clear()
                
        except Exception as e:
            self.logger.error(f"Failed to flush telemetry buffer: {e}")
    
    def create_session_summary(self, competition_result: Dict[str, Any] = None):
        """Create session summary for analysis"""
        session_duration = time.time() - self.session_start
        
        summary = {
            "session_metadata": {
                "session_dir": str(self.session_dir),
                "start_time": datetime.fromtimestamp(self.session_start).isoformat(),
                "duration_seconds": session_duration,
                "total_frames": self.frame_count,
                "average_fps": self.frame_count / session_duration if session_duration > 0 else 0
            },
            "detection_stats": {
                "total_balls_detected": self.total_balls_detected,
                "total_collections": self.total_collections,
                "collection_rate": self.total_collections / max(1, self.total_balls_detected),
                "errors_logged": self.errors_logged
            },
            "competition_result": competition_result or {},
            "files": {
                "telemetry_log": str(self.telemetry_file),
                "summary": str(self.summary_file)
            }
        }
        
        # Write summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_latest_logs(self, count: int = 10) -> List[Dict]:
        """Get the latest log entries for quick debugging"""
        with self.buffer_lock:
            recent_logs = self.telemetry_buffer[-count:] if self.telemetry_buffer else []
        
        # Also try to read from file if buffer is empty
        if not recent_logs and self.telemetry_file.exists():
            try:
                with open(self.telemetry_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-count:]:
                        recent_logs.append(json.loads(line.strip()))
            except:
                pass
        
        return recent_logs
    
    def export_for_analysis(self, output_file: str = None):
        """Export data in format suitable for analysis"""
        if output_file is None:
            output_file = self.session_dir / "analysis_export.json"
        
        # Flush any remaining buffer
        self._flush_buffer()
        
        # Read all telemetry data
        all_data = []
        if self.telemetry_file.exists():
            with open(self.telemetry_file, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line.strip()))
        
        # Create analysis-friendly export
        export_data = {
            "session_summary": self.create_session_summary(),
            "telemetry_data": all_data,
            "analysis_ready": True,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Telemetry exported to: {output_file}")
        print(f"📊 Total frames: {len(all_data)}")
        print(f"🎯 Total balls detected: {self.total_balls_detected}")
        print(f"🤖 Total collections: {self.total_collections}")
        print(f"❌ Total errors: {self.errors_logged}")
        
        return str(output_file)
    
    def cleanup(self):
        """Final cleanup and summary generation"""
        self._flush_buffer()
        summary = self.create_session_summary()
        
        self.logger.info(f"Telemetry session completed:")
        self.logger.info(f"  - Frames logged: {self.frame_count}")
        self.logger.info(f"  - Balls detected: {self.total_balls_detected}")
        self.logger.info(f"  - Collections: {self.total_collections}")
        self.logger.info(f"  - Errors: {self.errors_logged}")
        self.logger.info(f"  - Session data: {self.session_dir}")
        
        return summary