import cv2
import numpy as np
import time
from typing import List, Optional
import config

class GolfBotDashboard:
    """Clean dashboard interface for GolfBot with organized data panels - WHITE BALLS ONLY"""
    
    def __init__(self, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Dashboard layout
        self.panel_width = 300
        self.top_panel_height = 100
        self.dashboard_width = camera_width + self.panel_width
        self.dashboard_height = max(camera_height + self.top_panel_height, 600)
        
        # Colors
        self.bg_color = (40, 40, 40)      # Dark gray background
        self.panel_color = (60, 60, 60)    # Slightly lighter panels
        self.text_color = (255, 255, 255)  # White text
        self.accent_color = (0, 255, 255)  # Cyan accent
        self.success_color = (0, 255, 0)   # Green
        self.warning_color = (0, 165, 255) # Orange
        self.danger_color = (0, 0, 255)    # Red
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_large = 0.7
        self.font_scale_medium = 0.5
        self.font_scale_small = 0.4
        
        # Panel positions
        self.camera_x = 10
        self.camera_y = self.top_panel_height + 10
        self.right_panel_x = self.camera_width + 20
        self.right_panel_y = self.top_panel_height + 10
        
        # Initialize dashboard
        self.dashboard = np.full((self.dashboard_height, self.dashboard_width, 3), 
                                self.bg_color, dtype=np.uint8)
    
    def create_dashboard(self, camera_frame, robot_state, vision_system, hardware, telemetry=None):
        """Create complete dashboard with camera and data panels - WHITE BALLS ONLY"""
        
        # Create fresh dashboard
        self.dashboard = np.full((self.dashboard_height, self.dashboard_width, 3), 
                                self.bg_color, dtype=np.uint8)
        
        # 1. Place clean camera preview (already processed by vision system)
        self._place_camera_preview(camera_frame)
        
        # 2. Add top status bar
        self._add_top_status_bar(robot_state, hardware)
        
        # 3. Add right side panels
        self._add_vision_status_panel(vision_system)
        self._add_robot_status_panel(robot_state, hardware)
        self._add_detection_details_panel(vision_system)
        self._add_controls_legend_panel()
        
        # 4. Add performance info if available
        if telemetry:
            self._add_performance_panel(telemetry)
        
        return self.dashboard
    
    def _place_camera_preview(self, camera_frame):
        """Place camera preview in dashboard"""
        if camera_frame is not None and camera_frame.size > 0:
            # Resize if needed
            if camera_frame.shape[:2] != (self.camera_height, self.camera_width):
                camera_frame = cv2.resize(camera_frame, (self.camera_width, self.camera_height))
            
            # Place in dashboard
            end_y = self.camera_y + self.camera_height
            end_x = self.camera_x + self.camera_width
            self.dashboard[self.camera_y:end_y, self.camera_x:end_x] = camera_frame
            
            # Add border around camera
            cv2.rectangle(self.dashboard, 
                         (self.camera_x-2, self.camera_y-2), 
                         (end_x+1, end_y+1), 
                         self.accent_color, 2)
    
    def _add_top_status_bar(self, robot_state, hardware):
        """Add top status bar with critical info - WHITE BALLS ONLY"""
        # Background
        cv2.rectangle(self.dashboard, (0, 0), (self.dashboard_width, self.top_panel_height), 
                     self.panel_color, -1)
        
        # Title
        cv2.putText(self.dashboard, "GolfBot White Ball Collection Dashboard", 
                   (10, 30), self.font, self.font_scale_large, self.accent_color, 2)
        
        # Time and state info
        y_pos = 60
        
        # Time (if available)
        time_text = f"Running: {time.strftime('%H:%M:%S')}"
        cv2.putText(self.dashboard, time_text, (10, y_pos), 
                   self.font, self.font_scale_medium, self.text_color, 1)
        
        # Current state
        state_text = f"State: {robot_state.value.replace('_', ' ').title()}"
        state_color = self._get_state_color(robot_state)
        cv2.putText(self.dashboard, state_text, (200, y_pos), 
                   self.font, self.font_scale_medium, state_color, 1)
        
        # White balls collected
        ball_count = hardware.get_ball_count() if hardware else 0
        balls_text = f"White Balls: {ball_count}"
        cv2.putText(self.dashboard, balls_text, (400, y_pos), 
                   self.font, self.font_scale_medium, self.success_color, 1)
        
        # Emergency stop indicator
        cv2.putText(self.dashboard, "Press 'Q' to quit", (self.dashboard_width - 150, y_pos), 
                   self.font, self.font_scale_small, (180, 180, 180), 1)
    
    def _add_vision_status_panel(self, vision_system):
        """Add vision system status panel - WHITE BALLS ONLY"""
        panel_y = self.right_panel_y
        panel_height = 120
        
        # Panel background
        cv2.rectangle(self.dashboard, 
                     (self.right_panel_x, panel_y), 
                     (self.right_panel_x + self.panel_width, panel_y + panel_height), 
                     self.panel_color, -1)
        
        # Panel title
        cv2.putText(self.dashboard, "VISION STATUS", 
                   (self.right_panel_x + 5, panel_y + 20), 
                   self.font, self.font_scale_medium, self.accent_color, 1)
        
        y = panel_y + 40
        line_height = 18
        
        # Arena detection
        arena_status = "Detected" if getattr(vision_system, 'arena_detected', False) else "Fallback"
        arena_color = self.success_color if arena_status == "Detected" else self.warning_color
        cv2.putText(self.dashboard, f"Arena: {arena_status}", 
                   (self.right_panel_x + 5, y), self.font, self.font_scale_small, arena_color, 1)
        y += line_height
        
        # Detection method
        cv2.putText(self.dashboard, "Method: HoughCircles+Color", 
                   (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
        y += line_height
        
        # Current target (WHITE BALLS ONLY)
        if hasattr(vision_system, 'current_target') and vision_system.current_target:
            target = vision_system.current_target
            cv2.putText(self.dashboard, f"Target: WHITE BALL", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.warning_color, 1)
            y += line_height
            
            # Confidence
            cv2.putText(self.dashboard, f"Confidence: {target.confidence:.2f}", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
        else:
            cv2.putText(self.dashboard, "Target: SEARCHING", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, (128, 128, 128), 1)
    
        # In dashboard.py, find this section in _add_robot_status_panel method and replace it:

    def _add_robot_status_panel(self, robot_state, hardware):
        """Add robot hardware status panel"""
        panel_y = self.right_panel_y + 130
        panel_height = 140
        
        # Panel background
        cv2.rectangle(self.dashboard, 
                    (self.right_panel_x, panel_y), 
                    (self.right_panel_x + self.panel_width, panel_y + panel_height), 
                    self.panel_color, -1)
        
        # Panel title
        cv2.putText(self.dashboard, "ROBOT STATUS", 
                (self.right_panel_x + 5, panel_y + 20), 
                self.font, self.font_scale_medium, self.accent_color, 1)
        
        y = panel_y + 40
        line_height = 18
        
        # Current speed
        speed = getattr(hardware, 'current_speed', 0) if hardware else 0
        cv2.putText(self.dashboard, f"Speed: {speed*100:.0f}%", 
                (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
        y += line_height
        
        # Servo status - FIXED TO USE CORRECT NAMES
        if hardware and hasattr(hardware, 'get_servo_angles'):
            angles = hardware.get_servo_angles()
            # Use the correct keys: 'servo_ss' and 'servo_sf'
            ss_angle = angles.get('servo_ss', 90)
            sf_angle = angles.get('servo_sf', 90)
            cv2.putText(self.dashboard, f"Servos: SS {ss_angle:.0f}° SF {sf_angle:.0f}°", 
                    (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
        else:
            cv2.putText(self.dashboard, "Servos: N/A", 
                    (self.right_panel_x + 5, y), self.font, self.font_scale_small, (128, 128, 128), 1)
        y += line_height
        
        # Collection status (WHITE BALLS ONLY)
        ball_count = hardware.get_ball_count() if hardware else 0
        cv2.putText(self.dashboard, f"White Balls Collected: {ball_count}", 
                (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.success_color, 1)
        y += line_height
        
        # State details
        state_details = self._get_state_details(robot_state)
        cv2.putText(self.dashboard, f"Action: {state_details}", 
                (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
    
    def _add_detection_details_panel(self, vision_system):
        """Add detailed detection information panel - WHITE BALLS ONLY"""
        panel_y = self.right_panel_y + 280
        panel_height = 120
        
        # Panel background
        cv2.rectangle(self.dashboard, 
                     (self.right_panel_x, panel_y), 
                     (self.right_panel_x + self.panel_width, panel_y + panel_height), 
                     self.panel_color, -1)
        
        # Panel title
        cv2.putText(self.dashboard, "DETECTION DETAILS", 
                   (self.right_panel_x + 5, panel_y + 20), 
                   self.font, self.font_scale_medium, self.accent_color, 1)
        
        y = panel_y + 40
        line_height = 16
        
        # Ball count (WHITE BALLS ONLY)
        if hasattr(vision_system, '_last_detected_balls'):
            balls = getattr(vision_system, '_last_detected_balls', [])
            white_count = len(balls)  # All balls are white now
            
            cv2.putText(self.dashboard, f"White Balls Found: {white_count}", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
            y += line_height * 2
        else:
            cv2.putText(self.dashboard, "White Balls Found: 0", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, (128, 128, 128), 1)
            y += line_height * 2
        
        # Centering info (both X and Y) - Updated for more lenient tolerances
        x_tolerance = getattr(config, 'CENTERING_TOLERANCE', 25)
        y_tolerance = getattr(config, 'CENTERING_DISTANCE_TOLERANCE', 30)
        cv2.putText(self.dashboard, f"Centering: ±{x_tolerance}px X, ±{y_tolerance}px Y", 
                   (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.accent_color, 1)
        y += line_height
        
        # Wall detection
        wall_count = 0
        if hasattr(vision_system, 'detected_walls'):
            triggered_walls = [w for w in vision_system.detected_walls if w.get('triggered', False)]
            wall_count = len(triggered_walls)
        
        wall_color = self.danger_color if wall_count > 0 else self.success_color
        wall_status = "DANGER" if wall_count > 0 else "SAFE"
        cv2.putText(self.dashboard, f"Walls: {wall_status} ({wall_count})", 
                   (self.right_panel_x + 5, y), self.font, self.font_scale_small, wall_color, 1)
    
    def _add_controls_legend_panel(self):
        """Add controls and legend panel - WHITE BALLS ONLY"""
        panel_y = self.right_panel_y + 410
        panel_height = 100
        
        # Panel background
        cv2.rectangle(self.dashboard, 
                     (self.right_panel_x, panel_y), 
                     (self.right_panel_x + self.panel_width, panel_y + panel_height), 
                     self.panel_color, -1)
        
        # Panel title
        cv2.putText(self.dashboard, "LEGEND", 
                   (self.right_panel_x + 5, panel_y + 20), 
                   self.font, self.font_scale_medium, self.accent_color, 1)
        
        y = panel_y + 40
        line_height = 14
        
        # Legend items (WHITE BALLS ONLY)
        legend_items = [
            ("Green Zone: Collection area", self.success_color),
            ("Cyan Lines: Centering tolerance (X+Y)", self.accent_color),
            ("Cyan Arrow: Target direction", self.accent_color),
            ("B: White ball", self.text_color),
            ("Red Outline: Wall danger", self.danger_color),
        ]
        
        for item, color in legend_items:
            cv2.putText(self.dashboard, item, 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small-0.1, color, 1)
            y += line_height
    
    def _add_performance_panel(self, telemetry):
        """Add performance metrics panel"""
        panel_y = self.right_panel_y + 520
        panel_height = 60
        
        # Panel background
        cv2.rectangle(self.dashboard, 
                     (self.right_panel_x, panel_y), 
                     (self.right_panel_x + self.panel_width, panel_y + panel_height), 
                     self.panel_color, -1)
        
        # Panel title
        cv2.putText(self.dashboard, "PERFORMANCE", 
                   (self.right_panel_x + 5, panel_y + 20), 
                   self.font, self.font_scale_medium, self.accent_color, 1)
        
        y = panel_y + 40
        
        # Frame count and rate
        if hasattr(telemetry, 'frame_count'):
            cv2.putText(self.dashboard, f"Frames: {telemetry.frame_count}", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
            
            cv2.putText(self.dashboard, f"Collections: {telemetry.total_collections}", 
                       (self.right_panel_x + 120, y), self.font, self.font_scale_small, self.success_color, 1)
    
    def _get_state_color(self, robot_state):
        """Get color for robot state"""
        state_colors = {
            'SEARCHING': self.accent_color,
            'CENTERING_BALL': self.warning_color,
            'APPROACHING_BALL': self.success_color,
            'COLLECTING_BALL': self.success_color,
            'BLIND_COLLECTION': self.warning_color,
            'AVOIDING_BOUNDARY': self.danger_color,
            'EMERGENCY_STOP': self.danger_color,
        }
        return state_colors.get(robot_state.value.upper(), self.text_color)
    
    def _get_state_details(self, robot_state):
        """Get detailed description of current state"""
        state_details = {
            'SEARCHING': "Looking for white balls",
            'CENTERING_BALL': "Aligning X+Y for collection",
            'APPROACHING_BALL': "Moving toward target",
            'COLLECTING_BALL': "Enhanced collection sequence",
            'AVOIDING_BOUNDARY': "Avoiding walls",
            'EMERGENCY_STOP': "System stopped",
        }
        return state_details.get(robot_state.value.upper(), "Unknown")
    
    def show(self, window_name="GolfBot Dashboard"):
        """Display the dashboard"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.dashboard)
        return cv2.waitKey(1) & 0xFF
    
    def save_screenshot(self, filename=None):
        """Save dashboard screenshot"""
        if filename is None:
            filename = f"dashboard_{int(time.time())}.png"
        cv2.imwrite(filename, self.dashboard)
        return filename