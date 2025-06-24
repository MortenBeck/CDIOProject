import cv2
import numpy as np
import time
from typing import List, Optional
import config

class GolfBotDashboard:
    """Clean dashboard interface for GolfBot with delivery cycle tracking + Wall Danger Visualization - WHITE BALLS ONLY"""
    
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
        self.delivery_color = (255, 0, 255) # Magenta for delivery
        self.wall_safe_color = (0, 100, 0)   # Dark green for safe zones
        self.wall_danger_color = (0, 0, 200) # Dark red for danger zones
        
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
    
    def create_dashboard(self, camera_frame, robot_state, vision_system, hardware):
        """Create complete dashboard with camera and data panels + delivery cycle + Wall Danger Zones - WHITE BALLS ONLY"""
        
        # Create fresh dashboard
        self.dashboard = np.full((self.dashboard_height, self.dashboard_width, 3), 
                                self.bg_color, dtype=np.uint8)
        
        # 1. Add wall danger zone overlays to camera frame FIRST
        enhanced_camera_frame = self._add_wall_danger_overlays(camera_frame, vision_system)
        
        # 2. Place enhanced camera preview
        self._place_camera_preview(enhanced_camera_frame)
        
        # 3. Add top status bar with delivery cycle
        self._add_top_status_bar(robot_state, hardware)
        
        # 4. Add right side panels
        self._add_vision_status_panel(vision_system)
        self._add_robot_status_panel(robot_state, hardware)
        self._add_detection_details_panel(vision_system)
        self._add_delivery_cycle_panel(hardware, robot_state)
        self._add_wall_danger_panel(vision_system)  # NEW: Wall danger status panel
        self._add_controls_legend_panel()
        
        return self.dashboard
    
    def _add_wall_danger_overlays(self, camera_frame, vision_system):
        """Add visual wall danger zone overlays to camera frame"""
        if camera_frame is None:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        result = camera_frame.copy()
        h, w = result.shape[:2]
        
        # Get boundary system status
        boundary_system = vision_system.boundary_system
        boundary_status = boundary_system.get_status()
        walls_detected = boundary_status.get('walls_triggered', 0) > 0
        
        # Calculate danger zones (same logic as boundary_avoidance.py)
        target_zone_center_y = int(h * config.TARGET_ZONE_VERTICAL_POSITION)  # 65% down from top
        target_zone_height = getattr(config, 'TARGET_ZONE_HEIGHT', 45)
        target_zone_top_y = target_zone_center_y - (target_zone_height // 2)
        
        danger_start_y = target_zone_top_y  # Start detection from TOP of target zone
        danger_end_y = int(h * 0.95)  # Go almost to bottom (95%)
        
        # Zone colors based on danger status
        zone_color = self.wall_danger_color if walls_detected else self.wall_safe_color
        zone_alpha = 0.4 if walls_detected else 0.15
        
        # Create overlay for danger zones
        overlay = result.copy()
        
        # === ZONE 1: BOTTOM WALL DETECTION AREA ===
        # Exclude edge areas (10% margin from each side)
        bottom_left_margin = int(w * 0.1)
        bottom_right_margin = int(w * 0.1)
        bottom_detection_left = bottom_left_margin
        bottom_detection_right = w - bottom_right_margin
        
        danger_distance_vertical = int(h * 0.1)  # 10% of frame height
        bottom_region_start = max(danger_start_y, danger_end_y - danger_distance_vertical)
        
        # Draw bottom detection zone
        cv2.rectangle(overlay, 
                     (bottom_detection_left, bottom_region_start), 
                     (bottom_detection_right, danger_end_y), 
                     zone_color, -1)
        
        # === ZONE 2 & 3: LEFT AND RIGHT WALL DETECTION AREAS (BOTTOM 20% ONLY) ===
        detection_zone_height = danger_end_y - danger_start_y
        side_detection_height = int(detection_zone_height * 0.20)  # 20% of detection zone
        side_detection_start_y = danger_end_y - side_detection_height  # Start from bottom up
        
        danger_distance_horizontal = int(w * 0.08)  # 8% of frame width
        side_margin_reduction = int(w * 0.05)       # 5% margin from each side
        
        # Left side detection zone
        left_start_x = side_margin_reduction
        left_end_x = min(left_start_x + danger_distance_horizontal, w // 2)
        if left_end_x > left_start_x + 20:  # Only draw if reasonable width
            cv2.rectangle(overlay, 
                         (left_start_x, side_detection_start_y), 
                         (left_end_x, danger_end_y), 
                         zone_color, -1)
        
        # Right side detection zone
        right_end_x = w - side_margin_reduction
        right_start_x = max(right_end_x - danger_distance_horizontal, w // 2)
        if right_end_x > right_start_x + 20:  # Only draw if reasonable width
            cv2.rectangle(overlay, 
                         (right_start_x, side_detection_start_y), 
                         (right_end_x, danger_end_y), 
                         zone_color, -1)
        
        # === ZONE 4: CENTER FORWARD WALL DETECTION ===
        center_width = int(w * 0.4)         # Check center 40% of frame width
        center_start_x = int(w * 0.3)       # Start at 30% from left
        
        cv2.rectangle(overlay, 
                     (center_start_x, danger_start_y), 
                     (center_start_x + center_width, danger_end_y), 
                     zone_color, -1)
        
        # Blend overlay with original frame
        result = cv2.addWeighted(result, 1 - zone_alpha, overlay, zone_alpha, 0)
        
        # Add zone boundary lines
        line_color = (0, 0, 255) if walls_detected else (0, 255, 0)
        line_thickness = 3 if walls_detected else 1
        
        # Overall danger zone boundary
        cv2.rectangle(result, (0, danger_start_y), (w, danger_end_y), line_color, line_thickness)
        
        # Zone labels
        label_color = (255, 255, 255)
        if walls_detected:
            cv2.putText(result, "WALL DANGER ZONES - ACTIVE", (10, danger_start_y - 10), 
                       self.font, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(result, "Wall Detection Zones", (10, danger_start_y - 10), 
                       self.font, 0.4, (0, 200, 0), 1)
        
        # Show specific triggered zones
        if walls_detected and hasattr(boundary_system, 'detected_walls'):
            triggered_zones = [wall['zone'] for wall in boundary_system.detected_walls if wall.get('triggered', False)]
            if triggered_zones:
                zone_text = f"TRIGGERED: {', '.join(triggered_zones).upper()}"
                cv2.putText(result, zone_text, (10, danger_start_y + 20), 
                           self.font, 0.4, (0, 0, 255), 1)
        
        return result
    
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
        """Add top status bar with critical info + delivery cycle - WHITE BALLS ONLY"""
        # Background
        cv2.rectangle(self.dashboard, (0, 0), (self.dashboard_width, self.top_panel_height), 
                     self.panel_color, -1)
        
        # Title
        cv2.putText(self.dashboard, "GolfBot Collection + Delivery Cycle Dashboard", 
                   (10, 30), self.font, self.font_scale_large, self.accent_color, 2)
        
        # Time and state info
        y_pos = 60
        
        # Time (if available)
        time_text = f"Running: {time.strftime('%H:%M:%S')}"
        cv2.putText(self.dashboard, time_text, (10, y_pos), 
                   self.font, self.font_scale_medium, self.text_color, 1)
        
        # Current state with delivery emphasis
        state_text = f"State: {robot_state.value.replace('_', ' ').title()}"
        state_color = self._get_state_color(robot_state)
        cv2.putText(self.dashboard, state_text, (200, y_pos), 
                   self.font, self.font_scale_medium, state_color, 1)
        
        # Ball count with delivery progress
        ball_count = hardware.get_ball_count() if hardware else 0
        delivery_target = config.BALLS_BEFORE_DELIVERY
        balls_text = f"Balls: {ball_count}/{delivery_target}"
        
        # Color based on delivery status
        if robot_state.value in ['delivery_mode', 'post_delivery_turn']:
            balls_color = self.delivery_color
        elif ball_count >= delivery_target:
            balls_color = self.success_color
        elif ball_count >= delivery_target - 1:
            balls_color = self.warning_color
        else:
            balls_color = self.text_color
        
        cv2.putText(self.dashboard, balls_text, (400, y_pos), 
                   self.font, self.font_scale_medium, balls_color, 1)
        
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
            
            # Handle None values before formatting
            ss_text = f"{ss_angle:.0f}" if ss_angle is not None else "--"
            sf_text = f"{sf_angle:.0f}" if sf_angle is not None else "--"
            
            cv2.putText(self.dashboard, f"Servos: SS {ss_text}° SF {sf_text}°", 
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
        if hasattr(vision_system, 'boundary_system'):
            boundary_status = vision_system.boundary_system.get_status()
            wall_count = boundary_status.get('walls_triggered', 0)
        
        wall_color = self.danger_color if wall_count > 0 else self.success_color
        wall_status = "DANGER" if wall_count > 0 else "SAFE"
        cv2.putText(self.dashboard, f"Walls: {wall_status} ({wall_count})", 
                   (self.right_panel_x + 5, y), self.font, self.font_scale_small, wall_color, 1)
    
    def _add_delivery_cycle_panel(self, hardware, robot_state):
        """Add delivery cycle progress panel"""
        panel_y = self.right_panel_y + 410
        panel_height = 100
        
        # Panel background
        cv2.rectangle(self.dashboard, 
                     (self.right_panel_x, panel_y), 
                     (self.right_panel_x + self.panel_width, panel_y + panel_height), 
                     self.panel_color, -1)
        
        # Panel title
        cv2.putText(self.dashboard, "DELIVERY CYCLE", 
                   (self.right_panel_x + 5, panel_y + 20), 
                   self.font, self.font_scale_medium, self.accent_color, 1)
        
        y = panel_y + 40
        line_height = 16
        
        # Progress bar
        ball_count = hardware.get_ball_count() if hardware else 0
        delivery_target = config.BALLS_BEFORE_DELIVERY
        progress_ratio = min(1.0, ball_count / delivery_target)
        
        # Progress bar background
        bar_x = self.right_panel_x + 5
        bar_y = y
        bar_width = self.panel_width - 10
        bar_height = 12
        
        cv2.rectangle(self.dashboard, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progress bar fill
        fill_width = int(bar_width * progress_ratio)
        if robot_state.value in ['delivery_mode', 'post_delivery_turn']:
            fill_color = self.delivery_color
        elif ball_count >= delivery_target:
            fill_color = self.success_color
        else:
            fill_color = self.warning_color
        
        if fill_width > 0:
            cv2.rectangle(self.dashboard, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         fill_color, -1)
        
        y += bar_height + 8
        
        # Progress text
        cv2.putText(self.dashboard, f"Progress: {ball_count}/{delivery_target} balls", 
                   (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
        y += line_height
        
        # Cycle status
        if robot_state.value == 'delivery_mode':
            status_text = "DELIVERING BALLS"
            status_color = self.delivery_color
        elif robot_state.value == 'post_delivery_turn':
            status_text = "TURNING TO RESTART"
            status_color = self.delivery_color
        elif ball_count >= delivery_target:
            status_text = "READY FOR DELIVERY"
            status_color = self.success_color
        else:
            remaining = delivery_target - ball_count
            status_text = f"Need {remaining} more balls"
            status_color = self.text_color
        
        cv2.putText(self.dashboard, status_text, 
                   (self.right_panel_x + 5, y), self.font, self.font_scale_small, status_color, 1)
    
    def _add_wall_danger_panel(self, vision_system):
        """NEW: Add wall danger status panel with zone details"""
        panel_y = self.right_panel_y + 520
        panel_height = 100
        
        # Panel background
        cv2.rectangle(self.dashboard, 
                     (self.right_panel_x, panel_y), 
                     (self.right_panel_x + self.panel_width, panel_y + panel_height), 
                     self.panel_color, -1)
        
        # Panel title
        cv2.putText(self.dashboard, "WALL DANGER ZONES", 
                   (self.right_panel_x + 5, panel_y + 20), 
                   self.font, self.font_scale_medium, self.accent_color, 1)
        
        y = panel_y + 40
        line_height = 14
        
        # Get boundary system status
        if hasattr(vision_system, 'boundary_system'):
            boundary_status = vision_system.boundary_system.get_status()
            walls_triggered = boundary_status.get('walls_triggered', 0)
            danger_zones = boundary_status.get('danger_zones', [])
            is_safe = boundary_status.get('safe', True)
            
            # Overall status
            status_color = self.success_color if is_safe else self.danger_color
            status_text = "ALL CLEAR" if is_safe else "WALLS DETECTED"
            cv2.putText(self.dashboard, f"Status: {status_text}", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, status_color, 1)
            y += line_height
            
            # Triggered zones
            if danger_zones:
                zones_text = f"Zones: {', '.join(danger_zones).upper()}"
                cv2.putText(self.dashboard, zones_text, 
                           (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.danger_color, 1)
                y += line_height
            
            # Zone explanations
            cv2.putText(self.dashboard, "Detection Areas:", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, self.text_color, 1)
            y += line_height
            cv2.putText(self.dashboard, "• Bottom (center only)", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small-0.1, (200, 200, 200), 1)
            y += line_height - 2
            cv2.putText(self.dashboard, "• Left/Right (bottom 20%)", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small-0.1, (200, 200, 200), 1)
        else:
            cv2.putText(self.dashboard, "System: Not Available", 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small, (128, 128, 128), 1)
    
    def _add_controls_legend_panel(self):
        """Add controls and legend panel - WHITE BALLS ONLY + Wall Zones"""
        panel_y = self.right_panel_y + 630
        panel_height = 80
        
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
        line_height = 12
        
        # Legend items (WHITE BALLS ONLY + Wall zones)
        legend_items = [
            ("Green Zone: Collection area", self.success_color),
            ("Cyan Lines: Centering tolerance", self.accent_color),
            ("B: White ball", self.text_color),
            ("Red Overlay: Wall danger zones", self.danger_color),
        ]
        
        for item, color in legend_items:
            cv2.putText(self.dashboard, item, 
                       (self.right_panel_x + 5, y), self.font, self.font_scale_small-0.1, color, 1)
            y += line_height
    
    def _get_state_color(self, robot_state):
        """Get color for robot state"""
        state_colors = {
            'SEARCHING': self.accent_color,
            'CENTERING_BALL': self.warning_color,
            'APPROACHING_BALL': self.success_color,
            'COLLECTING_BALL': self.success_color,
            'AVOIDING_BOUNDARY': self.danger_color,
            'DELIVERY_MODE': self.delivery_color,
            'POST_DELIVERY_TURN': self.delivery_color,
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
            'DELIVERY_MODE': "Releasing balls",
            'POST_DELIVERY_TURN': "Turning to restart cycle",
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