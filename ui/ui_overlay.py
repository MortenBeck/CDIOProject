"""
Status overlay utilities for GolfBot
"""

import cv2
import config

class StatusOverlay:
    """Utility class for adding status overlays to video frames"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_large = 0.7
        self.font_scale_medium = 0.6
        self.font_scale_small = 0.5
        
        # Colors
        self.text_color = (0, 255, 255)  # Cyan
        self.success_color = (0, 255, 0)  # Green
        self.warning_color = (0, 165, 255)  # Orange
        self.danger_color = (0, 0, 255)  # Red
        self.white_color = (255, 255, 255)  # White
    
    def add_competition_status(self, frame, competition_manager, y_start=30):
        """Add competition status to frame"""
        y = y_start
        line_height = 25
        
        if competition_manager.is_active:
            remaining = competition_manager.get_time_remaining()
            elapsed = competition_manager.get_elapsed_time()
            
            # Time remaining
            cv2.putText(frame, f"Time: {remaining:.0f}s remaining", (10, y), 
                       self.font, self.font_scale_large, self.text_color, 2)
            y += line_height
            
            # Endgame warning
            if competition_manager.should_enter_endgame_mode():
                cv2.putText(frame, "ENDGAME MODE", (10, y), 
                           self.font, self.font_scale_medium, self.danger_color, 2)
                y += line_height
        else:
            cv2.putText(frame, "Competition Not Active", (10, y), 
                       self.font, self.font_scale_large, self.warning_color, 2)
            y += line_height
        
        return y
    
    def add_state_status(self, frame, state_machine, vision_system, y_start):
        """Add state machine status to frame"""
        y = y_start
        line_height = 25
        
        current_state = state_machine.get_current_state()
        state_text = f"State: {current_state.value.replace('_', ' ').title()}"
        
        # Add phase-specific info
        if current_state.value == 'CENTERING_1':
            state_text += " (X+Y Align)"
        elif current_state.value == 'CENTERING_2':
            state_text += " (Zone Position)"
        elif current_state.value == 'COLLECTING_BALL':
            state_text += " (Optimized)"
        
        # Add centering progress if applicable
        if (current_state.value in ['CENTERING_1', 'CENTERING_2'] and 
            vision_system.current_target):
            
            if current_state.value == 'CENTERING_1':
                x_offset = abs(vision_system.current_target.center[0] - vision_system.frame_center_x)
                y_offset = abs(vision_system.current_target.center[1] - vision_system.frame_center_y)
                x_ok = x_offset <= config.CENTERING_1_TOLERANCE
                y_ok = y_offset <= config.CENTERING_1_DISTANCE_TOLERANCE
                status_char = f"{'✓' if x_ok else 'X'}{'✓' if y_ok else 'Y'}"
                state_text += f" ({status_char})"
            elif current_state.value == 'CENTERING_2':
                in_zone = vision_system.is_in_collection_zone(vision_system.current_target.center)
                state_text += f" ({'✓' if in_zone else '→'})"
        
        # Color based on state
        state_color = self._get_state_color(current_state.value)
        cv2.putText(frame, state_text, (10, y), 
                   self.font, self.font_scale_large, state_color, 2)
        y += line_height
        
        return y
    
    def add_hardware_status(self, frame, hardware, y_start):
        """Add hardware status to frame"""
        y = y_start
        line_height = 25
        
        # Ball count
        ball_count = hardware.get_ball_count()
        cv2.putText(frame, f"Balls Collected: {ball_count}", (10, y), 
                   self.font, self.font_scale_large, self.success_color, 2)
        y += line_height
        
        # Servo status
        servo_angles = hardware.get_servo_angles()
        servo_ss_state = hardware.get_servo_ss_state()
        servo_text = f"Servos: SS={servo_angles['servo_ss']:.0f}° ({servo_ss_state}) SF={servo_angles['servo_sf']:.0f}°"
        cv2.putText(frame, servo_text, (10, y), 
                   self.font, self.font_scale_small, self.white_color, 1)
        y += line_height
        
        return y
    
    def add_vision_status(self, frame, vision_system, y_start):
        """Add vision system status to frame"""
        y = y_start
        line_height = 20
        
        # Arena detection status
        arena_status = "Detected" if vision_system.arena_detected else "Fallback"
        arena_color = self.success_color if vision_system.arena_detected else self.warning_color
        cv2.putText(frame, f"Arena: {arena_status}", (10, y), 
                   self.font, self.font_scale_medium, arena_color, 1)
        y += line_height
        
        # Current target info
        if vision_system.current_target:
            target = vision_system.current_target
            ball_type = "ORANGE" if target.object_type == 'orange_ball' else "WHITE"
            target_text = f"Target: {ball_type} (conf: {target.confidence:.2f})"
            cv2.putText(frame, target_text, (10, y), 
                       self.font, self.font_scale_small, self.warning_color, 1)
        else:
            cv2.putText(frame, "Target: SEARCHING", (10, y), 
                       self.font, self.font_scale_small, self.white_color, 1)
        y += line_height
        
        # Collection system info
        cv2.putText(frame, "Collection: Two-Phase (C1→C2→Collect)", (10, y), 
                   self.font, self.font_scale_small, self.white_color, 1)
        y += line_height
        
        return y
    
    def add_full_status_overlay(self, frame, competition_manager, state_machine, 
                              hardware, vision_system):
        """Add complete status overlay to frame"""
        y = 30
        
        # Competition status
        y = self.add_competition_status(frame, competition_manager, y)
        
        # State status
        y = self.add_state_status(frame, state_machine, vision_system, y)
        
        # Hardware status
        y = self.add_hardware_status(frame, hardware, y)
        
        # Vision status
        y = self.add_vision_status(frame, vision_system, y)
        
        return frame
    
    def _get_state_color(self, state_value):
        """Get color for robot state"""
        state_colors = {
            'SEARCHING': self.text_color,
            'CENTERING_1': self.warning_color,
            'CENTERING_2': self.warning_color,
            'COLLECTING_BALL': self.success_color,
            'AVOIDING_BOUNDARY': self.danger_color,
            'EMERGENCY_STOP': self.danger_color,
        }
        return state_colors.get(state_value.upper(), self.white_color)
