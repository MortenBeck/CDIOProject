#!/usr/bin/env python3
"""
GolfBot Delivery System - Green Target Detection and Navigation
Enhanced with oscillation prevention, wall detection, and proper function exports
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import config
from boundary_avoidance import BoundaryAvoidanceSystem

@dataclass
class GreenTarget:
    """Class to store detected green target information"""
    center: Tuple[int, int]
    area: int
    confidence: float
    distance_from_center: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height

class DeliveryVisionSystem:
    """Vision system specifically for green target detection with oscillation prevention and wall detection"""
    
    def __init__(self, vision_system):
        self.logger = logging.getLogger(__name__)
        self.vision = vision_system  # Reuse main vision system for camera
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        
        # Initialize boundary detection system
        self.boundary_system = BoundaryAvoidanceSystem()
        
        # Green detection parameters
        self.green_lower = np.array([40, 50, 50])   # Lower green HSV
        self.green_upper = np.array([80, 255, 255]) # Upper green HSV
        self.min_green_area = 500   # Minimum area for green target
        self.max_green_area = 50000 # Maximum area for green target
        
        # ADAPTIVE CENTERING TOLERANCES with oscillation prevention
        self.base_centering_tolerance_x = 30  # Base X tolerance
        self.base_centering_tolerance_y = 25  # Base Y tolerance
        
        # OSCILLATION DETECTION AND PREVENTION
        self.last_direction = None
        self.direction_change_count = 0
        self.centering_start_time = None
        self.centering_attempts = 0
        self.oscillation_detected = False
        self.stable_frames = 0
        self.min_stable_frames = 2  # Must be stable for 2 frames before proceeding
        
    def get_adaptive_tolerances(self, target: GreenTarget) -> tuple:
        """Get adaptive tolerances based on distance and oscillation state"""
        distance = target.distance_from_center
        
        # Base tolerances based on distance
        if distance > 100:
            # Far away - larger tolerances
            x_tolerance = self.base_centering_tolerance_x + 15
            y_tolerance = self.base_centering_tolerance_y + 10
        elif distance > 60:
            # Medium distance - standard tolerances
            x_tolerance = self.base_centering_tolerance_x
            y_tolerance = self.base_centering_tolerance_y
        else:
            # Close - tighter tolerances
            x_tolerance = self.base_centering_tolerance_x - 5
            y_tolerance = self.base_centering_tolerance_y - 5
        
        # OSCILLATION COMPENSATION - increase tolerances when oscillating
        if self.oscillation_detected or self.direction_change_count >= 3:
            self.logger.info("🔄 Oscillation detected - increasing tolerances")
            x_tolerance = int(x_tolerance * 1.8)  # 80% increase
            y_tolerance = int(y_tolerance * 1.6)  # 60% increase
        
        return max(15, x_tolerance), max(10, y_tolerance)
    
    def is_target_centered(self, target: GreenTarget) -> bool:
        """Check if green target is centered with adaptive tolerances"""
        x_tolerance, y_tolerance = self.get_adaptive_tolerances(target)
        
        x_offset = abs(target.center[0] - self.frame_center_x)
        y_offset = abs(target.center[1] - self.frame_center_y)
        
        centered = (x_offset <= x_tolerance and y_offset <= y_tolerance)
        
        if centered:
            self.stable_frames += 1
            if self.stable_frames >= self.min_stable_frames:
                return True
            else:
                self.logger.info(f"Target centered but verifying stability ({self.stable_frames}/{self.min_stable_frames})")
                return False
        else:
            self.stable_frames = 0
            return False
    
    def get_centering_direction(self, target: GreenTarget) -> tuple:
        """Get direction to center on green target with oscillation detection"""
        x_tolerance, y_tolerance = self.get_adaptive_tolerances(target)
        
        x_offset = target.center[0] - self.frame_center_x
        y_offset = target.center[1] - self.frame_center_y
        
        # Hysteresis to prevent rapid direction changes
        hysteresis = 5 if not self.oscillation_detected else 10
        
        # X-axis (turning) with hysteresis
        if abs(x_offset) <= x_tolerance:
            x_direction = 'centered'
        elif x_offset > (x_tolerance + hysteresis):
            x_direction = 'right'
        elif x_offset < -(x_tolerance + hysteresis):
            x_direction = 'left'
        else:
            x_direction = 'centered'  # In hysteresis zone - don't move
        
        # Y-axis (distance) with hysteresis
        if abs(y_offset) <= y_tolerance:
            y_direction = 'centered'
        elif y_offset > (y_tolerance + hysteresis):
            y_direction = 'backward'
        elif y_offset < -(y_tolerance + hysteresis):
            y_direction = 'forward'
        else:
            y_direction = 'centered'  # In hysteresis zone - don't move
        
        # OSCILLATION DETECTION
        current_direction = x_direction if x_direction != 'centered' else y_direction
        
        if (self.last_direction and 
            current_direction != 'centered' and 
            self.last_direction != 'centered' and
            current_direction != self.last_direction):
            
            self.direction_change_count += 1
            self.logger.info(f"Direction change detected: {self.last_direction} -> {current_direction} (count: {self.direction_change_count})")
            
            if self.direction_change_count >= 3:
                self.oscillation_detected = True
                self.logger.warning("🔄 OSCILLATION DETECTED - applying countermeasures")
        
        self.last_direction = current_direction
        return x_direction, y_direction
    
    def reset_centering_state(self):
        """Reset centering state when starting new target or after success"""
        self.last_direction = None
        self.direction_change_count = 0
        self.centering_start_time = None
        self.centering_attempts = 0
        self.oscillation_detected = False
        self.stable_frames = 0
        self.logger.info("🔄 Centering state reset")
        
    def detect_green_targets(self, frame) -> List[GreenTarget]:
        """Detect green targets in the frame"""
        green_targets = []
        
        if frame is None:
            return green_targets
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green colors
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_green_area < area < self.max_green_area:
                # Get bounding box
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                center = (x + w_rect // 2, y + h_rect // 2)
                
                # Calculate confidence based on area and shape
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # More rectangular = higher confidence for delivery zones
                    aspect_ratio = w_rect / max(h_rect, 1)
                    area_ratio = area / (w_rect * h_rect)
                    
                    # Confidence based on size and rectangular shape
                    size_confidence = min(1.0, area / 5000)  # Larger = more confident
                    shape_confidence = area_ratio * 0.7  # How well it fills bounding box
                    
                    confidence = (size_confidence + shape_confidence) / 2
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        distance_from_center = np.sqrt(
                            (center[0] - self.frame_center_x)**2 + 
                            (center[1] - self.frame_center_y)**2
                        )
                        
                        target = GreenTarget(
                            center=center,
                            area=area,
                            confidence=confidence,
                            distance_from_center=distance_from_center,
                            bbox=(x, y, w_rect, h_rect)
                        )
                        green_targets.append(target)
        
        # Sort by confidence and size (prefer larger, more confident targets)
        green_targets.sort(key=lambda t: (-t.confidence, -t.area))
        
        return green_targets[:3]  # Return top 3 targets
    
    def detect_walls(self, frame) -> bool:
        """Detect wall boundaries that should be avoided"""
        return self.boundary_system.detect_boundaries(frame)
    
    def get_wall_avoidance_command(self, frame) -> Optional[str]:
        """Get wall avoidance command if walls are detected"""
        return self.boundary_system.get_avoidance_command(frame)
    
    def draw_green_detection(self, frame, targets: List[GreenTarget]) -> np.ndarray:
        """Draw green target detection overlays with oscillation and wall info"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # === WALL DETECTION VISUALIZATION ===
        result = self.boundary_system.draw_boundary_visualization(result)
        
        # Draw adaptive centering zone for primary target
        if targets:
            primary_target = targets[0]
            x_tolerance, y_tolerance = self.get_adaptive_tolerances(primary_target)
            
            center_left = self.frame_center_x - x_tolerance
            center_right = self.frame_center_x + x_tolerance
            center_top = self.frame_center_y - y_tolerance
            center_bottom = self.frame_center_y + y_tolerance
            
            # Color based on oscillation state
            zone_color = (0, 165, 255) if self.oscillation_detected else (255, 255, 0)  # Orange if oscillating
            
            cv2.rectangle(result, (center_left, center_top), (center_right, center_bottom), zone_color, 2)
            
            zone_label = "ADAPTIVE CENTER ZONE"
            if self.oscillation_detected:
                zone_label += " (ANTI-OSCILLATION)"
            
            cv2.putText(result, zone_label, (center_left + 5, center_top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_color, 1)
        
        # Draw detected green targets
        for i, target in enumerate(targets):
            x, y, w_rect, h_rect = target.bbox
            
            # Color based on priority (first target is primary)
            if i == 0:
                color = (0, 255, 0)    # Bright green for primary target
                thickness = 3
                
                # Check if centered with adaptive tolerances
                centered = self.is_target_centered(target)
                if centered:
                    cv2.putText(result, "CENTERED - READY!", (target.center[0] - 50, target.center[1] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Show direction arrows with oscillation info
                    x_dir, y_dir = self.get_centering_direction(target)
                    direction_text = f"{x_dir.upper()}, {y_dir.upper()}"
                    
                    if self.oscillation_detected:
                        direction_text += " (DAMPED)"
                    
                    cv2.putText(result, direction_text, (target.center[0] - 40, target.center[1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Arrow to target
                    cv2.arrowedLine(result, (self.frame_center_x, self.frame_center_y), 
                                   target.center, (255, 255, 0), 2)
            else:
                color = (0, 150, 0)    # Darker green for secondary targets
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            
            # Draw center point
            cv2.circle(result, target.center, 5, color, -1)
            
            # Target info
            cv2.putText(result, f"G{i+1}", (target.center[0] - 10, target.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(result, f"C:{target.confidence:.2f}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Frame center crosshair
        cv2.line(result, (self.frame_center_x - 10, self.frame_center_y), 
                (self.frame_center_x + 10, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y - 10), 
                (self.frame_center_x, self.frame_center_y + 10), (255, 255, 255), 2)
        
        # Status overlay with oscillation and wall info
        overlay_height = 120  # Increased for wall info
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.6, overlay, 0.4, 0)
        
        # Status text
        cv2.putText(result, "DELIVERY MODE - Green Target Detection + Wall Avoidance", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        target_count = len(targets)
        primary_target = targets[0] if targets else None
        
        # Wall status
        wall_status = self.boundary_system.get_status()
        wall_danger = wall_status['walls_triggered'] > 0
        wall_text = f"Walls: {wall_status['walls_detected']} detected"
        if wall_danger:
            wall_text += " - DANGER!"
            wall_color = (0, 0, 255)
        else:
            wall_text += " - Safe"
            wall_color = (0, 255, 0)
        
        cv2.putText(result, wall_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, wall_color, 1)
        
        # Target status
        if primary_target:
            centered = self.is_target_centered(primary_target)
            if centered:
                status = "CENTERED - READY TO APPROACH"
                status_color = (0, 255, 0)
            elif self.oscillation_detected:
                status = "CENTERING (ANTI-OSCILLATION MODE)"
                status_color = (0, 165, 255)
            else:
                status = "CENTERING ON TARGET"
                status_color = (255, 255, 0)
        else:
            status = "SCANNING FOR GREEN TARGETS..."
            status_color = (255, 255, 255)
        
        cv2.putText(result, f"Targets: {target_count} | Status: {status}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Oscillation and wall debug info
        debug_info = []
        if self.oscillation_detected or self.direction_change_count > 0:
            debug_info.append(f"Dir Changes: {self.direction_change_count}")
            debug_info.append(f"Oscillation: {'YES' if self.oscillation_detected else 'NO'}")
        
        if wall_danger:
            triggered_zones = wall_status.get('danger_zones', [])
            debug_info.append(f"Wall zones: {', '.join(triggered_zones)}")
        
        if debug_info:
            debug_text = " | ".join(debug_info)
            cv2.putText(result, debug_text, (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        return result

class DeliverySystem:
    """Enhanced delivery system with oscillation prevention and wall avoidance"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = DeliveryVisionSystem(vision_system)
        
        # State management
        self.current_target = None
        self.delivery_active = False
        self.start_time = None
        
        # ADAPTIVE MOVEMENT PARAMETERS
        self.base_search_turn_duration = 0.8
        self.base_centering_turn_duration = 0.3
        self.approach_speed = 0.4
        self.search_speed = 0.5
        
    def get_adaptive_movement_params(self, target: Optional[GreenTarget] = None):
        """Get movement parameters adapted for oscillation state"""
        if target is None:
            return self.base_centering_turn_duration, self.search_speed
        
        # Base parameters
        turn_duration = self.base_centering_turn_duration
        speed = self.search_speed
        
        # Adapt based on distance
        distance = target.distance_from_center
        if distance > 100:
            # Far - faster, longer movements
            turn_duration *= 1.2
            speed *= 1.0
        elif distance < 40:
            # Close - slower, shorter movements
            turn_duration *= 0.7
            speed *= 0.8
        
        # OSCILLATION COMPENSATION
        if self.delivery_vision.oscillation_detected:
            self.logger.info("🔄 Applying oscillation compensation to movement")
            # Reduce movement duration and speed when oscillating
            turn_duration *= 0.5  # 50% reduction
            speed *= 0.7          # 30% speed reduction
        elif self.delivery_vision.direction_change_count >= 2:
            # Partial reduction when approaching oscillation
            turn_duration *= 0.8
            speed *= 0.9
        
        return turn_duration, speed
    
    def center_on_target(self, target: GreenTarget):
        """Center robot on green target with oscillation prevention"""
        x_direction, y_direction = self.delivery_vision.get_centering_direction(target)
        
        # Skip movement if both directions are centered
        if x_direction == 'centered' and y_direction == 'centered':
            return
        
        # Get adaptive movement parameters
        turn_duration, speed = self.get_adaptive_movement_params(target)
        
        if config.DEBUG_MOVEMENT:
            oscillation_status = " (DAMPED)" if self.delivery_vision.oscillation_detected else ""
            self.logger.info(f"🎯 Centering: {x_direction}, {y_direction}{oscillation_status} | Duration: {turn_duration:.2f}s")
        
        # Prioritize X-axis centering (turning)
        if x_direction == 'right':
            self.hardware.turn_right(duration=turn_duration, speed=speed)
        elif x_direction == 'left':
            self.hardware.turn_left(duration=turn_duration, speed=speed)
        elif y_direction == 'forward':
            self.hardware.move_forward(duration=turn_duration, speed=speed * 0.8)
        elif y_direction == 'backward':
            self.hardware.move_backward(duration=turn_duration, speed=speed * 0.8)
        
        # Adaptive post-movement delay
        if self.delivery_vision.oscillation_detected:
            time.sleep(0.2)  # Longer pause when oscillating
        else:
            time.sleep(0.1)  # Normal pause
    
    def handle_wall_avoidance(self, frame):
        """Handle wall avoidance during delivery"""
        avoidance_command = self.delivery_vision.get_wall_avoidance_command(frame)
        
        if avoidance_command:
            self.logger.warning(f"⚠️ Wall detected - executing avoidance: {avoidance_command}")
            
            # Stop current movement
            self.hardware.stop_motors()
            time.sleep(0.1)
            
            # Execute avoidance maneuver
            if avoidance_command == 'turn_right':
                self.hardware.turn_right(duration=0.4, speed=0.5)
            elif avoidance_command == 'turn_left':
                self.hardware.turn_left(duration=0.4, speed=0.5)
            elif avoidance_command == 'backup_and_turn':
                self.hardware.move_backward(duration=0.3, speed=0.4)
                time.sleep(0.1)
                self.hardware.turn_right(duration=0.6, speed=0.5)
            else:
                # Default avoidance
                self.hardware.move_backward(duration=0.3, speed=0.4)
                time.sleep(0.1)
                self.hardware.turn_right(duration=0.4, speed=0.5)
            
            time.sleep(0.2)
            return True
        
        return False
    
    def start_delivery_mode(self):
        """Start delivery mode with oscillation prevention and wall avoidance"""
        self.logger.info("🚚 STARTING DELIVERY MODE - Green Target Detection")
        self.logger.info("   Features: Adaptive tolerances, oscillation detection, wall avoidance")
        
        self.delivery_active = True
        self.start_time = time.time()
        
        try:
            self.delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Delivery mode interrupted by user")
        except Exception as e:
            self.logger.error(f"Delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def delivery_main_loop(self):
        """Main delivery loop with oscillation and wall handling"""
        search_direction = 1  # 1 for right, -1 for left
        frames_without_target = 0
        max_frames_without_target = 30
        
        while self.delivery_active:
            try:
                # Get current frame
                ret, frame = self.vision_system.get_frame()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # === PRIORITY 1: WALL AVOIDANCE ===
                wall_danger = self.delivery_vision.detect_walls(frame)
                if wall_danger:
                    wall_avoided = self.handle_wall_avoidance(frame)
                    if wall_avoided:
                        # Reset target tracking after wall avoidance
                        self.current_target = None
                        self.delivery_vision.reset_centering_state()
                        continue  # Skip this frame after avoidance
                
                # === PRIORITY 2: GREEN TARGET PROCESSING ===
                green_targets = self.delivery_vision.detect_green_targets(frame)
                
                # Create visualization
                debug_frame = self.delivery_vision.draw_green_detection(frame, green_targets)
                
                # Show frame if display available
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('GolfBot Delivery Mode (Anti-Oscillation + Wall Avoidance)', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # TARGET PROCESSING WITH OSCILLATION HANDLING
                if green_targets:
                    frames_without_target = 0
                    primary_target = green_targets[0]
                    
                    # Check if this is a new target
                    if (self.current_target is None or 
                        abs(primary_target.center[0] - self.current_target.center[0]) > 50 or
                        abs(primary_target.center[1] - self.current_target.center[1]) > 50):
                        
                        self.logger.info("🎯 New target detected - resetting centering state")
                        self.delivery_vision.reset_centering_state()
                    
                    self.current_target = primary_target
                    
                    # Initialize centering session if needed
                    if self.delivery_vision.centering_start_time is None:
                        self.delivery_vision.centering_start_time = time.time()
                        self.delivery_vision.centering_attempts = 0
                    
                    # Check if target is centered (with stability verification)
                    if self.delivery_vision.is_target_centered(primary_target):
                        elapsed_time = time.time() - self.delivery_vision.centering_start_time
                        self.logger.info(f"🎯 Target centered and stable! Approaching (took {elapsed_time:.1f}s)")
                        self.approach_target(primary_target)
                        self.delivery_vision.reset_centering_state()  # Reset for next target
                    else:
                        # TIMEOUT CHECK - prevent infinite centering
                        elapsed_time = time.time() - self.delivery_vision.centering_start_time
                        if elapsed_time > 15.0:  # 15 second timeout
                            self.logger.warning(f"⏰ Centering timeout after {elapsed_time:.1f}s - proceeding anyway")
                            self.approach_target(primary_target)
                            self.delivery_vision.reset_centering_state()
                        else:
                            # Continue centering with oscillation prevention
                            self.center_on_target(primary_target)
                
                else:
                    # No targets found - search
                    frames_without_target += 1
                    self.current_target = None
                    self.delivery_vision.reset_centering_state()  # Reset when losing target
                    
                    if frames_without_target >= max_frames_without_target:
                        search_direction *= -1
                        frames_without_target = 0
                        self.logger.info(f"🔍 Changing search direction: {'RIGHT' if search_direction > 0 else 'LEFT'}")
                    
                    self.search_for_targets(search_direction)
                
                # Control loop timing - adaptive based on state
                if self.current_target and self.delivery_vision.oscillation_detected:
                    time.sleep(0.15)  # Slower when oscillating
                elif wall_danger:
                    time.sleep(0.2)   # Slower when walls detected
                else:
                    time.sleep(0.1)   # Normal timing
                
            except Exception as e:
                self.logger.error(f"Delivery loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
    
    def approach_target(self, target: GreenTarget):
        """Approach the centered green target"""
        # Calculate approach distance based on target size
        approach_time = self.calculate_approach_time(target)
        
        self.logger.info(f"🚚 Approaching green delivery zone for {approach_time:.1f} seconds")
        
        # Move forward towards target
        self.hardware.move_forward(duration=approach_time, speed=self.approach_speed)
        
        # Brief pause
        time.sleep(0.5)
        
        # Check if we should release balls here
        if self.hardware.has_balls():
            self.logger.info("📦 Reached delivery zone - releasing balls!")
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls at green delivery zone")
            
            # Back away from delivery zone
            self.hardware.move_backward(duration=1.0, speed=self.approach_speed)
            time.sleep(0.5)
        else:
            self.logger.info("ℹ️  No balls to deliver - backing away from target")
            self.hardware.move_backward(duration=0.5, speed=self.approach_speed)
    
    def search_for_targets(self, direction: int):
        """Search for green targets by turning"""
        if direction > 0:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching right for green targets")
            self.hardware.turn_right(duration=self.base_search_turn_duration, speed=self.search_speed)
        else:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching left for green targets")
            self.hardware.turn_left(duration=self.base_search_turn_duration, speed=self.search_speed)
        
        time.sleep(0.2)
    
    def calculate_approach_time(self, target: GreenTarget) -> float:
        """Calculate how long to approach based on target size and distance"""
        # Larger targets are likely closer, smaller targets are farther
        # Base approach time inversely related to target area
        
        base_time = 2.0  # Base approach time
        area_factor = max(0.3, min(1.5, 5000 / max(target.area, 1000)))  # Scale by area
        
        approach_time = base_time * area_factor
        
        # Bounds
        return max(0.5, min(3.0, approach_time))
    
    def stop_delivery(self):
        """Stop delivery mode"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 DELIVERY MODE COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Final ball count: {self.hardware.get_ball_count()}")
        self.logger.info(f"   Oscillation events: {self.delivery_vision.direction_change_count}")
        
        # Wall avoidance stats
        wall_status = self.delivery_vision.boundary_system.get_status()
        self.logger.info(f"   Walls detected: {wall_status['walls_detected']}")
        self.logger.info(f"   Wall avoidance triggers: {wall_status['walls_triggered']}")
        
        cv2.destroyAllWindows()

def run_delivery_test():
    """Main entry point for delivery testing with wall avoidance"""
    print("\n🚚 GOLFBOT DELIVERY SYSTEM TEST (Anti-Oscillation + Wall Avoidance)")
    print("="*70)
    print("This mode will:")
    print("1. Search for GREEN targets (delivery zones)")
    print("2. Detect and avoid RED walls/boundaries")
    print("3. Center on detected green areas with oscillation prevention")
    print("4. Approach when properly aligned and walls are clear")
    print("5. Release balls if any are collected")
    print("\nNew Features:")
    print("• Wall/boundary detection and avoidance")
    print("• Oscillation detection and prevention")
    print("• Adaptive centering tolerances")
    print("• Movement damping when oscillating")
    print("• Wall avoidance priority over target centering")
    print("• Stability verification before approach")
    print("\nPress 'q' in the camera window to quit")
    print("="*70)
    
    input("Press Enter to start delivery test...")
    
    try:
        # Import and initialize systems
        from hardware import GolfBotHardware
        from vision import VisionSystem
        
        print("Initializing hardware and vision systems...")
        hardware = GolfBotHardware()
        vision = VisionSystem()
        
        if not vision.start():
            print("❌ Failed to initialize camera")
            return False
        
        print("✅ Systems initialized successfully!")
        
        # Create and start delivery system
        delivery_system = DeliverySystem(hardware, vision)
        delivery_system.start_delivery_mode()
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Delivery test interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Delivery test error: {e}")
        return False
    finally:
        # Cleanup
        try:
            if 'hardware' in locals():
                hardware.emergency_stop()
            if 'vision' in locals():
                vision.cleanup()
        except:
            pass

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_delivery_test()