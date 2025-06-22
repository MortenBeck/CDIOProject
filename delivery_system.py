def get_parallel_parking_command(self, target: GreenTarget) -> Optional[str]:
        """
        Get command for parallel parking approach using ACTUAL rectangle orientation
        
        Strategy: Find the closest short side and approach it perpendicularly
        """
        
        robot_x = self.frame_center_x
        robot_y = self.frame_center_y
        
        # If we have rotated rectangle data, use it for precise positioning
        if hasattr(target, 'box_points') and target.box_points is not None:
            box_pts = target.box_points
            
            # Calculate side lengths to identify short sides
            side_data = []
            for i in range(4):
                p1 = box_pts[i]
                p2 = box_pts[(i + 1) % 4]
                length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                
                # Calculate distance from robot to this side
                robot_dist = np.sqrt((robot_x - mid_x)**2 + (robot_y - mid_y)**2)
                
                side_data.append({
                    'length': length,
                    'midpoint': (mid_x, mid_y),
                    'robot_distance': robot_dist,
                    'side_index': i,
                    'p1': p1,
                    'p2': p2
                })
            
            # Sort by length to find short sides
            side_data.sort(key=lambda x: x['length'])
            short_sides = side_data[:2]  # Two shortest sides
            
            # Choose the short side closest to the robot
            closest_short_side = min(short_sides, key=lambda x: x['robot_distance'])
            
            target_midpoint = closest_short_side['midpoint']
            target_x, target_y = target_midpoint
            
            # Calculate the perpendicular approach vector
            # Vector along the short side
            p1, p2 = closest_short_side['p1'], closest_short_side['p2']
            side_vec_x = p2[0] - p1[0]
            side_vec_y = p2[1] - p1[1]
            
            # Perpendicular vector (rotate 90 degrees)
            perp_vec_x = -side_vec_y
            perp_vec_y = side_vec_x
            
            # Normalize perpendicular vector
            perp_length = np.sqrt(perp_vec_x**2 + perp_vec_y**2)
            if perp_length > 0:
                perp_vec_x /= perp_length
                perp_vec_y /= perp_length
            
            # Determine which direction to approach from (toward robot or away)
            # Calculate current robot vector from target
            robot_vec_x = robot_x - target_x
            robot_vec_y = robot_y - target_y
            
            # Choose perpendicular direction that points toward robot
            dot_product = robot_vec_x * perp_vec_x + robot_vec_y * perp_vec_y
            if dot_product < 0:
                perp_vec_x = -perp_vec_x
                perp_vec_y = -perp_vec_y
            
            # Desired position: approach_distance away from target in perpendicular direction
            desired_x = target_x + int(self.approach_distance * perp_vec_x)
            desired_y = target_y + int(self.approach_distance * perp_vec_y)
            
        else:
            # Fallback to simple logic if no rotated rectangle data
            target_x, target_y = target.center
            
            if target.orientation == 'horizontal':
                desired_x = target_x
                desired_y = target_y - self.approach_distance
            else:
                desired_x = target_x - self.approach_distance
                desired_y = target_y
        
        # Calculate positioning commands
        x_error = robot_x - desired_x
        y_error = robot_y - desired_y
        
        # Prioritize the larger error
        if abs(x_error) > abs(y_error):
            # X positioning needed
            if abs(x_error) > self.centering_tolerance:
                if x_error > 0:
                    return 'move_left'   # Robot too far right, turn left
                else:
                    return 'move_right'  # Robot too far left, turn right
        else:
            # Y positioning needed
            if abs(y_error) > self.centering_tolerance:
                if y_error > 0:
                    return 'move_backward'  # Robot too far down, move up
                else:
                    return 'move_forward'   # Robot too far up, move down
        
        # If we're close enough in both directions
        if (abs(x_error) <= self.centering_tolerance and 
            abs(y_error) <= self.centering_tolerance):
            return 'approach_target'
        
        return None
    
    def get_final_approach_command(self, target: GreenTarget) -> str:
        """Get command for final straight approach to the closest short side"""
        
        # If we have rotated rectangle data, calculate precise approach
        if hasattr(target, 'box_points') and target.box_points is not None:
            # The positioning phase should have set us up to approach the closest short side
            # Now we just need to drive straight toward it
            return 'approach_perpendicular'
        else:
            # Fallback to simple approach
            if target.orientation == 'horizontal':
                return 'approach_vertical'
            else:
                return 'approach_horizontal'#!/usr/bin/env python3
"""
Simple Green Target Delivery System - Parallel Parking Approach
Uses the working green detection to line up with green target's short end
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import config

@dataclass
class GreenTarget:
    """Simple green target data with rotated rectangle support"""
    center: Tuple[int, int]
    area: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height (approximate bounds)
    orientation: str  # 'horizontal' or 'vertical' 
    short_side_length: int
    long_side_length: int
    # Rotated rectangle data
    rotated_rect: Optional[Tuple] = None  # ((cx, cy), (w, h), angle)
    box_points: Optional[np.ndarray] = None  # 4 corner points
    rotation_angle: Optional[float] = None  # Rotation angle in degrees

class SimpleDeliveryVisionSystem:
    """Simplified vision system focused on green target detection and parallel parking approach"""
    
    def __init__(self, vision_system):
        self.logger = logging.getLogger(__name__)
        self.vision = vision_system
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        
        # Green detection parameters (working well)
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([80, 255, 255])
        self.min_green_area = 500
        self.max_green_area = 50000
        
        # Parallel parking approach parameters
        self.approach_distance = 120  # Distance to maintain during positioning
        self.alignment_tolerance = 15  # Tolerance for alignment in pixels
        self.centering_tolerance = 20  # Tolerance for centering
        
    def detect_green_targets(self, frame) -> List[GreenTarget]:
        """Detect green targets and determine their ACTUAL orientation using rotated rectangles"""
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
                # Use rotated rectangle to get ACTUAL orientation
                if len(contour) >= 5:  # Need at least 5 points for fitEllipse
                    try:
                        # Get the minimum area rotated rectangle
                        rotated_rect = cv2.minAreaRect(contour)
                        center_float, (width_float, height_float), angle = rotated_rect
                        
                        # Convert to integers
                        center = (int(center_float[0]), int(center_float[1]))
                        width = int(width_float)
                        height = int(height_float)
                        
                        # Determine TRUE orientation based on actual dimensions
                        if width > height:
                            orientation = 'horizontal'
                            short_side = height
                            long_side = width
                            # Normalize angle for horizontal rectangles
                            if angle < -45:
                                angle += 90
                        else:
                            orientation = 'vertical'  
                            short_side = width
                            long_side = height
                            # Normalize angle for vertical rectangles
                            if angle < -45:
                                angle += 90
                        
                        # Get the four corner points of the rotated rectangle
                        box_points = cv2.boxPoints(rotated_rect)
                        box_points = np.int0(box_points)
                        
                        # Use rotated rectangle for bbox instead of axis-aligned
                        x_coords = box_points[:, 0]
                        y_coords = box_points[:, 1]
                        bbox = (int(np.min(x_coords)), int(np.min(y_coords)), 
                               int(np.max(x_coords) - np.min(x_coords)), 
                               int(np.max(y_coords) - np.min(y_coords)))
                        
                        # Calculate confidence based on how rectangular the shape is
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            # For rotated rectangles, check how well contour fits the rotated rect
                            rect_area = width * height
                            contour_area = area
                            fill_ratio = contour_area / max(rect_area, 1)
                            
                            aspect_ratio = max(long_side, 1) / max(short_side, 1)
                            size_confidence = min(1.0, area / 3000)
                            shape_confidence = min(1.0, fill_ratio)
                            aspect_confidence = min(1.0, aspect_ratio / 4.0)  # Prefer rectangular shapes
                            
                            confidence = (size_confidence + shape_confidence + aspect_confidence) / 3
                            
                            if confidence > 0.3:
                                target = GreenTarget(
                                    center=center,
                                    area=area,
                                    confidence=confidence,
                                    bbox=bbox,
                                    orientation=orientation,
                                    short_side_length=short_side,
                                    long_side_length=long_side
                                )
                                # Store the rotated rectangle info for drawing
                                target.rotated_rect = rotated_rect
                                target.box_points = box_points
                                target.rotation_angle = angle
                                
                                green_targets.append(target)
                                
                    except Exception as e:
                        # Fallback to regular bounding box if rotated rect fails
                        if self.logger:
                            self.logger.warning(f"Rotated rectangle calculation failed: {e}")
                        continue
        
        # Sort by confidence and size
        green_targets.sort(key=lambda t: (-t.confidence, -t.area))
        return green_targets[:2]  # Keep top 2 targets
    
    def get_parallel_parking_command(self, target: GreenTarget) -> Optional[str]:
        """
        Get command for parallel parking approach to line up with green target's short end
        
        Strategy:
        1. Position robot to the side of the target (perpendicular to short side)
        2. Back up to create space
        3. Turn to align with the short side
        4. Drive straight toward the short side
        """
        
        robot_x = self.frame_center_x
        robot_y = self.frame_center_y
        target_x, target_y = target.center
        
        # Calculate positioning based on target orientation
        if target.orientation == 'horizontal':
            # Target is horizontal rectangle - approach from top or bottom (short sides)
            # Choose top approach for simplicity
            desired_x = target_x  # Same X as target
            desired_y = target_y - self.approach_distance  # Above the target
            
            # Phase 1: Get to the side of the target
            if abs(robot_x - target_x) > self.centering_tolerance:
                if robot_x < target_x:
                    return 'move_right'  # Turn right to get more to the right
                else:
                    return 'move_left'   # Turn left to get more to the left
            
            # Phase 2: Position at correct distance (above target)
            if robot_y > desired_y + self.alignment_tolerance:
                return 'move_forward'  # Too far below, move up
            elif robot_y < desired_y - self.alignment_tolerance:
                return 'move_backward'  # Too far above, move down
            
            # Phase 3: We're positioned - ready to approach
            return 'approach_target'
            
        else:  # vertical orientation
            # Target is vertical rectangle - approach from left or right (short sides)
            # Choose left approach for simplicity
            desired_x = target_x - self.approach_distance  # To the left of target
            desired_y = target_y  # Same Y as target
            
            # Phase 1: Get above/below the target first
            if abs(robot_y - target_y) > self.centering_tolerance:
                if robot_y < target_y:
                    return 'move_backward'  # Too far above, move down
                else:
                    return 'move_forward'   # Too far below, move up
            
            # Phase 2: Position at correct distance (left of target)
            if robot_x > desired_x + self.alignment_tolerance:
                return 'move_left'    # Too far right, move left
            elif robot_x < desired_x - self.alignment_tolerance:
                return 'move_right'   # Too far left, move right
            
            # Phase 3: We're positioned - ready to approach
            return 'approach_target'
    
    def get_final_approach_command(self, target: GreenTarget) -> str:
        """Get command for final straight approach to target"""
        robot_x = self.frame_center_x
        robot_y = self.frame_center_y
        target_x, target_y = target.center
        
        if target.orientation == 'horizontal':
            # Approach from above - drive straight down
            return 'approach_vertical'
        else:
            # Approach from left - drive straight right
            return 'approach_horizontal'
    
    def draw_delivery_visualization(self, frame, targets: List[GreenTarget], current_phase: str) -> np.ndarray:
        """Draw delivery system visualization with ACTUAL rotated rectangles"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw green targets with ACTUAL orientation info
        for i, target in enumerate(targets):
            if i == 0:
                color = (0, 255, 0)    # Bright green for primary target
                thickness = 3
                
                # Draw the ACTUAL rotated rectangle using the corner points
                if hasattr(target, 'box_points') and target.box_points is not None:
                    cv2.drawContours(result, [target.box_points], 0, color, thickness)
                    
                    # Calculate and highlight the short sides
                    box_pts = target.box_points
                    
                    # Calculate side lengths to identify short sides
                    side_lengths = []
                    for i in range(4):
                        p1 = box_pts[i]
                        p2 = box_pts[(i + 1) % 4]
                        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        side_lengths.append((length, i))
                    
                    # Sort to find the two shortest sides
                    side_lengths.sort()
                    short_side_indices = [side_lengths[0][1], side_lengths[1][1]]
                    
                    # Highlight the short sides in yellow
                    for side_idx in short_side_indices:
                        p1 = tuple(box_pts[side_idx])
                        p2 = tuple(box_pts[(side_idx + 1) % 4])
                        cv2.line(result, p1, p2, (0, 255, 255), 5)  # Thick yellow line
                    
                    # Show approach direction toward the closest short side
                    # Find the short side closest to the robot (frame center)
                    robot_pos = (self.frame_center_x, self.frame_center_y)
                    min_dist = float('inf')
                    closest_short_side = None
                    
                    for side_idx in short_side_indices:
                        p1 = box_pts[side_idx]
                        p2 = box_pts[(side_idx + 1) % 4]
                        # Calculate distance from robot to side midpoint
                        mid_x = (p1[0] + p2[0]) // 2
                        mid_y = (p1[1] + p2[1]) // 2
                        dist = np.sqrt((robot_pos[0] - mid_x)**2 + (robot_pos[1] - mid_y)**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_short_side = (p1, p2, (mid_x, mid_y))
                    
                    # Draw approach arrow to closest short side
                    if closest_short_side:
                        p1, p2, midpoint = closest_short_side
                        # Calculate approach position (30 pixels away from midpoint toward robot)
                        dx = robot_pos[0] - midpoint[0]
                        dy = robot_pos[1] - midpoint[1]
                        norm = np.sqrt(dx*dx + dy*dy)
                        if norm > 0:
                            approach_x = midpoint[0] + int(30 * dx / norm)
                            approach_y = midpoint[1] + int(30 * dy / norm)
                            cv2.arrowedLine(result, (approach_x, approach_y), midpoint, (0, 255, 255), 3)
                            cv2.putText(result, "APPROACH", (approach_x - 30, approach_y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                else:
                    # Fallback to regular bounding box if rotated rect not available
                    x, y, w_rect, h_rect = target.bbox
                    cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
                
                # Draw center point
                cv2.circle(result, target.center, 5, color, -1)
                
                # Show orientation and rotation info
                orientation_text = f"{target.orientation.upper()}"
                if hasattr(target, 'rotation_angle') and target.rotation_angle is not None:
                    orientation_text += f" ({target.rotation_angle:.1f}°)"
                    
                cv2.putText(result, orientation_text, (target.center[0] - 40, target.center[1] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                short_side_text = f"Short: {target.short_side_length}px"
                cv2.putText(result, short_side_text, (target.center[0] - 40, target.center[1] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
            else:
                color = (0, 150, 0)    # Darker green for secondary targets
                thickness = 2
                
                if hasattr(target, 'box_points') and target.box_points is not None:
                    cv2.drawContours(result, [target.box_points], 0, color, thickness)
                else:
                    x, y, w_rect, h_rect = target.bbox
                    cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
                
                cv2.circle(result, target.center, 3, color, -1)
            
            # Target label
            cv2.putText(result, f"G{i+1}", (target.center[0] - 10, target.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Frame center crosshair (robot position reference)
        cv2.line(result, (self.frame_center_x - 10, self.frame_center_y), 
                (self.frame_center_x + 10, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y - 10), 
                (self.frame_center_x, self.frame_center_y + 10), (255, 255, 255), 2)
        cv2.putText(result, "ROBOT", (self.frame_center_x - 20, self.frame_center_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Status overlay
        overlay_height = 120
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.6, overlay, 0.4, 0)
        
        # Status text
        cv2.putText(result, "SIMPLE DELIVERY - Rotated Rectangle Detection", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        target_count = len(targets)
        primary_target = targets[0] if targets else None
        
        if primary_target:
            angle_info = ""
            if hasattr(primary_target, 'rotation_angle') and primary_target.rotation_angle is not None:
                angle_info = f" (rotated {primary_target.rotation_angle:.1f}°)"
            status = f"Target: {primary_target.orientation.upper()} green rectangle{angle_info}"
            phase_status = f"Phase: {current_phase.upper()}"
        else:
            status = "Scanning for green targets..."
            phase_status = "Phase: SEARCHING"
        
        cv2.putText(result, f"Targets: {target_count} | {status}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(result, phase_status, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Instructions
        cv2.putText(result, "Yellow lines: Short sides to target | Yellow arrow: Approach direction", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return result

class SimpleDeliverySystem:
    """Simple delivery system using parallel parking approach"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = SimpleDeliveryVisionSystem(vision_system)
        
        # Simple state management
        self.current_target = None
        self.delivery_active = False
        self.start_time = None
        
        # Simple phases: search -> position -> approach -> deliver
        self.current_phase = "search"
        self.phase_start_time = None
        
    def start_simple_delivery_mode(self):
        """Start simple delivery mode"""
        self.logger.info("🚚 STARTING SIMPLE DELIVERY MODE - Parallel Parking to Green Targets")
        self.logger.info("   Strategy: Detect green rectangles -> Position to side -> Drive at short end")
        
        self.delivery_active = True
        self.start_time = time.time()
        self.current_phase = "search"
        
        try:
            self.simple_delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Simple delivery mode interrupted by user")
        except Exception as e:
            self.logger.error(f"Simple delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def simple_delivery_main_loop(self):
        """Main delivery loop with simple parallel parking logic"""
        search_direction = 1
        frames_without_target = 0
        max_frames_without_target = 25
        
        while self.delivery_active:
            try:
                # Get current frame
                ret, frame = self.vision_system.get_frame()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Detect green targets
                green_targets = self.delivery_vision.detect_green_targets(frame)
                
                # Create visualization
                debug_frame = self.delivery_vision.draw_delivery_visualization(
                    frame, green_targets, self.current_phase)
                
                # Show frame if display available
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('Simple Delivery - Parallel Parking', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # SIMPLE PHASE PROCESSING
                if green_targets:
                    frames_without_target = 0
                    primary_target = green_targets[0]
                    
                    # Check if this is a new target
                    if (self.current_target is None or 
                        abs(primary_target.center[0] - self.current_target.center[0]) > 50):
                        self.logger.info("🎯 New green target detected - starting parallel parking")
                        self.current_target = primary_target
                        self.current_phase = "position"
                        self.phase_start_time = time.time()
                    
                    self.current_target = primary_target
                    
                    # Execute current phase
                    if self.current_phase == "search":
                        self.current_phase = "position"
                        self.phase_start_time = time.time()
                    elif self.current_phase == "position":
                        self.handle_positioning_phase()
                    elif self.current_phase == "approach":
                        self.handle_approach_phase()
                    elif self.current_phase == "deliver":
                        self.handle_deliver_phase()
                
                else:
                    # No targets - search
                    frames_without_target += 1
                    self.current_target = None
                    self.current_phase = "search"
                    
                    if frames_without_target >= max_frames_without_target:
                        search_direction *= -1
                        frames_without_target = 0
                        self.logger.info(f"🔍 Changing search direction: {'RIGHT' if search_direction > 0 else 'LEFT'}")
                    
                    self.search_for_targets(search_direction)
                
                time.sleep(0.15)
                
            except Exception as e:
                self.logger.error(f"Simple delivery loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
    
    def handle_positioning_phase(self):
        """Handle parallel parking positioning"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        # Get parallel parking command
        parking_command = self.delivery_vision.get_parallel_parking_command(self.current_target)
        
        if parking_command == 'approach_target':
            self.logger.info("✅ Parallel parking position achieved - starting final approach")
            self.current_phase = "approach"
            self.phase_start_time = time.time()
            return
        
        # Execute positioning movement
        move_duration = 0.4
        move_speed = 0.4
        
        if parking_command == 'move_right':
            self.logger.info(f"🚗 Positioning: Turn right")
            self.hardware.turn_right(duration=move_duration, speed=move_speed)
        elif parking_command == 'move_left':
            self.logger.info(f"🚗 Positioning: Turn left")
            self.hardware.turn_left(duration=move_duration, speed=move_speed)
        elif parking_command == 'move_forward':
            self.logger.info(f"🚗 Positioning: Move forward")
            self.hardware.move_forward(duration=move_duration, speed=move_speed)
        elif parking_command == 'move_backward':
            self.logger.info(f"🚗 Positioning: Move backward")
            self.hardware.move_backward(duration=move_duration, speed=move_speed)
        
        # Timeout check
        if time.time() - self.phase_start_time > 20.0:
            self.logger.warning("⏰ Positioning timeout - attempting approach anyway")
            self.current_phase = "approach"
    
    def handle_approach_phase(self):
        """Handle final straight approach to green target's closest short side"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        # Get final approach command
        approach_command = self.delivery_vision.get_final_approach_command(self.current_target)
        
        self.logger.info(f"🚀 Final approach: {approach_command}")
        
        # Execute straight approach
        approach_duration = 1.2
        approach_speed = 0.35
        
        if approach_command == 'approach_perpendicular':
            # Drive straight toward the target (we should already be aligned)
            self.logger.info("Driving straight toward closest short side")
            self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
            
        elif approach_command == 'approach_vertical':
            # Drive straight toward horizontal target (from above)
            self.logger.info("Approaching horizontal target from above")
            self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
            
        elif approach_command == 'approach_horizontal':
            # For horizontal approach, we need to turn and drive
            # Since we can't move sideways, turn toward target and drive
            self.logger.info("Approaching vertical target with turn and drive")
            self.hardware.turn_right(duration=0.3, speed=approach_speed)
            time.sleep(0.1)
            self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
        
        # Move to delivery phase
        self.logger.info("📦 Reached target area - proceeding to delivery")
        self.current_phase = "deliver"
        self.phase_start_time = time.time()
    
    def handle_deliver_phase(self):
        """Handle ball delivery and reset"""
        self.logger.info("📦 Executing delivery sequence")
        
        # Release balls if we have any
        if self.hardware.has_balls():
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls at green target")
        else:
            self.logger.info("ℹ️  No balls to deliver")
        
        # Back away from the target
        self.logger.info("⬅️ Backing away from delivery zone")
        self.hardware.move_backward(duration=1.2, speed=0.4)
        
        # Turn to create separation
        self.hardware.turn_left(duration=0.6, speed=0.4)
        
        # Reset for next target
        self.current_phase = "search"
        self.current_target = None
        
        self.logger.info("🔄 Delivery complete - searching for next target")
    
    def search_for_targets(self, direction: int):
        """Search for green targets by turning"""
        if direction > 0:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching right for green targets")
            self.hardware.turn_right(duration=0.6, speed=0.5)
        else:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching left for green targets")
            self.hardware.turn_left(duration=0.6, speed=0.5)
        
        time.sleep(0.3)
    
    def stop_delivery(self):
        """Stop simple delivery mode"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 SIMPLE DELIVERY MODE COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Final ball count: {self.hardware.get_ball_count()}")
        self.logger.info(f"   Final phase: {self.current_phase}")
        
        cv2.destroyAllWindows()

def run_simple_delivery_test():
    """Main entry point for simple delivery testing"""
    print("\n🚚 SIMPLE GOLFBOT DELIVERY SYSTEM TEST")
    print("="*60)
    print("This simplified delivery system will:")
    print("1. Search for GREEN rectangular targets")
    print("2. Identify target orientation (horizontal/vertical)")
    print("3. Use 'parallel parking' approach to position robot")
    print("4. Drive straight toward the target's SHORT SIDE")
    print("5. Release balls and back away")
    print()
    print("Parallel Parking Strategy:")
    print("• For HORIZONTAL targets: Position above -> Drive down at short side")
    print("• For VERTICAL targets: Position to left -> Turn and drive at short side")
    print("• Uses only forward/backward/turn movements (no sideways)")
    print()
    print("Visual Indicators:")
    print("• Green rectangles: Detected delivery targets")
    print("• Yellow highlights: Short sides we want to hit")
    print("• Yellow arrows: Planned approach direction")
    print("• White crosshair: Robot position reference")
    print()
    print("Phases:")
    print("• SEARCH: Looking for green targets")
    print("• POSITION: Parallel parking positioning")
    print("• APPROACH: Final straight drive at target")
    print("• DELIVER: Release balls and back away")
    print()
    print("Press 'q' in the camera window to quit")
    print("="*60)
    
    input("Press Enter to start simple delivery test...")
    
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
        
        # Create and start simple delivery system
        delivery_system = SimpleDeliverySystem(hardware, vision)
        delivery_system.start_simple_delivery_mode()
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Simple delivery test interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Simple delivery test error: {e}")
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

def show_simple_delivery_info():
    """Show simple delivery system information"""
    print(f"\n🚚 Entering Simple Delivery System Test Mode...")
    
    print("\n🎯 Simple delivery system features:")
    print("   - Green rectangle detection using HSV color filtering")
    print("   - Automatic orientation detection (horizontal/vertical)")
    print("   - Parallel parking positioning strategy")
    print("   - Straight-line approach to target's short side")
    print("   - Simple search pattern (left/right scanning)")
    print("   - Automatic ball release at delivery targets")
    
    print(f"\n⚙️ Configuration:")
    print(f"   - Green HSV range: [40-80, 50-255, 50-255]")
    print(f"   - Approach distance: 120 pixels")
    print(f"   - Alignment tolerance: ±15px")
    print(f"   - Centering tolerance: ±20px")
    print(f"   - Min target area: 500 pixels")
    print(f"   - Max target area: 50,000 pixels")
    
    print("\n🚗 Parallel Parking Strategy:")
    print("   - HORIZONTAL targets: Position above → Drive down at short end")
    print("   - VERTICAL targets: Position to left → Turn and drive at short end")
    print("   - Uses only forward/backward/turn movements")
    
    print("\n🎮 Controls:")
    print("   - Press 'q' in camera window to quit")
    print("   - System automatically: Search → Position → Approach → Deliver")
    
    print("\nPress Enter to start simple delivery test...")
    input()

# Function to replace the complex delivery system
def run_delivery_test():
    """Replacement function for the main system"""
    return run_simple_delivery_test()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_simple_delivery_test()