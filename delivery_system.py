#!/usr/bin/env python3
"""
GolfBot Delivery System - Enhanced for Rectangular Hole Alignment
Detects green areas, then finds and aligns with rectangular holes for ball delivery
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

@dataclass
class RectangularHole:
    """Class to store rectangular hole information"""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area: int
    confidence: float
    aspect_ratio: float
    alignment_offset: Tuple[int, int]  # offset from robot center
    corners: List[Tuple[int, int]]  # corner points for precise alignment

class DeliveryVisionSystem:
    """Enhanced vision system for green detection and rectangular hole alignment"""
    
    def __init__(self, vision_system):
        self.logger = logging.getLogger(__name__)
        self.vision = vision_system
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        
        # Initialize boundary detection system
        self.boundary_system = BoundaryAvoidanceSystem()
        
        # Green detection parameters
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([80, 255, 255])
        self.min_green_area = 500
        self.max_green_area = 50000
        
        # Hole detection parameters
        self.hole_detection_enabled = False
        self.min_hole_area = 200
        self.max_hole_area = 5000
        self.hole_aspect_ratio_min = 1.2  # Width should be > height for rectangular hole
        self.hole_aspect_ratio_max = 4.0
        
        # Alignment parameters
        self.alignment_tolerance = 8  # pixels - very precise for hole alignment
        self.approach_distance_threshold = 100  # switch to hole detection when this close
        
        # State tracking
        self.current_mode = "green_search"  # "green_search", "hole_alignment", "approaching_hole"
        self.last_hole_seen = None
        self.alignment_stable_frames = 0
        self.min_stable_frames = 3
        
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
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                center = (x + w_rect // 2, y + h_rect // 2)
                
                # Calculate confidence based on area and shape
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    aspect_ratio = w_rect / max(h_rect, 1)
                    area_ratio = area / (w_rect * h_rect)
                    
                    size_confidence = min(1.0, area / 5000)
                    shape_confidence = area_ratio * 0.7
                    
                    confidence = (size_confidence + shape_confidence) / 2
                    
                    if confidence > 0.3:
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
        
        green_targets.sort(key=lambda t: (-t.confidence, -t.area))
        return green_targets[:3]
    
    def detect_rectangular_holes(self, frame) -> List[RectangularHole]:
        """Detect rectangular holes in the wall for precise alignment"""
        holes = []
        
        if frame is None or not self.hole_detection_enabled:
            return holes
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for finding hole boundaries
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_hole_area < area < self.max_hole_area:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular (4-6 corners)
                if 4 <= len(approx) <= 6:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    center = (x + w_rect // 2, y + h_rect // 2)
                    
                    # Check aspect ratio for rectangular hole
                    aspect_ratio = w_rect / max(h_rect, 1)
                    
                    if self.hole_aspect_ratio_min <= aspect_ratio <= self.hole_aspect_ratio_max:
                        # Calculate how well it fits a rectangle
                        rect_area = w_rect * h_rect
                        fill_ratio = area / rect_area if rect_area > 0 else 0
                        
                        if fill_ratio > 0.6:  # Should be fairly rectangular
                            # Get corner points
                            corners = [tuple(point[0]) for point in approx]
                            
                            # Calculate alignment offset from frame center
                            alignment_offset = (
                                center[0] - self.frame_center_x,
                                center[1] - self.frame_center_y
                            )
                            
                            # Confidence based on size, shape, and fill ratio
                            size_conf = min(1.0, area / 1000)
                            shape_conf = 1.0 / (abs(aspect_ratio - 2.0) + 1.0)  # Prefer ~2:1 ratio
                            fill_conf = fill_ratio
                            
                            confidence = (size_conf + shape_conf + fill_conf) / 3
                            
                            if confidence > 0.5:
                                hole = RectangularHole(
                                    center=center,
                                    bbox=(x, y, w_rect, h_rect),
                                    area=area,
                                    confidence=confidence,
                                    aspect_ratio=aspect_ratio,
                                    alignment_offset=alignment_offset,
                                    corners=corners
                                )
                                holes.append(hole)
        
        # Sort by confidence and distance from center
        holes.sort(key=lambda h: (-h.confidence, abs(h.alignment_offset[0]) + abs(h.alignment_offset[1])))
        return holes[:2]  # Return top 2 holes
    
    def is_hole_aligned(self, hole: RectangularHole) -> bool:
        """Check if hole is perfectly aligned with robot"""
        x_offset, y_offset = hole.alignment_offset
        
        # Very strict tolerance for hole alignment
        x_aligned = abs(x_offset) <= self.alignment_tolerance
        y_aligned = abs(y_offset) <= self.alignment_tolerance
        
        aligned = x_aligned and y_aligned
        
        if aligned:
            self.alignment_stable_frames += 1
            return self.alignment_stable_frames >= self.min_stable_frames
        else:
            self.alignment_stable_frames = 0
            return False
    
    def get_hole_alignment_direction(self, hole: RectangularHole) -> tuple:
        """Get precise alignment direction for hole"""
        x_offset, y_offset = hole.alignment_offset
        
        # Use smaller tolerance for micro-adjustments
        micro_tolerance = self.alignment_tolerance // 2
        
        # X-axis alignment (left/right turning)
        if abs(x_offset) <= micro_tolerance:
            x_direction = 'centered'
        elif x_offset > micro_tolerance:
            x_direction = 'right'
        else:
            x_direction = 'left'
        
        # Y-axis alignment (forward/backward)
        if abs(y_offset) <= micro_tolerance:
            y_direction = 'centered'
        elif y_offset > micro_tolerance:
            y_direction = 'backward'
        else:
            y_direction = 'forward'
        
        return x_direction, y_direction
    
    def should_switch_to_hole_mode(self, green_target: GreenTarget) -> bool:
        """Determine if robot is close enough to switch to hole detection mode"""
        if green_target is None:
            return False
        
        # Switch when green target is close and centered
        close_enough = green_target.distance_from_center < self.approach_distance_threshold
        x_centered = abs(green_target.center[0] - self.frame_center_x) < 30
        
        return close_enough and x_centered
    
    def draw_delivery_visualization(self, frame, green_targets: List[GreenTarget], holes: List[RectangularHole]) -> np.ndarray:
        """Draw enhanced visualization with hole alignment info"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # === WALL DETECTION VISUALIZATION ===
        result = self.boundary_system.draw_boundary_visualization(result)
        
        # === MODE-SPECIFIC VISUALIZATION ===
        if self.current_mode == "green_search":
            # Draw adaptive centering zone for green targets
            if green_targets:
                primary_target = green_targets[0]
                center_tolerance = 30
                
                center_left = self.frame_center_x - center_tolerance
                center_right = self.frame_center_x + center_tolerance
                center_top = self.frame_center_y - center_tolerance
                center_bottom = self.frame_center_y + center_tolerance
                
                cv2.rectangle(result, (center_left, center_top), (center_right, center_bottom), (255, 255, 0), 2)
                cv2.putText(result, "GREEN TARGET ZONE", (center_left + 5, center_top - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        elif self.current_mode == "hole_alignment":
            # Draw precise alignment zone for holes
            tolerance = self.alignment_tolerance
            
            align_left = self.frame_center_x - tolerance
            align_right = self.frame_center_x + tolerance
            align_top = self.frame_center_y - tolerance
            align_bottom = self.frame_center_y + tolerance
            
            cv2.rectangle(result, (align_left, align_top), (align_right, align_bottom), (0, 0, 255), 3)
            cv2.putText(result, "HOLE ALIGNMENT ZONE", (align_left + 5, align_top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # === GREEN TARGETS ===
        for i, target in enumerate(green_targets):
            x, y, w_rect, h_rect = target.bbox
            
            if i == 0:  # Primary target
                color = (0, 255, 0)
                thickness = 3
                
                # Check if should switch to hole mode
                should_switch = self.should_switch_to_hole_mode(target)
                
                if should_switch:
                    cv2.putText(result, "SWITCHING TO HOLE DETECTION", (target.center[0] - 80, target.center[1] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                else:
                    cv2.putText(result, f"APPROACH: {target.distance_from_center:.0f}px", (target.center[0] - 40, target.center[1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Arrow to target
                cv2.arrowedLine(result, (self.frame_center_x, self.frame_center_y), 
                               target.center, (0, 255, 0), 2)
            else:
                color = (0, 150, 0)
                thickness = 2
            
            # Draw bounding box and center
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            cv2.circle(result, target.center, 5, color, -1)
            cv2.putText(result, f"G{i+1}", (target.center[0] - 10, target.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # === RECTANGULAR HOLES ===
        for i, hole in enumerate(holes):
            x, y, w_rect, h_rect = hole.bbox
            
            if i == 0:  # Primary hole
                color = (0, 0, 255)  # Red for holes
                thickness = 4
                
                # Check alignment
                aligned = self.is_hole_aligned(hole)
                
                if aligned:
                    cv2.putText(result, "HOLE ALIGNED - READY!", (hole.center[0] - 70, hole.center[1] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Draw corners for perfect alignment visualization
                    for corner in hole.corners:
                        cv2.circle(result, corner, 3, (0, 255, 0), -1)
                else:
                    x_dir, y_dir = self.get_hole_alignment_direction(hole)
                    direction_text = f"{x_dir.upper()}, {y_dir.upper()}"
                    cv2.putText(result, f"ALIGN: {direction_text}", (hole.center[0] - 40, hole.center[1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    # Show offset values
                    x_offset, y_offset = hole.alignment_offset
                    cv2.putText(result, f"Offset: ({x_offset:+d}, {y_offset:+d})", (hole.center[0] - 50, hole.center[1] + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Alignment crosshair to hole center
                cv2.line(result, (self.frame_center_x, self.frame_center_y), 
                        hole.center, (0, 0, 255), 2)
            else:
                color = (0, 0, 150)
                thickness = 2
            
            # Draw hole bounding box
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            cv2.circle(result, hole.center, 3, color, -1)
            
            # Hole info
            cv2.putText(result, f"H{i+1}", (hole.center[0] - 8, hole.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(result, f"AR:{hole.aspect_ratio:.1f}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # === FRAME CENTER CROSSHAIR ===
        cv2.line(result, (self.frame_center_x - 10, self.frame_center_y), 
                (self.frame_center_x + 10, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y - 10), 
                (self.frame_center_x, self.frame_center_y + 10), (255, 255, 255), 2)
        
        # === STATUS OVERLAY ===
        overlay_height = 120
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.6, overlay, 0.4, 0)
        
        # Title
        cv2.putText(result, "DELIVERY MODE - Green Detection + Hole Alignment", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mode and status
        mode_text = f"Mode: {self.current_mode.replace('_', ' ').title()}"
        cv2.putText(result, mode_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Target counts
        green_count = len(green_targets)
        hole_count = len(holes) if self.hole_detection_enabled else 0
        cv2.putText(result, f"Green Targets: {green_count} | Holes: {hole_count} | Hole Detection: {'ON' if self.hole_detection_enabled else 'OFF'}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Wall status
        wall_status = self.boundary_system.get_status()
        wall_text = f"Walls: {wall_status['walls_detected']} detected"
        if wall_status['walls_triggered'] > 0:
            wall_text += " - AVOIDING!"
            wall_color = (0, 0, 255)
        else:
            wall_text += " - Safe"
            wall_color = (0, 255, 0)
        
        cv2.putText(result, wall_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, wall_color, 1)
        
        return result

class DeliverySystem:
    """Enhanced delivery system with rectangular hole alignment capability"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = DeliveryVisionSystem(vision_system)
        
        # State management
        self.current_target = None
        self.current_hole = None
        self.delivery_active = False
        self.start_time = None
        
        # Movement parameters for different phases
        self.search_turn_duration = 0.8
        self.green_centering_duration = 0.4
        self.hole_alignment_duration = 0.2  # Shorter for precise movements
        self.approach_speed = 0.4
        self.alignment_speed = 0.3  # Slower for precision
        
        # Triangulation parameters for hole alignment
        self.triangulation_steps = []
        self.max_triangulation_attempts = 5
        
    def center_on_green_target(self, target: GreenTarget):
        """Center robot on green target with moderate precision"""
        x_offset = target.center[0] - self.delivery_vision.frame_center_x
        y_offset = target.center[1] - self.delivery_vision.frame_center_y
        
        # Moderate tolerance for green centering
        tolerance = 25
        
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"🎯 Green centering: x_offset={x_offset}, y_offset={y_offset}")
        
        if abs(x_offset) > tolerance:
            if x_offset > 0:
                self.hardware.turn_right(duration=self.green_centering_duration, speed=self.approach_speed)
            else:
                self.hardware.turn_left(duration=self.green_centering_duration, speed=self.approach_speed)
            time.sleep(0.1)
        elif abs(y_offset) > tolerance:
            if y_offset > 0:
                self.hardware.move_backward(duration=self.green_centering_duration, speed=self.approach_speed)
            else:
                self.hardware.move_forward(duration=self.green_centering_duration, speed=self.approach_speed)
            time.sleep(0.1)
    
    def align_with_hole_triangulation(self, hole: RectangularHole):
        """Precise hole alignment using triangulation method"""
        x_offset, y_offset = hole.alignment_offset
        
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"🔧 Hole alignment: offset=({x_offset}, {y_offset}), tolerance={self.delivery_vision.alignment_tolerance}")
        
        # MICRO-MOVEMENTS for precise alignment
        if abs(x_offset) > self.delivery_vision.alignment_tolerance:
            # Lateral adjustment using back-turn-forward technique
            movement_magnitude = min(abs(x_offset) / 100.0, 0.5)  # Scale movement to offset
            
            if x_offset > 0:  # Hole is to the right
                self.logger.info("🔧 Triangulation: Moving right via back-turn-forward")
                self.hardware.move_backward(duration=0.2, speed=self.alignment_speed)
                time.sleep(0.1)
                self.hardware.turn_right(duration=self.hole_alignment_duration * movement_magnitude, speed=self.alignment_speed)
                time.sleep(0.1)
                self.hardware.move_forward(duration=0.2, speed=self.alignment_speed)
            else:  # Hole is to the left
                self.logger.info("🔧 Triangulation: Moving left via back-turn-forward")
                self.hardware.move_backward(duration=0.2, speed=self.alignment_speed)
                time.sleep(0.1)
                self.hardware.turn_left(duration=self.hole_alignment_duration * movement_magnitude, speed=self.alignment_speed)
                time.sleep(0.1)
                self.hardware.move_forward(duration=0.2, speed=self.alignment_speed)
            
            time.sleep(0.15)
            
        elif abs(y_offset) > self.delivery_vision.alignment_tolerance:
            # Forward/backward adjustment
            movement_duration = min(abs(y_offset) / 200.0, 0.3)
            
            if y_offset > 0:
                self.hardware.move_backward(duration=movement_duration, speed=self.alignment_speed)
            else:
                self.hardware.move_forward(duration=movement_duration, speed=self.alignment_speed)
            
            time.sleep(0.1)
    
    def approach_aligned_hole(self, hole: RectangularHole):
        """Final approach to aligned hole for ball delivery"""
        self.logger.info("🚚 Approaching aligned hole for ball delivery")
        
        # Calculate approach distance based on hole size
        approach_time = 1.5 + (hole.area / 2000.0)  # Larger holes = longer approach
        approach_time = min(approach_time, 3.0)  # Cap at 3 seconds
        
        self.logger.info(f"📏 Hole area: {hole.area}, approach time: {approach_time:.2f}s")
        
        # Approach in stages to monitor alignment
        stages = 3
        stage_time = approach_time / stages
        
        for stage in range(stages):
            self.logger.info(f"🚶 Approach stage {stage + 1}/{stages}")
            self.hardware.move_forward(duration=stage_time, speed=self.approach_speed)
            time.sleep(0.1)
            
            # Check for walls during approach
            ret, frame = self.vision_system.get_frame()
            if ret:
                wall_danger = self.delivery_vision.detect_walls(frame)
                if wall_danger:
                    self.logger.warning("⚠️ Wall detected during approach - stopping")
                    break
        
        # Release balls if we have any
        if self.hardware.has_balls():
            self.logger.info("📦 Releasing balls into hole!")
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls into rectangular hole")
            time.sleep(1.0)  # Allow balls to exit
        else:
            self.logger.info("ℹ️  No balls to deliver")
        
        # Back away from hole
        self.logger.info("⬅️ Backing away from delivery hole")
        self.hardware.move_backward(duration=1.0, speed=self.approach_speed)
        time.sleep(0.2)
    
    def handle_wall_avoidance(self, frame):
        """Handle wall avoidance during delivery"""
        avoidance_command = self.delivery_vision.get_wall_avoidance_command(frame)
        
        if avoidance_command:
            self.logger.warning(f"⚠️ Wall detected - executing avoidance: {avoidance_command}")
            
            self.hardware.stop_motors()
            time.sleep(0.1)
            
            if avoidance_command == 'turn_right':
                self.hardware.turn_right(duration=0.4, speed=0.5)
            elif avoidance_command == 'turn_left':
                self.hardware.turn_left(duration=0.4, speed=0.5)
            elif avoidance_command == 'backup_and_turn':
                self.hardware.move_backward(duration=0.3, speed=0.4)
                time.sleep(0.1)
                self.hardware.turn_right(duration=0.6, speed=0.5)
            else:
                self.hardware.move_backward(duration=0.3, speed=0.4)
                time.sleep(0.1)
                self.hardware.turn_right(duration=0.4, speed=0.5)
            
            time.sleep(0.2)
            return True
        
        return False
    
    def start_delivery_mode(self):
        """Start enhanced delivery mode with hole alignment"""
        self.logger.info("🚚 STARTING ENHANCED DELIVERY MODE")
        self.logger.info("   Phase 1: Green target detection and approach")
        self.logger.info("   Phase 2: Rectangular hole detection and precise alignment") 
        self.logger.info("   Phase 3: Ball delivery and retreat")
        
        self.delivery_active = True
        self.start_time = time.time()
        
        try:
            self.enhanced_delivery_loop()
        except KeyboardInterrupt:
            self.logger.info("Delivery mode interrupted by user")
        except Exception as e:
            self.logger.error(f"Delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def enhanced_delivery_loop(self):
        """Main delivery loop with green detection -> hole alignment -> delivery"""
        search_direction = 1
        frames_without_target = 0
        max_frames_without_target = 30
        triangulation_attempts = 0
        
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
                        self.current_target = None
                        self.current_hole = None
                        self.delivery_vision.current_mode = "green_search"
                        continue
                
                # === PRIORITY 2: PHASE-BASED PROCESSING ===
                green_targets = self.delivery_vision.detect_green_targets(frame)
                holes = self.delivery_vision.detect_rectangular_holes(frame)
                
                # Create visualization
                debug_frame = self.delivery_vision.draw_delivery_visualization(frame, green_targets, holes)
                
                # Show frame if display available
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('GolfBot Enhanced Delivery - Hole Alignment', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # === PHASE 1: GREEN TARGET SEARCH AND APPROACH ===
                if self.delivery_vision.current_mode == "green_search":
                    if green_targets:
                        frames_without_target = 0
                        primary_target = green_targets[0]
                        self.current_target = primary_target
                        
                        # Check if close enough to switch to hole detection
                        if self.delivery_vision.should_switch_to_hole_mode(primary_target):
                            self.logger.info("🎯 Switching to hole detection mode")
                            self.delivery_vision.current_mode = "hole_alignment"
                            self.delivery_vision.hole_detection_enabled = True
                            triangulation_attempts = 0
                            continue
                        
                        # Continue centering on green target
                        self.center_on_green_target(primary_target)
                        
                    else:
                        # No green targets - search
                        frames_without_target += 1
                        self.current_target = None
                        
                        if frames_without_target >= max_frames_without_target:
                            search_direction *= -1
                            frames_without_target = 0
                            self.logger.info(f"🔍 Changing search direction: {'RIGHT' if search_direction > 0 else 'LEFT'}")
                        
                        self.search_for_targets(search_direction)
                
                # === PHASE 2: HOLE DETECTION AND PRECISE ALIGNMENT ===
                elif self.delivery_vision.current_mode == "hole_alignment":
                    if holes:
                        primary_hole = holes[0]
                        self.current_hole = primary_hole
                        self.delivery_vision.last_hole_seen = time.time()
                        
                        # Check if hole is perfectly aligned
                        if self.delivery_vision.is_hole_aligned(primary_hole):
                            self.logger.info("🎯 Hole perfectly aligned! Starting final approach")
                            self.delivery_vision.current_mode = "approaching_hole"
                            continue
                        
                        # Perform precise alignment using triangulation
                        if triangulation_attempts < self.max_triangulation_attempts:
                            self.align_with_hole_triangulation(primary_hole)
                            triangulation_attempts += 1
                        else:
                            self.logger.warning(f"⏰ Max triangulation attempts reached - proceeding anyway")
                            self.delivery_vision.current_mode = "approaching_hole"
                        
                    else:
                        # No holes detected - might need to back up or search differently
                        if (self.delivery_vision.last_hole_seen and 
                            time.time() - self.delivery_vision.last_hole_seen > 3.0):
                            
                            self.logger.warning("❌ Lost hole detection - returning to green search")
                            self.delivery_vision.current_mode = "green_search"
                            self.delivery_vision.hole_detection_enabled = False
                            self.current_hole = None
                        else:
                            # Try small movements to find hole
                            self.logger.info("🔍 Searching for hole with micro-movements")
                            self.hardware.move_forward(duration=0.1, speed=0.3)
                            time.sleep(0.1)
                
                # === PHASE 3: FINAL APPROACH AND DELIVERY ===
                elif self.delivery_vision.current_mode == "approaching_hole":
                    if self.current_hole:
                        self.approach_aligned_hole(self.current_hole)
                        
                        # Mission complete - return to search mode
                        self.logger.info("✅ Delivery complete - returning to search mode")
                        self.delivery_vision.current_mode = "green_search"
                        self.delivery_vision.hole_detection_enabled = False
                        self.current_target = None
                        self.current_hole = None
                        triangulation_attempts = 0
                        
                        # Take a break before next search
                        time.sleep(2.0)
                    else:
                        self.logger.error("❌ No hole available for approach - resetting")
                        self.delivery_vision.current_mode = "green_search"
                        self.delivery_vision.hole_detection_enabled = False
                
                # Adaptive timing based on current mode
                if self.delivery_vision.current_mode == "hole_alignment":
                    time.sleep(0.05)  # Faster for precise alignment
                elif wall_danger:
                    time.sleep(0.2)   # Slower when walls detected
                else:
                    time.sleep(0.1)   # Normal timing
                
            except Exception as e:
                self.logger.error(f"Delivery loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
    
    def search_for_targets(self, direction: int):
        """Search for green targets by turning"""
        if direction > 0:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching right for green targets")
            self.hardware.turn_right(duration=self.search_turn_duration, speed=self.approach_speed)
        else:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching left for green targets")
            self.hardware.turn_left(duration=self.search_turn_duration, speed=self.approach_speed)
        
        time.sleep(0.2)
    
    def stop_delivery(self):
        """Stop delivery mode with comprehensive stats"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 ENHANCED DELIVERY MODE COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Final ball count: {self.hardware.get_ball_count()}")
        self.logger.info(f"   Final mode: {self.delivery_vision.current_mode}")
        self.logger.info(f"   Hole detection: {'Enabled' if self.delivery_vision.hole_detection_enabled else 'Disabled'}")
        
        # Wall avoidance stats
        wall_status = self.delivery_vision.boundary_system.get_status()
        self.logger.info(f"   Walls detected: {wall_status['walls_detected']}")
        self.logger.info(f"   Wall avoidance triggers: {wall_status['walls_triggered']}")
        
        cv2.destroyAllWindows()

def run_delivery_test():
    """Main entry point for enhanced delivery testing with hole alignment"""
    print("\n🚚 GOLFBOT ENHANCED DELIVERY SYSTEM TEST")
    print("="*70)
    print("PHASES:")
    print("1. GREEN TARGET DETECTION")
    print("   • Search for large green areas (delivery zones)")
    print("   • Approach and center on green targets")
    print("   • Switch to hole detection when close enough")
    print("")
    print("2. RECTANGULAR HOLE ALIGNMENT")
    print("   • Detect rectangular holes in walls using edge detection")
    print("   • Precise alignment using triangulation method:")
    print("     - Back up → Turn → Move forward (for lateral adjustments)")
    print("     - Forward/backward for distance adjustments")
    print("   • Very tight tolerance (±8 pixels) for perfect alignment")
    print("")
    print("3. BALL DELIVERY")
    print("   • Final approach to aligned hole")
    print("   • Automatic ball release")
    print("   • Safe retreat from delivery area")
    print("")
    print("TRIANGULATION METHOD:")
    print("• No direct left/right movement needed!")
    print("• Uses: backward → turn → forward sequence")
    print("• Creates lateral displacement without side movement")
    print("• Micro-adjustments based on hole offset")
    print("")
    print("FEATURES:")
    print("• Two-phase vision system (green → hole)")
    print("• Wall detection and avoidance throughout")
    print("• Adaptive movement speeds for each phase")
    print("• Comprehensive alignment verification")
    print("• Automatic mode switching based on proximity")
    print("")
    print("CONTROLS:")
    print("• Press 'q' in camera window to quit")
    print("• System will automatically progress through phases")
    print("="*70)
    
    input("Press Enter to start enhanced delivery test...")
    
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
        print("🎯 Starting with GREEN TARGET SEARCH phase...")
        
        # Create and start enhanced delivery system
        delivery_system = DeliverySystem(hardware, vision)
        delivery_system.start_delivery_mode()
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Enhanced delivery test interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Enhanced delivery test error: {e}")
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