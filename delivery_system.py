#!/usr/bin/env python3
"""
Enhanced GolfBot Delivery System - Hole/Opening Detection and Precise Alignment
Detects green area first, then finds the specific hole/opening for ball delivery
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
class DeliveryHole:
    """Class to store detected hole/opening information"""
    center: Tuple[int, int]
    area: int
    confidence: float
    distance_from_center: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    is_valid_opening: bool
    opening_width: int
    opening_height: int

@dataclass
class GreenTarget:
    """Class to store detected green target information"""
    center: Tuple[int, int]
    area: int
    confidence: float
    distance_from_center: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height

class EnhancedDeliveryVisionSystem:
    """Enhanced vision system for green area + hole detection with precise alignment"""
    
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
        
        # HOLE DETECTION PARAMETERS
        self.hole_detection_enabled = False  # Enable when in green area
        self.last_green_detection_time = None
        
        # ADAPTIVE CENTERING TOLERANCES
        self.hole_centering_tolerance_x = 15  # Precise X alignment for hole
        self.hole_centering_tolerance_y = 12  # Precise Y alignment for hole
        self.green_centering_tolerance_x = 30 # Broader X tolerance for green area
        self.green_centering_tolerance_y = 25 # Broader Y tolerance for green area
        
        # OSCILLATION PREVENTION
        self.last_direction = None
        self.direction_change_count = 0
        self.oscillation_detected = False
        self.stable_frames = 0
        self.min_stable_frames = 3
        
        # STATE TRACKING
        self.current_mode = "seeking_green"  # "seeking_green" or "aligning_hole"
        self.green_area_found = False
        
    def detect_green_areas(self, frame) -> List[GreenTarget]:
        """Detect green delivery areas in the frame"""
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
        
        # Sort by confidence and size
        green_targets.sort(key=lambda t: (-t.confidence, -t.area))
        
        return green_targets[:2]  # Return top 2 green areas
    
    def detect_delivery_holes(self, frame, green_area_bbox: Tuple[int, int, int, int] = None) -> List[DeliveryHole]:
        """Detect holes/openings within or near green delivery areas"""
        holes = []
        
        if frame is None:
            return holes
        
        h, w = frame.shape[:2]
        
        # If we have a green area, focus detection around it
        if green_area_bbox:
            x, y, w_rect, h_rect = green_area_bbox
            # Expand search area around green zone
            margin = 30
            search_x1 = max(0, x - margin)
            search_y1 = max(0, y - margin)
            search_x2 = min(w, x + w_rect + margin)
            search_y2 = min(h, y + h_rect + margin)
            
            search_frame = frame[search_y1:search_y2, search_x1:search_x2]
            offset_x, offset_y = search_x1, search_y1
        else:
            # Search entire frame
            search_frame = frame
            offset_x, offset_y = 0, 0
        
        if search_frame.size == 0:
            return holes
        
        # HOLE DETECTION METHOD 1: DARK OPENINGS/GAPS
        gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect very dark areas (potential holes/openings)
        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up to find solid dark regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find dark contours
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Hole should be reasonably sized (not too small, not too large)
            if 100 < area < 2000:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                center = (x + w_rect // 2 + offset_x, y + h_rect // 2 + offset_y)
                
                # Check aspect ratio - holes are usually more rectangular/square
                aspect_ratio = w_rect / max(h_rect, 1)
                
                # Validate as potential hole
                if 0.3 < aspect_ratio < 3.0 and w_rect > 8 and h_rect > 8:
                    # Calculate confidence based on darkness and shape regularity
                    roi = gray[y:y+h_rect, x:x+w_rect]
                    if roi.size > 0:
                        mean_darkness = np.mean(roi)
                        darkness_conf = max(0, (50 - mean_darkness) / 50)  # Darker = higher confidence
                        
                        shape_conf = min(1.0, area / 500)  # Reasonable size
                        
                        confidence = (darkness_conf + shape_conf) / 2
                        
                        if confidence > 0.4:
                            distance_from_center = np.sqrt(
                                (center[0] - self.frame_center_x)**2 + 
                                (center[1] - self.frame_center_y)**2
                            )
                            
                            hole = DeliveryHole(
                                center=center,
                                area=area,
                                confidence=confidence,
                                distance_from_center=distance_from_center,
                                bbox=(x + offset_x, y + offset_y, w_rect, h_rect),
                                is_valid_opening=True,
                                opening_width=w_rect,
                                opening_height=h_rect
                            )
                            holes.append(hole)
        
        # HOLE DETECTION METHOD 2: EDGE-BASED DETECTION FOR OPENINGS
        # Detect edges and look for gap patterns
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for vertical gaps (typical of wall openings)
        # Use morphology to find vertical structures
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Find contours in vertical edge structure
        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 80 < area < 1500:  # Different size range for edge-detected openings
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                center = (x + w_rect // 2 + offset_x, y + h_rect // 2 + offset_y)
                
                # Look for gap-like features (taller than wide for wall openings)
                aspect_ratio = h_rect / max(w_rect, 1)
                
                if aspect_ratio > 1.2 and h_rect > 15:  # Vertical opening characteristics
                    confidence = min(1.0, (area / 400) * (aspect_ratio / 3.0))
                    
                    if confidence > 0.3:
                        distance_from_center = np.sqrt(
                            (center[0] - self.frame_center_x)**2 + 
                            (center[1] - self.frame_center_y)**2
                        )
                        
                        # Check if this hole is too close to existing holes (avoid duplicates)
                        is_duplicate = False
                        for existing_hole in holes:
                            dist = np.sqrt((center[0] - existing_hole.center[0])**2 + 
                                         (center[1] - existing_hole.center[1])**2)
                            if dist < 20:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            hole = DeliveryHole(
                                center=center,
                                area=area,
                                confidence=confidence,
                                distance_from_center=distance_from_center,
                                bbox=(x + offset_x, y + offset_y, w_rect, h_rect),
                                is_valid_opening=True,
                                opening_width=w_rect,
                                opening_height=h_rect
                            )
                            holes.append(hole)
        
        # Sort holes by confidence and proximity to center
        holes.sort(key=lambda h: (-h.confidence, h.distance_from_center))
        
        return holes[:3]  # Return top 3 holes
    
    def is_green_area_centered(self, green_target: GreenTarget) -> bool:
        """FIXED: More lenient green area centering to prevent infinite loops"""
        x_tolerance = self.green_centering_tolerance_x
        y_tolerance = self.green_centering_tolerance_y
        
        x_offset = abs(green_target.center[0] - self.frame_center_x)
        y_offset = abs(green_target.center[1] - self.frame_center_y)
        
        centered = (x_offset <= x_tolerance and y_offset <= y_tolerance)
        
        if centered:
            self.stable_frames += 1
            # FIXED: Reduce required stable frames from 3 to 2 for faster progression
            return self.stable_frames >= max(2, self.min_stable_frames - 1)
        else:
            self.stable_frames = 0
            return False
    
    def is_hole_aligned(self, hole: DeliveryHole) -> bool:
        """Check if hole is precisely aligned for delivery"""
        x_tolerance = self.hole_centering_tolerance_x
        y_tolerance = self.hole_centering_tolerance_y
        
        x_offset = abs(hole.center[0] - self.frame_center_x)
        y_offset = abs(hole.center[1] - self.frame_center_y)
        
        aligned = (x_offset <= x_tolerance and y_offset <= y_tolerance)
        
        if aligned:
            self.stable_frames += 1
            return self.stable_frames >= self.min_stable_frames
        else:
            self.stable_frames = 0
            return False
    
    def get_centering_direction(self, target_center: Tuple[int, int], target_type: str = "green") -> tuple:
        """Get direction to center on target with appropriate tolerances"""
        if target_type == "hole":
            x_tolerance = self.hole_centering_tolerance_x
            y_tolerance = self.hole_centering_tolerance_y
        else:  # green area
            x_tolerance = self.green_centering_tolerance_x
            y_tolerance = self.green_centering_tolerance_y
        
        x_offset = target_center[0] - self.frame_center_x
        y_offset = target_center[1] - self.frame_center_y
        
        # X-axis (turning) with hysteresis
        hysteresis = 3 if target_type == "hole" else 5
        
        if abs(x_offset) <= x_tolerance:
            x_direction = 'centered'
        elif x_offset > (x_tolerance + hysteresis):
            x_direction = 'right'
        elif x_offset < -(x_tolerance + hysteresis):
            x_direction = 'left'
        else:
            x_direction = 'centered'  # In hysteresis zone
        
        # Y-axis (distance)
        if abs(y_offset) <= y_tolerance:
            y_direction = 'centered'
        elif y_offset > (y_tolerance + hysteresis):
            y_direction = 'backward'
        elif y_offset < -(y_tolerance + hysteresis):
            y_direction = 'forward'
        else:
            y_direction = 'centered'  # In hysteresis zone
        
        # OSCILLATION DETECTION
        current_direction = x_direction if x_direction != 'centered' else y_direction
        
        if (self.last_direction and 
            current_direction != 'centered' and 
            self.last_direction != 'centered' and
            current_direction != self.last_direction):
            
            self.direction_change_count += 1
            if self.direction_change_count >= 3:
                self.oscillation_detected = True
                self.logger.warning(f"🔄 OSCILLATION DETECTED in {target_type} centering")
        
        self.last_direction = current_direction
        return x_direction, y_direction
    
    def reset_centering_state(self):
        """Reset centering state when starting new target"""
        self.last_direction = None
        self.direction_change_count = 0
        self.oscillation_detected = False
        self.stable_frames = 0
        self.logger.info("🔄 Centering state reset")
    
    def detect_walls(self, frame) -> bool:
        """Detect wall boundaries that should be avoided"""
        return self.boundary_system.detect_boundaries(frame)
    
    def get_wall_avoidance_command(self, frame) -> Optional[str]:
        """Get wall avoidance command if walls are detected"""
        return self.boundary_system.get_avoidance_command(frame)
    
    def draw_enhanced_detection(self, frame, green_targets: List[GreenTarget], holes: List[DeliveryHole]) -> np.ndarray:
        """Draw enhanced detection visualization with green areas and holes"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # === WALL DETECTION VISUALIZATION ===
        result = self.boundary_system.draw_boundary_visualization(result)
        
        # === MODE-SPECIFIC CENTERING ZONES ===
        if self.current_mode == "seeking_green":
            # Show green area centering zone
            tolerance_x = self.green_centering_tolerance_x
            tolerance_y = self.green_centering_tolerance_y
            zone_color = (0, 255, 0)  # Green
            zone_label = "GREEN AREA CENTER ZONE"
        else:  # aligning_hole
            # Show hole alignment zone
            tolerance_x = self.hole_centering_tolerance_x
            tolerance_y = self.hole_centering_tolerance_y
            zone_color = (255, 255, 0)  # Yellow
            zone_label = "HOLE ALIGNMENT ZONE (PRECISE)"
        
        center_left = self.frame_center_x - tolerance_x
        center_right = self.frame_center_x + tolerance_x
        center_top = self.frame_center_y - tolerance_y
        center_bottom = self.frame_center_y + tolerance_y
        
        cv2.rectangle(result, (center_left, center_top), (center_right, center_bottom), zone_color, 2)
        cv2.putText(result, zone_label, (center_left + 5, center_top - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_color, 1)
        
        # === GREEN AREAS ===
        for i, green_target in enumerate(green_targets):
            x, y, w_rect, h_rect = green_target.bbox
            
            if i == 0:  # Primary green target
                color = (0, 255, 0)    # Bright green
                thickness = 3
                
                # Check if centered
                if self.current_mode == "seeking_green":
                    centered = self.is_green_area_centered(green_target)
                    if centered:
                        cv2.putText(result, "GREEN AREA CENTERED!", (green_target.center[0] - 60, green_target.center[1] - 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        x_dir, y_dir = self.get_centering_direction(green_target.center, "green")
                        direction_text = f"{x_dir.upper()}, {y_dir.upper()}"
                        cv2.putText(result, direction_text, (green_target.center[0] - 40, green_target.center[1] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Arrow to target
                        cv2.arrowedLine(result, (self.frame_center_x, self.frame_center_y), 
                                       green_target.center, (255, 255, 0), 2)
            else:
                color = (0, 150, 0)    # Darker green
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            cv2.circle(result, green_target.center, 5, color, -1)
            
            # Label
            cv2.putText(result, f"G{i+1}", (green_target.center[0] - 10, green_target.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # === DELIVERY HOLES ===
        for i, hole in enumerate(holes):
            x, y, w_rect, h_rect = hole.bbox
            
            if i == 0:  # Primary hole
                color = (0, 255, 255)  # Cyan for holes
                thickness = 3
                
                # Check if aligned
                if self.current_mode == "aligning_hole":
                    aligned = self.is_hole_aligned(hole)
                    if aligned:
                        cv2.putText(result, "HOLE ALIGNED - DELIVER!", (hole.center[0] - 70, hole.center[1] - 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        x_dir, y_dir = self.get_centering_direction(hole.center, "hole")
                        direction_text = f"ALIGN: {x_dir.upper()}, {y_dir.upper()}"
                        cv2.putText(result, direction_text, (hole.center[0] - 50, hole.center[1] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Precise alignment arrow
                        cv2.arrowedLine(result, (self.frame_center_x, self.frame_center_y), 
                                       hole.center, (0, 255, 255), 3)
            else:
                color = (0, 180, 180)  # Darker cyan
                thickness = 2
            
            # Draw hole detection
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            cv2.circle(result, hole.center, 3, color, -1)
            
            # Hole info
            cv2.putText(result, f"H{i+1}", (hole.center[0] - 8, hole.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(result, f"{hole.opening_width}x{hole.opening_height}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Frame center crosshair
        cv2.line(result, (self.frame_center_x - 15, self.frame_center_y), 
                (self.frame_center_x + 15, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y - 15), 
                (self.frame_center_x, self.frame_center_y + 15), (255, 255, 255), 2)
        
        # === STATUS OVERLAY ===
        overlay_height = 140
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.6, overlay, 0.4, 0)
        
        # Status text
        cv2.putText(result, "ENHANCED DELIVERY - Green Area + Hole Detection", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current mode
        mode_text = f"Mode: {self.current_mode.replace('_', ' ').title()}"
        mode_color = (0, 255, 0) if self.current_mode == "seeking_green" else (0, 255, 255)
        cv2.putText(result, mode_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Detection counts
        detection_text = f"Green Areas: {len(green_targets)} | Holes: {len(holes)}"
        cv2.putText(result, detection_text, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Wall status
        wall_status = self.boundary_system.get_status()
        wall_danger = wall_status['walls_triggered'] > 0
        wall_text = f"Walls: {'DANGER' if wall_danger else 'Safe'}"
        wall_color = (0, 0, 255) if wall_danger else (0, 255, 0)
        cv2.putText(result, wall_text, (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, wall_color, 1)
        
        # Progress indicator
        if self.current_mode == "seeking_green" and green_targets:
            progress_text = "Step 1: Centering on green area..."
        elif self.current_mode == "aligning_hole" and holes:
            progress_text = "Step 2: Aligning with delivery hole..."
        else:
            progress_text = "Scanning for delivery zone..."
        
        cv2.putText(result, progress_text, (10, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return result

class EnhancedDeliverySystem:
    """Enhanced delivery system with green area detection + precise hole alignment"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = EnhancedDeliveryVisionSystem(vision_system)
        
        # State management
        self.current_green_target = None
        self.current_hole_target = None
        self.delivery_active = False
        self.start_time = None
        
        # ENHANCED MOVEMENT PARAMETERS
        self.base_search_turn_duration = 0.8
        self.green_centering_turn_duration = 0.4
        self.hole_centering_turn_duration = 0.2  # More precise for hole alignment
        self.approach_speed = 0.3  # Slower for precision
        self.search_speed = 0.5
        
    def center_on_green_area(self, green_target: GreenTarget):
        """Center robot on green area (coarse centering)"""
        x_direction, y_direction = self.delivery_vision.get_centering_direction(green_target.center, "green")
        
        if x_direction == 'centered' and y_direction == 'centered':
            return
        
        turn_duration = self.green_centering_turn_duration
        speed = self.search_speed * 0.9
        
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"🎯 Green centering: {x_direction}, {y_direction}")
        
        # Execute movement
        if x_direction == 'right':
            self.hardware.turn_right(duration=turn_duration, speed=speed)
        elif x_direction == 'left':
            self.hardware.turn_left(duration=turn_duration, speed=speed)
        elif y_direction == 'forward':
            self.hardware.move_forward(duration=turn_duration, speed=speed * 0.8)
        elif y_direction == 'backward':
            self.hardware.move_backward(duration=turn_duration, speed=speed * 0.8)
        
        time.sleep(0.1)
    
    def align_with_hole(self, hole: DeliveryHole):
        """Precisely align robot with delivery hole"""
        x_direction, y_direction = self.delivery_vision.get_centering_direction(hole.center, "hole")
        
        if x_direction == 'centered' and y_direction == 'centered':
            return
        
        # Precise movement parameters for hole alignment
        turn_duration = self.hole_centering_turn_duration
        speed = self.approach_speed
        
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"🔧 PRECISE hole alignment: {x_direction}, {y_direction}")
        
        # Execute precise movement
        if x_direction == 'right':
            self.hardware.turn_right(duration=turn_duration, speed=speed)
        elif x_direction == 'left':
            self.hardware.turn_left(duration=turn_duration, speed=speed)
        elif y_direction == 'forward':
            self.hardware.move_forward(duration=turn_duration * 0.8, speed=speed * 0.8)
        elif y_direction == 'backward':
            self.hardware.move_backward(duration=turn_duration * 0.8, speed=speed * 0.8)
        
        time.sleep(0.15)  # Longer pause for stability
    
    def deliver_balls_to_hole(self, hole: DeliveryHole):
        """Deliver balls through the aligned hole"""
        self.logger.info("📦 DELIVERING BALLS THROUGH HOLE!")
        
        # Move forward slightly to get closer to hole
        self.logger.info("Step 1: Moving closer to hole")
        self.hardware.move_forward(duration=0.4, speed=self.approach_speed)
        time.sleep(0.2)
        
        # Release balls if we have any
        if self.hardware.has_balls():
            self.logger.info("Step 2: Releasing balls through hole")
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls through delivery hole!")
            
            # Wait for balls to exit
            time.sleep(1.0)
            
            # Back away from hole
            self.logger.info("Step 3: Backing away from delivery hole")
            for back_step in range(3):
                self.hardware.move_backward(duration=0.4, speed=self.approach_speed)
                time.sleep(0.2)
        else:
            self.logger.info("ℹ️  No balls to deliver - backing away from hole")
            self.hardware.move_backward(duration=0.5, speed=self.approach_speed)
            time.sleep(0.2)
    
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
    
    def search_for_delivery_zone(self, search_direction: int):
        """Search for green delivery areas"""
        if search_direction > 0:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching right for green delivery zone")
            self.hardware.turn_right(duration=self.base_search_turn_duration, speed=self.search_speed)
        else:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching left for green delivery zone")
            self.hardware.turn_left(duration=self.base_search_turn_duration, speed=self.search_speed)
        
        time.sleep(0.2)
    
    def start_enhanced_delivery_mode(self):
        """Start enhanced delivery mode with 2-stage targeting"""
        self.logger.info("🚚 STARTING ENHANCED DELIVERY MODE")
        self.logger.info("   Stage 1: Detect and center on GREEN delivery area")
        self.logger.info("   Stage 2: Detect and align with DELIVERY HOLE")
        self.logger.info("   Stage 3: Deliver balls through hole")
        
        self.delivery_active = True
        self.start_time = time.time()
        
        try:
            self.enhanced_delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Enhanced delivery mode interrupted by user")
        except Exception as e:
            self.logger.error(f"Enhanced delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def enhanced_delivery_main_loop(self):
        """FIXED: Main delivery loop with improved green area centering"""
        search_direction = 1  # 1 for right, -1 for left
        frames_without_target = 0
        max_frames_without_target = 25
        
        # FIXED: Add timeout and progress tracking for green centering
        green_centering_timeout = 8.0  # Maximum time to spend centering on green
        green_centering_start_time = None
        last_green_center = None
        
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
                        # Reset targeting after wall avoidance
                        self.current_green_target = None
                        self.current_hole_target = None
                        self.delivery_vision.reset_centering_state()
                        green_centering_start_time = None  # FIXED: Reset centering timeout
                        continue
                
                # === PRIORITY 2: DETECTION PHASE ===
                green_targets = self.delivery_vision.detect_green_areas(frame)
                holes = []
                
                # If we have green targets, look for holes within them
                if green_targets and self.delivery_vision.current_mode == "aligning_hole":
                    primary_green = green_targets[0]
                    holes = self.delivery_vision.detect_delivery_holes(frame, primary_green.bbox)
                
                # Create visualization
                debug_frame = self.delivery_vision.draw_enhanced_detection(frame, green_targets, holes)
                
                # Show frame if display available
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('Enhanced Delivery - Green Area + Hole Detection', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # === STAGE 1: GREEN AREA TARGETING (FIXED) ===
                if self.delivery_vision.current_mode == "seeking_green":
                    if green_targets:
                        frames_without_target = 0
                        primary_green = green_targets[0]
                        
                        # FIXED: Better "new target" detection - only reset if significantly different
                        is_new_target = False
                        if self.current_green_target is None:
                            is_new_target = True
                            green_centering_start_time = time.time()
                            self.logger.info("🎯 New green area detected - starting centering")
                        else:
                            # Check if this is actually a different green area (not just slight movement)
                            distance_from_last = np.sqrt(
                                (primary_green.center[0] - self.current_green_target.center[0])**2 +
                                (primary_green.center[1] - self.current_green_target.center[1])**2
                            )
                            
                            # FIXED: Only consider it "new" if it moved significantly (>100 pixels)
                            if distance_from_last > 100:
                                is_new_target = True
                                green_centering_start_time = time.time()
                                self.logger.info(f"🎯 Green area moved significantly ({distance_from_last:.0f}px) - resetting centering")
                        
                        if is_new_target:
                            self.delivery_vision.reset_centering_state()
                        
                        self.current_green_target = primary_green
                        
                        # FIXED: Add centering timeout to prevent infinite loops
                        if green_centering_start_time:
                            centering_elapsed = time.time() - green_centering_start_time
                            
                            # TIMEOUT: If we've been centering too long, force progression
                            if centering_elapsed > green_centering_timeout:
                                self.logger.warning(f"⏰ Green centering timeout ({centering_elapsed:.1f}s) - forcing progression to hole detection")
                                self.delivery_vision.current_mode = "aligning_hole"
                                self.delivery_vision.reset_centering_state()
                                green_centering_start_time = None
                                time.sleep(0.3)
                                continue
                        
                        # Check if green area is centered
                        if self.delivery_vision.is_green_area_centered(primary_green):
                            centering_elapsed = time.time() - green_centering_start_time if green_centering_start_time else 0
                            self.logger.info(f"✅ Green area centered! (took {centering_elapsed:.1f}s) Switching to hole detection mode...")
                            self.delivery_vision.current_mode = "aligning_hole"
                            self.delivery_vision.reset_centering_state()
                            green_centering_start_time = None
                            time.sleep(0.3)  # Brief pause for mode switch
                        else:
                            # FIXED: More aggressive centering to avoid getting stuck
                            centering_elapsed = time.time() - green_centering_start_time if green_centering_start_time else 0
                            
                            # Use more aggressive movements if we've been centering for a while
                            if centering_elapsed > 4.0:
                                self.logger.info(f"🚀 Using aggressive centering after {centering_elapsed:.1f}s")
                                # Continue centering on green area with increased aggression
                                self.center_on_green_area_aggressive(primary_green)
                            else:
                                # Normal centering
                                self.center_on_green_area(primary_green)
                    
                    else:
                        # No green areas found - search
                        frames_without_target += 1
                        self.current_green_target = None
                        green_centering_start_time = None  # FIXED: Reset timeout when no target
                        
                        if frames_without_target >= max_frames_without_target:
                            search_direction *= -1
                            frames_without_target = 0
                            self.logger.info(f"🔍 Changing search direction: {'RIGHT' if search_direction > 0 else 'LEFT'}")
                        
                        self.search_for_delivery_zone(search_direction)
                
                # === STAGE 2: HOLE ALIGNMENT ===
                elif self.delivery_vision.current_mode == "aligning_hole":
                    # First check if we still have the green area in view
                    if not green_targets:
                        self.logger.warning("❌ Lost sight of green area - returning to green seeking mode")
                        self.delivery_vision.current_mode = "seeking_green"
                        self.current_green_target = None
                        self.current_hole_target = None
                        self.delivery_vision.reset_centering_state()
                        green_centering_start_time = None  # FIXED: Reset timeout
                        continue
                    
                    # Look for holes within the green area
                    if holes:
                        frames_without_target = 0
                        primary_hole = holes[0]
                        
                        # Check if this is a new hole target
                        if (self.current_hole_target is None or 
                            abs(primary_hole.center[0] - self.current_hole_target.center[0]) > 30):
                            self.logger.info("🕳️  New delivery hole detected - resetting alignment")
                            self.delivery_vision.reset_centering_state()
                        
                        self.current_hole_target = primary_hole
                        
                        # Check if hole is precisely aligned
                        if self.delivery_vision.is_hole_aligned(primary_hole):
                            self.logger.info("🎯 HOLE PERFECTLY ALIGNED! Delivering balls...")
                            self.deliver_balls_to_hole(primary_hole)
                            
                            # Reset to green seeking mode for next delivery
                            self.delivery_vision.current_mode = "seeking_green"
                            self.current_green_target = None
                            self.current_hole_target = None
                            self.delivery_vision.reset_centering_state()
                            green_centering_start_time = None  # FIXED: Reset timeout
                        else:
                            # Continue precise alignment with hole
                            self.align_with_hole(primary_hole)
                    
                    else:
                        # No holes found in green area
                        frames_without_target += 1
                        self.current_hole_target = None
                        
                        if frames_without_target >= 15:  # Shorter timeout for hole detection
                            self.logger.warning("❌ No delivery hole found in green area - adjusting position")
                            
                            # Try moving slightly to get better view of holes
                            self.hardware.move_forward(duration=0.2, speed=0.3)
                            time.sleep(0.2)
                            frames_without_target = 0
                        
                        time.sleep(0.1)
                
                # === ADAPTIVE TIMING ===
                if self.delivery_vision.current_mode == "aligning_hole":
                    time.sleep(0.05)  # Faster when doing precise hole alignment
                elif wall_danger:
                    time.sleep(0.2)   # Slower when walls detected
                else:
                    time.sleep(0.1)   # Normal timing
                
            except Exception as e:
                self.logger.error(f"Enhanced delivery loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)

    def center_on_green_area_aggressive(self, green_target: GreenTarget):
        """FIXED: More aggressive centering for when normal centering gets stuck"""
        x_direction, y_direction = self.delivery_vision.get_centering_direction(green_target.center, "green")
        
        if x_direction == 'centered' and y_direction == 'centered':
            return
        
        # More aggressive parameters
        turn_duration = self.green_centering_turn_duration * 1.5  # 50% longer movements
        speed = self.search_speed  # Full speed
        
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"🚀 AGGRESSIVE green centering: {x_direction}, {y_direction}")
        
        # Execute more aggressive movement
        if x_direction == 'right':
            self.hardware.turn_right(duration=turn_duration, speed=speed)
        elif x_direction == 'left':
            self.hardware.turn_left(duration=turn_duration, speed=speed)
        elif y_direction == 'forward':
            self.hardware.move_forward(duration=turn_duration, speed=speed * 0.8)
        elif y_direction == 'backward':
            self.hardware.move_backward(duration=turn_duration, speed=speed * 0.8)
        
        time.sleep(0.2)  # Longer pause for aggressive movements
    
    def stop_delivery(self):
        """Stop enhanced delivery mode"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 ENHANCED DELIVERY MODE COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Final ball count: {self.hardware.get_ball_count()}")
        self.logger.info(f"   Final mode: {self.delivery_vision.current_mode}")
        
        # Enhanced delivery stats
        wall_status = self.delivery_vision.boundary_system.get_status()
        self.logger.info(f"   Walls detected: {wall_status['walls_detected']}")
        self.logger.info(f"   Wall avoidance triggers: {wall_status['walls_triggered']}")
        self.logger.info(f"   Oscillation events: {self.delivery_vision.direction_change_count}")
        
        cv2.destroyAllWindows()

def run_enhanced_delivery_test():
    """Main entry point for enhanced delivery testing with hole detection"""
    print("\n🚚 ENHANCED GOLFBOT DELIVERY SYSTEM TEST")
    print("="*70)
    print("ENHANCED 2-STAGE DELIVERY PROCESS:")
    print("="*70)
    print("STAGE 1 - GREEN AREA DETECTION:")
    print("• Search for GREEN delivery areas using HSV color detection")
    print("• Center robot on detected green area (coarse positioning)")
    print("• Switch to hole detection mode when green area is centered")
    print()
    print("STAGE 2 - HOLE/OPENING DETECTION:")
    print("• Detect dark openings/holes within or near green area")
    print("• Use multiple detection methods:")
    print("  - Dark region detection (holes appear darker)")
    print("  - Edge-based gap detection (wall openings)")
    print("• Precisely align robot with detected hole center")
    print("• Much tighter tolerances for hole alignment vs green centering")
    print()
    print("STAGE 3 - BALL DELIVERY:")
    print("• Move forward slightly to approach hole")
    print("• Release balls through precisely aligned hole")
    print("• Back away and reset to search for next delivery zone")
    print()
    print("ENHANCED FEATURES:")
    print("• Wall/boundary detection and avoidance")
    print("• Oscillation detection and prevention")
    print("• Adaptive movement (coarse for green, precise for holes)")
    print("• Visual feedback showing both green areas and detected holes")
    print("• Mode switching: 'seeking_green' → 'aligning_hole'")
    print("• Automatic retry if hole detection fails")
    print()
    print("TOLERANCES:")
    print(f"• Green area centering: ±30px X, ±25px Y (coarse)")
    print(f"• Hole alignment: ±15px X, ±12px Y (precise)")
    print(f"• Movement: 0.4s turns for green, 0.2s for holes")
    print("="*70)
    print("Press 'q' in the camera window to quit")
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
        
        # Create and start enhanced delivery system
        delivery_system = EnhancedDeliverySystem(hardware, vision)
        delivery_system.start_enhanced_delivery_mode()
        
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
    run_enhanced_delivery_test()