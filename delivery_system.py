#!/usr/bin/env python3
"""
Enhanced GolfBot Delivery System - Wall-Perpendicular Approach
Detects wall orientation and approaches green targets perpendicular to the wall
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
    wall_direction: Optional[str] = None  # 'top', 'bottom', 'left', 'right'
    approach_angle: Optional[float] = None  # Angle to approach perpendicular

@dataclass 
class WallSegment:
    """Class to store wall segment information"""
    direction: str  # 'top', 'bottom', 'left', 'right'
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    center_point: Tuple[int, int]
    length: int
    angle: float  # Wall angle in degrees

class EnhancedDeliveryVisionSystem:
    """Enhanced vision system with wall orientation detection for perpendicular approach"""
    
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
        
        # Wall detection parameters - BROADENED for better detection
        self.red_lower1 = np.array([0, 80, 80])     # More permissive red detection
        self.red_upper1 = np.array([20, 255, 255])
        self.red_lower2 = np.array([160, 80, 80])   # More permissive red detection
        self.red_upper2 = np.array([180, 255, 255])
        
        # Also detect orange/brown walls that might appear reddish
        self.orange_lower = np.array([5, 100, 100])
        self.orange_upper = np.array([25, 255, 255])
        
        # Approach state tracking
        self.approach_phase = "detect"  # "detect", "position", "align", "approach"
        self.target_approach_angle = None
        self.positioning_complete = False
        
    def detect_wall_segments(self, frame) -> List[WallSegment]:
        """Detect wall segments and their orientations around green targets - IMPROVED"""
        wall_segments = []
        
        if frame is None:
            return wall_segments
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create red/orange wall mask with broader detection
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask3 = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        red_mask = mask1 + mask2 + mask3
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # SIMPLIFIED WALL DETECTION - look for walls in specific regions
        # Check top region for horizontal wall
        top_region = red_mask[0:h//3, :]
        if np.sum(top_region > 0) > (w * h//3) * 0.05:  # 5% threshold
            # Found top wall
            wall_segment = WallSegment(
                direction='top',
                start_point=(0, h//6),
                end_point=(w, h//6),
                center_point=(w//2, h//6),
                length=w,
                angle=0.0
            )
            wall_segments.append(wall_segment)
            if config.DEBUG_VISION:
                self.logger.info("Detected TOP wall")
        
        # Check bottom region for horizontal wall
        bottom_region = red_mask[2*h//3:h, :]
        if np.sum(bottom_region > 0) > (w * h//3) * 0.05:
            # Found bottom wall
            wall_segment = WallSegment(
                direction='bottom',
                start_point=(0, 5*h//6),
                end_point=(w, 5*h//6),
                center_point=(w//2, 5*h//6),
                length=w,
                angle=0.0
            )
            wall_segments.append(wall_segment)
            if config.DEBUG_VISION:
                self.logger.info("Detected BOTTOM wall")
        
        # Check left region for vertical wall
        left_region = red_mask[:, 0:w//3]
        if np.sum(left_region > 0) > (h * w//3) * 0.05:
            # Found left wall
            wall_segment = WallSegment(
                direction='left',
                start_point=(w//6, 0),
                end_point=(w//6, h),
                center_point=(w//6, h//2),
                length=h,
                angle=90.0
            )
            wall_segments.append(wall_segment)
            if config.DEBUG_VISION:
                self.logger.info("Detected LEFT wall")
        
        # Check right region for vertical wall
        right_region = red_mask[:, 2*w//3:w]
        if np.sum(right_region > 0) > (h * w//3) * 0.05:
            # Found right wall
            wall_segment = WallSegment(
                direction='right',
                start_point=(5*w//6, 0),
                end_point=(5*w//6, h),
                center_point=(5*w//6, h//2),
                length=h,
                angle=90.0
            )
            wall_segments.append(wall_segment)
            if config.DEBUG_VISION:
                self.logger.info("Detected RIGHT wall")
        
        # FALLBACK: If no walls detected but green target exists, assume it's on the top wall
        if not wall_segments:
            if config.DEBUG_VISION:
                self.logger.warning("No walls detected - using fallback top wall assumption")
            wall_segment = WallSegment(
                direction='top',
                start_point=(0, h//4),
                end_point=(w, h//4),
                center_point=(w//2, h//4),
                length=w,
                angle=0.0
            )
            wall_segments.append(wall_segment)
        
        return wall_segments
    
    def _classify_wall_direction(self, x, y, w_rect, h_rect, frame_w, frame_h) -> Optional[str]:
        """Classify wall direction based on position and shape"""
        aspect_ratio = w_rect / max(h_rect, 1)
        
        # Determine if wall is more horizontal or vertical
        if aspect_ratio > 2.0:  # Wide wall (horizontal)
            if y < frame_h * 0.3:
                return 'top'
            elif y > frame_h * 0.7:
                return 'bottom'
        elif aspect_ratio < 0.5:  # Tall wall (vertical)
            if x < frame_w * 0.3:
                return 'left'
            elif x > frame_w * 0.7:
                return 'right'
        
        return None
    
    def detect_green_targets_with_walls(self, frame) -> List[GreenTarget]:
        """Detect green targets and determine which wall they're associated with"""
        green_targets = []
        
        if frame is None:
            return green_targets
        
        # First detect wall segments
        wall_segments = self.detect_wall_segments(frame)
        
        # Then detect green targets
        targets = self.detect_green_targets(frame)
        
        # Associate each green target with the nearest wall
        for target in targets:
            nearest_wall = self._find_nearest_wall(target, wall_segments)
            if nearest_wall:
                target.wall_direction = nearest_wall.direction
                target.approach_angle = self._calculate_approach_angle(target, nearest_wall)
            
            green_targets.append(target)
        
        return green_targets
    
    def detect_green_targets(self, frame) -> List[GreenTarget]:
        """Basic green target detection (unchanged from original)"""
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
                
                # Calculate confidence
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
        
        # Sort by confidence and size
        green_targets.sort(key=lambda t: (-t.confidence, -t.area))
        return green_targets[:3]
    
    def _find_nearest_wall(self, target: GreenTarget, wall_segments: List[WallSegment]) -> Optional[WallSegment]:
        """Find the nearest wall segment to a green target"""
        if not wall_segments:
            return None
        
        min_distance = float('inf')
        nearest_wall = None
        
        for wall in wall_segments:
            # Calculate distance from target to wall
            distance = self._point_to_line_distance(target.center, wall.start_point, wall.end_point)
            
            if distance < min_distance:
                min_distance = distance
                nearest_wall = wall
        
        return nearest_wall
    
    def _point_to_line_distance(self, point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
        """Calculate distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate line length
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Calculate distance
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / (line_length**2)))
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        return np.sqrt((x0 - projection_x)**2 + (y0 - projection_y)**2)
    
    def _calculate_approach_angle(self, target: GreenTarget, wall: WallSegment) -> float:
        """Calculate the angle needed to approach the target perpendicular to the wall"""
        if wall.direction == 'top':
            return 270.0  # Approach from bottom (upward)
        elif wall.direction == 'bottom':
            return 90.0   # Approach from top (downward)  
        elif wall.direction == 'left':
            return 0.0    # Approach from right (leftward)
        elif wall.direction == 'right':
            return 180.0  # Approach from left (rightward)
        else:
            return 0.0
    
    def get_positioning_command(self, target: GreenTarget) -> Optional[str]:
        """Get command to position robot for perpendicular approach"""
        if not target.wall_direction or target.approach_angle is None:
            return None
        
        robot_x = self.frame_center_x
        robot_y = self.frame_center_y
        target_x, target_y = target.center
        
        # Calculate where robot needs to be positioned for perpendicular approach
        approach_distance = 100  # Distance to maintain from target during positioning
        
        if target.wall_direction == 'top':
            # Position below target for upward approach
            desired_x = target_x
            desired_y = target_y + approach_distance
        elif target.wall_direction == 'bottom':
            # Position above target for downward approach
            desired_x = target_x
            desired_y = target_y - approach_distance
        elif target.wall_direction == 'left':
            # Position to the right of target for leftward approach
            desired_x = target_x + approach_distance
            desired_y = target_y
        elif target.wall_direction == 'right':
            # Position to the left of target for rightward approach
            desired_x = target_x - approach_distance
            desired_y = target_y
        else:
            return None
        
        # Calculate movement needed
        dx = desired_x - robot_x
        dy = desired_y - robot_y
        
        # Determine primary movement direction
        if abs(dx) > abs(dy):
            if dx > 20:
                return 'move_right'
            elif dx < -20:
                return 'move_left'
        else:
            if dy > 20:
                return 'move_backward'
            elif dy < -20:
                return 'move_forward'
        
        # If close enough, robot is positioned
        if abs(dx) < 20 and abs(dy) < 20:
            return 'positioned'
        
        return None
    
    def get_alignment_command(self, target: GreenTarget) -> Optional[str]:
        """Get command to align robot perpendicular to wall before approach"""
        if not target.wall_direction:
            return None
        
        # For simplicity, assume robot needs to turn to face the correct direction
        # In practice, you'd need to track robot's current orientation
        
        # This is a simplified alignment - you may need IMU/compass data for precise alignment
        if target.wall_direction in ['top', 'bottom']:
            return 'align_vertical'
        else:
            return 'align_horizontal'
    
    def draw_enhanced_detection(self, frame, targets: List[GreenTarget]) -> np.ndarray:
        """Draw enhanced detection with wall information and approach vectors"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw wall segments
        wall_segments = self.detect_wall_segments(frame)
        for wall in wall_segments:
            color = (0, 0, 255)  # Red for walls
            cv2.line(result, wall.start_point, wall.end_point, color, 3)
            
            # Label wall direction
            label_pos = (wall.center_point[0] - 20, wall.center_point[1] - 10)
            cv2.putText(result, wall.direction.upper(), label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw green targets with approach information
        for i, target in enumerate(targets):
            x, y, w_rect, h_rect = target.bbox
            
            if i == 0:
                color = (0, 255, 0)    # Bright green for primary target
                thickness = 3
                
                # Draw target
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
                cv2.circle(result, target.center, 5, color, -1)
                
                # Show wall association and approach angle
                if target.wall_direction:
                    info_text = f"Wall: {target.wall_direction.upper()}"
                    cv2.putText(result, info_text, (target.center[0] - 40, target.center[1] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if target.approach_angle is not None:
                        angle_text = f"Approach: {target.approach_angle:.0f}°"
                        cv2.putText(result, angle_text, (target.center[0] - 40, target.center[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Draw approach vector
                        approach_length = 60
                        angle_rad = np.radians(target.approach_angle)
                        end_x = target.center[0] + int(approach_length * np.cos(angle_rad))
                        end_y = target.center[1] + int(approach_length * np.sin(angle_rad))
                        
                        cv2.arrowedLine(result, target.center, (end_x, end_y), (255, 255, 0), 2)
                        cv2.putText(result, "APPROACH", (end_x - 30, end_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                color = (0, 150, 0)    # Darker green for secondary targets
                thickness = 2
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
                cv2.circle(result, target.center, 3, color, -1)
            
            # Target label
            cv2.putText(result, f"G{i+1}", (target.center[0] - 10, target.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Frame center crosshair
        cv2.line(result, (self.frame_center_x - 10, self.frame_center_y), 
                (self.frame_center_x + 10, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y - 10), 
                (self.frame_center_x, self.frame_center_y + 10), (255, 255, 255), 2)
        
        # Status overlay
        overlay_height = 140
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.6, overlay, 0.4, 0)
        
        # Status text
        cv2.putText(result, "ENHANCED DELIVERY - Perpendicular Wall Approach", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        target_count = len(targets)
        primary_target = targets[0] if targets else None
        
        if primary_target and primary_target.wall_direction:
            status = f"Target: Green at {primary_target.wall_direction.upper()} wall"
            approach_status = f"Approach Phase: {self.approach_phase.upper()}"
        else:
            status = "Scanning for green targets near walls..."
            approach_status = "Phase: DETECT"
        
        cv2.putText(result, f"Targets: {target_count} | {status}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(result, approach_status, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Wall detection info
        wall_count = len(wall_segments)
        cv2.putText(result, f"Walls detected: {wall_count}", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return result

class EnhancedDeliverySystem:
    """Enhanced delivery system with perpendicular wall approach"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = EnhancedDeliveryVisionSystem(vision_system)
        
        # Enhanced state management for perpendicular approach
        self.current_target = None
        self.delivery_active = False
        self.start_time = None
        
        # Approach phases: detect -> position -> align -> approach -> deliver
        self.approach_phase = "detect"
        self.phase_start_time = None
        
    def start_enhanced_delivery_mode(self):
        """Start enhanced delivery mode with perpendicular approach"""
        self.logger.info("🚚 STARTING ENHANCED DELIVERY MODE - Perpendicular Wall Approach")
        self.logger.info("   Features: Wall orientation detection, perpendicular positioning, straight approach")
        
        self.delivery_active = True
        self.start_time = time.time()
        self.approach_phase = "detect"
        
        try:
            self.enhanced_delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Enhanced delivery mode interrupted by user")
        except Exception as e:
            self.logger.error(f"Enhanced delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def enhanced_delivery_main_loop(self):
        """Main delivery loop with phase-based perpendicular approach"""
        search_direction = 1
        frames_without_target = 0
        max_frames_without_target = 30
        
        while self.delivery_active:
            try:
                # Get current frame
                ret, frame = self.vision_system.get_frame()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Detect green targets with wall associations
                green_targets = self.delivery_vision.detect_green_targets_with_walls(frame)
                
                # Create visualization
                debug_frame = self.delivery_vision.draw_enhanced_detection(frame, green_targets)
                
                # Show frame if display available
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('Enhanced Delivery - Perpendicular Approach', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # PHASE-BASED PROCESSING
                if green_targets:
                    frames_without_target = 0
                    primary_target = green_targets[0]
                    
                    # Check if this is a new target
                    if (self.current_target is None or 
                        abs(primary_target.center[0] - self.current_target.center[0]) > 50):
                        self.logger.info("🎯 New target detected - starting approach sequence")
                        self.current_target = primary_target
                        self.approach_phase = "detect"
                        self.phase_start_time = time.time()
                    
                    self.current_target = primary_target
                    
                    # Execute current phase
                    if self.approach_phase == "detect":
                        self.handle_detect_phase()
                    elif self.approach_phase == "position":
                        self.handle_position_phase()
                    elif self.approach_phase == "align":
                        self.handle_align_phase()
                    elif self.approach_phase == "approach":
                        self.handle_approach_phase()
                    elif self.approach_phase == "deliver":
                        self.handle_deliver_phase()
                
                else:
                    # No targets - search
                    frames_without_target += 1
                    self.current_target = None
                    self.approach_phase = "detect"
                    
                    if frames_without_target >= max_frames_without_target:
                        search_direction *= -1
                        frames_without_target = 0
                        self.logger.info(f"🔍 Changing search direction: {'RIGHT' if search_direction > 0 else 'LEFT'}")
                    
                    self.search_for_targets(search_direction)
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Enhanced delivery loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
    
    def handle_detect_phase(self):
        """Phase 1: Detect target and determine wall orientation - IMPROVED"""
        if not self.current_target:
            return
        
        # If wall direction is not determined, try to infer it from target position
        if not self.current_target.wall_direction:
            # Infer wall direction based on green target position
            target_x, target_y = self.current_target.center
            frame_w, frame_h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
            
            # Simple position-based inference
            if target_y < frame_h * 0.4:
                self.current_target.wall_direction = 'top'
                self.current_target.approach_angle = 270.0  # Approach upward
            elif target_y > frame_h * 0.6:
                self.current_target.wall_direction = 'bottom'
                self.current_target.approach_angle = 90.0   # Approach downward
            elif target_x < frame_w * 0.4:
                self.current_target.wall_direction = 'left'
                self.current_target.approach_angle = 0.0    # Approach rightward
            elif target_x > frame_w * 0.6:
                self.current_target.wall_direction = 'right'
                self.current_target.approach_angle = 180.0  # Approach leftward
            else:
                # Default to top wall if in center
                self.current_target.wall_direction = 'top'
                self.current_target.approach_angle = 270.0
            
            self.logger.info(f"📍 Inferred target at {self.current_target.wall_direction} wall based on position")
        
        self.logger.info(f"📍 Target detected at {self.current_target.wall_direction} wall - moving to positioning")
        self.approach_phase = "position"
        self.phase_start_time = time.time()
        self.delivery_vision.approach_phase = "position"
    
    def handle_position_phase(self):
        """Phase 2: Position robot for perpendicular approach"""
        if not self.current_target:
            self.approach_phase = "detect"
            return
        
        # Get positioning command
        pos_command = self.delivery_vision.get_positioning_command(self.current_target)
        
        if pos_command == 'positioned':
            self.logger.info("✅ Robot positioned for perpendicular approach - moving to alignment")
            self.approach_phase = "align"
            self.phase_start_time = time.time()
            self.delivery_vision.approach_phase = "align"
            return
        
        # Execute positioning movement
        if pos_command == 'move_right':
            self.hardware.turn_right(duration=0.3, speed=0.4)
        elif pos_command == 'move_left':
            self.hardware.turn_left(duration=0.3, speed=0.4)
        elif pos_command == 'move_forward':
            self.hardware.move_forward(duration=0.3, speed=0.4)
        elif pos_command == 'move_backward':
            self.hardware.move_backward(duration=0.3, speed=0.4)
        
        # Timeout check
        if time.time() - self.phase_start_time > 15.0:
            self.logger.warning("⏰ Positioning timeout - proceeding to alignment")
            self.approach_phase = "align"
    
    def handle_align_phase(self):
        """Phase 3: Align robot perpendicular to wall"""
        if not self.current_target:
            self.approach_phase = "detect"
            return
        
        # Get alignment command
        align_command = self.delivery_vision.get_alignment_command(self.current_target)
        
        if align_command == 'align_vertical':
            # For top/bottom walls, ensure robot is facing up/down
            if self.current_target.wall_direction == 'top':
                self.logger.info("🧭 Aligning to face upward (toward top wall)")
            else:
                self.logger.info("🧭 Aligning to face downward (toward bottom wall)")
            
            # Simple alignment turn (you may need more sophisticated alignment)
            self.hardware.turn_right(duration=0.2, speed=0.3)
            
        elif align_command == 'align_horizontal':
            # For left/right walls, ensure robot is facing left/right
            if self.current_target.wall_direction == 'left':
                self.logger.info("🧭 Aligning to face left (toward left wall)")
            else:
                self.logger.info("🧭 Aligning to face right (toward right wall)")
            
            # Simple alignment turn
            self.hardware.turn_left(duration=0.2, speed=0.3)
        
        # Move to approach phase after alignment attempt
        time.sleep(0.5)
        self.logger.info("⚡ Alignment complete - starting perpendicular approach")
        self.approach_phase = "approach"
        self.phase_start_time = time.time()
        self.delivery_vision.approach_phase = "approach"
    
    def handle_approach_phase(self):
        """Phase 4: Approach target perpendicular to wall"""
        if not self.current_target:
            self.approach_phase = "detect"
            return
        
        # Move straight toward the target (should now be perpendicular to wall)
        self.logger.info(f"🚀 Approaching {self.current_target.wall_direction} wall perpendicularly")
        
        # Execute perpendicular approach
        approach_duration = 0.4
        approach_speed = 0.35
        
        # Move forward toward the goal
        self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
        
        # Check if we've reached the target area
        current_distance = self.current_target.distance_from_center
        if current_distance < 50:  # Close enough to target
            self.logger.info("📦 Reached delivery zone - proceeding to delivery")
            self.approach_phase = "deliver"
            self.phase_start_time = time.time()
            self.delivery_vision.approach_phase = "deliver"
        
        # Timeout check
        if time.time() - self.phase_start_time > 10.0:
            self.logger.warning("⏰ Approach timeout - proceeding to delivery")
            self.approach_phase = "deliver"
    
    def handle_deliver_phase(self):
        """Phase 5: Deliver balls and back away"""
        self.logger.info("📦 Executing delivery sequence")
        
        # Release balls if we have any
        if self.hardware.has_balls():
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls at {self.current_target.wall_direction} wall goal")
        else:
            self.logger.info("ℹ️  No balls to deliver")
        
        # Back away from the goal in the opposite direction of approach
        self.logger.info("⬅️ Backing away from delivery zone")
        
        # Back away perpendicular to the wall (opposite of approach direction)
        if self.current_target.wall_direction == 'top':
            # Approached upward, back away downward
            self.hardware.move_backward(duration=1.0, speed=0.4)
        elif self.current_target.wall_direction == 'bottom':
            # Approached downward, back away upward  
            self.hardware.move_forward(duration=1.0, speed=0.4)
        elif self.current_target.wall_direction == 'left':
            # Approached leftward, back away rightward
            self.hardware.turn_right(duration=0.5, speed=0.4)
            self.hardware.move_forward(duration=0.8, speed=0.4)
        elif self.current_target.wall_direction == 'right':
            # Approached rightward, back away leftward
            self.hardware.turn_left(duration=0.5, speed=0.4) 
            self.hardware.move_forward(duration=0.8, speed=0.4)
        
        # Reset for next target
        self.approach_phase = "detect"
        self.current_target = None
        self.delivery_vision.approach_phase = "detect"
        
        self.logger.info("🔄 Delivery complete - searching for next target")
    
    def search_for_targets(self, direction: int):
        """Search for green targets by turning"""
        if direction > 0:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching right for green targets")
            self.hardware.turn_right(duration=0.8, speed=0.5)
        else:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching left for green targets")
            self.hardware.turn_left(duration=0.8, speed=0.5)
        
        time.sleep(0.2)
    
    def stop_delivery(self):
        """Stop enhanced delivery mode"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 ENHANCED DELIVERY MODE COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Final ball count: {self.hardware.get_ball_count()}")
        self.logger.info(f"   Final phase: {self.approach_phase}")
        
        cv2.destroyAllWindows()

def run_delivery_test():
    """Main entry point for enhanced delivery testing with perpendicular approach"""
    print("\n🚚 ENHANCED GOLFBOT DELIVERY SYSTEM TEST (Perpendicular Wall Approach)")
    print("="*80)
    print("This enhanced mode will:")
    print("1. Search for GREEN targets (delivery zones/goals)")
    print("2. Detect RED walls and determine wall orientation") 
    print("3. Position robot for PERPENDICULAR approach to the wall")
    print("4. Align robot to face the goal straight-on")
    print("5. Approach the goal PERPENDICULAR to the wall (straight |, not angled /\\)")
    print("6. Release balls and back away perpendicular to wall")
    print()
    print("Enhanced Approach Phases:")
    print("• DETECT: Find green target and identify which wall it's on")
    print("• POSITION: Move to optimal position for perpendicular approach")
    print("• ALIGN: Orient robot to face the goal straight-on")
    print("• APPROACH: Drive straight toward goal (perpendicular to wall)")
    print("• DELIVER: Release balls and back away straight")
    print()
    print("Key Improvements:")
    print("• Wall orientation detection (top/bottom/left/right)")
    print("• Smart positioning for perpendicular approach")
    print("• Straight-line approach instead of angled approach")
    print("• Proper backing away perpendicular to wall")
    print("• Phase-based state machine for reliable execution")
    print()
    print("Visual Indicators:")
    print("• Red lines: Detected wall segments with direction labels")
    print("• Yellow arrows: Planned approach vector (perpendicular to wall)")
    print("• Green rectangles: Detected delivery zones")
    print("• Phase indicator: Current execution phase")
    print()
    print("Press 'q' in the camera window to quit")
    print("="*80)
    
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

# Additional helper functions for wall analysis
def analyze_wall_structure(frame):
    """Analyze the overall wall structure of the arena"""
    if frame is None:
        return {}
    
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create red wall mask
    red_lower1 = np.array([0, 150, 100])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([165, 150, 100])
    red_upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = mask1 + mask2
    
    # Analyze wall coverage in each region
    regions = {
        'top': red_mask[0:h//4, :],
        'bottom': red_mask[3*h//4:h, :], 
        'left': red_mask[:, 0:w//4],
        'right': red_mask[:, 3*w//4:w]
    }
    
    wall_analysis = {}
    for region_name, region_mask in regions.items():
        wall_coverage = np.sum(region_mask > 0) / region_mask.size
        wall_analysis[region_name] = {
            'coverage': wall_coverage,
            'has_wall': wall_coverage > 0.1  # 10% threshold
        }
    
    return wall_analysis

def find_wall_gaps(wall_segments, frame_width, frame_height):
    """Find gaps in walls that might be goals"""
    gaps = []
    
    # Sort wall segments by position
    horizontal_walls = [w for w in wall_segments if w.direction in ['top', 'bottom']]
    vertical_walls = [w for w in wall_segments if w.direction in ['left', 'right']]
    
    # Find horizontal gaps (in top/bottom walls)
    for wall_type in ['top', 'bottom']:
        walls_of_type = [w for w in horizontal_walls if w.direction == wall_type]
        if len(walls_of_type) >= 2:
            # Sort by x position
            walls_of_type.sort(key=lambda w: w.start_point[0])
            
            for i in range(len(walls_of_type) - 1):
                gap_start = walls_of_type[i].end_point[0]
                gap_end = walls_of_type[i + 1].start_point[0]
                gap_width = gap_end - gap_start
                
                if gap_width > 50:  # Minimum gap width for a goal
                    gap_center_x = (gap_start + gap_end) // 2
                    gap_center_y = walls_of_type[i].center_point[1]
                    
                    gaps.append({
                        'type': 'horizontal_gap',
                        'wall_side': wall_type,
                        'center': (gap_center_x, gap_center_y),
                        'width': gap_width,
                        'start': gap_start,
                        'end': gap_end
                    })
    
    # Find vertical gaps (in left/right walls)  
    for wall_type in ['left', 'right']:
        walls_of_type = [w for w in vertical_walls if w.direction == wall_type]
        if len(walls_of_type) >= 2:
            # Sort by y position
            walls_of_type.sort(key=lambda w: w.start_point[1])
            
            for i in range(len(walls_of_type) - 1):
                gap_start = walls_of_type[i].end_point[1]
                gap_end = walls_of_type[i + 1].start_point[1] 
                gap_height = gap_end - gap_start
                
                if gap_height > 50:  # Minimum gap height for a goal
                    gap_center_x = walls_of_type[i].center_point[0]
                    gap_center_y = (gap_start + gap_end) // 2
                    
                    gaps.append({
                        'type': 'vertical_gap',
                        'wall_side': wall_type,
                        'center': (gap_center_x, gap_center_y),
                        'height': gap_height,
                        'start': gap_start,
                        'end': gap_end
                    })
    
    return gaps

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_delivery_test()