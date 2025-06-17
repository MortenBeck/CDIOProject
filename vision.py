import cv2
import numpy as np
import time
import subprocess
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import config

@dataclass
class DetectedObject:
    """Class to store detected object information"""
    object_type: str  # 'ball', 'orange_ball', 'boundary'
    center: Tuple[int, int]
    radius: int
    area: int
    confidence: float
    distance_from_center: float
    in_collection_zone: bool = False  # NEW: Is ball in collection zone?

class Pi5Camera:
    """Camera interface for Raspberry Pi 5 using libcamera"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_file = "/tmp/golfbot_frame.jpg"
        self.running = False
        
    def start_capture(self):
        """Initialize camera"""
        try:
            self.running = True
            self.logger.info("Pi 5 camera initialized with libcamera")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Pi 5 camera: {e}")
            return False
    
    def capture_frame(self):
        """Capture a single frame"""
        try:
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '100',  # Faster timeout
                '--width', str(config.CAMERA_WIDTH),
                '--height', str(config.CAMERA_HEIGHT),
                '--quality', '80',
                '--nopreview'  # No preview window
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=3)
            
            if result.returncode == 0 and os.path.exists(self.temp_file):
                frame = cv2.imread(self.temp_file)
                if frame is not None and frame.size > 0:
                    return True, frame
                else:
                    self.logger.warning("Empty frame captured")
                    return False, None
            else:
                self.logger.warning(f"Camera capture failed: {result.stderr}")
                return False, None
                
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return False, None
    
    def release(self):
        """Clean up camera resources"""
        self.running = False
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

class VisionSystem:
    """Main vision processing system for GolfBot"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.camera = Pi5Camera()
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        self.last_frame = None
        self.current_target = None  # Store currently targeted ball
        
        # Collection zone boundaries
        self.collection_zone = self._calculate_collection_zone()
        
        # Store wall detection results for visualization
        self.detected_walls = []
        self.red_mask = None
        
    def _calculate_collection_zone(self):
        """Calculate the collection zone boundaries (middle 25% horizontal, bottom 20% vertical)"""
        # Horizontal: middle 25% (37.5% margin on each side)
        horizontal_margin = config.CAMERA_WIDTH * 0.375
        left_boundary = int(horizontal_margin)
        right_boundary = int(config.CAMERA_WIDTH - horizontal_margin)
        
        vertical_threshold = int(config.CAMERA_HEIGHT * 0.6)
        bottom_boundary = config.CAMERA_HEIGHT
        
        return {
            'left': left_boundary,
            'right': right_boundary, 
            'top': vertical_threshold,
            'bottom': bottom_boundary
        }
    
    def is_in_collection_zone(self, ball_center: Tuple[int, int]) -> bool:
        """Check if ball center is in the collection zone"""
        x, y = ball_center
        zone = self.collection_zone
        
        horizontal_ok = zone['left'] <= x <= zone['right']
        vertical_ok = zone['top'] <= y <= zone['bottom']
        
        return horizontal_ok and vertical_ok
    
    def start(self):
        """Initialize vision system"""
        return self.camera.start_capture()
    
    def get_frame(self):
        """Get current camera frame"""
        ret, frame = self.camera.capture_frame()
        if ret:
            self.last_frame = frame
        return ret, frame
    
    def detect_balls(self, frame) -> List[DetectedObject]:
        """Detect ALL ping pong balls (white AND orange) in frame with improved filtering"""
        detected_objects = []
        
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # WHITE BALL DETECTION
        # More restrictive white/light color detection to reduce false positives
        ball_lower = np.array([0, 0, 200])    # Higher value threshold (brighter whites)
        ball_upper = np.array([180, 40, 255]) # Lower saturation (more white, less colored)
        white_mask = cv2.inRange(hsv, ball_lower, ball_upper)
        
        # ORANGE BALL DETECTION  
        # Orange color detection
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # COMBINE BOTH MASKS - treat all balls the same!
        combined_mask = cv2.bitwise_or(white_mask, orange_mask)
        
        # More aggressive noise cleanup
        kernel = np.ones((7,7), np.uint8)  # Larger kernel
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Additional: Remove small noise blobs
        combined_mask = cv2.medianBlur(combined_mask, 5)
        
        # Define arena boundaries (exclude areas outside the red boundary)
        h, w = frame.shape[:2]
        
        # Create arena mask - exclude top 15% and outer edges where non-arena objects appear
        arena_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define "safe arena area" - exclude top portion and outer edges
        top_margin = int(h * 0.15)      # Exclude top 15% where ceiling/wall objects appear
        bottom_margin = int(h * 0.05)   # Small bottom margin
        left_margin = int(w * 0.05)     # Small left margin  
        right_margin = int(w * 0.05)    # Small right margin
        
        # Create rectangular arena region
        arena_mask[top_margin:h-bottom_margin, left_margin:w-right_margin] = 255
        
        # Apply arena mask to ball detection mask
        combined_mask = cv2.bitwise_and(combined_mask, arena_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # More restrictive area filtering
            if config.BALL_MIN_AREA < area < config.BALL_MAX_AREA:
                # Enhanced circularity check
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Much stricter circularity requirement (ping pong balls are very round)
                    if circularity > 0.6:  # Increased from 0.3 to 0.6
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x), int(y))
                        radius = int(radius)
                        
                        if config.BALL_MIN_RADIUS < radius < config.BALL_MAX_RADIUS:
                            
                            # Additional validation: check if the detected circle matches the contour well
                            # Create a mask of the perfect circle and compare with contour
                            circle_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.circle(circle_mask, center, radius, 255, -1)
                            contour_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(contour_mask, [contour], 255)
                            
                            # Calculate overlap between perfect circle and actual contour
                            intersection = cv2.bitwise_and(circle_mask, contour_mask)
                            union = cv2.bitwise_or(circle_mask, contour_mask)
                            overlap_ratio = np.sum(intersection) / max(1, np.sum(union))
                            
                            # Only accept if the shape closely matches a circle
                            if overlap_ratio > 0.7:  # Must be at least 70% circle-like
                                
                                # Additional check: verify the object is actually white/bright OR orange in the original image
                                # Sample the center region and check color
                                sample_radius = max(3, radius // 3)
                                y1, y2 = max(0, center[1] - sample_radius), min(h, center[1] + sample_radius)
                                x1, x2 = max(0, center[0] - sample_radius), min(w, center[0] + sample_radius)
                                
                                center_region = frame[y1:y2, x1:x2]
                                if center_region.size > 0:
                                    # Check if it's white OR orange
                                    gray_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
                                    mean_brightness = np.mean(gray_region)
                                    
                                    # Check if it's in the orange region of original masks
                                    center_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
                                    orange_pixels = cv2.inRange(center_hsv, orange_lower, orange_upper)
                                    orange_ratio = np.sum(orange_pixels > 0) / max(1, orange_pixels.size)
                                    
                                    # Accept if it's bright (white) OR has significant orange content
                                    is_valid_ball = (mean_brightness > 180) or (orange_ratio > 0.3)
                                    
                                    if is_valid_ball:
                                        distance_from_center = np.sqrt(
                                            (center[0] - self.frame_center_x)**2 + 
                                            (center[1] - self.frame_center_y)**2
                                        )
                                        
                                        # Check if ball is in collection zone
                                        in_collection_zone = self.is_in_collection_zone(center)
                                        
                                        # Determine ball type for display purposes only
                                        ball_type = 'orange_ball' if orange_ratio > 0.3 else 'ball'
                                        
                                        ball = DetectedObject(
                                            object_type=ball_type,  # Just for display color
                                            center=center,
                                            radius=radius,
                                            area=area,
                                            confidence=circularity * overlap_ratio,  # Combined confidence score
                                            distance_from_center=distance_from_center,
                                            in_collection_zone=in_collection_zone
                                        )
                                        detected_objects.append(ball)
        
        # Sort by distance from center (closest first)
        detected_objects.sort(key=lambda x: x.distance_from_center)
        
        # Limit to maximum reasonable number of balls to prevent false positive spam
        max_balls = 6  # Arena shouldn't have more than 6 balls visible at once
        if len(detected_objects) > max_balls:
            # Keep only the closest and most confident detections
            detected_objects.sort(key=lambda x: (x.distance_from_center, -x.confidence))
            detected_objects = detected_objects[:max_balls]
        
        return detected_objects

    
    def detect_boundaries(self, frame) -> bool:
        """Detect red walls/boundaries with visualization data stored"""
        if frame is None:
            return False
        
        h, w = frame.shape[:2]
        
        # Clear previous wall detections
        self.detected_walls = []
        
        # Convert to HSV for red detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced red color detection with wider ranges
        # Lower red range (around 0-10)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([15, 255, 255])
        
        # Upper red range (around 170-180)
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        # Create combined red mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Store the red mask for visualization
        self.red_mask = red_mask.copy()
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # MUCH SMALLER danger zones - only trigger when very close to walls
        # Bottom danger zone: only bottom 10% of frame (was 30%)
        danger_distance = min(60, int(h * 0.12))  # Much smaller
        bottom_danger_y = h - danger_distance
        bottom_mask = red_mask[bottom_danger_y:h, :]
        
        # Side danger zones: only check very close to edges (was 15% of width)
        edge_width = min(40, int(w * 0.08))  # Much smaller
        left_mask = red_mask[:, 0:edge_width]
        right_mask = red_mask[:, w-edge_width:w]
        
        # Find contours in danger zones with stricter requirements
        danger_detected = False
        min_wall_area = 200  # INCREASED from 80 - need bigger wall segments
        
        # Check bottom area - only if wall takes up significant portion
        contours, _ = cv2.findContours(bottom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                length = max(w_rect, h_rect)
                # INCREASED minimum wall segment length (was 40)
                if length > 80 and w_rect > 60:  # Must be substantial horizontal wall
                    danger_detected = True
                    # Store wall detection for visualization
                    wall_info = {
                        'zone': 'bottom',
                        'contour': contour,
                        'area': area,
                        'bbox': (x, bottom_danger_y + y, w_rect, h_rect),  # Adjust y coordinate
                        'length': length,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.debug(f"Red wall detected in bottom danger zone (area: {area}, length: {length})")
                    break
                else:
                    # Store non-triggering walls too for visualization
                    wall_info = {
                        'zone': 'bottom',
                        'contour': contour,
                        'area': area,
                        'bbox': (x, bottom_danger_y + y, w_rect, h_rect),
                        'length': length,
                        'triggered': False
                    }
                    self.detected_walls.append(wall_info)
        
        # Check left edge - stricter requirements
        contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                # INCREASED minimum height (was 60) and added width requirement
                if h_rect > 100 and w_rect > 20:  # Must be substantial vertical wall
                    if not danger_detected:  # Only set if not already detected
                        danger_detected = True
                    wall_info = {
                        'zone': 'left',
                        'contour': contour,
                        'area': area,
                        'bbox': (x, y, w_rect, h_rect),
                        'length': h_rect,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.debug(f"Red wall detected on left edge (area: {area}, height: {h_rect})")
                else:
                    wall_info = {
                        'zone': 'left',
                        'contour': contour,
                        'area': area,
                        'bbox': (x, y, w_rect, h_rect),
                        'length': h_rect,
                        'triggered': False
                    }
                    self.detected_walls.append(wall_info)
        
        # Check right edge - stricter requirements  
        contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                # INCREASED minimum height (was 60) and added width requirement
                if h_rect > 100 and w_rect > 20:  # Must be substantial vertical wall
                    if not danger_detected:  # Only set if not already detected
                        danger_detected = True
                    wall_info = {
                        'zone': 'right',
                        'contour': contour,
                        'area': area,
                        'bbox': (w - edge_width + x, y, w_rect, h_rect),  # Adjust x coordinate
                        'length': h_rect,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.debug(f"Red wall detected on right edge (area: {area}, height: {h_rect})")
                else:
                    wall_info = {
                        'zone': 'right',
                        'contour': contour,
                        'area': area,
                        'bbox': (w - edge_width + x, y, w_rect, h_rect),
                        'length': h_rect,
                        'triggered': False
                    }
                    self.detected_walls.append(wall_info)
        
        return danger_detected
    
    def get_target_ball(self, balls: List[DetectedObject]) -> Optional[DetectedObject]:
        """Determine which ball the robot should target (simplified - no orange priority)"""
        
        # If we have any balls, target the closest one
        if balls:
            # Sort by distance from center (closest first)
            balls.sort(key=lambda x: x.distance_from_center)
            target = balls[0]
            self.current_target = target
            return target
        
        # No balls detected
        self.current_target = None
        return None
    
    def should_activate_servo(self) -> bool:
        """Check if servo1 should be activated based on target ball position"""
        if not self.current_target:
            return False
        
        return self.current_target.in_collection_zone
    
    def get_navigation_command(self, detected_objects: List[DetectedObject]) -> str:
        """Determine navigation command based on detected objects (simplified)"""
        
        target_ball = self.get_target_ball(detected_objects)
        
        if target_ball:
            # Check if ball is in collection zone (position-based)
            if target_ball.in_collection_zone:
                return "collect_ball"  # This will trigger servo activation
            else:
                return self._get_direction_to_object(target_ball)
        
        # No balls detected - search
        return "search"
    
    def _get_direction_to_object(self, obj: DetectedObject) -> str:
        """Get direction command to move toward object"""
        x_offset = obj.center[0] - self.frame_center_x
        y_offset = obj.center[1] - self.frame_center_y
        
        # Determine horizontal direction
        if abs(x_offset) > 30:  # Deadband for "centered"
            if x_offset > 0:
                return "turn_right"
            else:
                return "turn_left"
        else:
            # Object is centered, move forward
            return "forward"
    
    def draw_detections(self, frame, balls: List[DetectedObject]) -> np.ndarray:
        """Draw detection results on frame for debugging with enhanced wall visualization"""
        if not config.DEBUG_VISION:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # === WALL VISUALIZATION (NEW SECTION) ===
        
        # 1. Show the raw red mask as a semi-transparent overlay
        if self.red_mask is not None:
            # Create a colored version of the red mask
            red_overlay = np.zeros_like(result)
            red_overlay[:, :, 2] = self.red_mask  # Red channel
            
            # Apply semi-transparent overlay to show all detected red areas
            cv2.addWeighted(result, 0.85, red_overlay, 0.15, 0, result)
        
        # 2. Draw danger zone boundaries
        danger_distance = min(60, int(h * 0.12))
        danger_y = h - danger_distance
        edge_width = min(40, int(w * 0.08))
        
        # Bottom danger zone
        cv2.rectangle(result, (0, danger_y), (w, h), (0, 0, 255), 2)
        cv2.putText(result, f'BOTTOM DANGER ZONE ({danger_distance}px)', 
                   (10, danger_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Left danger zone
        cv2.rectangle(result, (0, 0), (edge_width, h), (0, 0, 255), 2)
        cv2.putText(result, f'L DANGER\n({edge_width}px)', 
                   (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Right danger zone
        cv2.rectangle(result, (w - edge_width, 0), (w, h), (0, 0, 255), 2)
        cv2.putText(result, f'R DANGER\n({edge_width}px)', 
                   (w - edge_width + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 3. Draw detected wall segments with detailed information
        wall_count = 0
        for wall in self.detected_walls:
            wall_count += 1
            x, y, w_rect, h_rect = wall['bbox']
            
            # Color coding: Red for triggering walls, Yellow for non-triggering
            color = (0, 0, 255) if wall['triggered'] else (0, 255, 255)
            thickness = 3 if wall['triggered'] else 2
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            
            # Draw contour outline
            if wall['zone'] == 'bottom':
                # Adjust contour coordinates for bottom zone
                adjusted_contour = wall['contour'].copy()
                adjusted_contour[:, :, 1] += danger_y
                cv2.drawContours(result, [adjusted_contour], -1, color, 1)
            elif wall['zone'] == 'left':
                cv2.drawContours(result, [wall['contour']], -1, color, 1)
            elif wall['zone'] == 'right':
                # Adjust contour coordinates for right zone
                adjusted_contour = wall['contour'].copy()
                adjusted_contour[:, :, 0] += (w - edge_width)
                cv2.drawContours(result, [adjusted_contour], -1, color, 1)
            
            # Add wall information label
            status = "TRIGGERED" if wall['triggered'] else "detected"
            label = f"WALL #{wall_count} ({wall['zone']}) - {status}"
            label_detail = f"Area:{wall['area']:.0f} Len:{wall['length']:.0f}"
            
            # Position label near the wall
            label_x = max(5, min(w - 200, x))
            label_y = max(20, y - 10) if y > 30 else y + h_rect + 20
            
            cv2.putText(result, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(result, label_detail, (label_x, label_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # === COLLECTION ZONE VISUALIZATION ===
        
        # Draw collection zone rectangle
        zone = self.collection_zone
        # Collection zone in semi-transparent green
        overlay = result.copy()
        cv2.rectangle(overlay, (zone['left'], zone['top']), 
                    (zone['right'], zone['bottom']), (0, 255, 0), -1)
        cv2.addWeighted(result, 0.8, overlay, 0.2, 0, result)
        
        # Collection zone border
        cv2.rectangle(result, (zone['left'], zone['top']), 
                    (zone['right'], zone['bottom']), (0, 255, 0), 3)
        
        # Collection zone labels
        cv2.putText(result, 'COLLECTION ZONE', (zone['left'] + 10, zone['top'] + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # === BALL VISUALIZATION ===
        
        # Draw all balls (white and orange treated the same)
        for ball in balls:
            # Check if this is the current target
            is_target = (self.current_target and 
                        self.current_target.center == ball.center)
            
            # Choose color based on ball type but treat functionally the same
            if ball.object_type == 'orange_ball':
                base_color = (0, 165, 255)  # Orange color for display
            else:
                base_color = (0, 255, 0)    # Green for white balls
            
            if is_target:
                # Highlight target ball with thick pulsing border
                thickness = 4
                color = base_color
                # Add pulsing effect based on time
                pulse = int(5 * (1 + np.sin(time.time() * 8)))  # Pulse between 0-10
                cv2.circle(result, ball.center, ball.radius + pulse, color, thickness)
                cv2.circle(result, ball.center, 5, color, -1)
                
                # Add arrow pointing to target
                arrow_start = (self.frame_center_x, self.frame_center_y)
                arrow_end = ball.center
                cv2.arrowedLine(result, arrow_start, arrow_end, (255, 255, 0), 3, tipLength=0.3)
                
                # Add "TARGET" label with collection zone status
                ball_name = 'ORANGE' if ball.object_type == 'orange_ball' else 'BALL'
                label = f'{ball_name} TARGET (IN ZONE)' if ball.in_collection_zone else f'{ball_name} TARGET'
                label_color = (255, 255, 0) if ball.in_collection_zone else (255, 255, 255)
                label_pos = (ball.center[0]-50, ball.center[1]-ball.radius-25)
                cv2.putText(result, label, label_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            else:
                # Regular ball display with zone status
                color = base_color if ball.in_collection_zone else tuple(int(c*0.7) for c in base_color)
                cv2.circle(result, ball.center, ball.radius, color, 2)
                cv2.circle(result, ball.center, 3, color, -1)
                
                ball_name = 'Orange' if ball.object_type == 'orange_ball' else 'Ball'
                label = f'{ball_name} (Zone)' if ball.in_collection_zone else ball_name
                cv2.putText(result, label, 
                        (ball.center[0]-20, ball.center[1]-ball.radius-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw center crosshair
        cv2.line(result, (self.frame_center_x-20, self.frame_center_y), 
                (self.frame_center_x+20, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y-20), 
                (self.frame_center_x, self.frame_center_y+20), (255, 255, 255), 2)
        
        # === STATUS INFORMATION OVERLAY ===
        
        # Add wall detection status in top-left corner
        wall_status = f"WALLS DETECTED: {len(self.detected_walls)}"
        triggered_walls = sum(1 for wall in self.detected_walls if wall['triggered'])
        boundary_status = f"BOUNDARY TRIGGERED: {'YES' if triggered_walls > 0 else 'NO'} ({triggered_walls})"
        
        cv2.putText(result, wall_status, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, boundary_status, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if triggered_walls > 0 else (255, 255, 255), 2)
        
        # Add detection thresholds info
        threshold_info = f"Min Wall Area: {200}, Min Length: 80/100"
        cv2.putText(result, threshold_info, (10, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add targeting information in the bottom-left corner
        if self.current_target:
            ball_name = 'ORANGE BALL' if self.current_target.object_type == 'orange_ball' else 'WHITE BALL'
            target_info = f"TARGET: {ball_name}"
            zone_info = f"IN COLLECTION ZONE: {'YES' if self.current_target.in_collection_zone else 'NO'}"
            servo_info = f"SERVO1 ACTIVE: {'YES' if self.should_activate_servo() else 'NO'}"
            
            cv2.putText(result, target_info, (10, h-80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(result, zone_info, (10, h-55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(result, servo_info, (10, h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            cv2.putText(result, "TARGET: NONE - SEARCHING", (10, h-35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add legend in top-right corner
        legend_x = w - 250
        legend_y = 25
        cv2.putText(result, "LEGEND:", (legend_x, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result, "Red Overlay: Raw red detection", (legend_x, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        cv2.putText(result, "Red Boxes: Triggering walls", (legend_x, legend_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(result, "Yellow Boxes: Non-triggering walls", (legend_x, legend_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(result, "Green Zone: Collection area", (legend_x, legend_y + 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return result
    
    def process_frame(self):
        """Process current frame and return detection results (simplified)"""
        ret, frame = self.get_frame()
        if not ret:
            return None, None, None, None, None
        
        # Detect all balls (white and orange together)
        balls = self.detect_balls(frame)
        orange_ball = None  # No longer used separately
        near_boundary = self.detect_boundaries(frame)
        
        # Get navigation command (this also sets the current target)
        nav_command = self.get_navigation_command(balls)
        
        # Create debug visualization with target highlighting and collection zone
        debug_frame = self.draw_detections(frame, balls)
        
        return balls, orange_ball, near_boundary, nav_command, debug_frame
    
    def cleanup(self):
        """Clean up vision system"""
        self.camera.release()
        cv2.destroyAllWindows()
        self.logger.info("Vision system cleanup completed")