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
        
    def _calculate_collection_zone(self):
        """Calculate the collection zone boundaries (middle 25% horizontal, bottom 20% vertical)"""
        # Horizontal: middle 25% (37.5% margin on each side)
        horizontal_margin = config.CAMERA_WIDTH * 0.375
        left_boundary = int(horizontal_margin)
        right_boundary = int(config.CAMERA_WIDTH - horizontal_margin)
        
        vertical_threshold = int(config.CAMERA_HEIGHT * 0.8)
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
        """Detect white ping pong balls in frame"""
        detected_objects = []
        
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for white/light colors
        mask = cv2.inRange(hsv, config.BALL_HSV_LOWER, config.BALL_HSV_UPPER)
        
        # Clean up noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if config.BALL_MIN_AREA < area < config.BALL_MAX_AREA:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.3:  # Reasonably circular
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x), int(y))
                        radius = int(radius)
                        
                        if config.BALL_MIN_RADIUS < radius < config.BALL_MAX_RADIUS:
                            distance_from_center = np.sqrt(
                                (center[0] - self.frame_center_x)**2 + 
                                (center[1] - self.frame_center_y)**2
                            )
                            
                            # Check if ball is in collection zone
                            in_collection_zone = self.is_in_collection_zone(center)
                            
                            ball = DetectedObject(
                                object_type='ball',
                                center=center,
                                radius=radius,
                                area=area,
                                confidence=circularity,
                                distance_from_center=distance_from_center,
                                in_collection_zone=in_collection_zone
                            )
                            detected_objects.append(ball)
        
        # Sort by distance from center (closest first)
        detected_objects.sort(key=lambda x: x.distance_from_center)
        return detected_objects
    
    def detect_orange_ball(self, frame) -> Optional[DetectedObject]:
        """Detect orange VIP ball specifically"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.ORANGE_HSV_LOWER, config.ORANGE_HSV_UPPER)
        
        # Clean up noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_orange_ball = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if config.BALL_MIN_AREA < area < config.BALL_MAX_AREA:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.3:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x), int(y))
                        radius = int(radius)
                        
                        if config.BALL_MIN_RADIUS < radius < config.BALL_MAX_RADIUS:
                            # Score based on size and circularity
                            score = area * circularity
                            
                            if score > best_score:
                                distance_from_center = np.sqrt(
                                    (center[0] - self.frame_center_x)**2 + 
                                    (center[1] - self.frame_center_y)**2
                                )
                                
                                # Check if ball is in collection zone
                                in_collection_zone = self.is_in_collection_zone(center)
                                
                                best_orange_ball = DetectedObject(
                                    object_type='orange_ball',
                                    center=center,
                                    radius=radius,
                                    area=area,
                                    confidence=circularity,
                                    distance_from_center=distance_from_center,
                                    in_collection_zone=in_collection_zone
                                )
                                best_score = score
        
        return best_orange_ball
    
    def detect_boundaries(self, frame) -> bool:
        """Detect red walls/boundaries using color detection with enhanced sensitivity"""
        if frame is None:
            return False
        
        h, w = frame.shape[:2]
        
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
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Enhanced danger zone detection - check multiple areas
        danger_distance = min(150, int(h * 0.3))
        
        # Check bottom area (primary danger zone)
        bottom_danger_y = h - danger_distance
        bottom_mask = red_mask[bottom_danger_y:h, :]
        
        # Also check left and right edges for walls
        edge_width = min(100, int(w * 0.15))
        left_mask = red_mask[:, 0:edge_width]
        right_mask = red_mask[:, w-edge_width:w]
        
        # Find contours in all danger zones
        danger_detected = False
        min_wall_area = 80  # Reduced minimum area for better sensitivity
        
        # Check bottom area
        contours, _ = cv2.findContours(bottom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                length = max(w_rect, h_rect)
                if length > 40:  # Reduced minimum wall segment length
                    danger_detected = True
                    if config.DEBUG_VISION:
                        self.logger.debug(f"Red wall detected in bottom danger zone (area: {area})")
                    break
        
        # Check left edge
        if not danger_detected:
            contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_wall_area:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    if h_rect > 60:  # Vertical wall segment
                        danger_detected = True
                        if config.DEBUG_VISION:
                            self.logger.debug(f"Red wall detected on left edge (area: {area})")
                        break
        
        # Check right edge
        if not danger_detected:
            contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_wall_area:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    if h_rect > 60:  # Vertical wall segment
                        danger_detected = True
                        if config.DEBUG_VISION:
                            self.logger.debug(f"Red wall detected on right edge (area: {area})")
                        break
        
        return danger_detected
    
    def get_target_ball(self, balls: List[DetectedObject], orange_ball: Optional[DetectedObject]) -> Optional[DetectedObject]:
        """Determine which ball the robot should target (stored for visualization)"""
        # Combine all balls into one list
        all_balls = []
        if balls:
            all_balls.extend(balls)
        if orange_ball:
            all_balls.append(orange_ball)
        
        # If we have any balls, target the closest one
        if all_balls:
            # Sort by distance from center (closest first)
            all_balls.sort(key=lambda x: x.distance_from_center)
            target = all_balls[0]
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
    
    def get_navigation_command(self, detected_objects: List[DetectedObject], 
                         orange_ball: Optional[DetectedObject]) -> str:
        """Determine navigation command based on detected objects"""
        
        target_ball = self.get_target_ball(detected_objects, orange_ball)
        
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
    
    def draw_detections(self, frame, balls: List[DetectedObject], 
                       orange_ball: Optional[DetectedObject]) -> np.ndarray:
        """Draw detection results on frame for debugging with collection zone visualization"""
        if not config.DEBUG_VISION:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
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
        cv2.putText(result, '50% horizontal, bottom 20%', (zone['left'] + 10, zone['top'] + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw danger zones for wall detection
        danger_distance = min(150, int(h * 0.3))
        danger_y = h - danger_distance
        edge_width = min(100, int(w * 0.15))
        
        # Bottom danger zone
        cv2.line(result, (0, danger_y), (w, danger_y), (0, 0, 255), 2)
        cv2.putText(result, 'WALL DANGER ZONE', (w - 200, danger_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Left and right edge danger zones
        cv2.line(result, (edge_width, 0), (edge_width, h), (0, 0, 255), 2)
        cv2.line(result, (w - edge_width, 0), (w - edge_width, h), (0, 0, 255), 2)
        
        # Draw regular balls in green
        for ball in balls:
            # Check if this is the current target
            is_target = (self.current_target and 
                        self.current_target.center == ball.center and 
                        self.current_target.object_type == 'ball')
            
            if is_target:
                # Highlight target ball with thick pulsing border
                thickness = 4
                color = (0, 255, 0)  # Bright green
                # Add pulsing effect based on time
                pulse = int(5 * (1 + np.sin(time.time() * 8)))  # Pulse between 0-10
                cv2.circle(result, ball.center, ball.radius + pulse, color, thickness)
                cv2.circle(result, ball.center, 5, color, -1)
                
                # Add arrow pointing to target
                arrow_start = (self.frame_center_x, self.frame_center_y)
                arrow_end = ball.center
                cv2.arrowedLine(result, arrow_start, arrow_end, (255, 255, 0), 3, tipLength=0.3)
                
                # Add "TARGET" label with collection zone status
                label = 'TARGET (IN ZONE)' if ball.in_collection_zone else 'TARGET'
                label_color = (255, 255, 0) if ball.in_collection_zone else (255, 255, 255)
                label_pos = (ball.center[0]-50, ball.center[1]-ball.radius-25)
                cv2.putText(result, label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            else:
                # Regular ball display with zone status
                color = (0, 255, 0) if ball.in_collection_zone else (100, 255, 100)
                cv2.circle(result, ball.center, ball.radius, color, 2)
                cv2.circle(result, ball.center, 3, color, -1)
                label = 'Ball (Zone)' if ball.in_collection_zone else 'Ball'
                cv2.putText(result, label, 
                           (ball.center[0]-20, ball.center[1]-ball.radius-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw orange ball with special handling for target and zone
        if orange_ball:
            # Check if orange ball is the current target
            is_target = (self.current_target and 
                        self.current_target.center == orange_ball.center and 
                        self.current_target.object_type == 'orange_ball')
            
            if is_target:
                # Highlight target orange ball with thick pulsing border
                thickness = 4
                color = (0, 165, 255)  # Orange
                # Add pulsing effect
                pulse = int(5 * (1 + np.sin(time.time() * 8)))
                cv2.circle(result, orange_ball.center, orange_ball.radius + pulse, color, thickness)
                cv2.circle(result, orange_ball.center, 5, color, -1)
                
                # Add arrow pointing to target
                arrow_start = (self.frame_center_x, self.frame_center_y)
                arrow_end = orange_ball.center
                cv2.arrowedLine(result, arrow_start, arrow_end, (255, 255, 0), 3, tipLength=0.3)
                
                # Add "VIP TARGET" label with zone status
                label = 'VIP TARGET (IN ZONE)' if orange_ball.in_collection_zone else 'VIP TARGET'
                label_color = (255, 255, 0) if orange_ball.in_collection_zone else (255, 255, 255)
                label_pos = (orange_ball.center[0]-60, orange_ball.center[1]-orange_ball.radius-25)
                cv2.putText(result, label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            else:
                # Regular orange ball display with zone status
                color = (0, 165, 255) if orange_ball.in_collection_zone else (100, 200, 255)
                cv2.circle(result, orange_ball.center, orange_ball.radius, color, 3)
                cv2.circle(result, orange_ball.center, 5, color, -1)
                label = 'VIP (Zone)' if orange_ball.in_collection_zone else 'VIP!'
                cv2.putText(result, label, 
                           (orange_ball.center[0]-20, orange_ball.center[1]-orange_ball.radius-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw center crosshair
        cv2.line(result, (self.frame_center_x-20, self.frame_center_y), 
                (self.frame_center_x+20, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y-20), 
                (self.frame_center_x, self.frame_center_y+20), (255, 255, 255), 2)
        
        # Add targeting information in the corner
        if self.current_target:
            target_info = f"TARGET: {self.current_target.object_type.upper()}"
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
        
        return result
    
    def process_frame(self):
        """Process current frame and return detection results"""
        ret, frame = self.get_frame()
        if not ret:
            return None, None, None, None, None
        
        # Detect all objects (no goals anymore)
        balls = self.detect_balls(frame)
        orange_ball = self.detect_orange_ball(frame)
        near_boundary = self.detect_boundaries(frame)
        
        # Get navigation command (this also sets the current target)
        nav_command = self.get_navigation_command(balls, orange_ball)
        
        # Create debug visualization with target highlighting and collection zone
        debug_frame = self.draw_detections(frame, balls, orange_ball)
        
        return balls, orange_ball, near_boundary, nav_command, debug_frame
    
    def cleanup(self):
        """Clean up vision system"""
        self.camera.release()
        cv2.destroyAllWindows()
        self.logger.info("Vision system cleanup completed")