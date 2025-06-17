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
    in_collection_zone: bool = False

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
                '--timeout', '100',
                '--width', str(config.CAMERA_WIDTH),
                '--height', str(config.CAMERA_HEIGHT),
                '--quality', '80',
                '--nopreview'
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
    """Enhanced vision processing system with HoughCircles and arena detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.camera = Pi5Camera()
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        self.last_frame = None
        self.current_target = None
        
        # Arena boundaries
        self.arena_mask = None
        self.arena_detected = False
        self.arena_contour = None
        
        # Collection zone boundaries
        self.collection_zone = self._calculate_collection_zone()
        
        # Store wall detection results for visualization
        self.detected_walls = []
        self.red_mask = None
        
        # Detection method tracking
        self.detection_method = "hybrid"
        
    def _calculate_collection_zone(self):
        """Calculate the collection zone boundaries"""
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
    
    def detect_arena_boundaries(self, frame) -> bool:
        """Detect arena boundaries from red walls to create detection mask"""
        if frame is None:
            return False
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red boundary detection with wider ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_boundary_mask = mask1 + mask2
        
        # Clean up the boundary mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_boundary_mask = cv2.morphologyEx(red_boundary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        red_boundary_mask = cv2.morphologyEx(red_boundary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find the arena boundary contour
        contours, _ = cv2.findContours(red_boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (should be arena boundary)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if this could be a valid arena boundary
            min_arena_area = (w * h) * 0.15  # Arena should be at least 15% of frame
            
            if area > min_arena_area:
                # Create arena mask - everything inside the red boundary
                self.arena_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(self.arena_mask, [largest_contour], 255)
                
                # Erode slightly to ensure we're well inside the boundary
                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                self.arena_mask = cv2.erode(self.arena_mask, erosion_kernel, iterations=1)
                
                self.arena_contour = largest_contour
                self.arena_detected = True
                
                if config.DEBUG_VISION:
                    self.logger.info(f"Arena boundary detected: area={area:.0f}")
                
                return True
        
        # Fallback: create a conservative arena mask
        if not self.arena_detected:
            self.arena_mask = np.ones((h, w), dtype=np.uint8) * 255
            # Exclude outer edges where objects might be outside arena
            margin_h = int(h * 0.12)  # 12% margin top/bottom
            margin_w = int(w * 0.08)  # 8% margin left/right
            
            self.arena_mask[:margin_h, :] = 0  # Top
            self.arena_mask[-margin_h:, :] = 0  # Bottom  
            self.arena_mask[:, :margin_w] = 0  # Left
            self.arena_mask[:, -margin_w:] = 0  # Right
            
            if config.DEBUG_VISION:
                self.logger.info("Using fallback arena mask (conservative edges)")
        
        return False
    
    def detect_balls_hough_circles(self, frame) -> List[DetectedObject]:
        """Primary detection method using HoughCircles for robust shape detection"""
        detected_objects = []
        
        if frame is None:
            return detected_objects
        
        # Ensure arena boundaries are set
        if self.arena_mask is None:
            self.detect_arena_boundaries(frame)
        
        h, w = frame.shape[:2]
        
        # Convert to grayscale for HoughCircles
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply arena mask to focus detection
        if self.arena_mask is not None:
            gray = cv2.bitwise_and(gray, self.arena_mask)
        
        # Apply Gaussian blur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using HoughCircles - optimized parameters for ping pong balls
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,                  # Accumulator resolution ratio
            minDist=25,              # Minimum distance between circle centers
            param1=50,               # Upper threshold for edge detection
            param2=28,               # Accumulator threshold for center detection (lower = more circles)
            minRadius=config.BALL_MIN_RADIUS,
            maxRadius=config.BALL_MAX_RADIUS
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, radius) in circles:
                center = (x, y)
                
                # Verify center is within arena
                if (self.arena_mask is not None and 
                    0 <= y < h and 0 <= x < w and
                    self.arena_mask[y, x] > 0):
                    
                    # Color verification to determine ball type and confidence
                    ball_type, confidence = self._verify_ball_color_advanced(frame, center, radius)
                    
                    if confidence > 0.3:  # Confidence threshold
                        distance_from_center = np.sqrt(
                            (center[0] - self.frame_center_x)**2 + 
                            (center[1] - self.frame_center_y)**2
                        )
                        
                        in_collection_zone = self.is_in_collection_zone(center)
                        area = int(np.pi * radius * radius)
                        
                        ball = DetectedObject(
                            object_type=ball_type,
                            center=center,
                            radius=radius,
                            area=area,
                            confidence=confidence,
                            distance_from_center=distance_from_center,
                            in_collection_zone=in_collection_zone
                        )
                        detected_objects.append(ball)
        
        return detected_objects
    
    def detect_balls_color_contours(self, frame) -> List[DetectedObject]:
        """Fallback detection using your existing color+contour method"""
        detected_objects = []
        
        if frame is None:
            return detected_objects
        
        # Ensure arena mask exists
        if self.arena_mask is None:
            self.detect_arena_boundaries(frame)
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Your existing color detection ranges
        ball_lower = np.array([0, 0, 200])
        ball_upper = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, ball_lower, ball_upper)
        
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        combined_mask = cv2.bitwise_or(white_mask, orange_mask)
        
        # CRITICAL: Apply arena mask to restrict detection to arena only
        if self.arena_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, self.arena_mask)
        
        # Your existing morphological operations
        kernel = np.ones((7, 7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.medianBlur(combined_mask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if config.BALL_MIN_AREA < area < config.BALL_MAX_AREA:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.6:  # Your existing threshold
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x), int(y))
                        radius = int(radius)
                        
                        if (config.BALL_MIN_RADIUS < radius < config.BALL_MAX_RADIUS and
                            0 <= center[1] < h and 0 <= center[0] < w and
                            self.arena_mask[center[1], center[0]] > 0):
                            
                            # Your existing circle-contour overlap validation
                            circle_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.circle(circle_mask, center, radius, 255, -1)
                            contour_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(contour_mask, [contour], 255)
                            
                            intersection = cv2.bitwise_and(circle_mask, contour_mask)
                            union = cv2.bitwise_or(circle_mask, contour_mask)
                            overlap_ratio = np.sum(intersection) / max(1, np.sum(union))
                            
                            if overlap_ratio > 0.7:
                                # Enhanced color verification
                                ball_type, confidence = self._verify_ball_color_advanced(frame, center, radius)
                                
                                if confidence > 0.4:  # Slightly higher threshold for contour method
                                    distance_from_center = np.sqrt(
                                        (center[0] - self.frame_center_x)**2 + 
                                        (center[1] - self.frame_center_y)**2
                                    )
                                    
                                    in_collection_zone = self.is_in_collection_zone(center)
                                    
                                    ball = DetectedObject(
                                        object_type=ball_type,
                                        center=center,
                                        radius=radius,
                                        area=area,
                                        confidence=confidence * circularity * overlap_ratio,
                                        distance_from_center=distance_from_center,
                                        in_collection_zone=in_collection_zone
                                    )
                                    detected_objects.append(ball)
        
        return detected_objects
    
    def _verify_ball_color_advanced(self, frame, center, radius) -> Tuple[str, float]:
        """Enhanced color verification for better ball type classification"""
        h, w = frame.shape[:2]
        
        # Extract region of interest around the ball
        roi_size = min(radius + 8, 30)  # Slightly larger than the ball
        x1, y1 = max(0, center[0] - roi_size), max(0, center[1] - roi_size)
        x2, y2 = min(w, center[0] + roi_size), min(h, center[1] + roi_size)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "ball", 0.0
        
        # Create circular mask for the ball area
        roi_h, roi_w = roi.shape[:2]
        roi_center = (roi_w // 2, roi_h // 2)
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        mask_radius = min(radius, min(roi_w//2, roi_h//2))
        cv2.circle(mask, roi_center, mask_radius, 255, -1)
        
        # Convert ROI to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Orange detection (VIP ball)
        orange_lower = np.array([10, 80, 80])  # Slightly relaxed thresholds
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
        orange_pixels = cv2.bitwise_and(orange_mask, mask)
        orange_ratio = np.sum(orange_pixels > 0) / max(1, np.sum(mask > 0))
        
        # White detection (regular ping pong ball)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray_roi, mask)
        
        if np.sum(mask > 0) > 0:
            mean_brightness = np.mean(masked_gray[mask > 0])
            
            # Check saturation (white objects have low saturation)
            s_channel = hsv_roi[:, :, 1]
            masked_saturation = cv2.bitwise_and(s_channel, mask)
            mean_saturation = np.mean(masked_saturation[mask > 0])
            
            # Enhanced orange ball detection
            if orange_ratio > 0.15:  # Lower threshold for better orange detection
                confidence = min(1.0, orange_ratio * 3.0)  # Boost orange confidence
                return "orange_ball", confidence
            
            # Enhanced white ball detection
            elif mean_brightness > 150 and mean_saturation < 80:
                # Calculate confidence based on brightness and low saturation
                brightness_conf = min(1.0, (mean_brightness - 150) / 105)  # 150-255 range
                saturation_conf = min(1.0, (80 - mean_saturation) / 80)   # Lower saturation = higher confidence
                
                # Combined confidence with slight boost for very bright, unsaturated objects
                combined_conf = (brightness_conf + saturation_conf) / 2
                if mean_brightness > 200 and mean_saturation < 40:
                    combined_conf = min(1.0, combined_conf * 1.2)  # Boost for very white objects
                
                return "ball", combined_conf
        
        return "ball", 0.0
    
    def detect_balls(self, frame) -> List[DetectedObject]:
        """Main detection method using hybrid approach"""
        # Primary: HoughCircles detection
        hough_balls = self.detect_balls_hough_circles(frame)
        
        # If HoughCircles finds few results, supplement with color-based detection
        if len(hough_balls) < 2:
            color_balls = self.detect_balls_color_contours(frame)
            
            # Merge results, avoiding duplicates
            for color_ball in color_balls:
                is_duplicate = False
                for hough_ball in hough_balls:
                    distance = np.sqrt(
                        (color_ball.center[0] - hough_ball.center[0])**2 +
                        (color_ball.center[1] - hough_ball.center[1])**2
                    )
                    if distance < 25:  # Too close, likely same ball
                        is_duplicate = True
                        break
                
                if not is_duplicate and len(hough_balls) < 6:
                    hough_balls.append(color_ball)
        
        # Sort by distance and confidence
        hough_balls.sort(key=lambda x: (x.distance_from_center, -x.confidence))
        
        # Limit to reasonable number of balls
        return hough_balls[:6]
    
    def detect_boundaries(self, frame) -> bool:
        """Detect if robot is too close to red walls (danger zones)"""
        if frame is None:
            return False
        
        h, w = frame.shape[:2]
        self.detected_walls = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red wall detection
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        self.red_mask = red_mask.copy()
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Smaller danger zones - only trigger when very close
        danger_distance = min(50, int(h * 0.1))
        bottom_danger_y = h - danger_distance
        bottom_mask = red_mask[bottom_danger_y:h, :]
        
        edge_width = min(30, int(w * 0.06))
        left_mask = red_mask[:, 0:edge_width]
        right_mask = red_mask[:, w-edge_width:w]
        
        danger_detected = False
        min_wall_area = 120  # Reduced slightly for better detection
        
        # Check bottom
        contours, _ = cv2.findContours(bottom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if w_rect > 40 and h_rect > 12:  # Horizontal wall
                    danger_detected = True
                    wall_info = {
                        'zone': 'bottom',
                        'contour': contour,
                        'area': area,
                        'bbox': (x, bottom_danger_y + y, w_rect, h_rect),
                        'length': w_rect,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    break
        
        # Check left
        contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if h_rect > 60 and w_rect > 12:  # Vertical wall
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
                    break
        
        # Check right
        contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if h_rect > 60 and w_rect > 12:  # Vertical wall
                    danger_detected = True
                    wall_info = {
                        'zone': 'right',
                        'contour': contour,
                        'area': area,
                        'bbox': (w - edge_width + x, y, w_rect, h_rect),
                        'length': h_rect,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    break
        
        return danger_detected
    
    def start(self):
        """Initialize vision system"""
        return self.camera.start_capture()
    
    def get_frame(self):
        """Get current camera frame"""
        ret, frame = self.camera.capture_frame()
        if ret:
            self.last_frame = frame
        return ret, frame
    
    def get_target_ball(self, balls: List[DetectedObject]) -> Optional[DetectedObject]:
        """Determine which ball to target"""
        if balls:
            # Filter for confident detections
            confident_balls = [ball for ball in balls if ball.confidence > 0.3]
            if confident_balls:
                confident_balls.sort(key=lambda x: x.distance_from_center)
                target = confident_balls[0]
                self.current_target = target
                return target
        
        self.current_target = None
        return None
    
    def should_activate_servo(self) -> bool:
        """Check if servo should be activated"""
        if not self.current_target:
            return False
        return self.current_target.in_collection_zone
    
    def get_navigation_command(self, detected_objects: List[DetectedObject]) -> str:
        """Get navigation command based on detections"""
        target_ball = self.get_target_ball(detected_objects)
        
        if target_ball:
            if target_ball.in_collection_zone:
                return "collect_ball"
            else:
                return self._get_direction_to_object(target_ball)
        
        return "search"
    
    def _get_direction_to_object(self, obj: DetectedObject) -> str:
        """Get direction to move toward object"""
        x_offset = obj.center[0] - self.frame_center_x
        
        if abs(x_offset) > 30:
            if x_offset > 0:
                return "turn_right"
            else:
                return "turn_left"
        else:
            return "forward"
    
    def draw_detections(self, frame, balls: List[DetectedObject]) -> np.ndarray:
        """Clean and organized detection visualization"""
        if not config.DEBUG_VISION:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # === CLEAN WALL VISUALIZATION ===
        
        # Show red walls with clear but subtle overlay
        if self.red_mask is not None:
            # Create clean red overlay - moderate opacity
            wall_overlay = np.zeros_like(result)
            wall_overlay[:, :, 2] = self.red_mask  # Red channel
            cv2.addWeighted(result, 0.75, wall_overlay, 0.25, 0, result)
            
            # Add clean outlines around red walls
            contours, _ = cv2.findContours(self.red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
        
        # === DANGER ZONES - CLEAN BORDERS ONLY ===
        
        danger_distance = min(50, int(h * 0.1))
        danger_y = h - danger_distance
        edge_width = min(30, int(w * 0.06))
        
        # Draw clean danger zone borders (no fill)
        cv2.rectangle(result, (0, danger_y), (w, h), (0, 100, 255), 2)  # Bottom
        cv2.rectangle(result, (0, 0), (edge_width, h), (0, 100, 255), 2)  # Left
        cv2.rectangle(result, (w - edge_width, 0), (w, h), (0, 100, 255), 2)  # Right
        
        # === TRIGGERED WALLS - MINIMAL BUT CLEAR ===
        
        triggered_walls = 0
        for wall in self.detected_walls:
            if wall['triggered']:
                triggered_walls += 1
                x, y, w_rect, h_rect = wall['bbox']
                # Simple bright red rectangle for triggered walls
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 4)
        
        # === COLLECTION ZONE - CLEAN ===
        
        zone = self.collection_zone
        cv2.rectangle(result, (zone['left'], zone['top']), 
                    (zone['right'], zone['bottom']), (0, 255, 0), 2)
        
        # === BALL DETECTION - SIMPLIFIED ===
        
        for ball in balls:
            is_target = (self.current_target and 
                        self.current_target.center == ball.center)
            
            if ball.object_type == 'orange_ball':
                color = (0, 165, 255)  # Orange
                ball_char = 'O'
            else:
                color = (0, 255, 0)    # Green for white balls
                ball_char = 'B'
            
            if is_target:
                # Target ball - prominent
                cv2.circle(result, ball.center, ball.radius + 2, color, 3)
                cv2.circle(result, ball.center, 4, (255, 255, 0), -1)
                
                # Simple arrow to target
                cv2.arrowedLine(result, (self.frame_center_x, self.frame_center_y), 
                            ball.center, (255, 255, 0), 2)
            else:
                # Other balls - simple
                cv2.circle(result, ball.center, ball.radius, color, 2)
                cv2.circle(result, ball.center, 2, color, -1)
            
            # Simple ball label
            cv2.putText(result, f'{ball_char}', (ball.center[0]-5, ball.center[1]+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # === CLEAN STATUS PANEL ===
        
        # Create semi-transparent status panel at top
        panel_height = 120
        panel_overlay = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel_overlay[:, :] = (0, 0, 0)  # Black background
        
        # Apply panel with transparency
        result[0:panel_height, 0:w] = cv2.addWeighted(
            result[0:panel_height, 0:w], 0.3, 
            panel_overlay, 0.7, 0
        )
        
        # Panel border
        cv2.rectangle(result, (0, 0), (w, panel_height), (100, 100, 100), 2)
        
        # === CLEAN STATUS TEXT - SPLIT BETWEEN LEFT AND RIGHT ===
        
        # LEFT SIDE STATUS
        y_pos_left = 25
        line_height = 22
        
        # System status
        cv2.putText(result, f"GolfBot Vision System", (10, y_pos_left), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos_left += line_height
        
        # Ball status
        ball_count = len(balls)
        target_text = "TARGET" if self.current_target else "SEARCHING"
        cv2.putText(result, f"Balls: {ball_count} | Status: {target_text}", (10, y_pos_left), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos_left += line_height
        
        # Arena status
        arena_status = "Detected" if self.arena_detected else "Fallback"
        cv2.putText(result, f"Arena: {arena_status} | Method: HoughCircles+Color", (10, y_pos_left), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # RIGHT SIDE STATUS
        y_pos_right = 25
        right_x = w - 300  # Start from right side with margin
        
        # Wall status
        wall_status = "DANGER" if triggered_walls > 0 else "SAFE"
        wall_color = (0, 0, 255) if triggered_walls > 0 else (0, 255, 0)
        cv2.putText(result, f"Walls: {len(self.detected_walls)} detected | Status: {wall_status}", 
                (right_x, y_pos_right), cv2.FONT_HERSHEY_SIMPLEX, 0.5, wall_color, 1)
        y_pos_right += line_height
        
        # Target info (if available)
        if self.current_target:
            ball_type = "ORANGE" if self.current_target.object_type == 'orange_ball' else "WHITE"
            zone_status = "IN ZONE" if self.current_target.in_collection_zone else "APPROACHING"
            confidence_text = f"Conf: {self.current_target.confidence:.2f}"
            cv2.putText(result, f"Target: {ball_type} | {zone_status}", (right_x, y_pos_right), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos_right += line_height - 5
            cv2.putText(result, confidence_text, (right_x, y_pos_right), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        else:
            cv2.putText(result, "Target: SEARCHING FOR BALLS", (right_x, y_pos_right), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # === MINIMAL RED MASK PREVIEW ===
        
        if self.red_mask is not None:
            # Small, clean preview in bottom-right
            mask_size = 120
            mask_resized = cv2.resize(self.red_mask, (mask_size, int(mask_size * h / w)))
            mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_HOT)
            
            # Position in bottom-right
            start_x = w - mask_size - 10
            start_y = h - mask_resized.shape[0] - 10
            end_x = start_x + mask_size
            end_y = start_y + mask_resized.shape[0]
            
            # Place mask preview
            result[start_y:end_y, start_x:end_x] = mask_colored
            cv2.rectangle(result, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1)
            
            # Small label
            cv2.putText(result, "Red Mask", (start_x, start_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # === CLEAN LEGEND (BOTTOM LEFT) ===
        
        legend_x = 10
        legend_y = h - 80
        legend_bg_height = 70
        
        # Legend background
        legend_overlay = np.zeros((legend_bg_height, 280, 3), dtype=np.uint8)
        result[legend_y-10:legend_y+legend_bg_height-10, legend_x:legend_x+280] = cv2.addWeighted(
            result[legend_y-10:legend_y+legend_bg_height-10, legend_x:legend_x+280], 0.3,
            legend_overlay, 0.7, 0
        )
        
        cv2.rectangle(result, (legend_x, legend_y-10), (legend_x+280, legend_y+legend_bg_height-10), 
                    (100, 100, 100), 1)
        
        # Legend content
        cv2.putText(result, "LEGEND", (legend_x+5, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, "Red: Wall areas", (legend_x+5, legend_y+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(result, "Blue: Danger zones", (legend_x+5, legend_y+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        cv2.putText(result, "Green: Collection zone", (legend_x+5, legend_y+45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(result, "Yellow: Current target", (legend_x+150, legend_y+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(result, "O: Orange ball", (legend_x+150, legend_y+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        cv2.putText(result, "B: White ball", (legend_x+150, legend_y+45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # === CENTER CROSSHAIR - MINIMAL ===
        
        cv2.line(result, (self.frame_center_x-10, self.frame_center_y), 
                (self.frame_center_x+10, self.frame_center_y), (255, 255, 255), 1)
        cv2.line(result, (self.frame_center_x, self.frame_center_y-10), 
                (self.frame_center_x, self.frame_center_y+10), (255, 255, 255), 1)
        
        # === ARENA BOUNDARY (IF DETECTED) ===
        
        if self.arena_detected and self.arena_contour is not None:
            cv2.drawContours(result, [self.arena_contour], -1, (0, 255, 255), 1)
        
        return result
    
    def process_frame(self):
        """Process current frame and return detection results"""
        ret, frame = self.get_frame()
        if not ret:
            return None, None, None, None, None
        
        # Detect all balls using hybrid method
        balls = self.detect_balls(frame)
        orange_ball = None  # Deprecated - handled in unified detection
        near_boundary = self.detect_boundaries(frame)
        
        # Get navigation command
        nav_command = self.get_navigation_command(balls)
        
        # Create enhanced debug visualization
        debug_frame = self.draw_detections(frame, balls)
        
        return balls, orange_ball, near_boundary, nav_command, debug_frame
    
    def cleanup(self):
        """Clean up vision system"""
        self.camera.release()
        cv2.destroyAllWindows()
        self.logger.info("Enhanced vision system cleanup completed")