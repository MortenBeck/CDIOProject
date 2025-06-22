import cv2
import numpy as np
import time
import subprocess
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
import config
from boundary_avoidance import BoundaryAvoidanceSystem

@dataclass
class DetectedObject:
    """Class to store detected object information"""
    object_type: str  # 'ball' or 'delivery_zone'
    center: Tuple[int, int]
    radius: int
    area: int
    confidence: float
    distance_from_center: float
    in_collection_zone: bool = False

@dataclass
class DeliveryZone:
    """Class to store delivery zone information"""
    center: Tuple[int, int]
    area: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance_from_center: float
    is_centered: bool

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
    """Enhanced vision processing system with delivery zone detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.camera = Pi5Camera()
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        self.last_frame = None
        self.current_target = None
        self.current_delivery_zone = None
        
        # Initialize boundary avoidance system
        self.boundary_system = BoundaryAvoidanceSystem()
        
        # Collection zone boundaries
        self.collection_zone = self._calculate_collection_zone()
        
        # Detection method tracking
        self.detection_method = "hybrid"
        
        # Dashboard support - store recent detections
        self._last_detected_balls = []
        self._last_detected_delivery_zones = []
        
        # Delegate arena properties to boundary system
        self.arena_mask = None
        self.arena_detected = False
        self.arena_contour = None
    
    def detect_delivery_zones(self, frame) -> List[DeliveryZone]:
        """Detect green delivery zones (extensions of outer wall)"""
        delivery_zones = []
        
        if frame is None:
            return delivery_zones
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green delivery zones
        green_mask = cv2.inRange(hsv, config.DELIVERY_ZONE_HSV_LOWER, config.DELIVERY_ZONE_HSV_UPPER)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Apply arena mask if available
        if self.arena_mask is not None:
            green_mask = cv2.bitwise_and(green_mask, self.arena_mask)
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if config.DELIVERY_ZONE_MIN_AREA < area < config.DELIVERY_ZONE_MAX_AREA:
                # Calculate center and bounding box
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    center = (center_x, center_y)
                    
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    bbox = (x, y, w_rect, h_rect)
                    
                    # Calculate distance from frame center
                    distance_from_center = np.sqrt(
                        (center_x - self.frame_center_x)**2 + 
                        (center_y - self.frame_center_y)**2
                    )
                    
                    # Check if centered
                    is_centered = distance_from_center <= config.DELIVERY_CENTERING_TOLERANCE
                    
                    # Calculate confidence based on area and shape
                    size_confidence = min(1.0, area / 5000)
                    aspect_ratio = w_rect / max(h_rect, 1)
                    shape_confidence = min(1.0, 1.0 / max(1, abs(aspect_ratio - 1.5)))  # Prefer rectangular
                    confidence = (size_confidence + shape_confidence) / 2
                    
                    if confidence > 0.3:
                        delivery_zone = DeliveryZone(
                            center=center,
                            area=area,
                            confidence=confidence,
                            bbox=bbox,
                            distance_from_center=distance_from_center,
                            is_centered=is_centered
                        )
                        delivery_zones.append(delivery_zone)
        
        # Sort by confidence and distance from center
        delivery_zones.sort(key=lambda z: (-z.confidence, z.distance_from_center))
        
        # Store for dashboard access
        self._last_detected_delivery_zones = delivery_zones
        
        return delivery_zones[:3]  # Limit to top 3 candidates
    
    def get_delivery_zone_centering_command(self, delivery_zone: DeliveryZone) -> Optional[str]:
        """Get command to center robot on delivery zone"""
        if delivery_zone.is_centered:
            return "centered"
        
        center_x, center_y = delivery_zone.center
        x_offset = center_x - self.frame_center_x
        
        # Simple left/right centering
        if abs(x_offset) > config.DELIVERY_CENTERING_TOLERANCE:
            if x_offset > 0:
                return "turn_right"
            else:
                return "turn_left"
        
        return "centered"
    
    # === EXISTING BALL DETECTION METHODS ===
    def is_ball_centered(self, ball: DetectedObject) -> bool:
        """Check if ball is centered enough to start collection (both X and Y)"""
        x_offset = abs(ball.center[0] - self.frame_center_x)
        y_offset = abs(ball.center[1] - self.frame_center_y)
        
        x_centered = x_offset <= config.CENTERING_TOLERANCE
        y_centered = y_offset <= config.CENTERING_DISTANCE_TOLERANCE
        
        return x_centered and y_centered
    
    def get_centering_adjustment(self, ball: DetectedObject) -> tuple:
        """Get centering adjustment directions (x_direction, y_direction)"""
        x_offset = ball.center[0] - self.frame_center_x
        y_offset = ball.center[1] - self.frame_center_y
        
        # X-axis centering
        if abs(x_offset) <= config.CENTERING_TOLERANCE:
            x_direction = 'centered'
        elif x_offset > 0:
            x_direction = 'right'
        else:
            x_direction = 'left'
        
        # Y-axis centering
        if abs(y_offset) <= config.CENTERING_DISTANCE_TOLERANCE:
            y_direction = 'centered'
        elif y_offset > 0:
            y_direction = 'backward'
        else:
            y_direction = 'forward'
        
        return x_direction, y_direction
    
    def calculate_drive_time_to_ball(self, ball: DetectedObject) -> float:
        """Calculate how long to drive to reach the ball"""
        collection_zone_bottom_center = (
            (self.collection_zone['left'] + self.collection_zone['right']) // 2,
            self.collection_zone['bottom'] - 20
        )
        
        dx = ball.center[0] - collection_zone_bottom_center[0]
        dy = ball.center[1] - collection_zone_bottom_center[1]
        pixel_distance = np.sqrt(dx*dx + dy*dy)
        
        drive_time = pixel_distance * config.COLLECTION_DRIVE_TIME_PER_PIXEL
        drive_time = max(config.MIN_COLLECTION_DRIVE_TIME, 
                        min(config.MAX_COLLECTION_DRIVE_TIME, drive_time))
        
        if config.DEBUG_COLLECTION:
            self.logger.info(f"Ball distance: {pixel_distance:.1f} pixels -> {drive_time:.2f}s drive time")
        
        return drive_time
    
    def detect_excluded_areas(self, frame):
        """Detect white containers/cages where balls should be excluded"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        bottom_start = int(h * 0.7)
        cropped_frame = frame[bottom_start:h, :]
        
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        exclusion_zones = []
        min_container_area = (w * h) * 0.02
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_container_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                aspect_ratio = w_rect / max(h_rect, 1)
                if 0.3 < aspect_ratio < 3.0 and w_rect > 50 and h_rect > 30:
                    margin = 10
                    exclusion_zone = {
                        'x': max(0, x - margin),
                        'y': max(0, y + bottom_start - margin),
                        'width': min(w - x + margin, w_rect + 2*margin),
                        'height': min(h - (y + bottom_start) + margin, h_rect + 2*margin),
                        'area': area
                    }
                    exclusion_zones.append(exclusion_zone)
        
        return exclusion_zones

    def is_ball_in_exclusion_zone(self, ball_center, exclusion_zones):
        """Check if a ball center is inside any exclusion zone"""
        if not exclusion_zones:
            return False
            
        x, y = ball_center
        
        for zone in exclusion_zones:
            if (zone['x'] <= x <= zone['x'] + zone['width'] and 
                zone['y'] <= y <= zone['y'] + zone['height']):
                return True
        
        return False

    def detect_balls_hough_circles(self, frame) -> List[DetectedObject]:
        """Primary detection method using HoughCircles with exclusion zones"""
        detected_objects = []
        
        if frame is None:
            return detected_objects
        
        if self.boundary_system.arena_mask is None:
            self.boundary_system.detect_arena_boundaries(frame)
        
        self.arena_mask = self.boundary_system.arena_mask
        self.arena_detected = self.boundary_system.arena_detected
        self.arena_contour = self.boundary_system.arena_contour
        
        exclusion_zones = self.detect_excluded_areas(frame)
        
        h, w = frame.shape[:2]
        bottom_exclusion_start = int(h * 0.7)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.arena_mask is not None:
            gray = cv2.bitwise_and(gray, self.arena_mask)
        
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=50,
            param2=28,
            minRadius=config.BALL_MIN_RADIUS,
            maxRadius=config.BALL_MAX_RADIUS
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, radius) in circles:
                center = (x, y)
                
                if y >= bottom_exclusion_start:
                    continue
                
                if (self.arena_mask is not None and 
                    0 <= y < h and 0 <= x < w and
                    self.arena_mask[y, x] > 0):
                    
                    if self.is_ball_in_exclusion_zone(center, exclusion_zones):
                        continue
                    
                    confidence = self._verify_white_ball_color(frame, center, radius)
                    
                    if confidence > 0.3:
                        distance_from_center = np.sqrt(
                            (center[0] - self.frame_center_x)**2 + 
                            (center[1] - self.frame_center_y)**2
                        )
                        
                        in_collection_zone = self.is_ball_in_target_zone(center)
                        area = int(np.pi * radius * radius)
                        
                        ball = DetectedObject(
                            object_type='ball',
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
        """Fallback detection using color+contour method"""
        detected_objects = []
        
        if frame is None:
            return detected_objects
        
        if self.boundary_system.arena_mask is None:
            self.boundary_system.detect_arena_boundaries(frame)
        
        self.arena_mask = self.boundary_system.arena_mask
        
        exclusion_zones = self.detect_excluded_areas(frame)
        
        h, w = frame.shape[:2]
        bottom_exclusion_start = int(h * 0.7)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        ball_lower = np.array([0, 0, 200])
        ball_upper = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, ball_lower, ball_upper)
        
        if self.arena_mask is not None:
            white_mask = cv2.bitwise_and(white_mask, self.arena_mask)
        
        kernel = np.ones((7, 7), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.medianBlur(white_mask, 5)
        
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if config.BALL_MIN_AREA < area < config.BALL_MAX_AREA:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.6:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x), int(y))
                        radius = int(radius)
                        
                        if center[1] >= bottom_exclusion_start:
                            continue
                        
                        if (config.BALL_MIN_RADIUS < radius < config.BALL_MAX_RADIUS and
                            0 <= center[1] < h and 0 <= center[0] < w and
                            self.arena_mask[center[1], center[0]] > 0):
                            
                            if self.is_ball_in_exclusion_zone(center, exclusion_zones):
                                continue
                            
                            circle_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.circle(circle_mask, center, radius, 255, -1)
                            contour_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(contour_mask, [contour], 255)
                            
                            intersection = cv2.bitwise_and(circle_mask, contour_mask)
                            union = cv2.bitwise_or(circle_mask, contour_mask)
                            overlap_ratio = np.sum(intersection) / max(1, np.sum(union))
                            
                            if overlap_ratio > 0.7:
                                confidence = self._verify_white_ball_color(frame, center, radius)
                                
                                if confidence > 0.4:
                                    distance_from_center = np.sqrt(
                                        (center[0] - self.frame_center_x)**2 + 
                                        (center[1] - self.frame_center_y)**2
                                    )
                                    
                                    in_collection_zone = self.is_ball_in_target_zone(center)
                                    
                                    ball = DetectedObject(
                                        object_type='ball',
                                        center=center,
                                        radius=radius,
                                        area=area,
                                        confidence=confidence * circularity * overlap_ratio,
                                        distance_from_center=distance_from_center,
                                        in_collection_zone=in_collection_zone
                                    )
                                    detected_objects.append(ball)
        
        return detected_objects
    
    def _verify_white_ball_color(self, frame, center, radius) -> float:
        """Simplified color verification for white balls only"""
        h, w = frame.shape[:2]
        
        roi_size = min(radius + 8, 30)
        x1, y1 = max(0, center[0] - roi_size), max(0, center[1] - roi_size)
        x2, y2 = min(w, center[0] + roi_size), min(h, center[1] + roi_size)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0
        
        roi_h, roi_w = roi.shape[:2]
        roi_center = (roi_w // 2, roi_h // 2)
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        mask_radius = min(radius, min(roi_w//2, roi_h//2))
        cv2.circle(mask, roi_center, mask_radius, 255, -1)
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray_roi, mask)
        
        if np.sum(mask > 0) > 0:
            mean_brightness = np.mean(masked_gray[mask > 0])
            
            s_channel = hsv_roi[:, :, 1]
            masked_saturation = cv2.bitwise_and(s_channel, mask)
            mean_saturation = np.mean(masked_saturation[mask > 0])
            
            if mean_brightness > 150 and mean_saturation < 80:
                brightness_conf = min(1.0, (mean_brightness - 150) / 105)
                saturation_conf = min(1.0, (80 - mean_saturation) / 80)
                
                combined_conf = (brightness_conf + saturation_conf) / 2
                if mean_brightness > 200 and mean_saturation < 40:
                    combined_conf = min(1.0, combined_conf * 1.2)
                
                return combined_conf
        
        return 0.0
    
    def detect_balls(self, frame) -> List[DetectedObject]:
        """Main detection method using hybrid approach"""
        hough_balls = self.detect_balls_hough_circles(frame)
        
        if len(hough_balls) < 2:
            color_balls = self.detect_balls_color_contours(frame)
            
            for color_ball in color_balls:
                is_duplicate = False
                for hough_ball in hough_balls:
                    distance = np.sqrt(
                        (color_ball.center[0] - hough_ball.center[0])**2 +
                        (color_ball.center[1] - hough_ball.center[1])**2
                    )
                    if distance < 25:
                        is_duplicate = True
                        break
                
                if not is_duplicate and len(hough_balls) < 6:
                    hough_balls.append(color_ball)
        
        hough_balls.sort(key=lambda x: (x.distance_from_center, -x.confidence))
        detected_balls = hough_balls[:6]
        self._last_detected_balls = detected_balls
        
        return detected_balls
    
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
            confident_balls = [ball for ball in balls if ball.confidence > 0.3]
            if confident_balls:
                confident_balls.sort(key=lambda x: x.distance_from_center)
                target = confident_balls[0]
                self.current_target = target
                return target
        
        self.current_target = None
        return None
    
    def get_target_delivery_zone(self, delivery_zones: List[DeliveryZone]) -> Optional[DeliveryZone]:
        """Determine which delivery zone to target"""
        if delivery_zones:
            delivery_zones.sort(key=lambda x: (-x.confidence, x.distance_from_center))
            target = delivery_zones[0]
            self.current_delivery_zone = target
            return target
        
        self.current_delivery_zone = None
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
            if self.is_ball_in_target_zone(target_ball.center):
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
    
    def draw_detections_legacy(self, frame, balls: List[DetectedObject]) -> np.ndarray:
        """Enhanced detection visualization with centering info (LEGACY MODE)"""
        if not config.DEBUG_VISION:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Boundary/wall visualization
        result = self.boundary_system.draw_boundary_visualization(result)
        
        # Collection zone
        zone = self.collection_zone
        cv2.rectangle(result, (zone['left'], zone['top']), 
                    (zone['right'], zone['bottom']), (0, 255, 0), 2)
        
        # Centering tolerance visualization
        tolerance_color = (255, 255, 0)
        left_line = self.frame_center_x - config.CENTERING_TOLERANCE
        right_line = self.frame_center_x + config.CENTERING_TOLERANCE
        top_line = self.frame_center_y - config.CENTERING_DISTANCE_TOLERANCE
        bottom_line = self.frame_center_y + config.CENTERING_DISTANCE_TOLERANCE
        
        cv2.line(result, (left_line, 0), (left_line, h), tolerance_color, 1)
        cv2.line(result, (right_line, 0), (right_line, h), tolerance_color, 1)
        cv2.line(result, (0, top_line), (w, top_line), tolerance_color, 1)
        cv2.line(result, (0, bottom_line), (w, bottom_line), tolerance_color, 1)
        
        cv2.putText(result, "CENTERING ZONE", (left_line + 5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, tolerance_color, 1)
        
        # Ball detection with centering info
        for ball in balls:
            is_target = (self.current_target and 
                        self.current_target.center == ball.center)
            
            color = (0, 255, 0)
            ball_char = 'B'
            
            if is_target:
                cv2.circle(result, ball.center, ball.radius + 2, color, 3)
                cv2.circle(result, ball.center, 4, (255, 255, 0), -1)
                
                cv2.arrowedLine(result, (self.frame_center_x, self.frame_center_y), 
                            ball.center, (255, 255, 0), 2)
                
                centered = self.is_ball_centered(ball)
                center_color = (0, 255, 0) if centered else (0, 0, 255)
                center_text = "CENTERED" if centered else "CENTERING"
                cv2.putText(result, center_text, (ball.center[0]-30, ball.center[1]-25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, center_color, 1)
                
                if centered:
                    drive_time = self.calculate_drive_time_to_ball(ball)
                    cv2.putText(result, f"{drive_time:.1f}s", (ball.center[0]-15, ball.center[1]+35), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                cv2.circle(result, ball.center, ball.radius, color, 2)
                cv2.circle(result, ball.center, 2, color, -1)
            
            cv2.putText(result, f'{ball_char}', (ball.center[0]-5, ball.center[1]+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw delivery zones if detected
        delivery_zones = getattr(self, '_last_detected_delivery_zones', [])
        for i, zone in enumerate(delivery_zones):
            color = (255, 0, 255) if i == 0 else (128, 0, 128)  # Magenta for delivery zones
            thickness = 3 if i == 0 else 2
            
            x, y, w_rect, h_rect = zone.bbox
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            cv2.circle(result, zone.center, 5, color, -1)
            cv2.putText(result, f'DZ{i+1}', (zone.center[0]-10, zone.center[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show if centered
            if zone.is_centered:
                cv2.putText(result, "CENTERED", (zone.center[0]-30, zone.center[1]+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Status panel and other visualization code...
        # (keeping existing status panel code but abbreviated for space)
        
        return result
    
    def draw_detections_clean(self, frame, balls: List[DetectedObject]) -> np.ndarray:
        """Clean detection visualization for dashboard mode"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Target zone visualization
        zone = self.collection_zone
        cv2.rectangle(result, (zone['target_left'], zone['target_top']), 
                    (zone['target_right'], zone['target_bottom']), (0, 255, 255), 3)
        
        target_cx, target_cy = zone['target_center_x'], zone['target_center_y']
        cv2.circle(result, (target_cx, target_cy), 3, (0, 255, 255), 2)
        cv2.line(result, (target_cx-8, target_cy), (target_cx+8, target_cy), (0, 255, 255), 1)
        cv2.line(result, (target_cx, target_cy-8), (target_cx, target_cy+8), (0, 255, 255), 1)
        
        # Wall detection
        if hasattr(self.boundary_system, 'detected_walls') and self.boundary_system.detected_walls:
            for wall in self.boundary_system.detected_walls:
                if wall.get('triggered', False):
                    x, y, w_rect, h_rect = wall['bbox']
                    cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 3)
        
        # Ball detections
        for ball in balls:
            is_target = (self.current_target and 
                        self.current_target.center == ball.center)
            
            color = (0, 255, 0)
            ball_char = 'B'
            
            if is_target:
                in_target_zone = self.is_ball_in_target_zone(ball.center)
                
                if in_target_zone:
                    cv2.circle(result, ball.center, ball.radius + 4, (0, 255, 0), 4)
                    cv2.circle(result, ball.center, 4, (0, 255, 0), -1)
                    cv2.putText(result, "COLLECT!", (ball.center[0]-30, ball.center[1]-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.circle(result, ball.center, ball.radius + 3, (255, 255, 0), 3)
                    cv2.circle(result, ball.center, 3, (0, 255, 255), -1)
                    
                    cv2.arrowedLine(result, ball.center, (target_cx, target_cy), (0, 255, 255), 2)
                    
                    distance = int(np.sqrt((ball.center[0] - target_cx)**2 + (ball.center[1] - target_cy)**2))
                    cv2.putText(result, f"{distance}px", (ball.center[0]-15, ball.center[1]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                cv2.circle(result, ball.center, ball.radius, color, 2)
                cv2.circle(result, ball.center, 2, color, -1)
            
            cv2.putText(result, ball_char, (ball.center[0]-5, ball.center[1]+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw delivery zones
        delivery_zones = getattr(self, '_last_detected_delivery_zones', [])
        for i, zone in enumerate(delivery_zones):
            color = (255, 0, 255) if i == 0 else (128, 0, 128)
            thickness = 3 if i == 0 else 2
            
            x, y, w_rect, h_rect = zone.bbox
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            cv2.circle(result, zone.center, 5, color, -1)
            cv2.putText(result, f'DZ', (zone.center[0]-10, zone.center[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if zone.is_centered:
                cv2.putText(result, "READY", (zone.center[0]-20, zone.center[1]+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Frame center crosshair
        cx, cy = self.frame_center_x, self.frame_center_y
        cv2.line(result, (cx-6, cy), (cx+6, cy), (255, 255, 255), 1)
        cv2.line(result, (cx, cy-6), (cx, cy+6), (255, 255, 255), 1)
        
        return result
    
    def process_frame(self, dashboard_mode=False):
        """Process current frame and return detection results"""
        ret, frame = self.get_frame()
        if not ret:
            return None, None, None, None, None, None
        
        # Detect balls
        balls = self.detect_balls(frame)
        
        # Detect delivery zones
        delivery_zones = self.detect_delivery_zones(frame)
        
        # Use boundary system for wall detection
        near_boundary = self.boundary_system.detect_boundaries(frame)
        
        # Get navigation command
        nav_command = self.get_navigation_command(balls)
        
        # Create appropriate visualization based on mode
        if dashboard_mode:
            debug_frame = self.draw_detections_clean(frame, balls)
        else:
            debug_frame = self.draw_detections_legacy(frame, balls)
        
        return balls, None, near_boundary, nav_command, debug_frame, delivery_zones
    
    def _calculate_collection_zone(self):
        """Calculate collection zone boundaries"""
        center_x = config.CAMERA_WIDTH // 2
        vertical_pos = getattr(config, 'TARGET_ZONE_VERTICAL_POSITION', 0.65)
        center_y = int(config.CAMERA_HEIGHT * vertical_pos)
        
        target_width = getattr(config, 'TARGET_ZONE_WIDTH', 60)
        target_height = getattr(config, 'TARGET_ZONE_HEIGHT', 45)
        
        target_left = center_x - (target_width // 2)
        target_right = center_x + (target_width // 2)
        target_top = center_y - (target_height // 2)
        target_bottom = center_y + (target_height // 2)
        
        return {
            'target_left': target_left,
            'target_right': target_right,
            'target_top': target_top,
            'target_bottom': target_bottom,
            'target_center_x': center_x,
            'target_center_y': center_y
        }

    def is_ball_in_target_zone(self, ball_center: Tuple[int, int]) -> bool:
        """Check if ball is in the precise target zone"""
        x, y = ball_center
        zone = self.collection_zone
        
        horizontal_ok = zone['target_left'] <= x <= zone['target_right']
        vertical_ok = zone['target_top'] <= y <= zone['target_bottom']
        
        return horizontal_ok and vertical_ok

    def is_ball_centered_for_collection(self, ball: DetectedObject) -> bool:
        """Check if ball is perfectly positioned in the small target zone"""
        return self.is_ball_in_target_zone(ball.center)

    def get_drive_time_to_collection(self) -> float:
        """Get fixed drive time from target zone to collection point"""
        return getattr(config, 'FIXED_COLLECTION_DRIVE_TIME', 1.0)

    def get_centering_adjustment_v2(self, ball: DetectedObject) -> tuple:
        """Adaptive centering tolerances based on distance"""
        ball_x, ball_y = ball.center
        zone = self.collection_zone
        
        target_center_x = zone['target_center_x']
        target_center_y = zone['target_center_y']
        
        distance_to_target = np.sqrt((ball_x - target_center_x)**2 + (ball_y - target_center_y)**2)
        
        if distance_to_target > 80:
            x_tolerance = 20
            y_tolerance = 15
            hysteresis = 5
        elif distance_to_target > 40:
            x_tolerance = 12
            y_tolerance = 10
            hysteresis = 3
        else:
            x_tolerance = 8
            y_tolerance = 6
            hysteresis = 2
        
        x_offset = ball_x - target_center_x
        if abs(x_offset) <= x_tolerance:
            x_direction = 'centered'
        elif x_offset > x_tolerance + hysteresis:
            x_direction = 'right'
        elif x_offset < -(x_tolerance + hysteresis):
            x_direction = 'left'
        else:
            x_direction = 'centered'
        
        y_offset = ball_y - target_center_y
        if abs(y_offset) <= y_tolerance:
            y_direction = 'centered'
        elif y_offset > y_tolerance + hysteresis:
            y_direction = 'backward'
        elif y_offset < -(y_tolerance + hysteresis):
            y_direction = 'forward'
        else:
            y_direction = 'centered'
        
        return x_direction, y_direction
    
    @property
    def detected_walls(self):
        """Delegate to boundary system for compatibility"""
        return self.boundary_system.detected_walls
    
    def cleanup(self):
        """Clean up vision system"""
        self.camera.release()
        cv2.destroyAllWindows()
        self.logger.info("Enhanced vision system (with delivery zones) cleanup completed")