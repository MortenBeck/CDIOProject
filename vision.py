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
    object_type: str  # 'ball', 'orange_ball', 'goal_a', 'goal_b', 'boundary'
    center: Tuple[int, int]
    radius: int
    area: int
    confidence: float
    distance_from_center: float

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
                '--timeout', '1',
                '--width', str(config.CAMERA_WIDTH),
                '--height', str(config.CAMERA_HEIGHT),
                '--quality', '80'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode == 0 and os.path.exists(self.temp_file):
                frame = cv2.imread(self.temp_file)
                return True, frame
            else:
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
                            
                            ball = DetectedObject(
                                object_type='ball',
                                center=center,
                                radius=radius,
                                area=area,
                                confidence=circularity,
                                distance_from_center=distance_from_center
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
                                
                                best_orange_ball = DetectedObject(
                                    object_type='orange_ball',
                                    center=center,
                                    radius=radius,
                                    area=area,
                                    confidence=circularity,
                                    distance_from_center=distance_from_center
                                )
                                best_score = score
        
        return best_orange_ball
    
    def detect_goals(self, frame) -> List[DetectedObject]:
        """Detect red tape goals (Goal A = smaller, Goal B = larger)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for red color (handle wrap-around at 0/180)
        mask1 = cv2.inRange(hsv, config.GOAL_HSV_LOWER, config.GOAL_HSV_UPPER)
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        goals = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 500:  # Minimum area for goals
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                
                # Classify goal based on size
                goal_type = 'goal_a' if area < 2000 else 'goal_b'  # Smaller area = Goal A
                
                distance_from_center = np.sqrt(
                    (center[0] - self.frame_center_x)**2 + 
                    (center[1] - self.frame_center_y)**2
                )
                
                goal = DetectedObject(
                    object_type=goal_type,
                    center=center,
                    radius=max(w, h)//2,
                    area=area,
                    confidence=1.0,
                    distance_from_center=distance_from_center
                )
                goals.append(goal)
        
        return goals
    
    def detect_boundaries(self, frame) -> bool:
        """Detect if robot is near immediate boundaries/obstacles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Define region of interest - focus on lower portion where immediate obstacles matter
        # Ignore top 40% of frame (distant walls) and focus on bottom 60%
        roi_top = int(h * 0.2)  # Start checking from 40% down
        roi_bottom = h
        roi_left = 0
        roi_right = w
        
        # Extract ROI
        roi = gray[roi_top:roi_bottom, roi_left:roi_right]
        roi_h, roi_w = roi.shape
        
        edge_thickness = min(config.BOUNDARY_DETECTION_THRESHOLD, roi_w//4, roi_h//4)
        
        # Check only immediate edges in ROI
        bottom_edge = np.mean(roi[roi_h-edge_thickness:roi_h, :])  # Very bottom of frame
        left_edge = np.mean(roi[:, 0:edge_thickness])              # Left edge in ROI
        right_edge = np.mean(roi[:, roi_w-edge_thickness:roi_w])   # Right edge in ROI
        
        # Center reference from middle of ROI
        center_y_start = roi_h//4
        center_y_end = 3*roi_h//4
        center_x_start = roi_w//4  
        center_x_end = 3*roi_w//4
        center = np.mean(roi[center_y_start:center_y_end, center_x_start:center_x_end])
        
        # More aggressive threshold for immediate obstacles
        boundary_threshold = center - 50  # Darker threshold for true obstacles
        
        # Only trigger if edges are significantly darker (immediate obstacle)
        near_boundary = (bottom_edge < boundary_threshold or 
                        left_edge < boundary_threshold or 
                        right_edge < boundary_threshold)
        
        # Debug visualization if enabled
        if config.DEBUG_VISION and near_boundary:
            self.logger.debug(f"Boundary detected - Bottom: {bottom_edge:.1f}, Left: {left_edge:.1f}, Right: {right_edge:.1f}, Center: {center:.1f}, Threshold: {boundary_threshold:.1f}")
        
        return near_boundary
    
    def get_navigation_command(self, detected_objects: List[DetectedObject], 
                             orange_ball: Optional[DetectedObject]) -> str:
        """Determine navigation command based on detected objects"""
        
        # Priority 1: Orange ball if not collected yet
        if orange_ball and orange_ball.distance_from_center < config.COLLECTION_DISTANCE_THRESHOLD:
            return "collect_orange"
        elif orange_ball:
            return self._get_direction_to_object(orange_ball)
        
        # Priority 2: Regular balls
        if detected_objects:
            closest_ball = detected_objects[0]  # Already sorted by distance
            if closest_ball.distance_from_center < config.COLLECTION_DISTANCE_THRESHOLD:
                return "collect_ball"
            else:
                return self._get_direction_to_object(closest_ball)
        
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
    
    def find_goal_direction(self, goals: List[DetectedObject], prefer_goal_a=False) -> Optional[str]:
        """Find direction to specified goal type"""
        target_goals = [g for g in goals if 
                       (g.object_type == 'goal_a' if prefer_goal_a else g.object_type == 'goal_b')]
        
        if target_goals:
            closest_goal = min(target_goals, key=lambda x: x.distance_from_center)
            return self._get_direction_to_object(closest_goal)
        
        return None
    
    def draw_detections(self, frame, balls: List[DetectedObject], 
                       orange_ball: Optional[DetectedObject], 
                       goals: List[DetectedObject]) -> np.ndarray:
        """Draw detection results on frame for debugging"""
        if not config.DEBUG_VISION:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw boundary detection ROI
        roi_top = int(h * 0.4)
        cv2.rectangle(result, (0, roi_top), (w, h), (255, 0, 0), 2)
        cv2.putText(result, 'Boundary ROI', (10, roi_top-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw regular balls in green
        for ball in balls:
            cv2.circle(result, ball.center, ball.radius, (0, 255, 0), 2)
            cv2.circle(result, ball.center, 3, (0, 255, 0), -1)
            cv2.putText(result, 'Ball', 
                       (ball.center[0]-20, ball.center[1]-ball.radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw orange ball in orange
        if orange_ball:
            cv2.circle(result, orange_ball.center, orange_ball.radius, (0, 165, 255), 3)
            cv2.circle(result, orange_ball.center, 5, (0, 165, 255), -1)
            cv2.putText(result, 'VIP!', 
                       (orange_ball.center[0]-15, orange_ball.center[1]-orange_ball.radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Draw goals in red with labels
        for goal in goals:
            color = (0, 0, 255)
            label = 'Goal A' if goal.object_type == 'goal_a' else 'Goal B'
            cv2.rectangle(result, 
                         (goal.center[0]-goal.radius, goal.center[1]-goal.radius),
                         (goal.center[0]+goal.radius, goal.center[1]+goal.radius),
                         color, 2)
            cv2.putText(result, label, 
                       (goal.center[0]-20, goal.center[1]-goal.radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw center crosshair
        cv2.line(result, (self.frame_center_x-20, self.frame_center_y), 
                (self.frame_center_x+20, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y-20), 
                (self.frame_center_x, self.frame_center_y+20), (255, 255, 255), 2)
        
        return result
    
    def process_frame(self):
        """Process current frame and return detection results"""
        ret, frame = self.get_frame()
        if not ret:
            return None, None, None, None, None, None
        
        # Detect all objects
        balls = self.detect_balls(frame)
        orange_ball = self.detect_orange_ball(frame)
        goals = self.detect_goals(frame)
        near_boundary = self.detect_boundaries(frame)
        
        # Get navigation command
        nav_command = self.get_navigation_command(balls, orange_ball)
        
        # Create debug visualization
        debug_frame = self.draw_detections(frame, balls, orange_ball, goals)
        
        return balls, orange_ball, goals, near_boundary, nav_command, debug_frame
    
    def cleanup(self):
        """Clean up vision system"""
        self.camera.release()
        cv2.destroyAllWindows()
        self.logger.info("Vision system cleanup completed")