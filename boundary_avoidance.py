import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

# --- config.py (simulated for demonstration) ---
class Config:
    DEBUG_VISION = True

config = Config()
# -------------------------------------------------

class BoundaryAvoidanceSystem:
    """Simplified system for detecting and avoiding arena boundaries/walls"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO if config.DEBUG_VISION else logging.WARNING,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Detection results
        self.detected_walls = []
        self.red_mask = None

        # Multi-step avoidance state
        self.avoidance_state = None  # 'backing_up' or 'turning_right'
        self.backup_duration_frames = 10
        self.backup_frame_count = 0

    def _create_red_mask(self, frame) -> np.ndarray:
        """Create mask for red boundaries"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red detection ranges - optimized for strong red walls
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return red_mask

    def detect_boundaries(self, frame) -> bool:
        """Detect if robot is too close to red walls in critical zones"""
        if frame is None:
            return False

        h, w = frame.shape[:2]
        self.detected_walls = []
        
        # Only check bottom 30% of frame where walls are actually dangerous
        danger_zone_top = int(h * 0.7)
        danger_zone_bottom = int(h * 0.95)  # Stop before very bottom edge
        
        self.red_mask = self._create_red_mask(frame)
        
        # Extract danger zone for analysis
        danger_region = self.red_mask[danger_zone_top:danger_zone_bottom, :]
        
        # Find wall contours in danger zone
        contours, _ = cv2.findContours(danger_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_wall_area = 200  # Minimum area to consider a significant wall
        danger_detected = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_wall_area:
                continue
                
            # Get bounding box in danger zone coordinates
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Convert back to full frame coordinates
            full_frame_y = danger_zone_top + y
            
            # Determine wall type and position
            wall_info = self._classify_wall(x, full_frame_y, w_rect, h_rect, w, contour, area)
            
            if wall_info:
                self.detected_walls.append(wall_info)
                danger_detected = True
                
                if config.DEBUG_VISION:
                    self.logger.info(f"Wall detected: {wall_info['zone']} - area={area}, size=({w_rect}x{h_rect})")

        return danger_detected

    def _classify_wall(self, x, y, w_rect, h_rect, frame_width, contour, area) -> Optional[Dict]:
        """Classify wall based on position and size"""
        # Determine wall zone based on position
        left_threshold = frame_width * 0.3
        right_threshold = frame_width * 0.7
        
        if x < left_threshold:
            zone = 'left'
        elif x > right_threshold:
            zone = 'right'
        else:
            zone = 'center'
        
        # Wall must be substantial enough to be a real obstacle
        if w_rect < 30 and h_rect < 30:
            return None
            
        return {
            'zone': zone,
            'contour': contour,
            'area': area,
            'bbox': (x, y, w_rect, h_rect),
            'triggered': True
        }

    def get_avoidance_command(self, frame) -> Optional[str]:
        """Get avoidance command based on wall detection"""
        # Handle multi-step avoidance sequence
        if self.avoidance_state == 'backing_up':
            self.backup_frame_count += 1
            if self.backup_frame_count < self.backup_duration_frames:
                return 'move_backward'
            else:
                # Finished backing up, start turning
                self.avoidance_state = 'turning_right'
                self.backup_frame_count = 0
                return 'turn_right'
                
        elif self.avoidance_state == 'turning_right':
            # Complete the turn and reset state
            self.avoidance_state = None
            return 'turn_right'

        # Check for new dangers
        danger_detected = self.detect_boundaries(frame)
        
        if not danger_detected:
            self.avoidance_state = None
            self.backup_frame_count = 0
            return None

        # Determine avoidance action based on wall positions
        triggered_zones = [wall['zone'] for wall in self.detected_walls if wall.get('triggered', False)]

        # Priority: side walls first (simple turn), then center walls (backup + turn)
        if 'left' in triggered_zones:
            self._reset_avoidance_state()
            return 'turn_right'
        elif 'right' in triggered_zones:
            self._reset_avoidance_state()
            return 'turn_left'
        elif 'center' in triggered_zones:
            if self.avoidance_state is None:
                # Start backup sequence for center obstacles
                self.avoidance_state = 'backing_up'
                self.backup_frame_count = 0
                return 'move_backward'
        
        # Default safe action
        self._reset_avoidance_state()
        return 'turn_right'

    def _reset_avoidance_state(self):
        """Reset avoidance state machine"""
        self.avoidance_state = None
        self.backup_frame_count = 0

    def draw_visualization(self, frame) -> np.ndarray:
        """Draw simple boundary detection overlay"""
        if frame is None:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        # Draw danger zone
        danger_zone_top = int(h * 0.7)
        danger_zone_bottom = int(h * 0.95)
        cv2.rectangle(result, (0, danger_zone_top), (w, danger_zone_bottom), (0, 255, 255), 2)

        # Highlight detected walls
        for wall in self.detected_walls:
            if wall.get('triggered'):
                x, y, w_rect, h_rect = wall['bbox']
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 3)
                
                # Label the wall zone
                cv2.putText(result, wall['zone'].upper(), (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show current state
        status_text = f"State: {self.avoidance_state or 'Normal'}"
        cv2.putText(result, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return result

    def get_status(self) -> Dict:
        """Get system status"""
        triggered_walls = [wall for wall in self.detected_walls if wall.get('triggered', False)]
        
        return {
            'walls_detected': len(self.detected_walls),
            'danger_zones': [wall['zone'] for wall in triggered_walls],
            'safe': len(triggered_walls) == 0,
            'avoidance_state': self.avoidance_state
        }

    def reset(self):
        """Reset detection state"""
        self.detected_walls = []
        self.red_mask = None
        self._reset_avoidance_state()

