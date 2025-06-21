import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import time

# --- config.py (simulated for demonstration) ---
class Config:
    DEBUG_VISION = True

config = Config()
# -------------------------------------------------

class BoundaryAvoidanceSystem:
    """Simplified boundary avoidance system - backs up then turns right when wall detected"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO if config.DEBUG_VISION else logging.WARNING,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Simple detection results
        self.detected_walls = []
        self.red_mask = None

        # Arena boundaries (keep this simple)
        self.arena_mask = None
        self.arena_detected = False
        self.arena_contour = None
        
        # Add compatibility properties for dashboard
        self.avoidance_state = None
        
        # NEW: Avoidance sequence tracking
        self.avoidance_sequence_active = False
        self.avoidance_step = 0  # 0 = not active, 1 = backing up, 2 = turning right
        self.avoidance_start_time = None

    def detect_arena_boundaries(self, frame) -> bool:
        """Simplified arena boundary detection"""
        if frame is None:
            return False

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Simple red boundary detection
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_boundary_mask = mask1 + mask2

        # Clean up the boundary mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_boundary_mask = cv2.morphologyEx(red_boundary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find the arena boundary contour
        contours, _ = cv2.findContours(red_boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Check if this could be a valid arena boundary
            min_arena_area = (w * h) * 0.15

            if area > min_arena_area:
                # Create arena mask
                self.arena_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(self.arena_mask, [largest_contour], 255)

                # Erode slightly to be safe
                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                self.arena_mask = cv2.erode(self.arena_mask, erosion_kernel, iterations=1)

                self.arena_contour = largest_contour
                self.arena_detected = True

                if config.DEBUG_VISION:
                    self.logger.info(f"Arena boundary detected: area={area:.0f}")
                return True

        # Fallback: create a simple arena mask
        if not self.arena_detected:
            self.arena_mask = np.ones((h, w), dtype=np.uint8) * 255
            # Simple margins
            margin_h = int(h * 0.10)
            margin_w = int(w * 0.08)

            self.arena_mask[:margin_h, :] = 0
            self.arena_mask[-margin_h:, :] = 0
            self.arena_mask[:, :margin_w] = 0
            self.arena_mask[:, -margin_w:] = 0

            if config.DEBUG_VISION:
                self.logger.info("Using fallback arena mask")

        return False

    def detect_boundaries(self, frame) -> bool:
        """Simplified boundary detection - only check for red walls in front of robot"""
        if frame is None:
            return False

        h, w = frame.shape[:2]
        self.detected_walls = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wall detection
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        self.red_mask = red_mask.copy()

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # SIMPLIFIED: Only check bottom center area where robot is likely to hit walls
        danger_zone_top = int(h * 0.6)     # Bottom 40% of image
        danger_zone_left = int(w * 0.25)   # Center 50% of image width
        danger_zone_right = int(w * 0.75)

        # Extract danger zone
        danger_region = red_mask[danger_zone_top:h, danger_zone_left:danger_zone_right]
        
        # Check if there's significant red in the danger zone
        red_pixels = np.sum(danger_region > 0)
        total_pixels = danger_region.shape[0] * danger_region.shape[1]
        red_percentage = red_pixels / total_pixels

        # If more than 5% of danger zone is red, we're too close to a wall
        danger_detected = red_percentage > 0.05

        if danger_detected:
            # Add a simple wall entry for visualization
            wall_info = {
                'zone': 'front',
                'area': red_pixels,
                'bbox': (danger_zone_left, danger_zone_top, 
                        danger_zone_right - danger_zone_left, 
                        h - danger_zone_top),
                'triggered': True
            }
            self.detected_walls.append(wall_info)
            
            if config.DEBUG_VISION:
                self.logger.info(f"Wall detected in front danger zone: {red_percentage:.1%} red")

        return danger_detected

    def get_avoidance_command(self, frame) -> Optional[str]:
        """NEW: Sequential avoidance - back up THEN turn right when wall detected"""
        danger_detected = self.detect_boundaries(frame)

        # If no danger and no active sequence, we're clear
        if not danger_detected and not self.avoidance_sequence_active:
            return None
        
        # If danger detected and no sequence active, start the sequence
        if danger_detected and not self.avoidance_sequence_active:
            if config.DEBUG_VISION:
                self.logger.info("Wall detected - starting avoidance sequence: STEP 1 - BACKING UP")
            self.avoidance_sequence_active = True
            self.avoidance_step = 1
            self.avoidance_start_time = time.time()
            return 'move_backward'
        
        # If sequence is active, manage the steps
        if self.avoidance_sequence_active:
            elapsed_time = time.time() - self.avoidance_start_time
            
            if self.avoidance_step == 1:  # Currently backing up
                if elapsed_time >= 0.4:  # Back up for 0.4 seconds
                    if config.DEBUG_VISION:
                        self.logger.info("Backup complete - starting STEP 2 - TURNING RIGHT")
                    self.avoidance_step = 2
                    self.avoidance_start_time = time.time()
                    return 'turn_right'
                else:
                    # Continue backing up
                    return 'move_backward'
            
            elif self.avoidance_step == 2:  # Currently turning right
                if elapsed_time >= 0.6:  # Turn right for 0.6 seconds
                    if config.DEBUG_VISION:
                        self.logger.info("Turn complete - avoidance sequence finished")
                    self.avoidance_sequence_active = False
                    self.avoidance_step = 0
                    self.avoidance_start_time = None
                    return None  # Sequence complete
                else:
                    # Continue turning right
                    return 'turn_right'
        
        return None

    def get_closest_boundary_distance(self) -> float:
        """Simplified distance calculation for compatibility with main.py"""
        if not self.detected_walls:
            return 100.0  # No walls detected, return safe distance
        
        # Simple approximation: if walls detected, assume close distance
        return 10.0  # Close enough to trigger critical boundary check

    def draw_boundary_visualization(self, frame) -> np.ndarray:
        """Simplified visualization with sequence status"""
        if frame is None:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        # Show red walls if detected
        if self.red_mask is not None:
            wall_overlay = np.zeros_like(result)
            wall_overlay[:, :, 2] = self.red_mask  # Red channel
            cv2.addWeighted(result, 0.8, wall_overlay, 0.2, 0, result)

        # Show danger zone
        danger_zone_top = int(h * 0.6)
        danger_zone_left = int(w * 0.25)
        danger_zone_right = int(w * 0.75)
        
        cv2.rectangle(result, (danger_zone_left, danger_zone_top), 
                     (danger_zone_right, h), (0, 255, 255), 2)
        cv2.putText(result, "DANGER ZONE", (danger_zone_left + 10, danger_zone_top + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show arena boundary if detected
        if self.arena_detected and self.arena_contour is not None:
            cv2.drawContours(result, [self.arena_contour], -1, (0, 255, 255), 1)

        # Show wall warnings with sequence status
        if self.detected_walls:
            if self.avoidance_sequence_active:
                if self.avoidance_step == 1:
                    status_text = "BACKING UP (STEP 1/2)"
                    color = (0, 165, 255)  # Orange
                elif self.avoidance_step == 2:
                    status_text = "TURNING RIGHT (STEP 2/2)"
                    color = (0, 100, 255)  # Red-orange
                else:
                    status_text = "WALL DETECTED - STARTING SEQUENCE"
                    color = (0, 0, 255)  # Red
            else:
                status_text = "WALL DETECTED - BACKING UP"
                color = (0, 0, 255)  # Red
                
            cv2.putText(result, status_text, (10, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return result

    def get_status(self) -> Dict:
        """Get simple status with sequence information"""
        triggered_walls = [wall for wall in self.detected_walls if wall.get('triggered', False)]

        return {
            'arena_detected': self.arena_detected,
            'walls_detected': len(self.detected_walls),
            'walls_triggered': len(triggered_walls),
            'danger_zones': [wall['zone'] for wall in triggered_walls],
            'safe': len(triggered_walls) == 0,
            'avoidance_sequence_active': self.avoidance_sequence_active,
            'avoidance_step': self.avoidance_step
        }

    def reset(self):
        """Reset detection state"""
        self.detected_walls = []
        self.red_mask = None
        # Also reset avoidance sequence
        self.avoidance_sequence_active = False
        self.avoidance_step = 0
        self.avoidance_start_time = None