import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

# --- config.py (simulated for demonstration) ---
class Config:
    DEBUG_VISION = True
    CAMERA_HEIGHT = 480
    CAMERA_WIDTH = 640
    TARGET_ZONE_VERTICAL_POSITION = 0.65  # 65% down from top

config = Config()
# -------------------------------------------------

class BoundaryAvoidanceSystem:
    """FOCUSED: Only detect boundaries below the collection zone middle"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO if config.DEBUG_VISION else logging.WARNING,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Wall detection results for visualization
        self.detected_walls = []
        self.red_mask = None

        # Arena boundaries
        self.arena_mask = None
        self.arena_detected = False
        self.arena_contour = None

    def detect_arena_boundaries(self, frame) -> bool:
        """Detect arena boundaries from red walls to create detection mask"""
        if frame is None:
            return False

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red boundary detection with wider ranges
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 150, 100])
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

    def detect_boundaries(self, frame) -> bool:
        """FOCUSED: Only detect walls below the collection zone middle"""
        if frame is None:
            return False

        h, w = frame.shape[:2]
        self.detected_walls = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wall detection - same parameters as before
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

        # FOCUSED DETECTION ZONE: Only below collection zone middle
        target_zone_center_y = int(h * config.TARGET_ZONE_VERTICAL_POSITION)  # 65% down from top
        danger_start_y = target_zone_center_y  # Start detection from target zone center
        danger_end_y = int(h * 0.95)  # Go almost to bottom (95%)

        if config.DEBUG_VISION:
            self.logger.info(f"FOCUSED wall detection zone: Y={danger_start_y} to {danger_end_y} (target center at {target_zone_center_y})")

        danger_detected = False
        min_wall_area = 200   # Minimum area to consider a wall

        # Danger zone sizes (reduced from original)
        danger_distance_vertical = int(h * 0.1)    # 10% of frame height (reduced)
        danger_distance_horizontal = int(w * 0.15) # 15% of frame width (slightly increased for sides)

        # === REGION 1: BOTTOM WALL DETECTION (only in lower area) ===
        bottom_region_start = max(danger_start_y, danger_end_y - danger_distance_vertical)
        if bottom_region_start < danger_end_y:
            bottom_region = red_mask[bottom_region_start:danger_end_y, :]
            contours, _ = cv2.findContours(bottom_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_wall_area:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    if w_rect > 60:   # Horizontal wall (must be reasonably wide)
                        danger_detected = True
                        wall_info = {
                            'zone': 'bottom',
                            'contour': contour,
                            'area': area,
                            'bbox': (x, bottom_region_start + y, w_rect, h_rect),
                            'length': w_rect,
                            'triggered': True
                        }
                        self.detected_walls.append(wall_info)
                        if config.DEBUG_VISION:
                            self.logger.info(f"Bottom wall detected: area={area}, width={w_rect}")
                        break

        # === REGION 2: LEFT WALL DETECTION (only in focused area) ===
        left_region = red_mask[danger_start_y:danger_end_y, 0:danger_distance_horizontal]
        contours, _ = cv2.findContours(left_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if h_rect > 40:   # Vertical wall (must be reasonably tall)
                    danger_detected = True
                    wall_info = {
                        'zone': 'left',
                        'contour': contour,
                        'area': area,
                        'bbox': (x, danger_start_y + y, w_rect, h_rect),
                        'length': h_rect,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.info(f"Left wall detected: area={area}, height={h_rect}")
                    break

        # === REGION 3: RIGHT WALL DETECTION (only in focused area) ===
        right_region = red_mask[danger_start_y:danger_end_y, w-danger_distance_horizontal:w]
        contours, _ = cv2.findContours(right_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if h_rect > 40:   # Vertical wall
                    danger_detected = True
                    wall_info = {
                        'zone': 'right',
                        'contour': contour,
                        'area': area,
                        'bbox': (w - danger_distance_horizontal + x, danger_start_y + y, w_rect, h_rect),
                        'length': h_rect,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.info(f"Right wall detected: area={area}, height={h_rect}")
                    break

        # === REGION 4: CENTER FORWARD WALL DETECTION (focused area only) ===
        center_width = int(w * 0.4)         # Check center 40% of frame width (reduced)
        center_start_x = int(w * 0.3)       # Start at 30% from left

        # Check center region in the focused area only
        center_region = red_mask[danger_start_y:danger_end_y, center_start_x:center_start_x + center_width]
        contours, _ = cv2.findContours(center_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                # Wall is dangerous if it has significant size
                if w_rect > 40 or h_rect > 30:   # Reasonable size thresholds
                    danger_detected = True
                    wall_info = {
                        'zone': 'center_forward',
                        'contour': contour,
                        'area': area,
                        'bbox': (center_start_x + x, danger_start_y + y, w_rect, h_rect),
                        'length': max(w_rect, h_rect),
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.info(f"CENTER FORWARD wall detected: area={area}, size=({w_rect}x{h_rect})")
                    break

        if config.DEBUG_VISION and danger_detected:
            triggered_zones = [wall['zone'] for wall in self.detected_walls if wall.get('triggered', False)]
            self.logger.info(f"FOCUSED wall avoidance triggered - zones: {triggered_zones}")

        return danger_detected

    def get_avoidance_command(self, frame) -> Optional[str]:
        """Get avoidance command based on wall detection"""
        danger_detected = self.detect_boundaries(frame)

        if not danger_detected:
            return None

        # Analyze which walls are triggered to determine best avoidance
        triggered_zones = [wall['zone'] for wall in self.detected_walls if wall.get('triggered', False)]

        # Simple priority system
        if 'center_forward' in triggered_zones:
            return 'backup_and_turn'      # Backup and turn for center walls
        elif 'left' in triggered_zones and 'right' in triggered_zones:
            return 'backup_and_turn'      # Both sides blocked - backup
        elif 'left' in triggered_zones:
            return 'turn_right'           # Turn away from left wall
        elif 'right' in triggered_zones:
            return 'turn_left'            # Turn away from right wall
        elif 'bottom' in triggered_zones:
            return 'turn_right'           # Simple turn for bottom wall
        else:
            return 'turn_right'           # Default safe action

    def draw_boundary_visualization(self, frame) -> np.ndarray:
        """Draw boundary detection overlays on frame - FOCUSED VERSION"""
        if frame is None:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        # === WALL VISUALIZATION ===
        if self.red_mask is not None:
            # Create red overlay
            wall_overlay = np.zeros_like(result)
            wall_overlay[:, :, 2] = self.red_mask  # Red channel
            cv2.addWeighted(result, 0.75, wall_overlay, 0.25, 0, result)

            # Add outlines around red walls
            contours, _ = cv2.findContours(self.red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)

        # === FOCUSED DETECTION ZONE ===
        target_zone_center_y = int(h * config.TARGET_ZONE_VERTICAL_POSITION)
        danger_start_y = target_zone_center_y
        danger_end_y = int(h * 0.95)
        danger_distance_horizontal = int(w * 0.15)

        # Draw the focused detection zone
        cv2.rectangle(result, (0, danger_start_y), (w, danger_end_y), (0, 255, 255), 2)
        cv2.putText(result, "FOCUSED DETECTION ZONE", (10, danger_start_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw target zone center line
        cv2.line(result, (0, target_zone_center_y), (w, target_zone_center_y), (255, 255, 0), 2)
        cv2.putText(result, "TARGET ZONE CENTER", (10, target_zone_center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw side detection zones (only in focused area)
        cv2.rectangle(result, (0, danger_start_y), (danger_distance_horizontal, danger_end_y), (0, 100, 255), 2)
        cv2.rectangle(result, (w - danger_distance_horizontal, danger_start_y), (w, danger_end_y), (0, 100, 255), 2)

        # Draw center forward detection zone (only in focused area)
        center_width = int(w * 0.4)
        center_start_x = int(w * 0.3)
        cv2.rectangle(result, (center_start_x, danger_start_y), 
                     (center_start_x + center_width, danger_end_y), (255, 150, 0), 2)

        # === TRIGGERED WALLS ===
        for wall in self.detected_walls:
            if wall['triggered']:
                x, y, w_rect, h_rect = wall['bbox']
                zone_color = {
                    'left': (0, 0, 255),          # Red
                    'right': (0, 0, 255),         # Red  
                    'bottom': (255, 0, 0),        # Blue
                    'center_forward': (0, 255, 255),     # Yellow
                }.get(wall['zone'], (255, 255, 255))
                
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), zone_color, 4)
                cv2.putText(result, wall['zone'].upper(), (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)

        # === ARENA BOUNDARY ===
        if self.arena_detected and self.arena_contour is not None:
            cv2.drawContours(result, [self.arena_contour], -1, (0, 255, 255), 1)

        return result

    def get_status(self) -> Dict:
        """Get boundary avoidance system status"""
        triggered_walls = [wall for wall in self.detected_walls if wall.get('triggered', False)]

        return {
            'arena_detected': self.arena_detected,
            'walls_detected': len(self.detected_walls),
            'walls_triggered': len(triggered_walls),
            'danger_zones': [wall['zone'] for wall in triggered_walls],
            'safe': len(triggered_walls) == 0,
        }

    def reset(self):
        """Reset detection state"""
        self.detected_walls = []
        self.red_mask = None