import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

# --- config.py (simulated for demonstration) ---
class Config:
    DEBUG_VISION = True

config = Config()
# -------------------------------------------------

class BoundaryAvoidanceSystem:
    """Dedicated system for detecting and avoiding arena boundaries/walls"""

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
        # UPDATED PARAMETERS for strong red (stop sign like)
        lower_red1 = np.array([0, 150, 100])  # Increased S and V minimums
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 150, 100]) # Increased S and V minimums
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
        """Detect if robot is too close to red walls (danger zones)
        UPDATED: Only trigger avoidance for walls in BOTTOM 30% of image"""
        if frame is None:
            return False

        h, w = frame.shape[:2]
        self.detected_walls = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wall detection - UPDATED PARAMETERS for strong red (stop sign like)
        lower_red1 = np.array([0, 150, 100])  # Increased S and V minimums
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 150, 100]) # Increased S and V minimums
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        self.red_mask = red_mask.copy()

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # CRITICAL CHANGE: Only check bottom 30% of image for all wall detection
        bottom_30_percent_y = int(h * 0.7)  # Start checking at 70% down from top
        collection_zone_y = int(h * 0.75)   # Collection zone starts at 75%

        danger_detected = False
        min_wall_area = 150   # Minimum area to consider a wall

        # Define danger zones ONLY in bottom 30%
        danger_distance_vertical = int(h * 0.1)   # 10% of frame height (reduced)
        danger_distance_horizontal = int(w * 0.08) # 8% of frame width (reduced)

        # === REGION 1: BOTTOM WALL DETECTION (in bottom 30% but above collection zone) ===
        bottom_danger_start = max(bottom_30_percent_y, collection_zone_y - danger_distance_vertical)
        if bottom_danger_start < collection_zone_y:
            bottom_region = red_mask[bottom_danger_start:collection_zone_y, :]
            contours, _ = cv2.findContours(bottom_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_wall_area:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    if w_rect > 30:   # Horizontal wall
                        danger_detected = True
                        wall_info = {
                            'zone': 'bottom',
                            'contour': contour,
                            'area': area,
                            'bbox': (x, bottom_danger_start + y, w_rect, h_rect),
                            'length': w_rect,
                            'triggered': True
                        }
                        self.detected_walls.append(wall_info)
                        if config.DEBUG_VISION:
                            self.logger.info(f"Bottom wall detected in BOTTOM 30%: area={area}, width={w_rect}")
                        break

        # === REGION 2: LEFT WALL DETECTION (ONLY in bottom 30%) ===
        left_region = red_mask[bottom_30_percent_y:collection_zone_y, 0:danger_distance_horizontal]
        contours, _ = cv2.findContours(left_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if h_rect > 30:   # Vertical wall
                    danger_detected = True
                    wall_info = {
                        'zone': 'left',
                        'contour': contour,
                        'area': area,
                        'bbox': (x, bottom_30_percent_y + y, w_rect, h_rect),
                        'length': h_rect,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.info(f"Left wall detected in BOTTOM 30%: area={area}, height={h_rect}")
                    break

        # === REGION 3: RIGHT WALL DETECTION (ONLY in bottom 30%) ===
        right_region = red_mask[bottom_30_percent_y:collection_zone_y, w-danger_distance_horizontal:w]
        contours, _ = cv2.findContours(right_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if h_rect > 30:   # Vertical wall
                    danger_detected = True
                    wall_info = {
                        'zone': 'right',
                        'contour': contour,
                        'area': area,
                        'bbox': (w - danger_distance_horizontal + x, bottom_30_percent_y + y, w_rect, h_rect),
                        'length': h_rect,
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.info(f"Right wall detected in BOTTOM 30%: area={area}, height={h_rect}")
                    break

        # === REGION 4: CENTER FORWARD WALL DETECTION (ONLY in bottom 30%) ===
        # Check center area for walls directly in front of robot
        center_width = int(w * 0.5)   # Check center 50% of frame width
        center_start_x = int(w * 0.25)   # Start at 25% from left

        # Only check bottom 30% area where walls are actually close to robot
        center_region = red_mask[bottom_30_percent_y:collection_zone_y, center_start_x:center_start_x + center_width]
        contours, _ = cv2.findContours(center_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                # Wall is dangerous if it has significant size
                if w_rect > 40 or h_rect > 20:   # Must be substantial wall segment
                    danger_detected = True
                    wall_info = {
                        'zone': 'center_forward',
                        'contour': contour,
                        'area': area,
                        'bbox': (center_start_x + x, bottom_30_percent_y + y, w_rect, h_rect),
                        'length': max(w_rect, h_rect),
                        'triggered': True
                    }
                    self.detected_walls.append(wall_info)
                    if config.DEBUG_VISION:
                        self.logger.info(f"CENTER wall detected in BOTTOM 30%: area={area}, pos=({center_start_x + x}, {bottom_30_percent_y + y})")
                    break

        if config.DEBUG_VISION and danger_detected:
            triggered_zones = [wall['zone'] for wall in self.detected_walls if wall.get('triggered', False)]
            self.logger.info(f"Wall avoidance triggered in BOTTOM 30% - zones: {triggered_zones}")

        return danger_detected

    def get_avoidance_command(self, frame) -> Optional[str]:
        """Get avoidance command based on wall detection - SIMPLIFIED VERSION"""
        danger_detected = self.detect_boundaries(frame)

        # REMOVED: All state machine handling
        # No more self.avoidance_state checks

        if not danger_detected:
            return None

        # Analyze which walls are triggered to determine best avoidance
        triggered_zones = [wall['zone'] for wall in self.detected_walls if wall.get('triggered', False)]

        # SIMPLIFIED: Direct commands without state tracking
        if 'left' in triggered_zones:
            return 'turn_right'       # Turn away from left wall
        elif 'right' in triggered_zones:
            return 'turn_left'        # Turn away from right wall
        elif 'center_forward' in triggered_zones:
            return 'backup_and_turn'  # NEW: Compound command for backing up + turning
        elif 'bottom' in triggered_zones:
            return 'turn_right'       # Simple turn for bottom wall
        else:
            return 'turn_right'       # Default safe action

    def draw_boundary_visualization(self, frame) -> np.ndarray:
        """Draw boundary detection overlays on frame"""
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

        # === DANGER ZONES ===
        h, w = frame.shape[:2]
        bottom_30_percent_y = int(h * 0.7)
        collection_zone_y = int(h * 0.75)
        danger_distance_horizontal = int(w * 0.08)

        # Draw danger zone borders
        cv2.rectangle(result, (0, bottom_30_percent_y), (w, collection_zone_y), (0, 100, 255), 2)
        cv2.rectangle(result, (0, bottom_30_percent_y), (danger_distance_horizontal, collection_zone_y), (0, 100, 255), 2)
        cv2.rectangle(result, (w - danger_distance_horizontal, bottom_30_percent_y), (w, collection_zone_y), (0, 100, 255), 2)

        # === TRIGGERED WALLS ===
        for wall in self.detected_walls:
            if wall['triggered']:
                x, y, w_rect, h_rect = wall['bbox']
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 4)

        # === ARENA BOUNDARY ===
        if self.arena_detected and self.arena_contour is not None:
            cv2.drawContours(result, [self.arena_contour], -1, (0, 255, 255), 1)

        # REMOVED: Display current avoidance state
        # No more state to display

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
            # REMOVED: 'current_avoidance_state': self.avoidance_state
        }

    def reset(self):
        """Reset detection state"""
        self.detected_walls = []
        self.red_mask = None

# --- Example of how to use the class in a main script ---
if __name__ == "__main__":
    # Initialize the camera
    cap = cv2.VideoCapture(0) # Use 0 for default webcam, or path to video file

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    # Initialize the boundary avoidance system
    boundary_system = BoundaryAvoidanceSystem()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect arena boundaries (often done once at startup or periodically)
        arena_found = boundary_system.detect_arena_boundaries(frame)

        # Get avoidance command based on current frame and internal state
        avoidance_command = boundary_system.get_avoidance_command(frame)

        # Get status for logging/debugging
        status = boundary_system.get_status()

        if config.DEBUG_VISION:
            # Draw visualizations
            visualized_frame = boundary_system.draw_boundary_visualization(frame)
            cv2.putText(visualized_frame, f"Arena Detected: {status['arena_detected']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(visualized_frame, f"Command: {avoidance_command if avoidance_command else 'None'}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(visualized_frame, f"Danger Zones: {', '.join(status['danger_zones']) if status['danger_zones'] else 'None'}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow('Boundary Avoidance System', visualized_frame)

        # Here you would integrate the 'avoidance_command' with your robot's movement logic
        if avoidance_command:
            boundary_system.logger.info(f"Robot should: {avoidance_command}")
            # Example:
            # if avoidance_command == 'turn_right':
            #     robot.turn_right()
            # elif avoidance_command == 'move_backward':
            #     robot.move_backward()
            # # Add other movement commands as needed
            # elif avoidance_command == 'turn_left':
            #     robot.turn_left()
        # else:
            # If no avoidance command, the robot can proceed with its main task
            # robot.move_forward() # or whatever its goal is

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()