import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import config

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

        # Range 1: Lower red hues (0-15)
        lower_red1 = np.array([0, 50, 50])      
        upper_red1 = np.array([15, 255, 255])   
        
        # Range 2: Upper red hues (160-180)
        lower_red2 = np.array([160, 50, 50])    
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_boundary_mask = mask1 + mask2

        self.red_mask = red_boundary_mask.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_boundary_mask = cv2.morphologyEx(red_boundary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        red_boundary_mask = cv2.morphologyEx(red_boundary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(red_boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Check if this could be a valid arena boundary
            min_arena_area = (w * h) * 0.10

            if area > min_arena_area:
                self.arena_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(self.arena_mask, [largest_contour], 255)

                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                self.arena_mask = cv2.erode(self.arena_mask, erosion_kernel, iterations=1)

                self.arena_contour = largest_contour
                self.arena_detected = True

                if config.DEBUG_VISION:
                    self.logger.info(f"✅ Arena boundary detected: area={area:.0f}")
                    self.logger.info(f"   Red range 1: H[0-15], S[50-255], V[50-255]")
                    self.logger.info(f"   Red range 2: H[160-180], S[50-255], V[50-255]")

                return True
            else:
                if config.DEBUG_VISION:
                    self.logger.warning(f"⚠️ Red area too small: {area:.0f} < {min_arena_area:.0f}")

        # Fallback: create a conservative arena mask
        if not self.arena_detected:
            self.arena_mask = np.ones((h, w), dtype=np.uint8) * 255
            margin_h = int(h * 0.12)
            margin_w = int(w * 0.08)

            self.arena_mask[:margin_h, :] = 0
            self.arena_mask[-margin_h:, :] = 0
            self.arena_mask[:, :margin_w] = 0
            self.arena_mask[:, -margin_w:] = 0

            if config.DEBUG_VISION:
                self.logger.warning("⚠️ Using fallback arena mask (no red walls detected)")

        return False

    def detect_boundaries(self, frame) -> bool:
        """FOCUSED: Only detect walls below the collection zone middle - EXCLUDE EDGE AREAS"""
        if frame is None:
            return False

        h, w = frame.shape[:2]
        self.detected_walls = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        self.red_mask = red_mask.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # FOCUSED DETECTION ZONE: From top of collection zone downward
        target_zone_center_y = int(h * config.TARGET_ZONE_VERTICAL_POSITION)
        target_zone_height = getattr(config, 'TARGET_ZONE_HEIGHT', 45)
        target_zone_top_y = target_zone_center_y - (target_zone_height // 2)
        
        danger_start_y = target_zone_top_y 
        danger_end_y = int(h * 0.95)

        if config.DEBUG_VISION:
            # Count total red pixels for debugging
            total_red_pixels = np.sum(red_mask > 0)
            self.logger.info(f"🔍 Red pixels detected: {total_red_pixels}")
            self.logger.info(f"🔍 FOCUSED wall detection zone: Y={danger_start_y} to {danger_end_y}")

        danger_detected = False
        min_wall_area = 200

        # Calculate detection zone height for percentage calculations
        detection_zone_height = danger_end_y - danger_start_y

        # === REGION 1: BOTTOM WALL DETECTION ===
        bottom_detection_height = int(detection_zone_height * 0.20) 
        bottom_region_start = danger_end_y - bottom_detection_height 
        
        # Calculate X boundaries for bottom detection - exclude purple edge areas
        bottom_left_margin = int(w * 0.1)  
        bottom_right_margin = int(w * 0.1) 
        bottom_detection_left = bottom_left_margin
        bottom_detection_right = w - bottom_right_margin
        
        if bottom_region_start < danger_end_y and bottom_detection_right > bottom_detection_left:
            bottom_region = red_mask[bottom_region_start:danger_end_y, bottom_detection_left:bottom_detection_right]
            contours, _ = cv2.findContours(bottom_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_wall_area:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    if w_rect > 60:
                        danger_detected = True
                        wall_info = {
                            'zone': 'bottom',
                            'contour': contour,
                            'area': area,
                            'bbox': (bottom_detection_left + x, bottom_region_start + y, w_rect, h_rect),
                            'length': w_rect,
                            'triggered': True
                        }
                        self.detected_walls.append(wall_info)
                        if config.DEBUG_VISION:
                            self.logger.info(f"🚨 Bottom wall detected: area={area}, width={w_rect}, height={bottom_detection_height}")
                        break

        # === REGION 2: CENTER FORWARD WALL DETECTION ===
        center_width = int(w * 0.4)
        center_start_x = int(w * 0.3)

        center_region = red_mask[danger_start_y:danger_end_y, center_start_x:center_start_x + center_width]
        contours, _ = cv2.findContours(center_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_wall_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                if w_rect > 40 or h_rect > 30:
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
                        self.logger.info(f"🚨 CENTER FORWARD wall detected: area={area}, size=({w_rect}x{h_rect})")
                    break

        if config.DEBUG_VISION and danger_detected:
            triggered_zones = [wall['zone'] for wall in self.detected_walls if wall.get('triggered', False)]
            self.logger.info(f"🚨 WALL DETECTION SUCCESS - zones: {triggered_zones}")
            self.logger.info(f"🚨 Bottom detection Y-range: [{bottom_region_start}:{danger_end_y}] (height: {bottom_detection_height})")
            self.logger.info(f"🚨 Bottom detection X-range: [{bottom_detection_left}:{bottom_detection_right}] (80% width)")
            self.logger.info(f"🚨 Center forward detection: [{center_start_x}:{center_start_x + center_width}] x [{danger_start_y}:{danger_end_y}]")

        return danger_detected

    def get_avoidance_command(self, frame) -> Optional[str]:
        """Get avoidance command based on wall detection"""
        danger_detected = self.detect_boundaries(frame)

        if not danger_detected:
            return None

        # Analyze which walls are triggered to determine best avoidance
        triggered_zones = [wall['zone'] for wall in self.detected_walls if wall.get('triggered', False)]

        if config.DEBUG_VISION:
            self.logger.info(f"Wall avoidance: detected zones {triggered_zones} -> backup_and_turn")

        return 'backup_and_turn'

    def draw_boundary_visualization(self, frame) -> np.ndarray:
        """Draw boundary detection overlays on frame - ENHANCED DEBUG VERSION"""
        if frame is None:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        # === RED MASK VISUALIZATION (ENHANCED) ===
        if self.red_mask is not None:
            # Show red detection in bright overlay
            red_overlay = np.zeros_like(result)
            red_overlay[:, :, 2] = self.red_mask
            cv2.addWeighted(result, 0.7, red_overlay, 0.3, 0, result)

            # Count and show red pixels
            total_red_pixels = np.sum(self.red_mask > 0)
            cv2.putText(result, f"Red pixels: {total_red_pixels}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add outlines around all red areas (not just walls)
            contours, _ = cv2.findContours(self.red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 50:
                    cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)  # Cyan outline
                    # Label each red area
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(result, f"R{i}:{area:.0f}", (cx-20, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # === HSV RANGE INFO ===
        cv2.putText(result, "HSV Range 1: H[0-15], S[50-255], V[50-255]", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(result, "HSV Range 2: H[160-180], S[50-255], V[50-255]", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # === DETECTION ZONE ===
        target_zone_center_y = int(h * config.TARGET_ZONE_VERTICAL_POSITION)
        target_zone_height = getattr(config, 'TARGET_ZONE_HEIGHT', 45)
        target_zone_top_y = target_zone_center_y - (target_zone_height // 2)
        
        danger_start_y = target_zone_top_y
        danger_end_y = int(h * 0.95)

        # Draw the focused detection zone
        cv2.rectangle(result, (0, danger_start_y), (w, danger_end_y), (0, 255, 255), 2)
        cv2.putText(result, "WALL DETECTION ZONE", (10, danger_start_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # === TRIGGERED WALLS ===
        for wall in self.detected_walls:
            if wall['triggered']:
                x, y, w_rect, h_rect = wall['bbox']
                zone_color = {
                    'bottom': (255, 0, 0),        # Blue
                    'center_forward': (0, 255, 255),     # Yellow
                }.get(wall['zone'], (255, 255, 255))
                
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), zone_color, 4)
                cv2.putText(result, f"{wall['zone'].upper()}: {wall['area']:.0f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)

        # === ARENA BOUNDARY STATUS ===
        arena_status = "DETECTED" if self.arena_detected else "FALLBACK"
        arena_color = (0, 255, 0) if self.arena_detected else (0, 165, 255)
        cv2.putText(result, f"Arena: {arena_status}", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, arena_color, 2)

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
            'red_pixels_detected': np.sum(self.red_mask > 0) if self.red_mask is not None else 0
        }

    def reset(self):
        """Reset detection state"""
        self.detected_walls = []
        self.red_mask = None