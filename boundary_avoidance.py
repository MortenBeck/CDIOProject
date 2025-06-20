import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import config

class BoundaryAvoidanceSystem:
    """Dedicated system for detecting and avoiding arena boundaries/walls"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
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
    
    def detect_boundaries(self, frame) -> bool:
        """Detect if robot is too close to red walls (danger zones)
        Now ignores walls in the collection zone (bottom 25% of frame)"""
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
        
        # Define collection zone boundary (bottom 25% of image)
        collection_zone_y = int(h * 0.75)  # Top of collection zone (75% down from top)
        
        # Smaller danger zones - only trigger when very close
        danger_distance = min(50, int(h * 0.1))
        bottom_danger_y = h - danger_distance
        
        # Only check bottom wall if it's ABOVE the collection zone
        if bottom_danger_y < collection_zone_y:
            bottom_mask = red_mask[bottom_danger_y:collection_zone_y, :]  # Stop at collection zone
        else:
            # If danger zone extends into collection area, skip bottom wall detection entirely
            bottom_mask = np.zeros((1, w), dtype=np.uint8)  # Empty mask
        
        edge_width = min(30, int(w * 0.06))
        
        # For side walls, only check the area ABOVE the collection zone
        left_mask = red_mask[0:collection_zone_y, 0:edge_width]  # Only upper 75%
        right_mask = red_mask[0:collection_zone_y, w-edge_width:w]  # Only upper 75%
        
        danger_detected = False
        min_wall_area = 120
        
        # Check bottom (only if not in collection zone)
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
                    if config.DEBUG_VISION:
                        self.logger.info(f"Bottom wall detected above collection zone: area={area}")
                    break
        
        # Check left (only upper portion)
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
                    if config.DEBUG_VISION:
                        self.logger.info(f"Left wall detected above collection zone: area={area}")
                    break
        
        # Check right (only upper portion)
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
                    if config.DEBUG_VISION:
                        self.logger.info(f"Right wall detected above collection zone: area={area}")
                    break
        
        return danger_detected
    
    def get_avoidance_command(self, frame) -> Optional[str]:
        """Get avoidance command based on wall detection"""
        danger_detected = self.detect_boundaries(frame)
        
        if not danger_detected:
            return None
        
        # Analyze which walls are triggered to determine best avoidance
        triggered_zones = [wall['zone'] for wall in self.detected_walls if wall.get('triggered', False)]
        
        if 'bottom' in triggered_zones:
            return 'move_backward'
        elif 'left' in triggered_zones:
            return 'turn_right'
        elif 'right' in triggered_zones:
            return 'turn_left'
        else:
            return 'move_backward'  # Default safe action
    
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
        danger_distance = min(50, int(h * 0.1))
        danger_y = h - danger_distance
        edge_width = min(30, int(w * 0.06))
        
        # Draw danger zone borders
        cv2.rectangle(result, (0, danger_y), (w, h), (0, 100, 255), 2)  # Bottom
        cv2.rectangle(result, (0, 0), (edge_width, h), (0, 100, 255), 2)  # Left
        cv2.rectangle(result, (w - edge_width, 0), (w, h), (0, 100, 255), 2)  # Right
        
        # === TRIGGERED WALLS ===
        for wall in self.detected_walls:
            if wall['triggered']:
                x, y, w_rect, h_rect = wall['bbox']
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 4)
        
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
            'safe': len(triggered_walls) == 0
        }
    
    def reset(self):
        """Reset detection state"""
        self.detected_walls = []
        self.red_mask = None