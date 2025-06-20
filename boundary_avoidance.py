import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import config

class BoundaryAvoidanceSystem:
    """Enhanced system for detecting and avoiding arena boundaries/walls"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Wall detection results for visualization
        self.detected_walls = []
        self.red_mask = None
        
        # Arena boundaries
        self.arena_mask = None
        self.arena_detected = False
        self.arena_contour = None
        
        # Arena boundary detection parameters
        self.arena_boundary_segments = []
        
    def detect_arena_boundaries(self, frame) -> bool:
        """Enhanced arena boundary detection that works with disconnected red segments"""
        if frame is None:
            return False
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced red boundary detection with multiple HSV ranges
        # Range 1: Pure reds (0-10)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([10, 255, 255])
        
        # Range 2: Deep reds (170-180)
        lower_red2 = np.array([170, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        # Range 3: Orange-reds (10-25) - sometimes walls appear orange-ish
        lower_red3 = np.array([10, 100, 80])
        upper_red3 = np.array([25, 255, 255])
        
        # Combine all red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
        red_boundary_mask = mask1 + mask2 + mask3
        
        # More aggressive morphological operations to connect segments
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        
        # First, close gaps aggressively
        red_boundary_mask = cv2.morphologyEx(red_boundary_mask, cv2.MORPH_CLOSE, kernel_large, iterations=5)
        red_boundary_mask = cv2.morphologyEx(red_boundary_mask, cv2.MORPH_DILATE, kernel_medium, iterations=3)
        red_boundary_mask = cv2.morphologyEx(red_boundary_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        
        # Find all boundary segments
        contours, _ = cv2.findContours(red_boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Strategy 1: Look for large contours that could be arena boundaries
        large_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (w * h) * 0.02:  # At least 2% of frame
                large_contours.append(contour)
        
        if large_contours:
            # Find the largest contour
            largest_contour = max(large_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if this could be a valid arena boundary
            min_arena_area = (w * h) * 0.08  # Reduced threshold to 8%
            
            if area > min_arena_area:
                self.arena_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(self.arena_mask, [largest_contour], 255)
                
                # Moderate erosion to ensure we're inside the boundary
                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                self.arena_mask = cv2.erode(self.arena_mask, erosion_kernel, iterations=1)
                
                self.arena_contour = largest_contour
                self.arena_detected = True
                
                if config.DEBUG_VISION:
                    self.logger.info(f"Arena boundary detected from large contour: area={area:.0f}")
                
                return True
        
        # Strategy 2: Try to construct arena from edge segments
        arena_constructed = self._construct_arena_from_segments(red_boundary_mask, w, h)
        if arena_constructed:
            return True
        
        # Strategy 3: Detect individual wall segments and create conservative boundary
        wall_segments = self._detect_wall_segments(red_boundary_mask, w, h)
        if len(wall_segments) >= 2:  # Need at least 2 wall segments
            self._create_arena_from_walls(wall_segments, w, h)
            if config.DEBUG_VISION:
                self.logger.info(f"Arena boundary created from {len(wall_segments)} wall segments")
            return True
        
        # Fallback: create a very conservative arena mask
        self._create_fallback_arena(w, h)
        if config.DEBUG_VISION:
            self.logger.info("Using fallback arena mask (no red walls detected)")
        
        return False
    
    def _construct_arena_from_segments(self, red_mask, w, h) -> bool:
        """Try to construct arena boundary by connecting edge segments"""
        
        # Detect segments along edges
        edge_segments = []
        
        # Top edge
        top_strip = red_mask[0:h//8, :]
        if np.sum(top_strip) > w * 10:  # Significant red on top
            edge_segments.append('top')
        
        # Bottom edge  
        bottom_strip = red_mask[7*h//8:h, :]
        if np.sum(bottom_strip) > w * 10:  # Significant red on bottom
            edge_segments.append('bottom')
        
        # Left edge
        left_strip = red_mask[:, 0:w//8]
        if np.sum(left_strip) > h * 10:  # Significant red on left
            edge_segments.append('left')
        
        # Right edge
        right_strip = red_mask[:, 7*w//8:w]
        if np.sum(right_strip) > h * 10:  # Significant red on right
            edge_segments.append('right')
        
        if len(edge_segments) >= 3:  # Need at least 3 edges for arena
            # Create arena mask based on detected edges
            margin = 30  # Pixels to stay away from detected walls
            
            self.arena_mask = np.ones((h, w), dtype=np.uint8) * 255
            
            if 'top' in edge_segments:
                self.arena_mask[0:margin, :] = 0
            if 'bottom' in edge_segments:
                self.arena_mask[h-margin:h, :] = 0
            if 'left' in edge_segments:
                self.arena_mask[:, 0:margin] = 0
            if 'right' in edge_segments:
                self.arena_mask[:, w-margin:w] = 0
            
            self.arena_detected = True
            
            if config.DEBUG_VISION:
                self.logger.info(f"Arena constructed from edge segments: {edge_segments}")
            
            return True
        
        return False
    
    def _detect_wall_segments(self, red_mask, w, h) -> List[Dict]:
        """Detect individual wall segments"""
        wall_segments = []
        
        # Find contours in the red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum wall segment area
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Classify segment type based on position and shape
                segment_type = "unknown"
                
                # Check if it's along an edge
                edge_threshold = 50
                
                if y < edge_threshold:  # Top edge
                    segment_type = "top_wall"
                elif y + h_rect > h - edge_threshold:  # Bottom edge
                    segment_type = "bottom_wall"
                elif x < edge_threshold:  # Left edge
                    segment_type = "left_wall"
                elif x + w_rect > w - edge_threshold:  # Right edge
                    segment_type = "right_wall"
                
                wall_segments.append({
                    'type': segment_type,
                    'contour': contour,
                    'bbox': (x, y, w_rect, h_rect),
                    'area': area
                })
        
        return wall_segments
    
    def _create_arena_from_walls(self, wall_segments, w, h):
        """Create arena mask based on detected wall segments"""
        # Start with full frame
        self.arena_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Exclude areas near detected walls
        margin = 40  # Stay away from walls
        
        for segment in wall_segments:
            x, y, w_rect, h_rect = segment['bbox']
            wall_type = segment['type']
            
            if wall_type == "top_wall":
                # Exclude top area
                exclude_height = min(y + h_rect + margin, h//3)
                self.arena_mask[0:exclude_height, :] = 0
            elif wall_type == "bottom_wall":
                # Exclude bottom area  
                exclude_start = max(y - margin, 2*h//3)
                self.arena_mask[exclude_start:h, :] = 0
            elif wall_type == "left_wall":
                # Exclude left area
                exclude_width = min(x + w_rect + margin, w//3)
                self.arena_mask[:, 0:exclude_width] = 0
            elif wall_type == "right_wall":
                # Exclude right area
                exclude_start = max(x - margin, 2*w//3)
                self.arena_mask[:, exclude_start:w] = 0
        
        self.arena_detected = True
    
    def _create_fallback_arena(self, w, h):
        """Create very conservative fallback arena"""
        self.arena_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Exclude outer edges conservatively
        margin_h = int(h * 0.15)  # 15% margin top/bottom
        margin_w = int(w * 0.12)  # 12% margin left/right
        
        self.arena_mask[:margin_h, :] = 0      # Top
        self.arena_mask[-margin_h:, :] = 0     # Bottom  
        self.arena_mask[:, :margin_w] = 0      # Left
        self.arena_mask[:, -margin_w:] = 0     # Right
        
        self.arena_detected = False  # Mark as fallback
    
    def detect_boundaries(self, frame) -> bool:
        """Detect if robot is too close to red walls (danger zones)"""
        if frame is None:
            return False
        
        h, w = frame.shape[:2]
        self.detected_walls = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced red detection with multiple ranges
        lower_red1 = np.array([0, 30, 30])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 30, 30])
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
        collection_zone_y = int(h * 0.75)
        
        # Danger zones - trigger when very close to walls
        danger_distance = min(60, int(h * 0.12))  # Slightly larger danger zone
        bottom_danger_y = h - danger_distance
        
        # Only check bottom wall if it's ABOVE the collection zone
        if bottom_danger_y < collection_zone_y:
            bottom_mask = red_mask[bottom_danger_y:collection_zone_y, :]
        else:
            bottom_mask = np.zeros((1, w), dtype=np.uint8)
        
        edge_width = min(40, int(w * 0.08))  # Slightly wider edge detection
        
        # For side walls, only check the area ABOVE the collection zone
        left_mask = red_mask[0:collection_zone_y, 0:edge_width]
        right_mask = red_mask[0:collection_zone_y, w-edge_width:w]
        
        danger_detected = False
        min_wall_area = 100  # Slightly lower threshold
        
        # Check each danger zone
        zones = [
            ('bottom', bottom_mask, (0, bottom_danger_y)),
            ('left', left_mask, (0, 0)),
            ('right', right_mask, (w-edge_width, 0))
        ]
        
        for zone_name, mask, offset in zones:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_wall_area:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    
                    # Adjust coordinates based on zone offset
                    actual_x = x + offset[0]
                    actual_y = y + offset[1]
                    
                    # Validate wall characteristics
                    is_valid_wall = False
                    if zone_name == 'bottom' and w_rect > 30 and h_rect > 8:
                        is_valid_wall = True
                    elif zone_name in ['left', 'right'] and h_rect > 40 and w_rect > 8:
                        is_valid_wall = True
                    
                    if is_valid_wall:
                        danger_detected = True
                        wall_info = {
                            'zone': zone_name,
                            'contour': contour,
                            'area': area,
                            'bbox': (actual_x, actual_y, w_rect, h_rect),
                            'length': max(w_rect, h_rect),
                            'triggered': True
                        }
                        self.detected_walls.append(wall_info)
                        
                        if config.DEBUG_VISION:
                            self.logger.info(f"{zone_name.capitalize()} wall detected: area={area:.0f}")
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
        """Enhanced boundary detection visualization"""
        if frame is None:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # === ARENA MASK VISUALIZATION ===
        if self.arena_mask is not None:
            # Show arena boundary as green tint
            arena_overlay = np.zeros_like(result)
            arena_overlay[:, :, 1] = self.arena_mask  # Green channel
            cv2.addWeighted(result, 0.85, arena_overlay, 0.15, 0, result)
            
            # Draw arena boundary contour if available
            if self.arena_contour is not None:
                cv2.drawContours(result, [self.arena_contour], -1, (0, 255, 255), 2)
        
        # === WALL VISUALIZATION ===
        if self.red_mask is not None:
            # Create red overlay for detected walls
            wall_overlay = np.zeros_like(result)
            wall_overlay[:, :, 2] = self.red_mask  # Red channel
            cv2.addWeighted(result, 0.75, wall_overlay, 0.25, 0, result)
            
            # Add outlines around red walls
            contours, _ = cv2.findContours(self.red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    cv2.drawContours(result, [contour], -1, (0, 0, 255), 1)
        
        # === DANGER ZONES ===
        danger_distance = min(60, int(h * 0.12))
        danger_y = h - danger_distance
        edge_width = min(40, int(w * 0.08))
        collection_zone_y = int(h * 0.75)
        
        # Draw danger zone borders
        cv2.rectangle(result, (0, danger_y), (w, collection_zone_y), (0, 100, 255), 1)  # Bottom
        cv2.rectangle(result, (0, 0), (edge_width, collection_zone_y), (0, 100, 255), 1)  # Left
        cv2.rectangle(result, (w - edge_width, 0), (w, collection_zone_y), (0, 100, 255), 1)  # Right
        
        # === TRIGGERED WALLS ===
        for wall in self.detected_walls:
            if wall['triggered']:
                x, y, w_rect, h_rect = wall['bbox']
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 3)
                
                # Add zone label
                label = f"{wall['zone'].upper()}"
                cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # === COLLECTION ZONE INDICATOR ===
        cv2.line(result, (0, collection_zone_y), (w, collection_zone_y), (255, 255, 0), 2)
        cv2.putText(result, "COLLECTION ZONE", (10, collection_zone_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return result
    
    def get_status(self) -> Dict:
        """Get enhanced boundary avoidance system status"""
        triggered_walls = [wall for wall in self.detected_walls if wall.get('triggered', False)]
        
        return {
            'arena_detected': self.arena_detected,
            'arena_method': 'constructed' if self.arena_detected else 'fallback',
            'walls_detected': len(self.detected_walls),
            'walls_triggered': len(triggered_walls),
            'danger_zones': [wall['zone'] for wall in triggered_walls],
            'safe': len(triggered_walls) == 0,
            'boundary_segments': len(getattr(self, 'arena_boundary_segments', []))
        }
    
    def reset(self):
        """Reset detection state"""
        self.detected_walls = []
        self.red_mask = None
        self.arena_boundary_segments = []