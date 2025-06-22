#!/usr/bin/env python3
"""
Triangular Green Target Delivery System
Designed for green triangles with their points facing toward the robot
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import config

@dataclass
class GreenTriangle:
    """Green triangle target data"""
    center: Tuple[int, int]
    area: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    triangle_tip: Optional[Tuple[int, int]] = None  # Point closest to robot
    triangle_base: Optional[Tuple[int, int]] = None  # Base center (farthest from robot)
    approach_direction: Optional[np.ndarray] = None  # Vector from base to tip
    distance_to_tip: Optional[float] = None  # Distance from robot to triangle tip
    contour: Optional[np.ndarray] = None

class TriangularDeliveryVisionSystem:
    """Vision system specialized for triangular green targets"""
    
    def __init__(self, vision_system):
        self.logger = logging.getLogger(__name__)
        self.vision = vision_system
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        
        # IMPORTANT: Robot position should be at bottom center of frame, not center
        # This is where the robot actually is relative to the camera view
        self.robot_position_x = config.CAMERA_WIDTH // 2
        self.robot_position_y = int(config.CAMERA_HEIGHT * 0.9)  # 90% down from top
        
        # Green detection parameters for triangles
        self.green_lower = np.array([35, 40, 40])  # Slightly broader green range
        self.green_upper = np.array([85, 255, 255])
        self.min_triangle_area = 300    # Smaller minimum for triangle tips
        self.max_triangle_area = 8000   # Reasonable maximum
        
        # Triangle approach parameters
        self.optimal_distance_from_tip = 80    # How close to get to triangle tip
        self.alignment_tolerance = 15          # Degrees for approach alignment
        self.position_tolerance = 20           # Pixels for position alignment
        self.final_approach_distance = 50      # Final distance before delivery
        
    def detect_green_triangles(self, frame) -> List[GreenTriangle]:
        """Detect triangular green targets and identify tip/base"""
        triangles = []
        
        if frame is None:
            return triangles
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green colors
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Use actual robot position (bottom center of frame)
        robot_pos = np.array([self.robot_position_x, self.robot_position_y])
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_triangle_area < area < self.max_triangle_area:
                # Check if this could be a triangle
                triangle_info = self._analyze_triangle_shape(contour, robot_pos)
                
                if triangle_info:
                    # Calculate center and bounding box
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        center = (center_x, center_y)
                    else:
                        center = triangle_info['tip']
                    
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    bbox = (x, y, w_rect, h_rect)
                    
                    # Calculate confidence based on triangle characteristics
                    confidence = self._calculate_triangle_confidence(contour, triangle_info, area)
                    
                    if confidence > 0.3:
                        triangle = GreenTriangle(
                            center=center,
                            area=area,
                            confidence=confidence,
                            bbox=bbox,
                            triangle_tip=triangle_info['tip'],
                            triangle_base=triangle_info['base'],
                            approach_direction=triangle_info['approach_vector'],
                            distance_to_tip=triangle_info['distance_to_tip'],
                            contour=contour
                        )
                        triangles.append(triangle)
        
        # Sort by confidence and proximity to robot
        triangles.sort(key=lambda t: (-t.confidence, t.distance_to_tip or float('inf')))
        return triangles[:3]  # Limit to top 3 candidates
    
    def _analyze_triangle_shape(self, contour, robot_pos):
        """Analyze contour to determine if it's a triangle and find tip/base"""
        # Approximate contour to reduce noise
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # For triangles, we expect 3-4 vertices after approximation
        if len(approx) < 3 or len(approx) > 6:
            return None
        
        # Get all points in the contour
        points = contour.reshape(-1, 2)
        
        # Find the convex hull to get the main triangle points
        hull = cv2.convexHull(contour)
        hull_points = hull.reshape(-1, 2)
        
        if len(hull_points) < 3:
            return None
        
        # IMPROVED TIP DETECTION: Find point closest to robot AND most "downward pointing"
        # The tip should be both close to robot and pointing toward the bottom of the image
        
        # First, find points that are in the bottom half of the triangle
        triangle_center_y = np.mean(hull_points[:, 1])
        bottom_candidates = []
        
        for i, point in enumerate(hull_points):
            distance_to_robot = np.linalg.norm(point - robot_pos)
            # Prefer points that are lower (higher Y value) and closer to robot
            y_score = point[1] - triangle_center_y  # Positive for points below center
            robot_distance_score = 1.0 / (distance_to_robot + 1)  # Higher for closer points
            
            # Combined score: prefer points that are both low and close to robot
            combined_score = y_score * 0.7 + robot_distance_score * 100  # Weight robot distance more
            
            bottom_candidates.append({
                'index': i,
                'point': point,
                'distance': distance_to_robot,
                'y_position': point[1],
                'score': combined_score
            })
        
        # Sort by combined score (highest = best tip candidate)
        bottom_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Take the best candidate as the tip
        tip_candidate = bottom_candidates[0]
        tip_point = tuple(tip_candidate['point'].astype(int))
        distance_to_tip = tip_candidate['distance']
        tip_idx = tip_candidate['index']
        
        # Find the base (points farthest from the tip)
        other_indices = [i for i in range(len(hull_points)) if i != tip_idx]
        
        if len(other_indices) < 2:
            # Fallback: use two points farthest from robot
            distances_to_robot = [np.linalg.norm(point - robot_pos) for point in hull_points]
            far_distances = [(i, dist) for i, dist in enumerate(distances_to_robot) if i != tip_idx]
            far_distances.sort(key=lambda x: x[1], reverse=True)
            
            if len(far_distances) >= 2:
                base_point1 = hull_points[far_distances[0][0]].astype(int)
                base_point2 = hull_points[far_distances[1][0]].astype(int)
            else:
                return None
        else:
            # Use the remaining points to find the base edge
            other_points = [hull_points[i] for i in other_indices]
            
            if len(other_points) == 2:
                base_point1, base_point2 = other_points[0].astype(int), other_points[1].astype(int)
            else:
                # Find the two points that are farthest from the tip (should be base points)
                tip_distances = []
                for i, point in enumerate(other_points):
                    dist_to_tip = np.linalg.norm(point - tip_candidate['point'])
                    tip_distances.append((i, dist_to_tip, point))
                
                tip_distances.sort(key=lambda x: x[1], reverse=True)
                
                if len(tip_distances) >= 2:
                    base_point1 = tip_distances[0][2].astype(int)
                    base_point2 = tip_distances[1][2].astype(int)
                else:
                    return None
        
        # Calculate base center
        base_center = tuple(((base_point1 + base_point2) // 2).astype(int))
        
        # Calculate approach vector (from base to tip - this is the direction to approach)
        approach_vector = np.array(tip_point) - np.array(base_center)
        approach_length = np.linalg.norm(approach_vector)
        if approach_length > 0:
            approach_vector_norm = approach_vector / approach_length
        else:
            approach_vector_norm = np.array([0, 1])  # Default downward
        
        return {
            'tip': tip_point,
            'base': base_center,
            'base_points': (tuple(base_point1), tuple(base_point2)),
            'approach_vector': approach_vector_norm,
            'distance_to_tip': distance_to_tip,
            'hull_points': hull_points
        }
    
    def _calculate_triangle_confidence(self, contour, triangle_info, area):
        """Calculate confidence that this is a valid triangle target"""
        # Base confidence from area
        size_confidence = min(1.0, area / 2000)
        
        # Shape confidence - check if it's reasonably triangular
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # Triangles should have lower circularity than circles
            shape_confidence = 1.0 - min(1.0, abs(circularity - 0.4) / 0.4)
        else:
            shape_confidence = 0.0
        
        # Aspect ratio confidence (triangles shouldn't be too elongated)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        fill_ratio = area / max(bbox_area, 1)
        fill_confidence = min(1.0, fill_ratio * 2)  # Triangles fill ~50% of bbox
        
        # Distance confidence (prefer closer triangles)
        distance_to_tip = triangle_info.get('distance_to_tip', float('inf'))
        max_reasonable_distance = 300
        distance_confidence = max(0.1, 1.0 - (distance_to_tip / max_reasonable_distance))
        
        # Combine confidences
        total_confidence = (size_confidence + shape_confidence + fill_confidence + distance_confidence) / 4
        
        return min(1.0, total_confidence)
    
    def get_triangle_positioning_command(self, triangle: GreenTriangle) -> Optional[str]:
        """Get positioning command to approach triangle tip properly"""
        if not triangle.triangle_tip or triangle.approach_direction is None:
            return self._fallback_positioning(triangle)
        
        try:
            robot_pos = np.array([self.robot_position_x, self.robot_position_y])  # Use actual robot position
            tip_pos = np.array(triangle.triangle_tip)
            approach_vector = triangle.approach_direction
            
            # Calculate ideal approach position (back from tip along approach vector)
            ideal_approach_pos = tip_pos - (approach_vector * self.optimal_distance_from_tip)
            
            # Current position error
            position_error = ideal_approach_pos - robot_pos
            position_distance = np.linalg.norm(position_error)
            
            # Current direction to triangle tip
            direction_to_tip = tip_pos - robot_pos
            tip_distance = np.linalg.norm(direction_to_tip)
            if tip_distance > 0:
                direction_to_tip_norm = direction_to_tip / tip_distance
            else:
                direction_to_tip_norm = np.array([0, 1])
            
            # Check alignment with approach vector
            alignment_dot = np.clip(np.dot(direction_to_tip_norm, approach_vector), -1.0, 1.0)
            alignment_angle = np.degrees(np.arccos(abs(alignment_dot)))
            
            self.logger.info(f"🎯 Triangle approach: pos_dist={position_distance:.1f}px, angle_error={alignment_angle:.1f}°")
            self.logger.info(f"🎯 Tip at {triangle.triangle_tip}, Robot at ({robot_pos[0]:.0f}, {robot_pos[1]:.0f})")
            
            # PHASE 1: Rough positioning - get into general area
            if position_distance > 60:
                # Move toward ideal approach position
                if abs(position_error[0]) > self.position_tolerance:
                    if position_error[0] > 0:
                        return 'move_right'
                    else:
                        return 'move_left'
                
                if abs(position_error[1]) > self.position_tolerance:
                    if position_error[1] > 0:
                        return 'move_backward'
                    else:
                        return 'move_forward'
            
            # PHASE 2: Alignment - ensure we're pointing toward triangle tip along approach vector
            elif alignment_angle > self.alignment_tolerance:
                # Determine turn direction using cross product
                cross_product = np.cross(direction_to_tip_norm, approach_vector)
                
                if cross_product > 0:
                    return 'align_turn_left'
                else:
                    return 'align_turn_right'
            
            # PHASE 3: Final distance check
            else:
                current_distance_to_tip = np.linalg.norm(robot_pos - tip_pos)
                
                if current_distance_to_tip > self.final_approach_distance:
                    return 'final_approach'
                else:
                    return 'approach_complete'
                    
        except Exception as e:
            self.logger.warning(f"Triangle positioning calculation error: {e}")
            return self._fallback_positioning(triangle)
    
    def _fallback_positioning(self, triangle: GreenTriangle) -> Optional[str]:
        """Fallback positioning if triangle analysis fails"""
        robot_x = self.robot_position_x  # Use actual robot position
        robot_y = self.robot_position_y
        target_x, target_y = triangle.center
        
        x_error = target_x - robot_x
        y_error = target_y - robot_y
        
        if abs(x_error) > 25:
            return 'move_right' if x_error > 0 else 'move_left'
        
        if abs(y_error) > 25:
            return 'move_backward' if y_error > 0 else 'move_forward'
        
        return 'approach_complete'
    
    def draw_triangle_visualization(self, frame, triangles: List[GreenTriangle], current_phase: str) -> np.ndarray:
        """Draw visualization showing triangle detection and approach vectors"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw triangles with approach information
        for i, triangle in enumerate(triangles):
            color = (0, 255, 0) if i == 0 else (0, 150, 0)
            thickness = 3 if i == 0 else 2
            
            # Draw triangle contour
            if triangle.contour is not None:
                cv2.drawContours(result, [triangle.contour], 0, color, thickness)
            
            # Draw center
            cv2.circle(result, triangle.center, 5 if i == 0 else 3, color, -1)
            cv2.putText(result, f"T{i+1}", (triangle.center[0] - 10, triangle.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if i == 0 and triangle.triangle_tip and triangle.triangle_base:  # Primary target
                # Highlight triangle tip (point facing robot)
                cv2.circle(result, triangle.triangle_tip, 8, (0, 255, 255), 3)
                cv2.putText(result, "TIP", (triangle.triangle_tip[0] - 15, triangle.triangle_tip[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Highlight triangle base
                cv2.circle(result, triangle.triangle_base, 6, (255, 0, 255), 2)
                cv2.putText(result, "BASE", (triangle.triangle_base[0] - 20, triangle.triangle_base[1] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Draw approach vector (from base to tip)
                if triangle.approach_direction is not None:
                    tip_pos = np.array(triangle.triangle_tip)
                    approach_end = tip_pos + (triangle.approach_direction * 60)
                    cv2.arrowedLine(result, triangle.triangle_tip, tuple(approach_end.astype(int)), 
                                   (255, 255, 0), 3)
                    cv2.putText(result, "APPROACH", tuple((approach_end + 10).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw ideal approach position
                ideal_pos = tip_pos - (triangle.approach_direction * self.optimal_distance_from_tip)
                cv2.circle(result, tuple(ideal_pos.astype(int)), 10, (255, 0, 255), 2)
                cv2.putText(result, "IDEAL", tuple((ideal_pos - [25, 0]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Robot position
                robot_pos = (self.frame_center_x, self.frame_center_y)
                cv2.circle(result, robot_pos, 8, (255, 255, 255), 2)
                cv2.putText(result, "ROBOT", (robot_pos[0] - 25, robot_pos[1] + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw line from robot to ideal position
                cv2.line(result, robot_pos, tuple(ideal_pos.astype(int)), (255, 0, 255), 2)
                
                # Show distance to tip
                if triangle.distance_to_tip:
                    cv2.putText(result, f"Dist: {triangle.distance_to_tip:.0f}px", 
                               (robot_pos[0] + 15, robot_pos[1] - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Status overlay
        overlay_height = 120
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.7, overlay, 0.3, 0)
        
        # Status text
        cv2.putText(result, "TRIANGULAR DELIVERY - Green Triangle Detection", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        triangle_count = len(triangles)
        primary_triangle = triangles[0] if triangles else None
        
        if primary_triangle and primary_triangle.triangle_tip:
            distance = primary_triangle.distance_to_tip or 0
            status = f"Triangle tip: {distance:.0f}px away"
            phase_status = f"Phase: {current_phase.upper()}"
        else:
            status = "Scanning for green triangles..."
            phase_status = "Phase: SEARCHING"
        
        cv2.putText(result, f"Triangles: {triangle_count} | {status}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(result, phase_status, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(result, "Yellow: Triangle tip (closest point) | Magenta: Base center", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return result

class TriangularDeliverySystem:
    """Delivery system specialized for triangular green targets"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = TriangularDeliveryVisionSystem(vision_system)
        
        self.current_target = None
        self.delivery_active = False
        self.start_time = None
        self.current_phase = "search"
        self.phase_start_time = None
        
    def start_triangular_delivery_mode(self):
        """Start triangular delivery mode"""
        self.logger.info("🔺 STARTING TRIANGULAR DELIVERY MODE")
        self.logger.info("Target: Green triangles with tips pointing toward robot")
        
        self.delivery_active = True
        self.start_time = time.time()
        self.current_phase = "search"
        
        try:
            self.triangular_delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Triangular delivery mode interrupted")
        except Exception as e:
            self.logger.error(f"Triangular delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def triangular_delivery_main_loop(self):
        """Main delivery loop for triangular targets"""
        search_direction = 1
        frames_without_target = 0
        max_frames_without_target = 30
        
        while self.delivery_active:
            try:
                # Get frame
                ret, frame = self.vision_system.get_frame()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Detect triangular targets
                triangles = self.delivery_vision.detect_green_triangles(frame)
                
                # Create visualization
                debug_frame = self.delivery_vision.draw_triangle_visualization(
                    frame, triangles, self.current_phase)
                
                # Show frame
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('Triangular Delivery - Green Triangle Detection', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Process triangles
                if triangles:
                    frames_without_target = 0
                    primary_triangle = triangles[0]
                    
                    # Check for new target - fix numpy array comparison
                    target_changed = False
                    if self.current_target is None:
                        target_changed = True
                    else:
                        try:
                            # Safe comparison of centers
                            curr_x, curr_y = self.current_target.center
                            new_x, new_y = primary_triangle.center
                            if abs(new_x - curr_x) > 80 or abs(new_y - curr_y) > 80:
                                target_changed = True
                        except (TypeError, AttributeError):
                            target_changed = True
                    
                    if target_changed:
                        self.logger.info("🔺 New triangle detected - starting positioning")
                        self.current_target = primary_triangle
                        self.current_phase = "triangle_position"
                        self.phase_start_time = time.time()
                    
                    self.current_target = primary_triangle
                    
                    # Execute phase
                    if self.current_phase == "search":
                        self.current_phase = "triangle_position"
                        self.phase_start_time = time.time()
                    elif self.current_phase == "triangle_position":
                        self.handle_triangle_positioning_phase()
                    elif self.current_phase == "approach":
                        self.handle_triangle_approach_phase()
                    elif self.current_phase == "deliver":
                        self.handle_triangle_deliver_phase()
                
                else:
                    # No triangles - search
                    frames_without_target += 1
                    self.current_target = None
                    self.current_phase = "search"
                    
                    if frames_without_target >= max_frames_without_target:
                        search_direction *= -1
                        frames_without_target = 0
                        self.logger.info(f"🔍 Changing triangle search direction")
                    
                    self.search_for_triangles(search_direction)
                
                time.sleep(0.12)
                
            except Exception as e:
                self.logger.error(f"Triangular delivery loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
    
    def handle_triangle_positioning_phase(self):
        """Handle positioning relative to triangle tip"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        positioning_command = self.delivery_vision.get_triangle_positioning_command(self.current_target)
        
        if positioning_command == 'approach_complete':
            self.logger.info("✅ Triangle positioning complete - aligned with tip")
            self.current_phase = "approach"
            self.phase_start_time = time.time()
            return
        
        # Execute movement commands
        move_duration = 0.4
        move_speed = 0.45
        
        if positioning_command in ['align_turn_left', 'align_turn_right']:
            # Fine alignment
            move_duration = 0.3
            move_speed = 0.45
        
        # Execute the movement
        if positioning_command == 'move_right':
            self.logger.info("🔺 Triangle positioning: Turn right")
            self.hardware.turn_right(duration=move_duration, speed=move_speed)
        elif positioning_command == 'move_left':
            self.logger.info("🔺 Triangle positioning: Turn left")
            self.hardware.turn_left(duration=move_duration, speed=move_speed)
        elif positioning_command == 'move_forward':
            self.logger.info("🔺 Triangle positioning: Move forward")
            self.hardware.move_forward(duration=move_duration, speed=move_speed)
        elif positioning_command == 'move_backward':
            self.logger.info("🔺 Triangle positioning: Move backward")
            self.hardware.move_backward(duration=move_duration, speed=move_speed)
        elif positioning_command == 'align_turn_right':
            self.logger.info("🎯 Triangle alignment: Turn right")
            self.hardware.turn_right(duration=move_duration, speed=move_speed)
        elif positioning_command == 'align_turn_left':
            self.logger.info("🎯 Triangle alignment: Turn left")
            self.hardware.turn_left(duration=move_duration, speed=move_speed)
        elif positioning_command == 'final_approach':
            self.logger.info("🚀 Moving to final approach distance")
            self.hardware.move_forward(duration=0.6, speed=0.4)
        
        # Timeout check
        if time.time() - self.phase_start_time > 30.0:
            self.logger.warning("⏰ Triangle positioning timeout - proceeding anyway")
            self.current_phase = "approach"
    
    def handle_triangle_approach_phase(self):
        """Handle final approach to triangle tip"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        self.logger.info("🚀 Final approach to triangle tip")
        
        # Approach the triangle tip
        approach_duration = 1.2
        approach_speed = 0.35
        
        self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
        
        self.logger.info("📦 Reached triangle - proceeding to delivery")
        self.current_phase = "deliver"
        self.phase_start_time = time.time()
    
    def handle_triangle_deliver_phase(self):
        """Handle delivery at triangle"""
        self.logger.info("📦 Executing delivery at triangle")
        
        if self.hardware.has_balls():
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls at triangle")
        else:
            self.logger.info("ℹ️  No balls to deliver")
        
        # Back away and turn
        self.hardware.move_backward(duration=1.5, speed=0.4)
        self.hardware.turn_left(duration=0.8, speed=0.4)
        
        # Reset for next target
        self.current_phase = "search"
        self.current_target = None
        self.logger.info("🔄 Triangle delivery complete - searching for next target")
    
    def search_for_triangles(self, direction: int):
        """Search for triangular targets"""
        if direction > 0:
            self.hardware.turn_right(duration=0.7, speed=0.45)
        else:
            self.hardware.turn_left(duration=0.7, speed=0.45)
        time.sleep(0.3)
    
    def stop_delivery(self):
        """Stop triangular delivery"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 TRIANGULAR DELIVERY COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Ball count: {self.hardware.get_ball_count()}")
        
        cv2.destroyAllWindows()

# Updated test function
def run_triangular_delivery_test():
    """Main entry point for triangular delivery system"""
    print("\n🔺 TRIANGULAR GOLFBOT DELIVERY SYSTEM")
    print("="*60)
    print("SPECIALIZED FOR:")
    print("• Green triangular targets")
    print("• Triangle tips pointing toward robot")
    print("• Automatic tip/base detection")
    print("• Approach vector calculation")
    print("• Precise alignment with triangle orientation")
    print("="*60)
    
    input("Press Enter to start triangular delivery test...")
    
    try:
        from hardware import GolfBotHardware
        from vision import VisionSystem
        
        print("Initializing triangular delivery systems...")
        hardware = GolfBotHardware()
        vision = VisionSystem()
        
        if not vision.start():
            print("❌ Camera initialization failed")
            return False
        
        print("✅ Triangular delivery systems ready!")
        
        delivery_system = TriangularDeliverySystem(hardware, vision)
        delivery_system.start_triangular_delivery_mode()
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        try:
            if 'hardware' in locals():
                hardware.emergency_stop()
            if 'vision' in locals():
                vision.cleanup()
        except:
            pass

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_triangular_delivery_test()