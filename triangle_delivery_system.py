#!/usr/bin/env python3
"""
PRECISION TRIANGLE TIP ALIGNMENT SYSTEM
Designed for DEAD-STRAIGHT delivery through precise holes in walls
The robot centers on triangle tips and approaches in perfectly straight lines
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import config

@dataclass
class PrecisionTriangleTarget:
    """Triangle target optimized for precision tip alignment"""
    center: Tuple[int, int]
    area: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    tip_point: Tuple[int, int]  # THE MOST IMPORTANT POINT - where we need to aim
    tip_confidence: float       # How confident we are this is the real tip
    distance_to_tip: float      # Distance from robot to tip
    contour: Optional[np.ndarray] = None
    
    # Precision alignment data
    x_alignment_error: float = 0.0      # Horizontal offset from center
    y_distance_to_optimal: float = 0.0  # Distance to optimal approach position
    is_perfectly_centered: bool = False # Ready for straight-line approach
    approach_angle_error: float = 0.0   # How far off straight we are

class PrecisionTriangleVision:
    """Vision system specialized for PRECISION triangle tip alignment"""
    
    def __init__(self, vision_system):
        self.logger = logging.getLogger(__name__)
        self.vision = vision_system
        
        # ROBOT POSITION: Bottom center of frame (where robot actually is)
        self.robot_x = config.CAMERA_WIDTH // 2
        self.robot_y = int(config.CAMERA_HEIGHT * 0.85)  # 85% down from top
        
        # PRECISION ALIGNMENT PARAMETERS
        self.precision_x_tolerance = 8      # Pixels - VERY tight X centering
        self.precision_y_tolerance = 12     # Pixels - Tight Y positioning  
        self.optimal_approach_distance = 120 # Pixels from tip for final approach
        self.final_approach_distance = 60   # Minimum distance before delivery
        
        # MOVEMENT PARAMETERS - Tuned for precision
        self.coarse_turn_duration = 0.4     # Large corrections
        self.coarse_turn_speed = 0.45
        self.fine_turn_duration = 0.25      # Small corrections  
        self.fine_turn_speed = 0.35
        self.micro_turn_duration = 0.15     # Micro corrections
        self.micro_turn_speed = 0.25
        self.positioning_move_duration = 0.3
        self.positioning_move_speed = 0.4
        
        # GREEN DETECTION - Optimized for triangle detection
        self.green_lower = np.array([35, 50, 50])
        self.green_upper = np.array([80, 255, 255])
        self.min_triangle_area = 400
        self.max_triangle_area = 15000
        
        # PRECISION STATES
        self.alignment_state = "searching"  # searching, coarse_align, fine_align, ready, approaching
        
    def detect_precision_triangles(self, frame) -> List[PrecisionTriangleTarget]:
        """Detect triangles with EXTREME focus on tip precision"""
        targets = []
        
        if frame is None:
            return targets
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced green detection
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Aggressive cleanup for better triangle detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        green_mask = cv2.medianBlur(green_mask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        robot_pos = np.array([self.robot_x, self.robot_y])
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_triangle_area < area < self.max_triangle_area:
                # PRECISION TIP DETECTION
                tip_analysis = self._precision_tip_detection(contour, robot_pos)
                
                if tip_analysis and tip_analysis['tip_confidence'] > 0.6:
                    # Calculate center and bbox
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        center = (center_x, center_y)
                    else:
                        center = tip_analysis['tip_point']
                    
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    bbox = (x, y, w_rect, h_rect)
                    
                    # Calculate overall confidence
                    confidence = self._calculate_precision_confidence(contour, tip_analysis, area)
                    
                    if confidence > 0.4:
                        # PRECISION ALIGNMENT CALCULATIONS
                        tip_point = tip_analysis['tip_point']
                        x_error = tip_point[0] - self.robot_x
                        y_distance = np.linalg.norm(np.array(tip_point) - robot_pos)
                        y_optimal_error = y_distance - self.optimal_approach_distance
                        
                        # Check if perfectly centered
                        is_centered = (abs(x_error) <= self.precision_x_tolerance and 
                                     abs(y_optimal_error) <= self.precision_y_tolerance)
                        
                        # Calculate approach angle error
                        tip_vector = np.array(tip_point) - robot_pos
                        if np.linalg.norm(tip_vector) > 0:
                            tip_vector_norm = tip_vector / np.linalg.norm(tip_vector)
                            straight_vector = np.array([0, -1])  # Straight up
                            angle_error = np.degrees(np.arccos(np.clip(np.dot(tip_vector_norm, straight_vector), -1, 1)))
                        else:
                            angle_error = 0
                        
                        target = PrecisionTriangleTarget(
                            center=center,
                            area=area,
                            confidence=confidence,
                            bbox=bbox,
                            tip_point=tip_point,
                            tip_confidence=tip_analysis['tip_confidence'],
                            distance_to_tip=tip_analysis['distance_to_tip'],
                            contour=contour,
                            x_alignment_error=x_error,
                            y_distance_to_optimal=y_optimal_error,
                            is_perfectly_centered=is_centered,
                            approach_angle_error=angle_error
                        )
                        targets.append(target)
        
        # Sort by tip confidence and alignment quality
        targets.sort(key=lambda t: (-t.tip_confidence, abs(t.x_alignment_error), -t.confidence))
        return targets[:2]  # Limit to top 2 candidates
    
    def _precision_tip_detection(self, contour, robot_pos):
        """ENHANCED tip detection using multiple methods for maximum precision"""
        
        # METHOD 1: Convex hull + closest point (primary method)
        hull = cv2.convexHull(contour)
        hull_points = hull.reshape(-1, 2)
        
        if len(hull_points) < 3:
            return None
        
        # Find closest point to robot (primary tip candidate)
        distances_to_robot = [np.linalg.norm(point - robot_pos) for point in hull_points]
        closest_idx = np.argmin(distances_to_robot)
        primary_tip = tuple(hull_points[closest_idx].astype(int))
        primary_distance = distances_to_robot[closest_idx]
        
        # METHOD 2: Extreme point detection for validation
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        
        extreme_points = [leftmost, rightmost, topmost, bottommost]
        
        # METHOD 3: Angle-based tip detection
        angle_tips = []
        for i, point in enumerate(hull_points):
            if i == 0 or i == len(hull_points) - 1:
                continue
            
            prev_point = hull_points[i-1]
            next_point = hull_points[i+1]
            
            # Calculate angle at this point
            v1 = prev_point - point
            v2 = next_point - point
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)
                angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))
                angle_degrees = np.degrees(angle)
                
                # Sharp angles (< 90 degrees) are likely tips
                if angle_degrees < 90:
                    distance_to_robot = np.linalg.norm(point - robot_pos)
                    angle_tips.append({
                        'point': tuple(point.astype(int)),
                        'angle': angle_degrees,
                        'distance': distance_to_robot
                    })
        
        # VALIDATION: Combine all methods to find the BEST tip
        tip_candidates = []
        
        # Add primary tip (closest point)
        tip_candidates.append({
            'point': primary_tip,
            'confidence': 0.8,  # High confidence for closest point
            'distance': primary_distance,
            'method': 'closest'
        })
        
        # Add angle-based tips
        for angle_tip in angle_tips:
            confidence = 0.6 + (90 - angle_tip['angle']) / 90 * 0.3  # Sharper = higher confidence
            tip_candidates.append({
                'point': angle_tip['point'],
                'confidence': confidence,
                'distance': angle_tip['distance'],
                'method': 'angle'
            })
        
        # Find the BEST tip candidate
        best_tip = None
        best_score = 0
        
        for candidate in tip_candidates:
            # Score = confidence * distance_factor
            distance_factor = max(0.1, 1.0 - (candidate['distance'] / 400))  # Closer is better
            score = candidate['confidence'] * distance_factor
            
            if score > best_score:
                best_score = score
                best_tip = candidate
        
        if best_tip:
            self.logger.info(f"🎯 Tip detected: {best_tip['point']} (method: {best_tip['method']}, conf: {best_tip['confidence']:.2f})")
            
            return {
                'tip_point': best_tip['point'],
                'tip_confidence': best_tip['confidence'],
                'distance_to_tip': best_tip['distance'],
                'detection_method': best_tip['method']
            }
        
        return None
    
    def _calculate_precision_confidence(self, contour, tip_analysis, area):
        """Calculate confidence with emphasis on tip quality"""
        
        # Base confidence from area
        size_confidence = min(1.0, area / 3000)
        
        # Shape confidence (triangular-ness)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        shape_confidence = min(1.0, 1.0 / max(1, abs(len(approx) - 3) + 1))
        
        # Tip confidence (from tip analysis)
        tip_confidence = tip_analysis['tip_confidence']
        
        # Distance confidence (prefer closer targets)
        distance = tip_analysis['distance_to_tip']
        distance_confidence = max(0.2, 1.0 - (distance / 300))
        
        # Combined confidence with tip emphasis
        combined = (size_confidence * 0.2 + 
                   shape_confidence * 0.2 + 
                   tip_confidence * 0.4 +        # TIP IS MOST IMPORTANT
                   distance_confidence * 0.2)
        
        return min(1.0, combined)
    
    def get_precision_alignment_command(self, target: PrecisionTriangleTarget) -> Optional[str]:
        """Get PRECISION alignment command for dead-straight approach"""
        
        tip_x, tip_y = target.tip_point
        x_error = target.x_alignment_error
        y_error = target.y_distance_to_optimal
        distance_to_tip = target.distance_to_tip
        
        self.logger.info(f"🎯 PRECISION: tip=({tip_x},{tip_y}), x_err={x_error:.1f}, y_err={y_error:.1f}, dist={distance_to_tip:.1f}")
        
        # PHASE 1: COARSE ALIGNMENT (big corrections)
        if abs(x_error) > 25 or abs(y_error) > 40:
            self.alignment_state = "coarse_align"
            
            # Prioritize X alignment first (most critical for straight approach)
            if abs(x_error) > 25:
                if x_error > 0:
                    return 'coarse_turn_right'
                else:
                    return 'coarse_turn_left'
            
            # Then handle Y positioning
            if y_error > 40:  # Too far, move closer
                return 'move_forward'
            elif y_error < -40:  # Too close, back up
                return 'move_backward'
        
        # PHASE 2: FINE ALIGNMENT (medium corrections)
        elif abs(x_error) > 12 or abs(y_error) > 20:
            self.alignment_state = "fine_align"
            
            if abs(x_error) > 12:
                if x_error > 0:
                    return 'fine_turn_right'
                else:
                    return 'fine_turn_left'
            
            if y_error > 20:
                return 'fine_move_forward'
            elif y_error < -20:
                return 'fine_move_backward'
        
        # PHASE 3: MICRO ALIGNMENT (tiny corrections)
        elif abs(x_error) > self.precision_x_tolerance or abs(y_error) > self.precision_y_tolerance:
            self.alignment_state = "micro_align"
            
            if abs(x_error) > self.precision_x_tolerance:
                if x_error > 0:
                    return 'micro_turn_right'
                else:
                    return 'micro_turn_left'
            
            if y_error > self.precision_y_tolerance:
                return 'micro_move_forward'
            elif y_error < -self.precision_y_tolerance:
                return 'micro_move_backward'
        
        # PHASE 4: PERFECT ALIGNMENT - READY FOR STRAIGHT APPROACH
        else:
            if distance_to_tip > self.final_approach_distance:
                self.alignment_state = "ready"
                return 'straight_approach'
            else:
                self.alignment_state = "complete"
                return 'delivery_position_reached'
        
        return None
    
    def draw_precision_visualization(self, frame, targets: List[PrecisionTriangleTarget], current_phase: str) -> np.ndarray:
        """Draw PRECISION alignment visualization"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # === PRECISION ALIGNMENT GRID ===
        # Draw robot position
        robot_pos = (self.robot_x, self.robot_y)
        cv2.circle(result, robot_pos, 10, (255, 255, 255), 3)
        cv2.putText(result, "ROBOT", (robot_pos[0] - 25, robot_pos[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw precision alignment zone
        precision_zone_color = (100, 100, 255)
        left_bound = self.robot_x - self.precision_x_tolerance
        right_bound = self.robot_x + self.precision_x_tolerance
        cv2.line(result, (left_bound, 0), (left_bound, h), precision_zone_color, 2)
        cv2.line(result, (right_bound, 0), (right_bound, h), precision_zone_color, 2)
        cv2.putText(result, f"±{self.precision_x_tolerance}px", (left_bound + 5, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, precision_zone_color, 1)
        
        # === TRIANGLE TARGETS ===
        for i, target in enumerate(targets):
            color = (0, 255, 0) if i == 0 else (0, 150, 0)
            thickness = 3 if i == 0 else 2
            
            # Draw triangle contour
            if target.contour is not None:
                cv2.drawContours(result, [target.contour], 0, color, thickness)
            
            # === PRECISION TIP HIGHLIGHTING ===
            tip_color = (0, 255, 255)  # Bright cyan for tip
            cv2.circle(result, target.tip_point, 12, tip_color, 4)
            cv2.circle(result, target.tip_point, 3, tip_color, -1)
            cv2.putText(result, "TIP", (target.tip_point[0] - 15, target.tip_point[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, tip_color, 2)
            
            if i == 0:  # Primary target
                # === PRECISION ALIGNMENT INDICATORS ===
                tip_x, tip_y = target.tip_point
                
                # X-alignment line (horizontal from robot to tip X position)
                alignment_color = (0, 255, 0) if abs(target.x_alignment_error) <= self.precision_x_tolerance else (0, 0, 255)
                cv2.line(result, (self.robot_x, robot_pos[1]), (tip_x, robot_pos[1]), alignment_color, 3)
                
                # Vertical line from tip to robot level
                cv2.line(result, (tip_x, tip_y), (tip_x, robot_pos[1]), alignment_color, 2, cv2.LINE_DASHED)
                
                # Distance arc showing optimal approach distance
                optimal_y = robot_pos[1] - self.optimal_approach_distance
                if optimal_y > 0:
                    cv2.circle(result, robot_pos, self.optimal_approach_distance, (255, 0, 255), 2)
                    cv2.putText(result, f"OPTIMAL: {self.optimal_approach_distance}px", 
                               (robot_pos[0] + 30, optimal_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # === STRAIGHT APPROACH LINE ===
                if target.is_perfectly_centered:
                    # Green line showing straight approach path
                    approach_end = (tip_x, max(0, tip_y - 50))
                    cv2.arrowedLine(result, robot_pos, approach_end, (0, 255, 0), 4)
                    cv2.putText(result, "STRAIGHT APPROACH READY", (tip_x + 20, tip_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Red/yellow line showing current alignment
                    alignment_color = (0, 255, 255) if abs(target.x_alignment_error) <= 15 else (0, 0, 255)
                    cv2.arrowedLine(result, robot_pos, target.tip_point, alignment_color, 2)
                
                # === PRECISION ERROR DISPLAY ===
                error_x = w - 200
                error_y = 50
                error_bg = np.zeros((120, 190, 3), dtype=np.uint8)
                result[error_y-10:error_y+110, error_x:error_x+190] = cv2.addWeighted(
                    result[error_y-10:error_y+110, error_x:error_x+190], 0.3, error_bg, 0.7, 0)
                
                cv2.putText(result, "PRECISION ERRORS", (error_x + 5, error_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                x_color = (0, 255, 0) if abs(target.x_alignment_error) <= self.precision_x_tolerance else (0, 0, 255)
                cv2.putText(result, f"X: {target.x_alignment_error:+.1f}px", (error_x + 5, error_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, x_color, 1)
                
                y_color = (0, 255, 0) if abs(target.y_distance_to_optimal) <= self.precision_y_tolerance else (0, 0, 255)
                cv2.putText(result, f"Y: {target.y_distance_to_optimal:+.1f}px", (error_x + 5, error_y + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, y_color, 1)
                
                cv2.putText(result, f"Dist: {target.distance_to_tip:.1f}px", (error_x + 5, error_y + 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                status_color = (0, 255, 0) if target.is_perfectly_centered else (255, 255, 0)
                status_text = "READY" if target.is_perfectly_centered else "ALIGNING"
                cv2.putText(result, status_text, (error_x + 5, error_y + 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # === STATUS OVERLAY ===
        overlay_height = 100
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.7, overlay, 0.3, 0)
        
        cv2.putText(result, "PRECISION TRIANGLE TIP ALIGNMENT SYSTEM", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        triangle_count = len(targets)
        primary_target = targets[0] if targets else None
        
        if primary_target:
            status = f"Tip confidence: {primary_target.tip_confidence:.2f}"
            phase_status = f"Phase: {self.alignment_state.upper()}"
            
            centered_status = "PERFECTLY CENTERED - READY FOR DELIVERY" if primary_target.is_perfectly_centered else "ALIGNING FOR PRECISION"
            centered_color = (0, 255, 0) if primary_target.is_perfectly_centered else (255, 255, 0)
        else:
            status = "Scanning for green triangles..."
            phase_status = "Phase: SEARCHING"
            centered_status = "NO TARGET"
            centered_color = (128, 128, 128)
        
        cv2.putText(result, f"Triangles: {triangle_count} | {status}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(result, phase_status, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(result, centered_status, (400, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, centered_color, 2)
        
        return result

class PrecisionTriangleDeliverySystem:
    """PRECISION delivery system for dead-straight triangle tip alignment"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.precision_vision = PrecisionTriangleVision(vision_system)
        
        self.current_target = None
        self.delivery_active = False
        self.start_time = None
        self.current_phase = "search"
        self.phase_start_time = None
        
        # Precision tracking
        self.alignment_attempts = 0
        self.max_alignment_attempts = 100
        self.stable_alignment_count = 0
        self.required_stable_frames = 3
        
    def start_precision_delivery_mode(self):
        """Start PRECISION triangle delivery mode"""
        self.logger.info("🎯 STARTING PRECISION TRIANGLE TIP ALIGNMENT SYSTEM")
        self.logger.info("🎯 Target: DEAD-STRAIGHT delivery through precise holes")
        self.logger.info("🎯 Method: Extreme precision tip centering + straight-line approach")
        
        self.delivery_active = True
        self.start_time = time.time()
        self.current_phase = "search"
        
        try:
            self.precision_delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Precision delivery mode interrupted")
        except Exception as e:
            self.logger.error(f"Precision delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def precision_delivery_main_loop(self):
        """Main loop for PRECISION triangle delivery"""
        search_direction = 1
        frames_without_target = 0
        max_frames_without_target = 25
        
        while self.delivery_active:
            try:
                # Get frame
                ret, frame = self.vision_system.get_frame()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Detect precision triangle targets
                triangles = self.precision_vision.detect_precision_triangles(frame)
                
                # Create visualization
                debug_frame = self.precision_vision.draw_precision_visualization(
                    frame, triangles, self.current_phase)
                
                # Show frame
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('PRECISION Triangle Tip Alignment', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Process triangles
                if triangles:
                    frames_without_target = 0
                    primary_triangle = triangles[0]
                    
                    # Check for new target
                    if (self.current_target is None or 
                        np.linalg.norm(np.array(primary_triangle.tip_point) - np.array(self.current_target.tip_point)) > 50):
                        self.logger.info("🎯 New precision target detected - starting alignment")
                        self.current_target = primary_triangle
                        self.current_phase = "precision_align"
                        self.phase_start_time = time.time()
                        self.alignment_attempts = 0
                        self.stable_alignment_count = 0
                    
                    self.current_target = primary_triangle
                    
                    # Execute phase
                    if self.current_phase == "search":
                        self.current_phase = "precision_align"
                        self.phase_start_time = time.time()
                    elif self.current_phase == "precision_align":
                        self.handle_precision_alignment_phase()
                    elif self.current_phase == "straight_approach":
                        self.handle_straight_approach_phase()
                    elif self.current_phase == "deliver":
                        self.handle_precision_deliver_phase()
                
                else:
                    # No triangles - search
                    frames_without_target += 1
                    self.current_target = None
                    self.current_phase = "search"
                    
                    if frames_without_target >= max_frames_without_target:
                        search_direction *= -1
                        frames_without_target = 0
                        self.logger.info(f"🔍 Changing precision search direction")
                    
                    self.search_for_precision_targets(search_direction)
                
                time.sleep(0.08)  # Faster loop for precision
                
            except Exception as e:
                self.logger.error(f"Precision delivery loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
    
    def handle_precision_alignment_phase(self):
        """Handle PRECISION alignment with dead-straight accuracy"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        self.alignment_attempts += 1
        
        # Check for perfect alignment
        if self.current_target.is_perfectly_centered:
            self.stable_alignment_count += 1
            self.logger.info(f"🎯 PERFECT ALIGNMENT achieved! Stability: {self.stable_alignment_count}/{self.required_stable_frames}")
            
            if self.stable_alignment_count >= self.required_stable_frames:
                self.logger.info("✅ PRECISION ALIGNMENT STABLE - INITIATING STRAIGHT APPROACH")
                self.current_phase = "straight_approach"
                self.phase_start_time = time.time()
                return
        else:
            self.stable_alignment_count = 0  # Reset stability counter
        
        # Get precision alignment command
        alignment_command = self.precision_vision.get_precision_alignment_command(self.current_target)
        
        if alignment_command == 'delivery_position_reached':
            self.logger.info("🎯 DELIVERY POSITION REACHED - PERFECT ALIGNMENT ACHIEVED")
            self.current_phase = "deliver"
            self.phase_start_time = time.time()
            return
        
        # Execute precision movements with different parameters
        if alignment_command:
            self.execute_precision_movement(alignment_command)
        
        # Timeout protection
        if self.alignment_attempts > self.max_alignment_attempts:
            self.logger.warning("⏰ Maximum alignment attempts reached")
            if abs(self.current_target.x_alignment_error) <= 20:  # Close enough
                self.logger.info("🎯 Proceeding with close-enough alignment")
                self.current_phase = "straight_approach"
            else:
                self.logger.warning("🔄 Alignment failed - searching for new target")
                self.current_phase = "search"
    
    def execute_precision_movement(self, command):
        """Execute precision movement commands with appropriate parameters"""
        vision = self.precision_vision
        
        # COARSE MOVEMENTS (big corrections)
        if command == 'coarse_turn_right':
            self.logger.info("🔄 COARSE turn right")
            self.hardware.turn_right(duration=vision.coarse_turn_duration, speed=vision.coarse_turn_speed)
        elif command == 'coarse_turn_left':
            self.logger.info("🔄 COARSE turn left")
            self.hardware.turn_left(duration=vision.coarse_turn_duration, speed=vision.coarse_turn_speed)
        elif command == 'move_forward':
            self.logger.info("⬆️ COARSE move forward")
            self.hardware.move_forward(duration=vision.positioning_move_duration, speed=vision.positioning_move_speed)
        elif command == 'move_backward':
            self.logger.info("⬇️ COARSE move backward")
            self.hardware.move_backward(duration=vision.positioning_move_duration, speed=vision.positioning_move_speed)
        
        # FINE MOVEMENTS (medium corrections)
        elif command == 'fine_turn_right':
            self.logger.info("🔧 FINE turn right")
            self.hardware.turn_right(duration=vision.fine_turn_duration, speed=vision.fine_turn_speed)
        elif command == 'fine_turn_left':
            self.logger.info("🔧 FINE turn left")
            self.hardware.turn_left(duration=vision.fine_turn_duration, speed=vision.fine_turn_speed)
        elif command == 'fine_move_forward':
            self.logger.info("⬆️ FINE move forward")
            self.hardware.move_forward(duration=vision.fine_turn_duration, speed=vision.fine_turn_speed)
        elif command == 'fine_move_backward':
            self.logger.info("⬇️ FINE move backward")
            self.hardware.move_backward(duration=vision.fine_turn_duration, speed=vision.fine_turn_speed)
        
        # MICRO MOVEMENTS (tiny corrections)
        elif command == 'micro_turn_right':
            self.logger.info("🎯 MICRO turn right")
            self.hardware.turn_right(duration=vision.micro_turn_duration, speed=vision.micro_turn_speed)
        elif command == 'micro_turn_left':
            self.logger.info("🎯 MICRO turn left")
            self.hardware.turn_left(duration=vision.micro_turn_duration, speed=vision.micro_turn_speed)
        elif command == 'micro_move_forward':
            self.logger.info("⬆️ MICRO move forward")
            self.hardware.move_forward(duration=vision.micro_turn_duration, speed=vision.micro_turn_speed)
        elif command == 'micro_move_backward':
            self.logger.info("⬇️ MICRO move backward")
            self.hardware.move_backward(duration=vision.micro_turn_duration, speed=vision.micro_turn_speed)
        
        # STRAIGHT APPROACH (final approach)
        elif command == 'straight_approach':
            self.logger.info("🚀 STRAIGHT approach to tip")
            self.hardware.move_forward(duration=0.5, speed=0.4)
        
        # Small delay for stability
        time.sleep(0.05)
    
    def handle_straight_approach_phase(self):
        """Handle the final STRAIGHT approach to the triangle tip"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        self.logger.info("🚀 EXECUTING STRAIGHT-LINE APPROACH TO TRIANGLE TIP")
        self.logger.info("🎯 Robot is perfectly aligned - approaching hole in wall")
        
        # Calculate approach distance based on current distance to tip
        approach_distance = max(0.8, min(2.0, self.current_target.distance_to_tip / 100))
        approach_speed = 0.35  # Steady, controlled speed
        
        self.logger.info(f"🚀 Straight approach: {approach_distance:.1f}s at {approach_speed*100:.0f}% speed")
        
        # CRITICAL: Pure straight-line movement - NO TURNING
        self.hardware.move_forward(duration=approach_distance, speed=approach_speed)
        
        self.logger.info("📦 Reached triangle tip - proceeding to delivery")
        self.current_phase = "deliver"
        self.phase_start_time = time.time()
    
    def handle_precision_deliver_phase(self):
        """Handle precision delivery at triangle tip"""
        self.logger.info("📦 EXECUTING PRECISION DELIVERY AT TRIANGLE TIP")
        
        if self.hardware.has_balls():
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ SUCCESSFULLY DELIVERED {released_count} balls through hole!")
        else:
            self.logger.info("ℹ️  No balls to deliver")
        
        # Enhanced backing sequence for precision scenarios
        self.logger.info("🔄 Executing precision retreat sequence")
        self.hardware.move_backward(duration=1.8, speed=0.4)  # Longer retreat
        time.sleep(0.2)
        self.hardware.turn_left(duration=0.9, speed=0.4)      # More turn for clearance
        time.sleep(0.2)
        
        # Reset all tracking
        self.current_phase = "search"
        self.current_target = None
        self.alignment_attempts = 0
        self.stable_alignment_count = 0
        self.precision_vision.alignment_state = "searching"
        
        self.logger.info("🔄 PRECISION DELIVERY COMPLETE - Ready for next target")
    
    def search_for_precision_targets(self, direction: int):
        """Search for precision triangle targets"""
        vision = self.precision_vision
        
        if direction > 0:
            self.hardware.turn_right(duration=0.7, speed=0.45)
        else:
            self.hardware.turn_left(duration=0.7, speed=0.45)
        time.sleep(0.2)
    
    def stop_delivery(self):
        """Stop precision delivery system"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 PRECISION TRIANGLE DELIVERY COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Ball count: {self.hardware.get_ball_count()}")
        self.logger.info(f"   Alignment attempts: {self.alignment_attempts}")
        self.logger.info(f"   Final alignment state: {self.precision_vision.alignment_state}")
        
        cv2.destroyAllWindows()

# INTEGRATION WITH EXISTING DELIVERY SYSTEM
def run_precision_triangular_delivery_test():
    """Entry point for PRECISION triangular delivery system"""
    print("\n🎯 PRECISION TRIANGLE TIP ALIGNMENT SYSTEM")
    print("="*70)
    print("🎯 SPECIALIZED FOR DEAD-STRAIGHT DELIVERY:")
    print("• Green triangular targets with tips pointing toward robot")
    print("• EXTREME PRECISION tip detection and alignment")
    print("• Dead-straight approach for precise hole delivery")
    print("• Multi-phase alignment: Coarse → Fine → Micro → Straight")
    print("• Real-time precision error monitoring")
    print("• Stability verification before approach")
    print("="*70)
    print("🎯 PRECISION PARAMETERS:")
    print(f"• X-axis tolerance: ±8 pixels (VERY TIGHT)")
    print(f"• Y-axis tolerance: ±12 pixels") 
    print(f"• Optimal approach distance: 120 pixels")
    print(f"• Required stable frames: 3")
    print(f"• Movement phases: Coarse/Fine/Micro adjustments")
    print("="*70)
    
    input("Press Enter to start PRECISION triangular delivery test...")
    
    try:
        from hardware import GolfBotHardware
        from vision import VisionSystem
        
        print("Initializing PRECISION triangular delivery systems...")
        hardware = GolfBotHardware()
        vision = VisionSystem()
        
        if not vision.start():
            print("❌ Camera initialization failed")
            return False
        
        print("✅ PRECISION triangular delivery systems ready!")
        print("🎯 Robot will now perform EXTREME PRECISION alignment")
        
        delivery_system = PrecisionTriangleDeliverySystem(hardware, vision)
        delivery_system.start_precision_delivery_mode()
        
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

# ENHANCED DELIVERY SYSTEM SELECTOR
def run_delivery_test():
    """Enhanced delivery system selector with precision option"""
    print("\n🚚 GOLFBOT DELIVERY SYSTEM v4.0 - PRECISION EDITION")
    print("="*70)
    print("DELIVERY OPTIONS:")
    print("1. PRECISION Triangular Targets (NEW) - Dead-straight alignment")
    print("2. Standard Triangular Targets - Tips pointing toward robot")
    print("3. Rectangular Green Targets (Legacy)")
    print("="*70)
    
    while True:
        try:
            choice = input("Select delivery mode (1, 2, or 3): ").strip()
            
            if choice == '1':
                print("\n🎯 PRECISION TRIANGULAR DELIVERY MODE SELECTED")
                print("Target: Green triangles with EXTREME precision tip alignment")
                print("Purpose: Dead-straight delivery through precise holes in walls")
                return run_precision_triangular_delivery_test()
                
            elif choice == '2':
                print("\n🔺 STANDARD TRIANGULAR DELIVERY MODE SELECTED")
                print("Target: Green triangles with tips pointing toward robot")
                # Import and run the standard triangular system
                from triangle_delivery_system import run_triangular_delivery_test
                return run_triangular_delivery_test()
                
            elif choice == '3':
                print("\n🟫 RECTANGULAR DELIVERY MODE SELECTED")
                print("Target: Green rectangular zones")
                # Import and run the rectangular system  
                from delivery_system import run_rectangular_delivery_test
                return run_rectangular_delivery_test()
                
            else:
                print("Invalid choice. Enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return True
        except EOFError:
            print("\nExiting...")
            return True

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_delivery_test()