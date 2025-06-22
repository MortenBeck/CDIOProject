#!/usr/bin/env python3
"""
Clean Simple Green Target Delivery System - Rotated Rectangle Detection
Properly detects rotated green rectangles and approaches their actual short sides
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import config

@dataclass
class GreenTarget:
    """Green target data with rotated rectangle support"""
    center: Tuple[int, int]
    area: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    orientation: str
    short_side_length: int
    long_side_length: int
    rotated_rect: Optional[Tuple] = None
    box_points: Optional[np.ndarray] = None
    rotation_angle: Optional[float] = None

class SimpleDeliveryVisionSystem:
    """Vision system with rotated rectangle detection"""
    
    def __init__(self, vision_system):
        self.logger = logging.getLogger(__name__)
        self.vision = vision_system
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        
        # Green detection parameters
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([80, 255, 255])
        self.min_green_area = 500
        self.max_green_area = 50000
        
        # Approach parameters
        self.approach_distance = 120
        self.alignment_tolerance = 15
        self.centering_tolerance = 20
        
    def detect_green_targets(self, frame) -> List[GreenTarget]:
        """Detect green targets using rotated rectangle detection"""
        green_targets = []
        
        if frame is None:
            return green_targets
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green colors
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_green_area < area < self.max_green_area:
                if len(contour) >= 5:
                    try:
                        # Get rotated rectangle
                        rotated_rect = cv2.minAreaRect(contour)
                        center_float, (width_float, height_float), angle = rotated_rect
                        
                        center = (int(center_float[0]), int(center_float[1]))
                        width = int(width_float)
                        height = int(height_float)
                        
                        # Determine orientation
                        if width > height:
                            orientation = 'horizontal'
                            short_side = height
                            long_side = width
                        else:
                            orientation = 'vertical'
                            short_side = width
                            long_side = height
                        
                        # Get corner points
                        box_points = cv2.boxPoints(rotated_rect)
                        box_points = np.int0(box_points)
                        
                        # Calculate bbox
                        x_coords = box_points[:, 0]
                        y_coords = box_points[:, 1]
                        bbox = (int(np.min(x_coords)), int(np.min(y_coords)), 
                               int(np.max(x_coords) - np.min(x_coords)), 
                               int(np.max(y_coords) - np.min(y_coords)))
                        
                        # Calculate confidence
                        rect_area = width * height
                        fill_ratio = area / max(rect_area, 1)
                        aspect_ratio = max(long_side, 1) / max(short_side, 1)
                        
                        size_confidence = min(1.0, area / 3000)
                        shape_confidence = min(1.0, fill_ratio)
                        aspect_confidence = min(1.0, aspect_ratio / 4.0)
                        
                        confidence = (size_confidence + shape_confidence + aspect_confidence) / 3
                        
                        if confidence > 0.3:
                            target = GreenTarget(
                                center=center,
                                area=area,
                                confidence=confidence,
                                bbox=bbox,
                                orientation=orientation,
                                short_side_length=short_side,
                                long_side_length=long_side,
                                rotated_rect=rotated_rect,
                                box_points=box_points,
                                rotation_angle=angle
                            )
                            green_targets.append(target)
                            
                    except Exception as e:
                        self.logger.warning(f"Rotated rectangle calculation failed: {e}")
                        continue
        
        # Sort by confidence and size
        green_targets.sort(key=lambda t: (-t.confidence, -t.area))
        return green_targets[:2]
    
    def get_parallel_parking_command(self, target: GreenTarget) -> Optional[str]:
        """Get positioning command for parallel parking approach"""
        robot_x = self.frame_center_x
        robot_y = self.frame_center_y
        
        if hasattr(target, 'box_points') and target.box_points is not None:
            box_pts = target.box_points
            
            # Calculate side lengths to identify short sides
            side_data = []
            for i in range(4):
                p1 = box_pts[i]
                p2 = box_pts[(i + 1) % 4]
                length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                robot_dist = np.sqrt((robot_x - mid_x)**2 + (robot_y - mid_y)**2)
                
                side_data.append({
                    'length': length,
                    'midpoint': (mid_x, mid_y),
                    'robot_distance': robot_dist,
                    'p1': p1,
                    'p2': p2
                })
            
            # Find shortest sides
            side_data.sort(key=lambda x: x['length'])
            short_sides = side_data[:2]
            
            # Choose closest short side
            closest_short_side = min(short_sides, key=lambda x: x['robot_distance'])
            target_x, target_y = closest_short_side['midpoint']
            
            # Calculate perpendicular approach vector
            p1, p2 = closest_short_side['p1'], closest_short_side['p2']
            side_vec_x = p2[0] - p1[0]
            side_vec_y = p2[1] - p1[1]
            
            # Perpendicular vector
            perp_vec_x = -side_vec_y
            perp_vec_y = side_vec_x
            
            # Normalize
            perp_length = np.sqrt(perp_vec_x**2 + perp_vec_y**2)
            if perp_length > 0:
                perp_vec_x /= perp_length
                perp_vec_y /= perp_length
            
            # Choose direction toward robot
            robot_vec_x = robot_x - target_x
            robot_vec_y = robot_y - target_y
            dot_product = robot_vec_x * perp_vec_x + robot_vec_y * perp_vec_y
            if dot_product < 0:
                perp_vec_x = -perp_vec_x
                perp_vec_y = -perp_vec_y
            
            # Desired position
            desired_x = target_x + int(self.approach_distance * perp_vec_x)
            desired_y = target_y + int(self.approach_distance * perp_vec_y)
            
        else:
            # Fallback
            target_x, target_y = target.center
            if target.orientation == 'horizontal':
                desired_x = target_x
                desired_y = target_y - self.approach_distance
            else:
                desired_x = target_x - self.approach_distance
                desired_y = target_y
        
        # Calculate positioning commands
        x_error = robot_x - desired_x
        y_error = robot_y - desired_y
        
        # Prioritize larger error
        if abs(x_error) > abs(y_error):
            if abs(x_error) > self.centering_tolerance:
                return 'move_left' if x_error > 0 else 'move_right'
        else:
            if abs(y_error) > self.centering_tolerance:
                return 'move_backward' if y_error > 0 else 'move_forward'
        
        # Close enough
        if (abs(x_error) <= self.centering_tolerance and 
            abs(y_error) <= self.centering_tolerance):
            return 'approach_target'
        
        return None
    
    def get_final_approach_command(self, target: GreenTarget) -> str:
        """Get final approach command"""
        if hasattr(target, 'box_points') and target.box_points is not None:
            return 'approach_perpendicular'
        else:
            return 'approach_vertical' if target.orientation == 'horizontal' else 'approach_horizontal'
    
    def draw_delivery_visualization(self, frame, targets: List[GreenTarget], current_phase: str) -> np.ndarray:
        """Draw visualization with rotated rectangles"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw targets
        for i, target in enumerate(targets):
            color = (0, 255, 0) if i == 0 else (0, 150, 0)
            thickness = 3 if i == 0 else 2
            
            if hasattr(target, 'box_points') and target.box_points is not None:
                # Draw rotated rectangle
                cv2.drawContours(result, [target.box_points], 0, color, thickness)
                
                if i == 0:  # Primary target
                    box_pts = target.box_points
                    
                    # Calculate side lengths to find short sides
                    side_lengths = []
                    for j in range(4):
                        p1 = box_pts[j]
                        p2 = box_pts[(j + 1) % 4]
                        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        side_lengths.append((length, j))
                    
                    # Highlight short sides
                    side_lengths.sort()
                    short_side_indices = [side_lengths[0][1], side_lengths[1][1]]
                    
                    for side_idx in short_side_indices:
                        p1 = tuple(box_pts[side_idx])
                        p2 = tuple(box_pts[(side_idx + 1) % 4])
                        cv2.line(result, p1, p2, (0, 255, 255), 5)
                    
                    # Draw approach arrow to closest short side
                    robot_pos = (self.frame_center_x, self.frame_center_y)
                    min_dist = float('inf')
                    closest_short_side = None
                    
                    for side_idx in short_side_indices:
                        p1 = box_pts[side_idx]
                        p2 = box_pts[(side_idx + 1) % 4]
                        mid_x = (p1[0] + p2[0]) // 2
                        mid_y = (p1[1] + p2[1]) // 2
                        dist = np.sqrt((robot_pos[0] - mid_x)**2 + (robot_pos[1] - mid_y)**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_short_side = (p1, p2, (mid_x, mid_y))
                    
                    if closest_short_side:
                        p1, p2, midpoint = closest_short_side
                        dx = robot_pos[0] - midpoint[0]
                        dy = robot_pos[1] - midpoint[1]
                        norm = np.sqrt(dx*dx + dy*dy)
                        if norm > 0:
                            approach_x = midpoint[0] + int(30 * dx / norm)
                            approach_y = midpoint[1] + int(30 * dy / norm)
                            cv2.arrowedLine(result, (approach_x, approach_y), midpoint, (0, 255, 255), 3)
                            cv2.putText(result, "APPROACH", (approach_x - 30, approach_y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
            else:
                # Fallback rectangle
                x, y, w_rect, h_rect = target.bbox
                cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            
            # Draw center and label
            cv2.circle(result, target.center, 5 if i == 0 else 3, color, -1)
            cv2.putText(result, f"G{i+1}", (target.center[0] - 10, target.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show info for primary target
            if i == 0:
                orientation_text = f"{target.orientation.upper()}"
                if hasattr(target, 'rotation_angle') and target.rotation_angle is not None:
                    orientation_text += f" ({target.rotation_angle:.1f}°)"
                cv2.putText(result, orientation_text, (target.center[0] - 40, target.center[1] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                short_side_text = f"Short: {target.short_side_length}px"
                cv2.putText(result, short_side_text, (target.center[0] - 40, target.center[1] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Robot crosshair
        cv2.line(result, (self.frame_center_x - 10, self.frame_center_y), 
                (self.frame_center_x + 10, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y - 10), 
                (self.frame_center_x, self.frame_center_y + 10), (255, 255, 255), 2)
        cv2.putText(result, "ROBOT", (self.frame_center_x - 20, self.frame_center_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Status overlay
        overlay_height = 120
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.6, overlay, 0.4, 0)
        
        cv2.putText(result, "SIMPLE DELIVERY - Rotated Rectangle Detection", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        target_count = len(targets)
        primary_target = targets[0] if targets else None
        
        if primary_target:
            angle_info = ""
            if hasattr(primary_target, 'rotation_angle') and primary_target.rotation_angle is not None:
                angle_info = f" (rotated {primary_target.rotation_angle:.1f}°)"
            status = f"Target: {primary_target.orientation.upper()} green rectangle{angle_info}"
            phase_status = f"Phase: {current_phase.upper()}"
        else:
            status = "Scanning for green targets..."
            phase_status = "Phase: SEARCHING"
        
        cv2.putText(result, f"Targets: {target_count} | {status}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(result, phase_status, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(result, "Yellow lines: Short sides | Yellow arrow: Approach direction", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return result

class SimpleDeliverySystem:
    """Simple delivery system with parallel parking approach"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = SimpleDeliveryVisionSystem(vision_system)
        
        self.current_target = None
        self.delivery_active = False
        self.start_time = None
        self.current_phase = "search"
        self.phase_start_time = None
        
    def start_simple_delivery_mode(self):
        """Start delivery mode"""
        self.logger.info("🚚 STARTING SIMPLE DELIVERY MODE - Rotated Rectangle Detection")
        
        self.delivery_active = True
        self.start_time = time.time()
        self.current_phase = "search"
        
        try:
            self.simple_delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Delivery mode interrupted")
        except Exception as e:
            self.logger.error(f"Delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def simple_delivery_main_loop(self):
        """Main delivery loop"""
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
                
                # Detect targets
                green_targets = self.delivery_vision.detect_green_targets(frame)
                
                # Create visualization
                debug_frame = self.delivery_vision.draw_delivery_visualization(
                    frame, green_targets, self.current_phase)
                
                # Show frame
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('Simple Delivery - Rotated Rectangle Detection', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Process targets
                if green_targets:
                    frames_without_target = 0
                    primary_target = green_targets[0]
                    
                    # Check for new target
                    if (self.current_target is None or 
                        abs(primary_target.center[0] - self.current_target.center[0]) > 50):
                        self.logger.info("🎯 New target detected")
                        self.current_target = primary_target
                        self.current_phase = "position"
                        self.phase_start_time = time.time()
                    
                    self.current_target = primary_target
                    
                    # Execute phase
                    if self.current_phase == "search":
                        self.current_phase = "position"
                        self.phase_start_time = time.time()
                    elif self.current_phase == "position":
                        self.handle_positioning_phase()
                    elif self.current_phase == "approach":
                        self.handle_approach_phase()
                    elif self.current_phase == "deliver":
                        self.handle_deliver_phase()
                
                else:
                    # No targets - search
                    frames_without_target += 1
                    self.current_target = None
                    self.current_phase = "search"
                    
                    if frames_without_target >= max_frames_without_target:
                        search_direction *= -1
                        frames_without_target = 0
                        self.logger.info(f"🔍 Changing search direction")
                    
                    self.search_for_targets(search_direction)
                
                time.sleep(0.15)
                
            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
    
    def handle_positioning_phase(self):
        """Handle positioning"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        parking_command = self.delivery_vision.get_parallel_parking_command(self.current_target)
        
        if parking_command == 'approach_target':
            self.logger.info("✅ Position achieved - starting approach")
            self.current_phase = "approach"
            self.phase_start_time = time.time()
            return
        
        # Execute movement
        move_duration = 0.4
        move_speed = 0.4
        
        if parking_command == 'move_right':
            self.hardware.turn_right(duration=move_duration, speed=move_speed)
        elif parking_command == 'move_left':
            self.hardware.turn_left(duration=move_duration, speed=move_speed)
        elif parking_command == 'move_forward':
            self.hardware.move_forward(duration=move_duration, speed=move_speed)
        elif parking_command == 'move_backward':
            self.hardware.move_backward(duration=move_duration, speed=move_speed)
        
        # Timeout
        if time.time() - self.phase_start_time > 20.0:
            self.logger.warning("⏰ Positioning timeout")
            self.current_phase = "approach"
    
    def handle_approach_phase(self):
        """Handle approach"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        approach_command = self.delivery_vision.get_final_approach_command(self.current_target)
        self.logger.info(f"🚀 Final approach: {approach_command}")
        
        approach_duration = 1.2
        approach_speed = 0.35
        
        if approach_command == 'approach_perpendicular':
            self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
        elif approach_command == 'approach_vertical':
            self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
        elif approach_command == 'approach_horizontal':
            self.hardware.turn_right(duration=0.3, speed=approach_speed)
            time.sleep(0.1)
            self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
        
        self.logger.info("📦 Reached target - proceeding to delivery")
        self.current_phase = "deliver"
        self.phase_start_time = time.time()
    
    def handle_deliver_phase(self):
        """Handle delivery"""
        self.logger.info("📦 Executing delivery")
        
        if self.hardware.has_balls():
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls")
        else:
            self.logger.info("ℹ️  No balls to deliver")
        
        # Back away
        self.hardware.move_backward(duration=1.2, speed=0.4)
        self.hardware.turn_left(duration=0.6, speed=0.4)
        
        # Reset
        self.current_phase = "search"
        self.current_target = None
        self.logger.info("🔄 Delivery complete")
    
    def search_for_targets(self, direction: int):
        """Search for targets"""
        if direction > 0:
            self.hardware.turn_right(duration=0.6, speed=0.5)
        else:
            self.hardware.turn_left(duration=0.6, speed=0.5)
        time.sleep(0.3)
    
    def stop_delivery(self):
        """Stop delivery"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 DELIVERY COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Ball count: {self.hardware.get_ball_count()}")
        
        cv2.destroyAllWindows()

def run_simple_delivery_test():
    """Main entry point"""
    print("\n🚚 ENHANCED GOLFBOT DELIVERY SYSTEM")
    print("="*50)
    print("Features:")
    print("• Rotated rectangle detection")
    print("• True short side identification")
    print("• Perpendicular approach")
    print("• Visual feedback")
    print("="*50)
    
    input("Press Enter to start...")
    
    try:
        from hardware import GolfBotHardware
        from vision import VisionSystem
        
        print("Initializing systems...")
        hardware = GolfBotHardware()
        vision = VisionSystem()
        
        if not vision.start():
            print("❌ Camera initialization failed")
            return False
        
        print("✅ Systems ready!")
        
        delivery_system = SimpleDeliverySystem(hardware, vision)
        delivery_system.start_simple_delivery_mode()
        
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

def run_delivery_test():
    """Main system integration"""
    return run_simple_delivery_test()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_simple_delivery_test()