#!/usr/bin/env python3
"""
Enhanced Green Target Delivery System - Proper Perpendicular Alignment
Ensures robot approaches perpendicular to the short sides of rotated rectangles
"""

import cv2
import numpy as np
import time
import logging
from triangle_delivery_system import TriangularDeliverySystem, run_triangular_delivery_test
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
    closest_short_side: Optional[dict] = None  # New: closest short side info

class EnhancedDeliveryVisionSystem:
    """Enhanced vision system with proper perpendicular alignment"""
    
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
        
        # Enhanced approach parameters
        self.target_distance = 100        # Distance to maintain from target
        self.alignment_tolerance = 10     # Degrees for angle alignment
        self.position_tolerance = 15      # Pixels for position alignment
        self.final_approach_distance = 60 # Distance for final approach
        
    def detect_green_targets(self, frame) -> List[GreenTarget]:
        """Detect green targets and calculate closest short side for each"""
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
                            # Calculate closest short side info
                            closest_short_side = self._find_closest_short_side(box_points)
                            
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
                                rotation_angle=angle,
                                closest_short_side=closest_short_side
                            )
                            green_targets.append(target)
                            
                    except Exception as e:
                        self.logger.warning(f"Rotated rectangle calculation failed: {e}")
                        continue
        
        # Sort by confidence and size
        green_targets.sort(key=lambda t: (-t.confidence, -t.area))
        return green_targets[:2]
    
    def _find_closest_short_side(self, box_points):
        """Find the closest short side to the robot and calculate approach info"""
        robot_pos = np.array([self.frame_center_x, self.frame_center_y])
        
        # Calculate side lengths to identify short sides
        side_info = []
        for i in range(4):
            p1 = box_points[i]
            p2 = box_points[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            midpoint = (p1 + p2) / 2
            distance_to_robot = np.linalg.norm(midpoint - robot_pos)
            
            # Calculate side vector and normal (perpendicular)
            side_vector = p2 - p1
            side_vector_norm = side_vector / np.linalg.norm(side_vector)
            
            # Normal vector (perpendicular to side, pointing outward)
            normal_vector = np.array([-side_vector_norm[1], side_vector_norm[0]])
            
            side_info.append({
                'index': i,
                'p1': p1,
                'p2': p2,
                'length': length,
                'midpoint': midpoint,
                'distance_to_robot': distance_to_robot,
                'side_vector': side_vector_norm,
                'normal_vector': normal_vector
            })
        
        # Sort by length to find short sides
        side_info.sort(key=lambda x: x['length'])
        short_sides = side_info[:2]  # Two shortest sides
        
        # Find the closest short side to the robot
        closest_short_side = min(short_sides, key=lambda x: x['distance_to_robot'])
        
        # Calculate ideal approach position (along the normal, at target distance)
        approach_offset = closest_short_side['normal_vector'] * self.target_distance
        ideal_approach_pos = closest_short_side['midpoint'] + approach_offset
        
        return {
            'midpoint': closest_short_side['midpoint'],
            'p1': closest_short_side['p1'],
            'p2': closest_short_side['p2'],
            'normal_vector': closest_short_side['normal_vector'],
            'side_vector': closest_short_side['side_vector'],
            'ideal_approach_pos': ideal_approach_pos,
            'length': closest_short_side['length']
        }
    
    def get_enhanced_positioning_command(self, target: GreenTarget) -> Optional[str]:
        """Enhanced positioning that ensures perpendicular alignment to short side"""
        if not target.closest_short_side:
            return self._fallback_positioning(target)
        
        robot_pos = np.array([self.frame_center_x, self.frame_center_y])
        
        short_side = target.closest_short_side
        ideal_pos = short_side['ideal_approach_pos']
        side_midpoint = short_side['midpoint']
        normal_vector = short_side['normal_vector']
        
        # Calculate current position relative to ideal approach position
        position_error = ideal_pos - robot_pos
        position_distance = np.linalg.norm(position_error)
        
        # Calculate current angle relative to target normal
        robot_to_target = side_midpoint - robot_pos
        robot_to_target_norm = robot_to_target / max(np.linalg.norm(robot_to_target), 1e-6)
        
        # Angle between robot direction and ideal approach direction (normal)
        dot_product = np.dot(robot_to_target_norm, normal_vector)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_error = np.degrees(np.arccos(abs(dot_product)))
        
        # Debug info
        self.logger.info(f"🎯 Position error: {position_distance:.1f}px, Angle error: {angle_error:.1f}°")
        self.logger.info(f"🎯 Ideal pos: ({ideal_pos[0]:.0f}, {ideal_pos[1]:.0f}), Robot: ({robot_pos[0]}, {robot_pos[1]})")
        
        # PHASE 1: Rough positioning - get close to ideal approach area
        if position_distance > 50:
            # Move toward ideal approach position
            if abs(position_error[0]) > self.position_tolerance:
                if position_error[0] > 0:
                    return 'move_right'
                else:
                    return 'move_left'
            
            if abs(position_error[1]) > self.position_tolerance:
                if position_error[1] > 0:
                    return 'move_backward'  # Ideal pos is below robot
                else:
                    return 'move_forward'   # Ideal pos is above robot
        
        # PHASE 2: Fine alignment - ensure perpendicular approach angle
        elif angle_error > self.alignment_tolerance:
            # Calculate which direction to turn for better alignment
            # Cross product to determine turn direction
            cross_product = np.cross(robot_to_target_norm, normal_vector)
            
            if cross_product > 0:
                return 'align_turn_left'   # Turn left to align with normal
            else:
                return 'align_turn_right'  # Turn right to align with normal
        
        # PHASE 3: Final approach distance check
        else:
            current_distance_to_target = np.linalg.norm(robot_pos - side_midpoint)
            
            if current_distance_to_target > self.final_approach_distance:
                return 'final_approach'
            else:
                return 'approach_complete'
    
    def _fallback_positioning(self, target: GreenTarget) -> Optional[str]:
        """Fallback positioning if short side calculation fails"""
        robot_x = self.frame_center_x
        robot_y = self.frame_center_y
        target_x, target_y = target.center
        
        x_error = target_x - robot_x
        y_error = target_y - robot_y
        
        if abs(x_error) > 20:
            return 'move_right' if x_error > 0 else 'move_left'
        
        if abs(y_error) > 20:
            return 'move_backward' if y_error > 0 else 'move_forward'
        
        return 'approach_complete'
    
    def draw_enhanced_visualization(self, frame, targets: List[GreenTarget], current_phase: str) -> np.ndarray:
        """Enhanced visualization showing alignment vectors and approach paths"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw targets with enhanced approach visualization
        for i, target in enumerate(targets):
            color = (0, 255, 0) if i == 0 else (0, 150, 0)
            thickness = 3 if i == 0 else 2
            
            if hasattr(target, 'box_points') and target.box_points is not None:
                # Draw rotated rectangle
                cv2.drawContours(result, [target.box_points], 0, color, thickness)
                
                if i == 0 and target.closest_short_side:  # Primary target with approach info
                    short_side = target.closest_short_side
                    
                    # Highlight the closest short side in bright yellow
                    p1 = tuple(short_side['p1'].astype(int))
                    p2 = tuple(short_side['p2'].astype(int))
                    midpoint = tuple(short_side['midpoint'].astype(int))
                    
                    cv2.line(result, p1, p2, (0, 255, 255), 6)
                    cv2.circle(result, midpoint, 5, (0, 255, 255), -1)
                    
                    # Draw normal vector (approach direction)
                    normal_end = (short_side['midpoint'] + short_side['normal_vector'] * 60).astype(int)
                    cv2.arrowedLine(result, midpoint, tuple(normal_end), (255, 255, 0), 3)
                    cv2.putText(result, "APPROACH", tuple(normal_end + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    # Draw ideal approach position
                    ideal_pos = tuple(short_side['ideal_approach_pos'].astype(int))
                    cv2.circle(result, ideal_pos, 8, (255, 0, 255), 3)
                    cv2.putText(result, "IDEAL", (ideal_pos[0] - 20, ideal_pos[1] - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    
                    # Draw current robot position
                    robot_pos = (self.frame_center_x, self.frame_center_y)
                    cv2.circle(result, robot_pos, 6, (255, 255, 255), 2)
                    cv2.putText(result, "ROBOT", (robot_pos[0] - 25, robot_pos[1] + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Draw positioning line from robot to ideal position
                    cv2.line(result, robot_pos, ideal_pos, (255, 0, 255), 2)
                    
                    # Calculate and display distances/angles
                    robot_pos_np = np.array(robot_pos)
                    position_distance = np.linalg.norm(short_side['ideal_approach_pos'] - robot_pos_np)
                    cv2.putText(result, f"Dist: {position_distance:.0f}px", 
                               (robot_pos[0] + 15, robot_pos[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw center and basic info
            cv2.circle(result, target.center, 5 if i == 0 else 3, color, -1)
            cv2.putText(result, f"G{i+1}", (target.center[0] - 10, target.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status overlay
        overlay_height = 140
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.6, overlay, 0.4, 0)
        
        cv2.putText(result, "ENHANCED DELIVERY - Perpendicular Alignment", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        target_count = len(targets)
        primary_target = targets[0] if targets else None
        
        if primary_target and primary_target.closest_short_side:
            robot_pos_np = np.array([self.frame_center_x, self.frame_center_y])
            position_distance = np.linalg.norm(primary_target.closest_short_side['ideal_approach_pos'] - robot_pos_np)
            
            status = f"Target: {primary_target.orientation.upper()} rectangle, Distance: {position_distance:.0f}px"
            phase_status = f"Phase: {current_phase.upper()}"
        else:
            status = "Scanning for green targets..."
            phase_status = "Phase: SEARCHING"
        
        cv2.putText(result, f"Targets: {target_count} | {status}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(result, phase_status, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(result, "Yellow: Closest short side | Magenta: Ideal approach position", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(result, "Blue arrow: Perpendicular approach direction", (10, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return result

class EnhancedDeliverySystem:
    """Enhanced delivery system with proper perpendicular alignment"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = EnhancedDeliveryVisionSystem(vision_system)
        
        self.current_target = None
        self.delivery_active = False
        self.start_time = None
        self.current_phase = "search"
        self.phase_start_time = None
        
    def start_enhanced_delivery_mode(self):
        """Start enhanced delivery mode"""
        self.logger.info("🚚 STARTING ENHANCED DELIVERY MODE - Perpendicular Alignment")
        
        self.delivery_active = True
        self.start_time = time.time()
        self.current_phase = "search"
        
        try:
            self.enhanced_delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Delivery mode interrupted")
        except Exception as e:
            self.logger.error(f"Delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def enhanced_delivery_main_loop(self):
        """Enhanced main delivery loop with proper alignment"""
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
                debug_frame = self.delivery_vision.draw_enhanced_visualization(
                    frame, green_targets, self.current_phase)
                
                # Show frame
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('Enhanced Delivery - Perpendicular Alignment', debug_frame)
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
                        self.logger.info("🎯 New target detected - starting enhanced positioning")
                        self.current_target = primary_target
                        self.current_phase = "enhanced_position"
                        self.phase_start_time = time.time()
                    
                    self.current_target = primary_target
                    
                    # Execute phase
                    if self.current_phase == "search":
                        self.current_phase = "enhanced_position"
                        self.phase_start_time = time.time()
                    elif self.current_phase == "enhanced_position":
                        self.handle_enhanced_positioning_phase()
                    elif self.current_phase == "approach":
                        self.handle_final_approach_phase()
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
    
    def handle_enhanced_positioning_phase(self):
        """Handle enhanced positioning with perpendicular alignment"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        positioning_command = self.delivery_vision.get_enhanced_positioning_command(self.current_target)
        
        if positioning_command == 'approach_complete':
            self.logger.info("✅ Enhanced positioning complete - perpendicular alignment achieved")
            self.current_phase = "approach"
            self.phase_start_time = time.time()
            return
        
        # Execute movement with different parameters for different commands
        if positioning_command in ['move_right', 'move_left', 'move_forward', 'move_backward']:
            # Standard positioning movements
            move_duration = 0.4
            move_speed = 0.4
        elif positioning_command in ['align_turn_left', 'align_turn_right']:
            # Fine alignment turns
            move_duration = 0.3  # Shorter duration for fine alignment
            move_speed = 0.35    # Slower speed for precision
        else:
            move_duration = 0.4
            move_speed = 0.4
        
        # Execute the movement
        if positioning_command == 'move_right':
            self.logger.info("🚗 Enhanced positioning: Turn right")
            self.hardware.turn_right(duration=move_duration, speed=move_speed)
        elif positioning_command == 'move_left':
            self.logger.info("🚗 Enhanced positioning: Turn left")
            self.hardware.turn_left(duration=move_duration, speed=move_speed)
        elif positioning_command == 'move_forward':
            self.logger.info("🚗 Enhanced positioning: Move forward")
            self.hardware.move_forward(duration=move_duration, speed=move_speed)
        elif positioning_command == 'move_backward':
            self.logger.info("🚗 Enhanced positioning: Move backward")
            self.hardware.move_backward(duration=move_duration, speed=move_speed)
        elif positioning_command == 'align_turn_right':
            self.logger.info("🎯 Fine alignment: Turn right")
            self.hardware.turn_right(duration=move_duration, speed=move_speed)
        elif positioning_command == 'align_turn_left':
            self.logger.info("🎯 Fine alignment: Turn left")
            self.hardware.turn_left(duration=move_duration, speed=move_speed)
        elif positioning_command == 'final_approach':
            self.logger.info("🚀 Moving to final approach distance")
            self.hardware.move_forward(duration=0.5, speed=0.4)
        
        # Timeout check
        if time.time() - self.phase_start_time > 25.0:
            self.logger.warning("⏰ Enhanced positioning timeout - proceeding anyway")
            self.current_phase = "approach"
    
    def handle_final_approach_phase(self):
        """Handle final approach - straight line to target"""
        if not self.current_target:
            self.current_phase = "search"
            return
        
        self.logger.info("🚀 Final perpendicular approach to green target")
        
        # Straight approach since we're already aligned
        approach_duration = 1.0
        approach_speed = 0.35
        
        self.hardware.move_forward(duration=approach_duration, speed=approach_speed)
        
        self.logger.info("📦 Reached target - proceeding to delivery")
        self.current_phase = "deliver"
        self.phase_start_time = time.time()
    
    def handle_deliver_phase(self):
        """Handle delivery phase"""
        self.logger.info("📦 Executing delivery")
        
        if self.hardware.has_balls():
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls")
        else:
            self.logger.info("ℹ️  No balls to deliver")
        
        # Back away and turn
        self.hardware.move_backward(duration=1.2, speed=0.4)
        self.hardware.turn_left(duration=0.6, speed=0.4)
        
        # Reset
        self.current_phase = "search"
        self.current_target = None
        self.logger.info("🔄 Delivery complete - ready for next target")
    
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
        self.logger.info("🏁 ENHANCED DELIVERY COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Ball count: {self.hardware.get_ball_count()}")
        
        cv2.destroyAllWindows()

# Update the main test function to use enhanced system
def run_delivery_test():
    """Main entry point for delivery system - now supports triangular targets"""
    print("\n🚚 GOLFBOT DELIVERY SYSTEM v3.0")
    print("="*60)
    print("DELIVERY OPTIONS:")
    print("1. Triangular Green Targets (NEW) - Tips pointing toward robot")
    print("2. Rectangular Green Targets (Legacy)")
    print("="*60)
    
    while True:
        try:
            choice = input("Select delivery mode (1 or 2): ").strip()
            
            if choice == '1':
                print("\n🔺 TRIANGULAR DELIVERY MODE SELECTED")
                print("Target: Green triangles with tips pointing toward robot")
                return run_triangular_delivery_test()
                
            elif choice == '2':
                print("\n🟫 RECTANGULAR DELIVERY MODE SELECTED")
                print("Target: Green rectangular zones")
                return run_rectangular_delivery_test()  # Your existing system
                
            else:
                print("Invalid choice. Enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return True
        except EOFError:
            print("\nExiting...")
            return True

def run_rectangular_delivery_test():
    """Your existing rectangular delivery system"""
    print("\n🟫 RECTANGULAR GOLFBOT DELIVERY SYSTEM")
    print("="*60)
    print("FEATURES:")
    print("• Perpendicular alignment to short sides")
    print("• Smart approach positioning calculation")
    print("• Fine angle alignment before approach")
    print("• Enhanced visual feedback with vectors")
    print("• Improved approach accuracy")
    print("="*60)
    
    input("Press Enter to start rectangular delivery test...")
    
    try:
        from hardware import GolfBotHardware
        from vision import VisionSystem
        
        print("Initializing rectangular delivery systems...")
        hardware = GolfBotHardware()
        vision = VisionSystem()
        
        if not vision.start():
            print("❌ Camera initialization failed")
            return False
        
        print("✅ Rectangular delivery systems ready!")
        
        # Use your existing EnhancedDeliverySystem
        delivery_system = EnhancedDeliverySystem(hardware, vision)
        delivery_system.start_enhanced_delivery_mode()
        
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
    run_delivery_test()