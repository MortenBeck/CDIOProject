#!/usr/bin/env python3
"""
GolfBot Delivery System - Green Target Detection and Navigation
Standalone system for finding and moving towards green delivery zones
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
    """Class to store detected green target information"""
    center: Tuple[int, int]
    area: int
    confidence: float
    distance_from_center: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height

class DeliveryVisionSystem:
    """Vision system specifically for green target detection"""
    
    def __init__(self, vision_system):
        self.logger = logging.getLogger(__name__)
        self.vision = vision_system  # Reuse main vision system for camera
        self.frame_center_x = config.CAMERA_WIDTH // 2
        self.frame_center_y = config.CAMERA_HEIGHT // 2
        
        # Green detection parameters
        self.green_lower = np.array([40, 50, 50])   # Lower green HSV
        self.green_upper = np.array([80, 255, 255]) # Upper green HSV
        self.min_green_area = 500   # Minimum area for green target
        self.max_green_area = 50000 # Maximum area for green target
        
        # Centering tolerances for delivery
        self.centering_tolerance_x = 30  # pixels
        self.centering_tolerance_y = 25  # pixels
        
    def detect_green_targets(self, frame) -> List[GreenTarget]:
        """Detect green targets in the frame"""
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
                # Get bounding box
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                center = (x + w_rect // 2, y + h_rect // 2)
                
                # Calculate confidence based on area and shape
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # More rectangular = higher confidence for delivery zones
                    aspect_ratio = w_rect / max(h_rect, 1)
                    area_ratio = area / (w_rect * h_rect)
                    
                    # Confidence based on size and rectangular shape
                    size_confidence = min(1.0, area / 5000)  # Larger = more confident
                    shape_confidence = area_ratio * 0.7  # How well it fills bounding box
                    
                    confidence = (size_confidence + shape_confidence) / 2
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        distance_from_center = np.sqrt(
                            (center[0] - self.frame_center_x)**2 + 
                            (center[1] - self.frame_center_y)**2
                        )
                        
                        target = GreenTarget(
                            center=center,
                            area=area,
                            confidence=confidence,
                            distance_from_center=distance_from_center,
                            bbox=(x, y, w_rect, h_rect)
                        )
                        green_targets.append(target)
        
        # Sort by confidence and size (prefer larger, more confident targets)
        green_targets.sort(key=lambda t: (-t.confidence, -t.area))
        
        return green_targets[:3]  # Return top 3 targets
    
    def is_target_centered(self, target: GreenTarget) -> bool:
        """Check if green target is centered for approach"""
        x_offset = abs(target.center[0] - self.frame_center_x)
        y_offset = abs(target.center[1] - self.frame_center_y)
        
        return (x_offset <= self.centering_tolerance_x and 
                y_offset <= self.centering_tolerance_y)
    
    def get_centering_direction(self, target: GreenTarget) -> tuple:
        """Get direction to center on green target"""
        x_offset = target.center[0] - self.frame_center_x
        y_offset = target.center[1] - self.frame_center_y
        
        # X-axis (turning)
        if abs(x_offset) <= self.centering_tolerance_x:
            x_direction = 'centered'
        elif x_offset > 0:
            x_direction = 'right'
        else:
            x_direction = 'left'
        
        # Y-axis (distance)
        if abs(y_offset) <= self.centering_tolerance_y:
            y_direction = 'centered'
        elif y_offset > 0:
            y_direction = 'backward'
        else:
            y_direction = 'forward'
        
        return x_direction, y_direction
    
    def draw_green_detection(self, frame, targets: List[GreenTarget]) -> np.ndarray:
        """Draw green target detection overlays"""
        if frame is None:
            return np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw centering zone
        center_left = self.frame_center_x - self.centering_tolerance_x
        center_right = self.frame_center_x + self.centering_tolerance_x
        center_top = self.frame_center_y - self.centering_tolerance_y
        center_bottom = self.frame_center_y + self.centering_tolerance_y
        
        cv2.rectangle(result, (center_left, center_top), (center_right, center_bottom), 
                     (255, 255, 0), 2)
        cv2.putText(result, "DELIVERY CENTER ZONE", (center_left + 5, center_top - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw detected green targets
        for i, target in enumerate(targets):
            x, y, w_rect, h_rect = target.bbox
            
            # Color based on priority (first target is primary)
            if i == 0:
                color = (0, 255, 0)    # Bright green for primary target
                thickness = 3
                
                # Check if centered
                centered = self.is_target_centered(target)
                if centered:
                    cv2.putText(result, "CENTERED - READY!", (target.center[0] - 50, target.center[1] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Show direction arrows
                    x_dir, y_dir = self.get_centering_direction(target)
                    direction_text = f"{x_dir.upper()}, {y_dir.upper()}"
                    cv2.putText(result, direction_text, (target.center[0] - 40, target.center[1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Arrow to target
                    cv2.arrowedLine(result, (self.frame_center_x, self.frame_center_y), 
                                   target.center, (255, 255, 0), 2)
            else:
                color = (0, 150, 0)    # Darker green for secondary targets
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), color, thickness)
            
            # Draw center point
            cv2.circle(result, target.center, 5, color, -1)
            
            # Target info
            cv2.putText(result, f"G{i+1}", (target.center[0] - 10, target.center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(result, f"C:{target.confidence:.2f}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Frame center crosshair
        cv2.line(result, (self.frame_center_x - 10, self.frame_center_y), 
                (self.frame_center_x + 10, self.frame_center_y), (255, 255, 255), 2)
        cv2.line(result, (self.frame_center_x, self.frame_center_y - 10), 
                (self.frame_center_x, self.frame_center_y + 10), (255, 255, 255), 2)
        
        # Status overlay
        overlay_height = 80
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        result[0:overlay_height, :] = cv2.addWeighted(result[0:overlay_height, :], 0.6, overlay, 0.4, 0)
        
        # Status text
        cv2.putText(result, "DELIVERY MODE - Green Target Detection", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        target_count = len(targets)
        primary_target = targets[0] if targets else None
        
        if primary_target:
            centered = self.is_target_centered(primary_target)
            status = "CENTERED - READY TO APPROACH" if centered else "CENTERING ON TARGET"
            status_color = (0, 255, 0) if centered else (255, 255, 0)
        else:
            status = "SCANNING FOR GREEN TARGETS..."
            status_color = (255, 255, 255)
        
        cv2.putText(result, f"Targets: {target_count} | Status: {status}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        return result

class DeliverySystem:
    """Main delivery system for green target approach"""
    
    def __init__(self, hardware, vision_system):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision_system = vision_system
        self.delivery_vision = DeliveryVisionSystem(vision_system)
        
        # State management
        self.current_target = None
        self.delivery_active = False
        self.start_time = None
        
        # Movement parameters
        self.search_turn_duration = 0.8  # Time to turn while searching
        self.centering_turn_duration = 0.3  # Time for centering adjustments
        self.approach_speed = 0.4  # Speed when approaching target
        self.search_speed = 0.5   # Speed when searching
        
    def start_delivery_mode(self):
        """Start delivery mode - search and approach green targets"""
        self.logger.info("🚚 STARTING DELIVERY MODE - Green Target Detection")
        self.logger.info("   Searching for green delivery zones...")
        self.logger.info("   Will center on target and approach when found")
        
        self.delivery_active = True
        self.start_time = time.time()
        
        try:
            self.delivery_main_loop()
        except KeyboardInterrupt:
            self.logger.info("Delivery mode interrupted by user")
        except Exception as e:
            self.logger.error(f"Delivery mode error: {e}")
        finally:
            self.stop_delivery()
    
    def delivery_main_loop(self):
        """Main delivery loop - search, center, approach"""
        search_direction = 1  # 1 for right, -1 for left
        frames_without_target = 0
        max_frames_without_target = 30  # ~3 seconds at 10fps
        
        while self.delivery_active:
            try:
                # Get current frame
                ret, frame = self.vision_system.get_frame()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Detect green targets
                green_targets = self.delivery_vision.detect_green_targets(frame)
                
                # Create visualization
                debug_frame = self.delivery_vision.draw_green_detection(frame, green_targets)
                
                # Show frame if display available
                if config.SHOW_CAMERA_FEED:
                    cv2.imshow('GolfBot Delivery Mode', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # TARGET PROCESSING
                if green_targets:
                    frames_without_target = 0
                    primary_target = green_targets[0]
                    self.current_target = primary_target
                    
                    # Check if target is centered
                    if self.delivery_vision.is_target_centered(primary_target):
                        self.logger.info(f"🎯 Target centered! Approaching green zone (conf: {primary_target.confidence:.2f})")
                        self.approach_target(primary_target)
                    else:
                        # Center on target
                        self.center_on_target(primary_target)
                
                else:
                    # No targets found - search
                    frames_without_target += 1
                    self.current_target = None
                    
                    if frames_without_target >= max_frames_without_target:
                        # Change search direction periodically
                        search_direction *= -1
                        frames_without_target = 0
                        self.logger.info(f"🔍 Changing search direction: {'RIGHT' if search_direction > 0 else 'LEFT'}")
                    
                    self.search_for_targets(search_direction)
                
                # Control loop timing
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                self.logger.error(f"Delivery loop error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
    
    def center_on_target(self, target: GreenTarget):
        """Center robot on green target"""
        x_direction, y_direction = self.delivery_vision.get_centering_direction(target)
        
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"🎯 Centering on green target: {x_direction}, {y_direction}")
        
        # Prioritize X-axis centering (turning)
        if x_direction == 'right':
            self.hardware.turn_right(duration=self.centering_turn_duration, speed=self.search_speed)
        elif x_direction == 'left':
            self.hardware.turn_left(duration=self.centering_turn_duration, speed=self.search_speed)
        elif y_direction == 'forward':
            self.hardware.move_forward(duration=self.centering_turn_duration, speed=self.approach_speed)
        elif y_direction == 'backward':
            self.hardware.move_backward(duration=self.centering_turn_duration, speed=self.approach_speed)
        
        time.sleep(0.1)  # Brief pause between movements
    
    def approach_target(self, target: GreenTarget):
        """Approach the centered green target"""
        # Calculate approach distance based on target size
        approach_time = self.calculate_approach_time(target)
        
        self.logger.info(f"🚚 Approaching green delivery zone for {approach_time:.1f} seconds")
        
        # Move forward towards target
        self.hardware.move_forward(duration=approach_time, speed=self.approach_speed)
        
        # Brief pause
        time.sleep(0.5)
        
        # Check if we should release balls here
        if self.hardware.has_balls():
            self.logger.info("📦 Reached delivery zone - releasing balls!")
            released_count = self.hardware.release_balls()
            self.logger.info(f"✅ Released {released_count} balls at green delivery zone")
            
            # Back away from delivery zone
            self.hardware.move_backward(duration=1.0, speed=self.approach_speed)
            time.sleep(0.5)
        else:
            self.logger.info("ℹ️  No balls to deliver - backing away from target")
            self.hardware.move_backward(duration=0.5, speed=self.approach_speed)
    
    def search_for_targets(self, direction: int):
        """Search for green targets by turning"""
        if direction > 0:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching right for green targets")
            self.hardware.turn_right(duration=self.search_turn_duration, speed=self.search_speed)
        else:
            if config.DEBUG_MOVEMENT:
                self.logger.debug("🔍 Searching left for green targets")
            self.hardware.turn_left(duration=self.search_turn_duration, speed=self.search_speed)
        
        time.sleep(0.2)
    
    def calculate_approach_time(self, target: GreenTarget) -> float:
        """Calculate how long to approach based on target size and distance"""
        # Larger targets are likely closer, smaller targets are farther
        # Base approach time inversely related to target area
        
        base_time = 2.0  # Base approach time
        area_factor = max(0.3, min(1.5, 5000 / max(target.area, 1000)))  # Scale by area
        
        approach_time = base_time * area_factor
        
        # Bounds
        return max(0.5, min(3.0, approach_time))
    
    def stop_delivery(self):
        """Stop delivery mode"""
        self.delivery_active = False
        self.hardware.stop_motors()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("🏁 DELIVERY MODE COMPLETED")
        self.logger.info(f"   Total time: {elapsed:.1f} seconds")
        self.logger.info(f"   Final ball count: {self.hardware.get_ball_count()}")
        
        cv2.destroyAllWindows()

def run_delivery_test():
    """Main entry point for delivery testing"""
    print("\n🚚 GOLFBOT DELIVERY SYSTEM TEST")
    print("="*50)
    print("This mode will:")
    print("1. Search for GREEN targets (delivery zones)")
    print("2. Center on detected green areas")
    print("3. Approach when properly aligned")
    print("4. Release balls if any are collected")
    print("\nPress 'q' in the camera window to quit")
    print("="*50)
    
    input("Press Enter to start delivery test...")
    
    try:
        # Import and initialize systems
        from hardware import GolfBotHardware
        from vision import VisionSystem
        
        print("Initializing hardware and vision systems...")
        hardware = GolfBotHardware()
        vision = VisionSystem()
        
        if not vision.start():
            print("❌ Failed to initialize camera")
            return False
        
        print("✅ Systems initialized successfully!")
        
        # Create and start delivery system
        delivery_system = DeliverySystem(hardware, vision)
        delivery_system.start_delivery_mode()
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Delivery test interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Delivery test error: {e}")
        return False
    finally:
        # Cleanup
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