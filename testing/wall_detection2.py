#!/usr/bin/env python3
"""
GolfBot Wall Avoidance System - INTEGRATED MOTOR CONTROL
Pi 5 Compatible with automatic wall detection and motor stopping

FEATURES:
- White ball detection (from original code)
- Red wall/boundary detection for navigation
- DC Motor control integration from second_test.py
- Automatic stopping when walls detected
- Manual motor control with keyboard
- Real-time performance monitoring

EXPECTED: 15 FPS with dual detection and motor control
"""

import cv2
import numpy as np
import time
import subprocess
import os
import sys

# Motor control imports
from gpiozero import PWMOutputDevice

# Try to import picamera2 (best performance)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    print("✓ picamera2 available - will use for best performance")
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  picamera2 not available - using fallback methods")

# === PERFORMANCE SETTINGS ===
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_WIDTH = 320      # Process at half resolution for speed
PROCESS_HEIGHT = 240
TARGET_FPS = 15
DISPLAY_FRAME_SKIP = 1   # Display every Nth frame (1 = all frames)

# Detection settings
MIN_BALL_AREA = 150      # White ball minimum area
MIN_WALL_AREA = 100      # Red wall minimum area
BALL_CIRCULARITY_THRESHOLD = 0.25  # Ball circularity
WALL_MIN_LENGTH = 50     # Minimum wall segment length
ENABLE_PERFORMANCE_STATS = True

# Motor control settings
MOTOR_SPEED_SLOW = 0.3    # 30% speed
MOTOR_SPEED_MEDIUM = 0.5  # 50% speed
MOTOR_SPEED_FAST = 0.8    # 80% speed
DEFAULT_SPEED = MOTOR_SPEED_SLOW  # Default speed for manual control

# Wall avoidance settings
WALL_DANGER_DISTANCE = 50   # Pixels from bottom of frame to consider "close" (reduced from 150)
AUTO_STOP_ENABLED = True    # Enable automatic stopping
WALL_AVOIDANCE_ACTIVE = True  # Enable wall avoidance system

# === DC MOTOR SETUP ===
print("Initializing motor control...")
try:
    motor_in1 = PWMOutputDevice(19)  # GPIO 19 (Pin 35) - Motor A direction 1
    motor_in2 = PWMOutputDevice(26)  # GPIO 26 (Pin 37) - Motor A direction 2
    motor_in3 = PWMOutputDevice(20)  # GPIO 20 (Pin 38) - Motor B direction 1
    motor_in4 = PWMOutputDevice(21)  # GPIO 21 (Pin 40) - Motor B direction 2
    MOTORS_AVAILABLE = True
    print("✓ Motors initialized successfully")
except Exception as e:
    MOTORS_AVAILABLE = False
    print(f"⚠️  Motor initialization failed: {e}")

# === MOTOR CONTROL FUNCTIONS ===
def stop_motors():
    """Stop all motors immediately"""
    if not MOTORS_AVAILABLE:
        return
    motor_in1.off()
    motor_in2.off()
    motor_in3.off()
    motor_in4.off()

def motor_a_forward(speed=DEFAULT_SPEED):
    """Motor A forward"""
    if not MOTORS_AVAILABLE:
        return
    motor_in1.value = speed
    motor_in2.off()

def motor_a_reverse(speed=DEFAULT_SPEED):
    """Motor A reverse"""
    if not MOTORS_AVAILABLE:
        return
    motor_in1.off()
    motor_in2.value = speed

def motor_b_forward(speed=DEFAULT_SPEED):
    """Motor B forward"""
    if not MOTORS_AVAILABLE:
        return
    motor_in3.value = speed
    motor_in4.off()

def motor_b_reverse(speed=DEFAULT_SPEED):
    """Motor B reverse"""
    if not MOTORS_AVAILABLE:
        return
    motor_in3.off()
    motor_in4.value = speed

# CORRECTED FUNCTIONS - Both motors same direction for straight movement
def both_motors_forward(speed=DEFAULT_SPEED):
    """Move forward (straight) - CORRECTED for proper wiring"""
    if not MOTORS_AVAILABLE:
        return
    motor_a_forward(speed)
    motor_b_reverse(speed)  # CHANGED: Motor B needs to be reverse for forward motion
    print(f"Moving forward at {int(speed*100)}% speed")

def both_motors_reverse(speed=DEFAULT_SPEED):
    """Move reverse (straight) - CORRECTED for proper wiring"""
    if not MOTORS_AVAILABLE:
        return
    motor_a_reverse(speed)
    motor_b_forward(speed)  # CHANGED: Motor B needs to be forward for reverse motion
    print(f"Moving reverse at {int(speed*100)}% speed")

def turn_right(speed=DEFAULT_SPEED):
    """Turn right - CORRECTED direction"""
    if not MOTORS_AVAILABLE:
        return
    motor_a_forward(speed)   # Left motor forward
    motor_b_forward(speed)   # Right motor forward (same direction = turn right)
    print(f"Turning right at {int(speed*100)}% speed")

def turn_left(speed=DEFAULT_SPEED):
    """Turn left - CORRECTED direction"""
    if not MOTORS_AVAILABLE:
        return
    motor_a_reverse(speed)   # Left motor reverse  
    motor_b_reverse(speed)   # Right motor reverse (same direction = turn left)
    print(f"Turning left at {int(speed*100)}% speed")

# Visualization modes
class VisualizationMode:
    BOTH = 0          # Show both balls and walls
    BALLS_ONLY = 1    # Show only balls
    WALLS_ONLY = 2    # Show only walls
    DEBUG = 3         # Show debug masks

# Motor states for display
class MotorState:
    STOPPED = 0
    FORWARD = 1
    REVERSE = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4

# === PERFORMANCE MONITORING ===
class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.last_time = time.time()
        self.fps = 0
        self.avg_latency = 0
        
    def update(self):
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        
        # Keep only last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
        if len(self.frame_times) > 1:
            self.fps = 1.0 / np.mean(self.frame_times)
            self.avg_latency = np.mean(self.frame_times) * 1000  # ms
            
        self.last_time = current_time

# === OPTIMIZED CAMERA CLASS ===
class FastPiCamera2:
    """Ultra-fast camera using picamera2 - BEST PERFORMANCE"""
    
    def __init__(self):
        self.picam2 = None
        self.running = False
        
    def start_capture(self):
        try:
            self.picam2 = Picamera2()
            
            # Use BGR888 format (what the camera actually outputs)
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT), 
                    "format": "BGR888"  # Camera natively outputs BGR
                },
                controls={
                    "FrameRate": TARGET_FPS,
                    "ExposureTime": 20000,  # Slightly longer exposure
                    "AnalogueGain": 1.0,
                    "AwbEnable": True,      # Auto white balance
                    "AwbMode": 0,           # Auto white balance mode
                    "AeEnable": True,       # Auto exposure
                    "Brightness": 0.0,      # Normal brightness
                    "Contrast": 1.0         # Normal contrast
                }
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            self.running = True
            
            # Longer warm-up for color adjustment
            time.sleep(2.0)  
            
            print(f"✓ picamera2 initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {TARGET_FPS}FPS")
            return True
            
        except Exception as e:
            print(f"❌ picamera2 failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture frame with minimal latency and proper color correction"""
        if not self.running or not self.picam2:
            return False, None
            
        try:
            # Direct array capture
            frame = self.picam2.capture_array()
            
            # Convert BGR to RGB (Mode 3 - the one that worked!)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return True, frame
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            return False, None
    
    def release(self):
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        self.running = False

# === FALLBACK CAMERA CLASS ===
class FallbackLibCamera:
    """Fallback libcamera using optimized still capture"""
    
    def __init__(self):
        self.temp_file = "/tmp/golfbot_frame_optimized.jpg"
        self.running = False
        
    def start_capture(self):
        try:
            # Test capture
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '1',
                '--width', str(CAMERA_WIDTH),
                '--height', str(CAMERA_HEIGHT),
                '--quality', '70',
                '--immediate',
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=3)
            
            if result.returncode == 0 and os.path.exists(self.temp_file):
                self.running = True
                print(f"✓ libcamera-still optimized mode: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                os.remove(self.temp_file)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ libcamera-still failed: {e}")
            return False
    
    def capture_frame(self):
        """Optimized single frame capture"""
        if not self.running:
            return False, None
            
        try:
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '1',
                '--width', str(CAMERA_WIDTH),
                '--height', str(CAMERA_HEIGHT),
                '--quality', '60',
                '--immediate',
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode == 0 and os.path.exists(self.temp_file):
                frame = cv2.imread(self.temp_file)
                return True, frame
            else:
                return False, None
                
        except Exception as e:
            return False, None
    
    def release(self):
        self.running = False
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

# === DETECTION FUNCTIONS ===
def detect_white_balls_fast(frame):
    """
    Optimized white ball detection
    """
    if frame is None:
        return []
    
    # Resize for processing (major speed improvement)
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    
    # Convert to HSV
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # Optimized white detection range
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 25, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Simplified morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    scale_x = CAMERA_WIDTH / PROCESS_WIDTH
    scale_y = CAMERA_HEIGHT / PROCESS_HEIGHT
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_BALL_AREA:
            # Quick circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > BALL_CIRCULARITY_THRESHOLD:
                    # Get center and radius
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Scale back to original resolution
                    center_x = int(x * scale_x)
                    center_y = int(y * scale_y)
                    radius_scaled = int(radius * max(scale_x, scale_y))
                    
                    if 5 < radius_scaled < 150:
                        balls.append((center_x, center_y, radius_scaled))
    
    return balls

def detect_red_walls_fast(frame):
    """
    Optimized red wall/boundary detection with danger zone analysis
    """
    if frame is None:
        return [], None, False
    
    # Resize for processing
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    
    # Convert to HSV
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # Red color detection (two ranges for red hue wrap-around)
    # Lower red range (0-10)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    
    # Upper red range (170-180)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    walls = []
    danger_detected = False
    scale_x = CAMERA_WIDTH / PROCESS_WIDTH
    scale_y = CAMERA_HEIGHT / PROCESS_HEIGHT
    
    # Define danger zone (bottom portion of frame)
    danger_y_threshold = PROCESS_HEIGHT - (WALL_DANGER_DISTANCE / scale_y)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_WALL_AREA:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Scale back to original resolution
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            
            # Check if it's long enough to be a wall segment
            length = max(w_scaled, h_scaled)
            if length > WALL_MIN_LENGTH:
                walls.append((x_scaled, y_scaled, w_scaled, h_scaled))
                
                # Check if wall is in danger zone (close to robot)
                wall_bottom_y = y + h
                if wall_bottom_y > danger_y_threshold:
                    danger_detected = True
    
    # Return walls, debug mask, and danger status
    debug_mask = cv2.resize(red_mask, (CAMERA_WIDTH, CAMERA_HEIGHT))
    return walls, debug_mask, danger_detected

def draw_detections_with_motors(frame, balls, walls, motor_state, danger_detected, mode=VisualizationMode.BOTH):
    """Draw detections with motor status and danger indicators"""
    if frame is None:
        return frame
    
    display_frame = frame.copy()
    
    # Draw danger zone line
    if WALL_AVOIDANCE_ACTIVE:
        danger_y = CAMERA_HEIGHT - WALL_DANGER_DISTANCE
        color = (0, 0, 255) if danger_detected else (0, 255, 255)  # Red if danger, yellow if safe
        cv2.line(display_frame, (0, danger_y), (CAMERA_WIDTH, danger_y), color, 2)
        cv2.putText(display_frame, "DANGER ZONE", (CAMERA_WIDTH - 150, danger_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw balls
    if mode in [VisualizationMode.BOTH, VisualizationMode.BALLS_ONLY]:
        for x, y, radius in balls:
            cv2.circle(display_frame, (x, y), radius, (0, 255, 0), 2)  # Green circle
            cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)      # Green center dot
            # Add label
            cv2.putText(display_frame, "BALL", (x-20, y-radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw walls with danger highlighting
    if mode in [VisualizationMode.BOTH, VisualizationMode.WALLS_ONLY]:
        for x, y, w, h in walls:
            # Check if this wall is in danger zone
            wall_in_danger = (y + h) > (CAMERA_HEIGHT - WALL_DANGER_DISTANCE)
            wall_color = (0, 0, 255) if wall_in_danger else (255, 0, 0)  # Bright red if dangerous
            thickness = 3 if wall_in_danger else 2
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), wall_color, thickness)
            # Add label  
            label = "DANGER!" if wall_in_danger else "WALL"
            cv2.putText(display_frame, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, wall_color, 2)
    
    # Motor status indicator
    motor_colors = {
        MotorState.STOPPED: (128, 128, 128),     # Gray
        MotorState.FORWARD: (0, 255, 0),         # Green
        MotorState.REVERSE: (255, 255, 0),       # Yellow
        MotorState.TURN_LEFT: (255, 0, 255),     # Magenta
        MotorState.TURN_RIGHT: (0, 255, 255)     # Cyan
    }
    
    motor_texts = {
        MotorState.STOPPED: "STOPPED",
        MotorState.FORWARD: "FORWARD", 
        MotorState.REVERSE: "REVERSE",
        MotorState.TURN_LEFT: "TURN LEFT",
        MotorState.TURN_RIGHT: "TURN RIGHT"
    }
    
    motor_color = motor_colors.get(motor_state, (255, 255, 255))
    motor_text = motor_texts.get(motor_state, "UNKNOWN")
    
    # Draw motor status box
    cv2.rectangle(display_frame, (CAMERA_WIDTH - 150, 10), (CAMERA_WIDTH - 10, 50), motor_color, -1)
    cv2.rectangle(display_frame, (CAMERA_WIDTH - 150, 10), (CAMERA_WIDTH - 10, 50), (255, 255, 255), 2)
    cv2.putText(display_frame, motor_text, (CAMERA_WIDTH - 145, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return display_frame

# === MAIN WALL AVOIDANCE FUNCTION ===
def wall_avoidance_system():
    """Main wall avoidance system with motor control"""
    global AUTO_STOP_ENABLED, WALL_AVOIDANCE_ACTIVE
    
    print("=== WALL AVOIDANCE SYSTEM ===")
    print("Detecting walls and controlling motors for collision avoidance")
    print("Controls:")
    print("  'q' - Quit")
    print("  'v' - Cycle visualization modes")
    print("  's' - Show/hide performance stats")
    print("  'a' - Toggle auto-stop")
    print("  'w' - Toggle wall avoidance")
    print("  ↑ - Move forward")
    print("  ↓ - Move reverse")
    print("  ← - Turn left") 
    print("  → - Turn right")
    print("  SPACE or X - Stop motors")
    print("  'x' - Stop motors")
    print("  '1/2/3' - Speed control")
    
    # Performance monitoring
    perf_monitor = PerformanceMonitor()
    show_stats = ENABLE_PERFORMANCE_STATS
    vis_mode = VisualizationMode.BOTH
    motor_state = MotorState.STOPPED
    current_speed = DEFAULT_SPEED  # Use the configured default speed
    
    # Initialize camera
    camera = None
    camera_type = "Unknown"
    
    print("\n🔍 Detecting camera...")
    
    # Try picamera2 first
    if PICAMERA2_AVAILABLE:
        print("Trying picamera2...")
        camera = FastPiCamera2()
        if camera.start_capture():
            camera_type = "picamera2 (Ultra-Fast)"
        else:
            camera = None
    
    # Fallback to libcamera
    if camera is None:
        print("Trying libcamera-still...")
        camera = FallbackLibCamera()
        if camera.start_capture():
            camera_type = "libcamera-still (Fallback)"
        else:
            camera = None
    
    if camera is None:
        print("❌ No camera available!")
        return
    
    print(f"✓ Using: {camera_type}")
    print(f"✓ Motors: {'Available' if MOTORS_AVAILABLE else 'Disabled'}")
    print(f"✓ Auto-stop: {'Enabled' if AUTO_STOP_ENABLED else 'Disabled'}")
    print(f"✓ Target FPS: {TARGET_FPS} (optimized for stability)")
    
    # Initialize motors
    stop_motors()
    
    # Visualization mode names
    mode_names = ["BOTH", "BALLS ONLY", "WALLS ONLY", "DEBUG"]
    
    try:
        frame_count = 0
        start_time = time.time()
        last_danger_time = 0
        
        while True:
            # Capture frame
            ret, frame = camera.capture_frame()
            if not ret or frame is None:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            perf_monitor.update()
            
            # Detect balls and walls
            balls = detect_white_balls_fast(frame)
            walls, wall_debug_mask, danger_detected = detect_red_walls_fast(frame)
            
            # Wall avoidance logic
            current_time = time.time()
            if WALL_AVOIDANCE_ACTIVE and AUTO_STOP_ENABLED and danger_detected:
                if motor_state != MotorState.STOPPED:
                    stop_motors()
                    motor_state = MotorState.STOPPED
                    last_danger_time = current_time
                    print("🚨 WALL DETECTED - EMERGENCY STOP!")
            
            # Draw display
            display_frame = draw_detections_with_motors(frame, balls, walls, motor_state, danger_detected, vis_mode)
            
            # Add status overlay
            ball_color = (0, 255, 0) if len(balls) > 0 else (100, 100, 100)
            wall_color = (0, 0, 255) if len(walls) > 0 else (100, 100, 100)
            
            cv2.putText(display_frame, f"Balls: {len(balls)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, ball_color, 2)
            cv2.putText(display_frame, f"Walls: {len(walls)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, wall_color, 2)
            
            # System status
            auto_status = "AUTO-STOP: ON" if AUTO_STOP_ENABLED else "AUTO-STOP: OFF"
            wall_status = "AVOIDANCE: ON" if WALL_AVOIDANCE_ACTIVE else "AVOIDANCE: OFF"
            cv2.putText(display_frame, auto_status, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(display_frame, wall_status, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Speed indicator
            cv2.putText(display_frame, f"Speed: {int(current_speed*100)}%", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Danger warning
            if danger_detected:
                cv2.putText(display_frame, "⚠️ DANGER ZONE ⚠️", (CAMERA_WIDTH//2 - 100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Performance stats
            if show_stats:
                fps_text = f"FPS: {perf_monitor.fps:.1f}"
                latency_text = f"Latency: {perf_monitor.avg_latency:.1f}ms"
                cv2.putText(display_frame, fps_text, (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, latency_text, (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display frame
            window_name = 'GolfBot Wall Avoidance System'
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                vis_mode = (vis_mode + 1) % 3  # Cycle through first 3 modes
                print(f"Visualization mode: {mode_names[vis_mode]}")
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"Performance stats: {'ON' if show_stats else 'OFF'}")
            elif key == ord('a'):
                AUTO_STOP_ENABLED = not AUTO_STOP_ENABLED
                print(f"Auto-stop: {'ON' if AUTO_STOP_ENABLED else 'OFF'}")
            elif key == ord('r'):
                WALL_AVOIDANCE_ACTIVE = not WALL_AVOIDANCE_ACTIVE
                print(f"Wall avoidance: {'ON' if WALL_AVOIDANCE_ACTIVE else 'OFF'}")
            
            # Manual motor control (only if not in danger or auto-stop disabled)
            elif not (danger_detected and AUTO_STOP_ENABLED):
                if key == 82:  # Up arrow - Forward
                    both_motors_forward(current_speed)
                    motor_state = MotorState.FORWARD
                    print("Manual: Forward")
                elif key == 84:  # Down arrow - Reverse
                    both_motors_reverse(current_speed)
                    motor_state = MotorState.REVERSE
                    print("Manual: Reverse")
                elif key == 81:  # Left arrow - Turn left
                    turn_left(current_speed)
                    motor_state = MotorState.TURN_LEFT
                    print("Manual: Turn left")
                elif key == 83:  # Right arrow - Turn right
                    turn_right(current_speed)
                    motor_state = MotorState.TURN_RIGHT
                    print("Manual: Turn right")
                elif key == ord(' ') or key == ord('x'):  # Space bar or X - Stop
                    stop_motors()
                    motor_state = MotorState.STOPPED
                    print("Manual: Stop")
            
            # Speed control
            if key == ord('1'):
                current_speed = MOTOR_SPEED_SLOW
                print(f"Speed: {int(current_speed*100)}%")
            elif key == ord('2'):
                current_speed = MOTOR_SPEED_MEDIUM
                print(f"Speed: {int(current_speed*100)}%")
            elif key == ord('3'):
                current_speed = MOTOR_SPEED_FAST
                print(f"Speed: {int(current_speed*100)}%")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        stop_motors()
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        
        # Close motor connections
        if MOTORS_AVAILABLE:
            motor_in1.close()
            motor_in2.close()
            motor_in3.close()
            motor_in4.close()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"✓ Cleanup complete")
        print(f"📊 Final stats:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Camera type: {camera_type}")
        print(f"   Motors: {'Available' if MOTORS_AVAILABLE else 'Disabled'}")

if __name__ == "__main__":
    print("GolfBot Wall Avoidance System with Motor Control")
    print("This system will:")
    print("  🟢 Detect white balls")
    print("  🔴 Detect red walls/boundaries")
    print("  🛑 Automatically stop when walls are too close")
    print("  🎮 Allow manual motor control with Arrow Keys")
    print("  ⚡ Run at optimized 15 FPS for stable performance")
    print()
    
    # Safety check
    if not MOTORS_AVAILABLE:
        print("⚠️  Motors not available - running in vision-only mode")
        print("   Motor control commands will be ignored")
    
    # Speed configuration
    print("=== SPEED CONFIGURATION ===")
    print(f"Current default speed: {int(DEFAULT_SPEED*100)}%")
    print("Available presets:")
    print("  1 - Slow (30%)")
    print("  2 - Medium (50%)")  
    print("  3 - Fast (80%)")
    print("  c - Custom percentage (10-100%)")
    print("  ENTER - Use current default")
    
    try:
        speed_choice = input("Choose speed setting: ").strip().lower()
        
        if speed_choice == "1":
            DEFAULT_SPEED = MOTOR_SPEED_SLOW
            print(f"✓ Speed set to: {int(DEFAULT_SPEED*100)}% (Slow)")
        elif speed_choice == "2":
            DEFAULT_SPEED = MOTOR_SPEED_MEDIUM
            print(f"✓ Speed set to: {int(DEFAULT_SPEED*100)}% (Medium)")
        elif speed_choice == "3":
            DEFAULT_SPEED = MOTOR_SPEED_FAST
            print(f"✓ Speed set to: {int(DEFAULT_SPEED*100)}% (Fast)")
        elif speed_choice == "c":
            while True:
                try:
                    custom_percent = int(input("Enter speed percentage (10-100): "))
                    if 10 <= custom_percent <= 100:
                        DEFAULT_SPEED = custom_percent / 100.0
                        print(f"✓ Speed set to: {custom_percent}% (Custom)")
                        break
                    else:
                        print("Please enter a value between 10 and 100")
                except ValueError:
                    print("Please enter a valid number")
        elif speed_choice == "":
            print(f"✓ Using default speed: {int(DEFAULT_SPEED*100)}%")
        else:
            print(f"✓ Invalid choice, using default: {int(DEFAULT_SPEED*100)}%")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        exit()
    
    print(f"\n🚀 Starting system with {int(DEFAULT_SPEED*100)}% motor speed...")
    print("Remember: You can still change speed during operation with keys 1, 2, 3")
    time.sleep(1)
    
    try:
        wall_avoidance_system()
            
    except KeyboardInterrupt:
        print("\nExiting...")
        stop_motors()
    except Exception as e:
        print(f"Error: {e}")
        stop_motors()
        import traceback
        traceback.print_exc()
