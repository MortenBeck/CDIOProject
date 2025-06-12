#!/usr/bin/env python3
"""
GolfBot Autonomous Wall Avoidance System - AUTONOMOUS OPERATION
Pi 5 Compatible with automatic ball collection and wall avoidance

FEATURES:
- Autonomous operation with state machine
- White ball detection and movement
- Red wall/boundary detection and avoidance
- Automatic ball collection (servo placeholder)
- State-based behavior: SEARCHING -> MOVING -> COLLECTING
- Emergency wall avoidance override

STATES:
- SEARCHING: Rotate to look for balls
- MOVING: Move towards closest detected ball
- COLLECTING: Activate collection mechanism
- AVOIDING: Emergency wall avoidance
"""

import cv2
import numpy as np
import time
import subprocess
import os
import sys
import math

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
DEFAULT_SPEED = MOTOR_SPEED_SLOW  # Default speed for autonomous operation

# Autonomous behavior settings
SEARCH_ROTATION_SPEED = 0.4      # Speed when searching for balls
MOVE_SPEED = 0.5                 # Speed when moving to ball
COLLECTION_TIME = 3.0            # Time to spend collecting (seconds)
SEARCH_TIMEOUT = 10.0            # Max time to search before giving up (seconds)
BALL_REACHED_DISTANCE = 30       # Pixels - when ball is "reached"
WALL_DANGER_DISTANCE = 80        # Pixels from bottom - closer than before for more aggressive behavior
WALL_AVOIDANCE_TURN_TIME = 1.5   # Time to turn when avoiding walls
BALL_CONFIRMATION_TIME = 2.0     # Time to observe ball before moving (seconds)
BALL_POSITION_TOLERANCE = 30     # Pixels - how much ball can move and still be "same" ball
AUTO_ENABLED = True              # Enable autonomous mode
HEADLESS_MODE = False            # Run without display (for remote operation)

# === ROBOT STATES ===
class RobotState:
    SEARCHING = 0    # Looking for balls by rotating
    MOVING = 1       # Moving towards a detected ball
    COLLECTING = 2   # Collecting the ball
    AVOIDING = 3     # Emergency wall avoidance

# Motor states for display
class MotorState:
    STOPPED = 0
    FORWARD = 1
    REVERSE = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4

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

def both_motors_forward(speed=DEFAULT_SPEED):
    """Move forward (straight) - both motors forward"""
    if not MOTORS_AVAILABLE:
        return
    motor_a_forward(speed)
    motor_b_forward(speed)

def both_motors_reverse(speed=DEFAULT_SPEED):
    """Move reverse (straight) - both motors reverse"""
    if not MOTORS_AVAILABLE:
        return
    motor_a_reverse(speed)
    motor_b_reverse(speed)

def turn_right(speed=DEFAULT_SPEED):
    """Turn right - one forward, one reverse"""
    if not MOTORS_AVAILABLE:
        return
    motor_a_forward(speed)
    motor_b_reverse(speed)

def turn_left(speed=DEFAULT_SPEED):
    """Turn left - one reverse, one forward"""
    if not MOTORS_AVAILABLE:
        return
    motor_a_reverse(speed)
    motor_b_forward(speed)

# === SERVO CONTROL (PLACEHOLDER) ===
def activate_collection_servo():
    """Activate ball collection mechanism - PLACEHOLDER"""
    print("🔧 Collection servo activated (placeholder)")
    # TODO: Add actual servo control here
    # Example: servo.angle = 90, wait, servo.angle = 0

def deactivate_collection_servo():
    """Deactivate ball collection mechanism - PLACEHOLDER"""
    print("🔧 Collection servo deactivated (placeholder)")
    # TODO: Add actual servo control here

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

# === AUTONOMOUS BEHAVIOR CLASS ===
class AutonomousController:
    def __init__(self):
        self.state = RobotState.SEARCHING
        self.state_start_time = time.time()
        self.motor_state = MotorState.STOPPED
        self.target_ball = None
        self.last_ball_positions = []
        self.wall_avoidance_start_time = 0
        
        # Ball confirmation system
        self.candidate_ball = None           # Ball being observed for confirmation
        self.candidate_start_time = None     # When we started observing the candidate
        self.confirmed_ball = None           # Ball that has been confirmed for 2+ seconds
        self.ball_observation_history = []   # History of ball positions for tracking
        
    def is_same_ball(self, ball1, ball2):
        """Check if two ball detections are likely the same ball"""
        if ball1 is None or ball2 is None:
            return False
        
        x1, y1, r1 = ball1
        x2, y2, r2 = ball2
        
        # Calculate distance between centers
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Check if distance is within tolerance and radius is similar
        radius_diff = abs(r1 - r2)
        return (distance < BALL_POSITION_TOLERANCE and radius_diff < 20)
    
    def update_ball_confirmation(self, balls):
        """Update ball confirmation system - requires 2 seconds of consistent detection"""
        current_time = time.time()
        closest_ball = self.get_closest_ball(balls) if balls else None
        
        if closest_ball is None:
            # No balls detected - reset confirmation
            self.candidate_ball = None
            self.candidate_start_time = None
            self.confirmed_ball = None
            return None
        
        if self.candidate_ball is None:
            # Start observing new candidate
            self.candidate_ball = closest_ball
            self.candidate_start_time = current_time
            print(f"🔍 Observing candidate ball at {closest_ball[:2]} - need {BALL_CONFIRMATION_TIME}s confirmation")
            return None
        
        # Check if current ball is same as candidate
        if self.is_same_ball(closest_ball, self.candidate_ball):
            # Same ball - check if enough time has passed
            observation_time = current_time - self.candidate_start_time
            
            if observation_time >= BALL_CONFIRMATION_TIME:
                # Ball confirmed!
                if self.confirmed_ball is None:
                    print(f"✅ Ball CONFIRMED after {observation_time:.1f}s! Moving to target.")
                self.confirmed_ball = closest_ball
                return self.confirmed_ball
            else:
                # Still observing
                remaining_time = BALL_CONFIRMATION_TIME - observation_time
                if int(remaining_time * 10) % 5 == 0:  # Print every 0.5 seconds
                    print(f"⏳ Confirming ball... {remaining_time:.1f}s remaining")
                return None
        else:
            # Different ball detected - start over with new candidate
            self.candidate_ball = closest_ball
            self.candidate_start_time = current_time
            print(f"🔍 New candidate ball detected at {closest_ball[:2]} - restarting confirmation")
            self.confirmed_ball = None
            return None
    
    def get_closest_ball(self, balls):
        """Find the closest ball to the center bottom of the frame"""
        if not balls:
            return None
            
        center_x = CAMERA_WIDTH // 2
        bottom_y = CAMERA_HEIGHT - 50  # Reference point near bottom center
        
        closest_ball = None
        closest_distance = float('inf')
        
        for ball_x, ball_y, radius in balls:
            # Calculate distance to reference point
            distance = math.sqrt((ball_x - center_x)**2 + (ball_y - bottom_y)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_ball = (ball_x, ball_y, radius)
                
        return closest_ball
    
    def calculate_steering(self, ball_x):
        """Calculate which direction to turn based on ball position"""
        center_x = CAMERA_WIDTH // 2
        threshold = 50  # Pixels from center to consider "aligned"
        
        if ball_x < center_x - threshold:
            return "left"
        elif ball_x > center_x + threshold:
            return "right"
        else:
            return "aligned"
    
    def is_ball_reached(self, ball_y, ball_radius):
        """Check if ball is close enough to collect"""
        # Ball is reached if it's large (close) and near bottom of frame
        return ball_radius > 40 and ball_y > (CAMERA_HEIGHT - BALL_REACHED_DISTANCE - ball_radius)
    
    def update_autonomous_behavior(self, balls, danger_detected):
        """Main autonomous behavior state machine with ball confirmation"""
        current_time = time.time()
        state_duration = current_time - self.state_start_time
        
        # EMERGENCY WALL AVOIDANCE - Overrides all other states
        if danger_detected and self.state != RobotState.AVOIDING:
            print("🚨 EMERGENCY: Wall detected! Switching to avoidance mode")
            self.state = RobotState.AVOIDING
            self.state_start_time = current_time
            self.wall_avoidance_start_time = current_time
            stop_motors()
            self.motor_state = MotorState.STOPPED
            # Reset ball confirmation when avoiding
            self.candidate_ball = None
            self.candidate_start_time = None
            self.confirmed_ball = None
            return
        
        # Update ball confirmation system
        confirmed_ball = self.update_ball_confirmation(balls)
        
        # STATE MACHINE
        if self.state == RobotState.SEARCHING:
            self.handle_searching_state(confirmed_ball, state_duration)
            
        elif self.state == RobotState.MOVING:
            self.handle_moving_state(confirmed_ball, state_duration)
            
        elif self.state == RobotState.COLLECTING:
            self.handle_collecting_state(state_duration)
            
        elif self.state == RobotState.AVOIDING:
            self.handle_avoiding_state(danger_detected, state_duration)
    
    def handle_searching_state(self, confirmed_ball, duration):
        """Handle SEARCHING state - rotate to look for balls, wait for confirmation"""
        if confirmed_ball:
            # Found and confirmed ball! Switch to moving
            self.target_ball = confirmed_ball
            print(f"🎯 Confirmed ball found! Switching to MOVING state. Target: {self.target_ball}")
            self.state = RobotState.MOVING
            self.state_start_time = time.time()
            stop_motors()
            self.motor_state = MotorState.STOPPED
        else:
            # Keep searching by rotating (but slower if we're observing a candidate)
            if duration > SEARCH_TIMEOUT:
                # Search timeout - reverse direction and continue
                print("⏰ Search timeout - changing direction")
                self.state_start_time = time.time()
                # Reset confirmation when changing search direction
                self.candidate_ball = None
                self.candidate_start_time = None
                self.confirmed_ball = None
            
            # Rotate right to search (slower if observing candidate)
            search_speed = SEARCH_ROTATION_SPEED
            if self.candidate_ball is not None:
                search_speed = SEARCH_ROTATION_SPEED * 0.5  # Slower rotation while confirming
            
            turn_right(search_speed)
            self.motor_state = MotorState.TURN_RIGHT
    
    def handle_moving_state(self, confirmed_ball, duration):
        """Handle MOVING state - move towards confirmed target ball"""
        if not confirmed_ball:
            # Lost the confirmed ball - go back to searching
            print("❌ Lost confirmed ball! Switching back to SEARCHING")
            self.state = RobotState.SEARCHING
            self.state_start_time = time.time()
            self.target_ball = None
            self.confirmed_ball = None
            self.candidate_ball = None
            self.candidate_start_time = None
            stop_motors()
            self.motor_state = MotorState.STOPPED
            return
        
        # Use the confirmed ball as target
        ball_x, ball_y, ball_radius = confirmed_ball
        self.target_ball = confirmed_ball
        
        # Check if we've reached the ball
        if self.is_ball_reached(ball_y, ball_radius):
            print("🎉 Ball reached! Switching to COLLECTING")
            self.state = RobotState.COLLECTING
            self.state_start_time = time.time()
            stop_motors()
            self.motor_state = MotorState.STOPPED
            return
        
        # Determine steering direction
        steering = self.calculate_steering(ball_x)
        
        if steering == "aligned":
            # Move forward towards ball
            both_motors_forward(MOVE_SPEED)
            self.motor_state = MotorState.FORWARD
        elif steering == "left":
            # Turn left towards ball
            turn_left(MOVE_SPEED)
            self.motor_state = MotorState.TURN_LEFT
        elif steering == "right":
            # Turn right towards ball
            turn_right(MOVE_SPEED)
            self.motor_state = MotorState.TURN_RIGHT
    
    def handle_collecting_state(self, duration):
        """Handle COLLECTING state - activate collection mechanism"""
        if duration < COLLECTION_TIME:
            # Still collecting
            if duration < 0.5:  # First 0.5 seconds
                activate_collection_servo()
                stop_motors()
                self.motor_state = MotorState.STOPPED
        else:
            # Collection complete - go back to searching
            print("✅ Collection complete! Switching back to SEARCHING")
            deactivate_collection_servo()
            self.state = RobotState.SEARCHING
            self.state_start_time = time.time()
            # Reset ball confirmation
            self.candidate_ball = None
            self.candidate_start_time = None
            self.confirmed_ball = None
            self.target_ball = None
            stop_motors()
            self.motor_state = MotorState.STOPPED
    
    def handle_avoiding_state(self, danger_detected, duration):
        """Handle AVOIDING state - turn away from walls"""
        if not danger_detected and duration > WALL_AVOIDANCE_TURN_TIME:
            # Wall avoided - go back to searching
            print("✅ Wall avoided! Switching back to SEARCHING")
            self.state = RobotState.SEARCHING
            self.state_start_time = time.time()
            # Reset ball confirmation
            self.candidate_ball = None
            self.candidate_start_time = None
            self.confirmed_ball = None
            stop_motors()
            self.motor_state = MotorState.STOPPED
        else:
            # Keep turning away from wall
            if duration < WALL_AVOIDANCE_TURN_TIME:
                turn_left(MOVE_SPEED)  # Turn left by default, could be smarter
                self.motor_state = MotorState.TURN_LEFT
            else:
                stop_motors()
                self.motor_state = MotorState.STOPPED

# === CAMERA CLASSES ===
class FastPiCamera2:
    """Ultra-fast camera using picamera2 - BEST PERFORMANCE"""
    
    def __init__(self):
        self.picam2 = None
        self.running = False
        
    def start_capture(self):
        try:
            self.picam2 = Picamera2()
            
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT), 
                    "format": "BGR888"
                },
                controls={
                    "FrameRate": TARGET_FPS,
                    "ExposureTime": 20000,
                    "AnalogueGain": 1.0,
                    "AwbEnable": True,
                    "AwbMode": 0,
                    "AeEnable": True,
                    "Brightness": 0.0,
                    "Contrast": 1.0
                }
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            self.running = True
            time.sleep(2.0)
            
            print(f"✓ picamera2 initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {TARGET_FPS}FPS")
            return True
            
        except Exception as e:
            print(f"❌ picamera2 failed: {e}")
            return False
    
    def capture_frame(self):
        if not self.running or not self.picam2:
            return False, None
            
        try:
            frame = self.picam2.capture_array()
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

class FallbackLibCamera:
    """Fallback libcamera using optimized still capture"""
    
    def __init__(self):
        self.temp_file = "/tmp/golfbot_frame_optimized.jpg"
        self.running = False
        
    def start_capture(self):
        try:
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
    """Optimized white ball detection"""
    if frame is None:
        return []
    
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 25, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    scale_x = CAMERA_WIDTH / PROCESS_WIDTH
    scale_y = CAMERA_HEIGHT / PROCESS_HEIGHT
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_BALL_AREA:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > BALL_CIRCULARITY_THRESHOLD:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    center_x = int(x * scale_x)
                    center_y = int(y * scale_y)
                    radius_scaled = int(radius * max(scale_x, scale_y))
                    
                    if 5 < radius_scaled < 150:
                        balls.append((center_x, center_y, radius_scaled))
    
    return balls

def detect_red_walls_fast(frame):
    """Optimized red wall/boundary detection with danger zone analysis"""
    if frame is None:
        return [], None, False
    
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    walls = []
    danger_detected = False
    scale_x = CAMERA_WIDTH / PROCESS_WIDTH
    scale_y = CAMERA_HEIGHT / PROCESS_HEIGHT
    
    danger_y_threshold = PROCESS_HEIGHT - (WALL_DANGER_DISTANCE / scale_y)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_WALL_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            
            length = max(w_scaled, h_scaled)
            if length > WALL_MIN_LENGTH:
                walls.append((x_scaled, y_scaled, w_scaled, h_scaled))
                
                wall_bottom_y = y + h
                if wall_bottom_y > danger_y_threshold:
                    danger_detected = True
    
    debug_mask = cv2.resize(red_mask, (CAMERA_WIDTH, CAMERA_HEIGHT))
    return walls, debug_mask, danger_detected

def draw_autonomous_display(frame, balls, walls, robot_state, motor_state, danger_detected, target_ball=None, candidate_ball=None, confirmation_progress=0.0):
    """Draw autonomous robot status and detections with ball confirmation info"""
    if frame is None:
        return frame
    
    display_frame = frame.copy()
    
    # Draw danger zone line
    danger_y = CAMERA_HEIGHT - WALL_DANGER_DISTANCE
    color = (0, 0, 255) if danger_detected else (0, 255, 255)
    cv2.line(display_frame, (0, danger_y), (CAMERA_WIDTH, danger_y), color, 2)
    cv2.putText(display_frame, "DANGER ZONE", (CAMERA_WIDTH - 150, danger_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw all balls
    for i, (x, y, radius) in enumerate(balls):
        ball_color = (0, 255, 0)  # Default green
        ball_thickness = 2
        label = f"BALL {i+1}"
        
        # Highlight candidate ball being observed
        if candidate_ball and (x, y, radius) == candidate_ball:
            ball_color = (0, 255, 255)  # Yellow for candidate
            ball_thickness = 3
            label = f"OBSERVING {confirmation_progress:.1f}s"
            # Draw progress circle
            progress_radius = radius + 10
            progress_angle = int(360 * (confirmation_progress / BALL_CONFIRMATION_TIME))
            if progress_angle > 0:
                # Draw arc to show confirmation progress
                cv2.ellipse(display_frame, (x, y), (progress_radius, progress_radius), 
                           0, 0, progress_angle, (0, 255, 255), 3)
        
        # Highlight confirmed target ball
        if target_ball and (x, y, radius) == target_ball:
            cv2.circle(display_frame, (x, y), radius + 8, (255, 255, 0), 4)  # Yellow highlight
            ball_color = (0, 255, 0)  # Keep green but with yellow border
            label = "TARGET"
        
        cv2.circle(display_frame, (x, y), radius, ball_color, ball_thickness)
        cv2.circle(display_frame, (x, y), 2, ball_color, -1)
        cv2.putText(display_frame, label, (x-30, y-radius-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_color, 1)
    
    # Draw walls with danger highlighting
    for x, y, w, h in walls:
        wall_in_danger = (y + h) > (CAMERA_HEIGHT - WALL_DANGER_DISTANCE)
        wall_color = (0, 0, 255) if wall_in_danger else (255, 0, 0)
        thickness = 3 if wall_in_danger else 2
        
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), wall_color, thickness)
        label = "DANGER!" if wall_in_danger else "WALL"
        cv2.putText(display_frame, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, wall_color, 2)
    
    # Robot state indicator
    state_names = {
        RobotState.SEARCHING: "SEARCHING",
        RobotState.MOVING: "MOVING", 
        RobotState.COLLECTING: "COLLECTING",
        RobotState.AVOIDING: "AVOIDING"
    }
    
    state_colors = {
        RobotState.SEARCHING: (255, 255, 0),    # Yellow
        RobotState.MOVING: (0, 255, 0),         # Green
        RobotState.COLLECTING: (255, 0, 255),   # Magenta
        RobotState.AVOIDING: (0, 0, 255)        # Red
    }
    
    state_name = state_names.get(robot_state, "UNKNOWN")
    state_color = state_colors.get(robot_state, (255, 255, 255))
    
    # Draw state box
    cv2.rectangle(display_frame, (10, 10), (250, 50), state_color, -1)
    cv2.rectangle(display_frame, (10, 10), (250, 50), (255, 255, 255), 2)
    cv2.putText(display_frame, f"STATE: {state_name}", (15, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Ball confirmation status
    if candidate_ball and robot_state == RobotState.SEARCHING:
        remaining_time = BALL_CONFIRMATION_TIME - confirmation_progress
        cv2.putText(display_frame, f"Confirming: {remaining_time:.1f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Ball and wall counts
    y_offset = 90 if candidate_ball and robot_state == RobotState.SEARCHING else 70
    cv2.putText(display_frame, f"Balls: {len(balls)}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Walls: {len(walls)}", (10, y_offset + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Autonomous mode indicator
    cv2.putText(display_frame, "AUTONOMOUS MODE", (10, y_offset + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if danger_detected:
        cv2.putText(display_frame, "⚠️ DANGER ZONE ⚠️", (CAMERA_WIDTH//2 - 100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    return display_frame

# === MAIN AUTONOMOUS FUNCTION ===
def autonomous_wall_avoidance_system():
    """Main autonomous system"""
    global AUTO_ENABLED, HEADLESS_MODE
    
    print("=== AUTONOMOUS WALL AVOIDANCE SYSTEM ===")
    print("Robot will run autonomously to find and collect balls")
    
    if not HEADLESS_MODE:
        print("Controls:")
        print("  'q' - Quit")
        print("  'm' - Toggle manual/autonomous mode")
        print("  's' - Show/hide performance stats")
        print("  SPACE - Emergency stop")
    else:
        print("Running in HEADLESS mode - use Ctrl+C to stop")
    
    # Initialize components
    perf_monitor = PerformanceMonitor()
    autonomous_controller = AutonomousController()
    show_stats = ENABLE_PERFORMANCE_STATS and not HEADLESS_MODE
    
    # Initialize camera
    camera = None
    camera_type = "Unknown"
    
    print("\n🔍 Detecting camera...")
    
    if PICAMERA2_AVAILABLE:
        print("Trying picamera2...")
        camera = FastPiCamera2()
        if camera.start_capture():
            camera_type = "picamera2 (Ultra-Fast)"
        else:
            camera = None
    
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
    print(f"✓ Autonomous: {'Enabled' if AUTO_ENABLED else 'Disabled'}")
    print(f"✓ Target FPS: {TARGET_FPS}")
    
    # Initialize motors
    stop_motors()
    
    try:
        frame_count = 0
        start_time = time.time()
        
        print("\n🚀 Starting autonomous operation...")
        
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
            
            # Run autonomous behavior
            if AUTO_ENABLED:
                autonomous_controller.update_autonomous_behavior(balls, danger_detected)
            else:
                # Manual mode - stop motors
                if autonomous_controller.motor_state != MotorState.STOPPED:
                    stop_motors()
                    autonomous_controller.motor_state = MotorState.STOPPED
            
            # Display only if not headless
            if not HEADLESS_MODE:
                # Calculate confirmation progress for display
                confirmation_progress = 0.0
                if autonomous_controller.candidate_start_time:
                    confirmation_progress = time.time() - autonomous_controller.candidate_start_time
                
                # Draw display
                display_frame = draw_autonomous_display(
                    frame, balls, walls, 
                    autonomous_controller.state, 
                    autonomous_controller.motor_state, 
                    danger_detected, 
                    autonomous_controller.target_ball,
                    autonomous_controller.candidate_ball,
                    confirmation_progress
                )
                
                # Performance stats
                if show_stats:
                    fps_text = f"FPS: {perf_monitor.fps:.1f}"
                    latency_text = f"Latency: {perf_monitor.avg_latency:.1f}ms"
                    cv2.putText(display_frame, fps_text, (10, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display_frame, latency_text, (10, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Mode indicator
                mode_text = "AUTO" if AUTO_ENABLED else "MANUAL"
                mode_color = (0, 255, 0) if AUTO_ENABLED else (0, 0, 255)
                cv2.putText(display_frame, f"MODE: {mode_text}", (CAMERA_WIDTH - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
                
                # Display frame
                cv2.imshow('Autonomous GolfBot System', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    AUTO_ENABLED = not AUTO_ENABLED
                    if not AUTO_ENABLED:
                        stop_motors()
                        autonomous_controller.motor_state = MotorState.STOPPED
                    print(f"Mode: {'AUTONOMOUS' if AUTO_ENABLED else 'MANUAL'}")
                elif key == ord('s'):
                    show_stats = not show_stats
                    print(f"Performance stats: {'ON' if show_stats else 'OFF'}")
                elif key == ord(' '):  # Emergency stop
                    print("🛑 EMERGENCY STOP")
                    stop_motors()
                    autonomous_controller.motor_state = MotorState.STOPPED
                    autonomous_controller.state = RobotState.SEARCHING
                    autonomous_controller.state_start_time = time.time()
                    # Reset ball confirmation
                    autonomous_controller.candidate_ball = None
                    autonomous_controller.candidate_start_time = None
                    autonomous_controller.confirmed_ball = None
            else:
                # Headless mode - print periodic status with confirmation info
                if frame_count % (TARGET_FPS * 2) == 0:  # Every 2 seconds
                    state_names = {
                        RobotState.SEARCHING: "SEARCHING",
                        RobotState.MOVING: "MOVING", 
                        RobotState.COLLECTING: "COLLECTING",
                        RobotState.AVOIDING: "AVOIDING"
                    }
                    state_name = state_names.get(autonomous_controller.state, "UNKNOWN")
                    
                    status_msg = f"Status: {state_name} | Balls: {len(balls)} | Walls: {len(walls)} | FPS: {perf_monitor.fps:.1f}"
                    
                    # Add confirmation info
                    if autonomous_controller.candidate_ball:
                        confirmation_progress = time.time() - autonomous_controller.candidate_start_time
                        remaining = BALL_CONFIRMATION_TIME - confirmation_progress
                        status_msg += f" | Confirming ball: {remaining:.1f}s"
                    elif autonomous_controller.confirmed_ball:
                        status_msg += " | Ball confirmed!"
                    
                    print(status_msg)
                    
                    if danger_detected:
                        print("⚠️  DANGER: Wall detected!")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        stop_motors()
        deactivate_collection_servo()
        
        if camera:
            camera.release()
        
        if not HEADLESS_MODE:
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
    print("🤖 Autonomous GolfBot Wall Avoidance System")
    print("===========================================")
    print("This system will autonomously:")
    print("  🔍 SEARCH for white balls by rotating")
    print("  🎯 MOVE towards the closest detected ball")
    print("  🔧 COLLECT balls using servo mechanism")
    print("  🚨 AVOID red walls/boundaries automatically")
    print("  🔄 REPEAT the cycle continuously")
    print()
    
    # Safety check
    if not MOTORS_AVAILABLE:
        print("⚠️  Motors not available - running in vision-only mode")
        print("   Motor control commands will be ignored")
    
    print("=== AUTONOMOUS BEHAVIOR SETTINGS ===")
    print(f"Search rotation speed: {int(SEARCH_ROTATION_SPEED*100)}%")
    print(f"Movement speed: {int(MOVE_SPEED*100)}%")
    print(f"Collection time: {COLLECTION_TIME}s")
    print(f"Search timeout: {SEARCH_TIMEOUT}s")
    print(f"Ball confirmation time: {BALL_CONFIRMATION_TIME}s")
    print(f"Ball reach distance: {BALL_REACHED_DISTANCE} pixels")
    print(f"Wall danger distance: {WALL_DANGER_DISTANCE} pixels")
    print(f"Wall avoidance turn time: {WALL_AVOIDANCE_TURN_TIME}s")
    print()
    
    print("=== CONFIGURATION OPTIONS ===")
    print("  ENTER - Start with default settings")
    print("  'c' - Customize behavior settings")
    print("  'v' - Vision-only mode (no motor control)")
    print("  'h' - Headless mode (no display, for remote operation)")
    
    try:
        config_choice = input("Choose configuration: ").strip().lower()
        
        if config_choice == "c":
            print("\n=== CUSTOM CONFIGURATION ===")
            try:
                move_speed_input = input(f"Movement speed % (current: {int(MOVE_SPEED*100)}): ").strip()
                if move_speed_input:
                    MOVE_SPEED = int(move_speed_input) / 100.0
                    
                search_speed_input = input(f"Search rotation speed % (current: {int(SEARCH_ROTATION_SPEED*100)}): ").strip()
                if search_speed_input:
                    SEARCH_ROTATION_SPEED = int(search_speed_input) / 100.0
                    
                collection_time_input = input(f"Collection time seconds (current: {COLLECTION_TIME}): ").strip()
                if collection_time_input:
                    COLLECTION_TIME = float(collection_time_input)
                    
                confirmation_time_input = input(f"Ball confirmation time seconds (current: {BALL_CONFIRMATION_TIME}): ").strip()
                if confirmation_time_input:
                    BALL_CONFIRMATION_TIME = float(confirmation_time_input)
                    
                danger_distance_input = input(f"Wall danger distance pixels (current: {WALL_DANGER_DISTANCE}): ").strip()
                if danger_distance_input:
                    WALL_DANGER_DISTANCE = int(danger_distance_input)
                    
                print("✓ Custom configuration applied")
            except ValueError:
                print("⚠️  Invalid input, using defaults")
                
        elif config_choice == "v":
            print("✓ Vision-only mode selected")
            MOTORS_AVAILABLE = False
            
        elif config_choice == "h":
            print("✓ Headless mode selected - no display will be shown")
            HEADLESS_MODE = True
            
        elif config_choice == "":
            print("✓ Using default configuration")
        else:
            print("✓ Invalid choice, using defaults")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        exit()
    
    print(f"\n🚀 Starting autonomous system...")
    if HEADLESS_MODE:
        print("Running in headless mode - use Ctrl+C to stop")
    else:
        print("Press 'q' to quit, 'm' to toggle manual mode, SPACE for emergency stop")
    time.sleep(2)
    
    try:
        autonomous_wall_avoidance_system()
    except KeyboardInterrupt:
        print("\nExiting...")
        stop_motors()
    except Exception as e:
        print(f"Error: {e}")
        stop_motors()
        import traceback
        traceback.print_exc()
