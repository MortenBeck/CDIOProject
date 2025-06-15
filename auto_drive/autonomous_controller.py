#!/usr/bin/env python3
"""
Autonomous Controller for GolfBot
Implements the state machine for autonomous ball collection behavior
OPTIMIZED: Still search -> rotate only when needed
"""

import time
import math
from robot_states import RobotState, MotorState
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, SEARCH_ROTATION_SPEED, MOVE_SPEED,
    COLLECTION_TIME, SEARCH_TIMEOUT, BALL_REACHED_DISTANCE,
    WALL_AVOIDANCE_TURN_TIME, BALL_CONFIRMATION_TIME,
    BALL_POSITION_TOLERANCE
)

class AutonomousController:
    """Main autonomous behavior controller with state machine"""
    
    def __init__(self, motor_controller, servo_controller):
        self.motor_controller = motor_controller
        self.servo_controller = servo_controller
        
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
        
        # NEW: Still search system
        self.still_search_duration = 5.0    # Stay still for 5 seconds to search
        self.rotation_amount = 30            # Degrees to rotate when no balls found
        self.current_search_start = time.time()  # When current still search started
        self.is_currently_rotating = False   # Flag to track if we're in rotation phase
        self.rotation_start_time = None      # When rotation started
        self.rotation_duration = 0.5        # How long to rotate (seconds)
    
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
            self.motor_controller.stop_motors()
            self.motor_state = MotorState.STOPPED
            # Reset ball confirmation when avoiding
            self.candidate_ball = None
            self.candidate_start_time = None
            self.confirmed_ball = None
            # Reset search state
            self.is_currently_rotating = False
            self.current_search_start = current_time
            return
        
        # Update ball confirmation system
        confirmed_ball = self.update_ball_confirmation(balls)
        
        # STATE MACHINE
        if self.state == RobotState.SEARCHING:
            self.handle_searching_state_optimized(confirmed_ball, state_duration)
            
        elif self.state == RobotState.MOVING:
            self.handle_moving_state(confirmed_ball, state_duration)
            
        elif self.state == RobotState.COLLECTING:
            self.handle_collecting_state(state_duration)
            
        elif self.state == RobotState.AVOIDING:
            self.handle_avoiding_state(danger_detected, state_duration)
    
    def handle_searching_state_optimized(self, confirmed_ball, duration):
        """
        OPTIMIZED SEARCHING: Stay still for 5 seconds, then rotate briefly if no balls found
        """
        current_time = time.time()
        
        if confirmed_ball:
            # Found and confirmed ball! Switch to moving
            self.target_ball = confirmed_ball
            print(f"🎯 Confirmed ball found! Switching to MOVING state. Target: {self.target_ball}")
            self.state = RobotState.MOVING
            self.state_start_time = time.time()
            self.motor_controller.stop_motors()
            self.motor_state = MotorState.STOPPED
            # Reset search state
            self.is_currently_rotating = False
            self.current_search_start = current_time
            return
        
        # Check if we're currently in rotation phase
        if self.is_currently_rotating:
            # We're rotating - check if rotation is complete
            rotation_elapsed = current_time - self.rotation_start_time
            
            if rotation_elapsed >= self.rotation_duration:
                # Rotation complete - stop and start new still search
                print("🔄 Rotation complete - starting new still search phase")
                self.motor_controller.stop_motors()
                self.motor_state = MotorState.STOPPED
                self.is_currently_rotating = False
                self.current_search_start = current_time
                # Reset ball confirmation for new position
                self.candidate_ball = None
                self.candidate_start_time = None
                self.confirmed_ball = None
            else:
                # Continue rotating
                self.motor_controller.turn_right(SEARCH_ROTATION_SPEED)
                self.motor_state = MotorState.TURN_RIGHT
        else:
            # We're in still search phase
            still_search_elapsed = current_time - self.current_search_start
            
            if still_search_elapsed >= self.still_search_duration:
                # Still search time is up - start rotation
                print(f"⏰ Still search timeout ({self.still_search_duration}s) - starting brief rotation")
                self.is_currently_rotating = True
                self.rotation_start_time = current_time
                self.motor_controller.turn_right(SEARCH_ROTATION_SPEED)
                self.motor_state = MotorState.TURN_RIGHT
                # Reset ball confirmation when starting rotation
                self.candidate_ball = None
                self.candidate_start_time = None
                self.confirmed_ball = None
            else:
                # Stay still and search
                self.motor_controller.stop_motors()
                self.motor_state = MotorState.STOPPED
                
                # Optional: Print remaining time every second
                if int(still_search_elapsed) != int(still_search_elapsed - 0.1):
                    remaining_time = self.still_search_duration - still_search_elapsed
                    if remaining_time > 0 and int(remaining_time * 10) % 10 == 0:
                        print(f"🔍 Still searching... {remaining_time:.1f}s remaining")
    
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
            # Reset search state
            self.is_currently_rotating = False
            self.current_search_start = time.time()
            self.motor_controller.stop_motors()
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
            self.motor_controller.stop_motors()
            self.motor_state = MotorState.STOPPED
            return
        
        # Determine steering direction
        steering = self.calculate_steering(ball_x)
        
        if steering == "aligned":
            # Move forward towards ball
            self.motor_controller.both_motors_forward(MOVE_SPEED)
            self.motor_state = MotorState.FORWARD
        elif steering == "left":
            # Turn left towards ball
            self.motor_controller.turn_left(MOVE_SPEED)
            self.motor_state = MotorState.TURN_LEFT
        elif steering == "right":
            # Turn right towards ball
            self.motor_controller.turn_right(MOVE_SPEED)
            self.motor_state = MotorState.TURN_RIGHT
    
    def handle_collecting_state(self, duration):
        """Handle COLLECTING state - activate collection mechanism"""
        if duration < COLLECTION_TIME:
            # Still collecting
            if duration < 0.5:  # First 0.5 seconds
                self.servo_controller.activate_collection_servo()
                self.motor_controller.stop_motors()
                self.motor_state = MotorState.STOPPED
        else:
            # Collection complete - go back to searching
            print("✅ Collection complete! Switching back to SEARCHING")
            self.servo_controller.deactivate_collection_servo()
            self.state = RobotState.SEARCHING
            self.state_start_time = time.time()
            # Reset ball confirmation
            self.candidate_ball = None
            self.candidate_start_time = None
            self.confirmed_ball = None
            self.target_ball = None
            # Reset search state
            self.is_currently_rotating = False
            self.current_search_start = time.time()
            self.motor_controller.stop_motors()
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
            # Reset search state
            self.is_currently_rotating = False
            self.current_search_start = time.time()
            self.motor_controller.stop_motors()
            self.motor_state = MotorState.STOPPED
        else:
            # Keep turning away from wall
            if duration < WALL_AVOIDANCE_TURN_TIME:
                self.motor_controller.turn_left(MOVE_SPEED)  # Turn left by default, could be smarter
                self.motor_state = MotorState.TURN_LEFT
            else:
                self.motor_controller.stop_motors()
                self.motor_state = MotorState.STOPPED
    
    def get_confirmation_progress(self):
        """Get current ball confirmation progress (0.0 to BALL_CONFIRMATION_TIME)"""
        if self.candidate_start_time is None:
            return 0.0
        return time.time() - self.candidate_start_time
    
    def get_search_progress(self):
        """Get current still search progress (0.0 to still_search_duration)"""
        if self.is_currently_rotating:
            return self.still_search_duration  # Full progress during rotation
        return min(time.time() - self.current_search_start, self.still_search_duration)
    
    def is_in_rotation_phase(self):
        """Check if currently in rotation phase of searching"""
        return self.is_currently_rotating
    
    def emergency_stop(self):
        """Emergency stop - immediately halt all movement and reset state"""
        print("🛑 EMERGENCY STOP")
        self.motor_controller.stop_motors()
        self.motor_state = MotorState.STOPPED
        self.state = RobotState.SEARCHING
        self.state_start_time = time.time()
        # Reset ball confirmation
        self.candidate_ball = None
        self.candidate_start_time = None
        self.confirmed_ball = None
        # Reset search state
        self.is_currently_rotating = False
        self.current_search_start = time.time()