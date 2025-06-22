import time
import logging
import numpy as np
from enum import Enum
from typing import Optional, List
import config

class RobotState(Enum):
    SEARCHING = "searching"
    CENTERING_BALL = "centering_ball"
    APPROACHING_BALL = "approaching_ball"
    COLLECTING_BALL = "collecting_ball"
    AVOIDING_BOUNDARY = "avoiding_boundary"
    EMERGENCY_STOP = "emergency_stop"

class RobotStateMachine:
    """Handles all robot state transitions and behavior logic"""
    
    def __init__(self, hardware, vision):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision = vision
        
        # State management
        self.state = RobotState.SEARCHING
        self.search_pattern_index = 0
        self.last_ball_seen_time = None
        
        # Centering state tracking
        self.centering_history = []
        self.centering_stable_count = 0
        self.last_centering_direction = None
        self.direction_change_count = 0
        self.min_stable_frames = 3
        
        # Distance-based and timeout system
        self.centering_start_time = None
        self.centering_attempts = 0
        self.last_significant_progress = None
    
    def execute_state_machine(self, balls, near_boundary, nav_command):
        """Execute state logic with ball detection priority - WHITE BALLS ONLY"""
        
        # HIGH PRIORITY: Never interrupt active collection
        if self.state == RobotState.COLLECTING_BALL:
            self.handle_collecting_ball()
            return
        
        # MEDIUM PRIORITY: Protect ball operations if we have good targets
        confident_balls = [ball for ball in balls if ball.confidence > 0.4] if balls else []
        has_good_target = bool(confident_balls)
        
        # SMART BOUNDARY AVOIDANCE
        should_avoid_boundary = (
            near_boundary and (
                not has_good_target or 
                self.state == RobotState.SEARCHING or
                self._is_boundary_critical()
            )
        )
        
        if should_avoid_boundary:
            reason = "no_balls" if not has_good_target else "searching" if self.state == RobotState.SEARCHING else "critical_proximity"
            if config.DEBUG_MOVEMENT:
                self.logger.info(f"⚠️ Boundary avoidance triggered: {reason}")
            self.state = RobotState.AVOIDING_BOUNDARY
        
        # Execute current state
        if self.state == RobotState.SEARCHING:
            self.handle_searching(balls, nav_command)
            
        elif self.state == RobotState.CENTERING_BALL:
            # Check for centering timeout and force progression
            if hasattr(self, 'centering_start_time') and self.centering_start_time:
                elapsed = time.time() - self.centering_start_time
                if elapsed > 10.0:  # Hard timeout - give up and search
                    self.logger.warning(f"HARD centering timeout ({elapsed:.1f}s) - returning to search")
                    self.state = RobotState.SEARCHING
                    self._reset_centering_state()
                    return
            
            self.handle_centering_ball(balls, nav_command)
            
        elif self.state == RobotState.APPROACHING_BALL:
            self.handle_approaching_ball(balls, nav_command)
            
        elif self.state == RobotState.AVOIDING_BOUNDARY:
            self.handle_avoiding_boundary(near_boundary)
            
        elif self.state == RobotState.EMERGENCY_STOP:
            self.hardware.emergency_stop()

    def handle_searching(self, balls, nav_command):
        """Handle searching with centering requirement - WHITE BALLS ONLY"""
        if balls:
            # Filter for high confidence balls only
            confident_balls = [ball for ball in balls if ball.confidence > 0.4]
            
            if confident_balls:
                ball_count = len(confident_balls)
                avg_confidence = sum(ball.confidence for ball in confident_balls) / ball_count
                
                self.logger.info(f"Found {ball_count} confident white ball(s) (avg conf: {avg_confidence:.2f})")
                self.state = RobotState.CENTERING_BALL
                return
        
        # No confident balls found
        self.execute_search_pattern()
    
    def handle_centering_ball(self, balls, nav_command):
        """ADAPTIVE CENTERING: Distance-based movement + timeout system"""
        if not balls:
            self.logger.info("Lost sight of ball during centering - returning to search")
            self.state = RobotState.SEARCHING
            self._reset_centering_state()
            return
        
        # Filter for confident balls
        confident_balls = [ball for ball in balls if ball.confidence > 0.3]
        
        if not confident_balls:
            self.logger.info("No confident ball detections during centering - returning to search")
            self.state = RobotState.SEARCHING
            self._reset_centering_state()
            return
        
        # Target the closest confident ball
        target_ball = confident_balls[0]
        
        # Initialize centering session if needed
        if self.centering_start_time is None:
            self.centering_start_time = time.time()
            self.centering_attempts = 0
            self.logger.info("Starting centering session")
        
        # Check if ball is in the precise target zone
        if self.vision.is_ball_centered_for_collection(target_ball):
            self.centering_stable_count += 1
            
            if self.centering_stable_count >= self.min_stable_frames:
                elapsed_time = time.time() - self.centering_start_time
                self.logger.info(f"White ball STABLE in target zone! (took {elapsed_time:.1f}s, {self.centering_attempts} attempts)")
                self.state = RobotState.COLLECTING_BALL
                self._reset_centering_state()
                return
            else:
                self.logger.info(f"Ball in target zone - verifying stability ({self.centering_stable_count}/{self.min_stable_frames})")
                time.sleep(0.1)
                return
        else:
            # Ball moved out of target zone - reset stability counter
            self.centering_stable_count = 0
        
        # Calculate distance from ball to target zone center
        ball_x, ball_y = target_ball.center
        target_x = self.vision.collection_zone['target_center_x']
        target_y = self.vision.collection_zone['target_center_y']
        
        distance_to_target = np.sqrt((ball_x - target_x)**2 + (ball_y - target_y)**2)
        x_error = abs(ball_x - target_x)
        y_error = abs(ball_y - target_y)
        
        # ADAPTIVE STRATEGY based on distance and time
        elapsed_time = time.time() - self.centering_start_time
        self.centering_attempts += 1
        
        # Get movement direction
        x_direction, y_direction = self.vision.get_centering_adjustment_v2(target_ball)
        
        # === STRATEGY 1: BALL IS FAR AWAY (>80 pixels) - AGGRESSIVE APPROACH ===
        if distance_to_target > 80:
            self.logger.info(f"Ball far from target ({distance_to_target:.0f}px) - using aggressive centering")
            movement_duration = config.CENTERING_TURN_DURATION
            movement_speed = config.CENTERING_SPEED
            post_delay = 0.03
            min_x_error = 15
            min_y_error = 12
        
        # === STRATEGY 2: BALL IS CLOSE (40-80 pixels) - BALANCED APPROACH ===
        elif distance_to_target > 40:
            self.logger.info(f"Ball moderately close ({distance_to_target:.0f}px) - using balanced centering")
            movement_duration = config.CENTERING_TURN_DURATION * 0.8
            movement_speed = config.CENTERING_SPEED * 0.9
            post_delay = 0.05
            min_x_error = 10
            min_y_error = 8
        
        # === STRATEGY 3: BALL IS VERY CLOSE (<40 pixels) - PRECISE APPROACH ===
        else:
            self.logger.info(f"Ball very close ({distance_to_target:.0f}px) - using precise centering")
            
            # Apply oscillation detection only for close-range precision work
            if self.direction_change_count >= 3:
                self.logger.warning("OSCILLATION DETECTED in precision mode - using micro movements")
                movement_duration = config.CENTERING_TURN_DURATION * 0.4
                movement_speed = config.CENTERING_SPEED * 0.6
                post_delay = 0.15
                min_x_error = 12
                min_y_error = 10
            else:
                movement_duration = config.CENTERING_TURN_DURATION * 0.7
                movement_speed = config.CENTERING_SPEED * 0.8
                post_delay = 0.08
                min_x_error = 6
                min_y_error = 5

        # === TIMEOUT SYSTEM: GIVE UP ON DIFFICULT BALLS ===
        max_centering_time = 12.0
        max_attempts_without_progress = 30

        if (elapsed_time > max_centering_time or 
            self.centering_attempts > max_attempts_without_progress):
            
            self.logger.warning(f"Centering timeout ({elapsed_time:.1f}s, {self.centering_attempts} attempts)")
            
            # ONLY proceed if ball is actually in target zone
            if self.vision.is_ball_in_target_zone(target_ball.center):
                self.logger.info("✅ Ball IS in target zone despite timeout - proceeding with collection")
                self.state = RobotState.COLLECTING_BALL
                self._reset_centering_state()
                return
            else:
                self.logger.info("❌ Ball NOT in target zone after timeout - abandoning this ball")
                self.state = RobotState.SEARCHING
                self._reset_centering_state()
                return
        
        # === MOVEMENT EXECUTION ===
        movement_made = False
        
        # Track direction changes for oscillation detection
        current_direction = x_direction if x_direction != 'centered' else y_direction
        if (self.last_centering_direction and 
            current_direction != 'centered' and 
            self.last_centering_direction != 'centered' and
            current_direction != self.last_centering_direction):
            self.direction_change_count += 1
        
        # PRIORITIZE X-AXIS (turning) when ball is far
        if x_direction != 'centered' and x_error > min_x_error:
            if x_direction == 'right':
                self.hardware.turn_right(duration=movement_duration, speed=movement_speed)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"Centering: turn right (x_error: {x_error:.0f}px, dist: {distance_to_target:.0f}px)")
            elif x_direction == 'left':
                self.hardware.turn_left(duration=movement_duration, speed=movement_speed)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"Centering: turn left (x_error: {x_error:.0f}px, dist: {distance_to_target:.0f}px)")
            
            movement_made = True
            self.last_centering_direction = x_direction
        
        # Y-AXIS (forward/backward) only if X is reasonably centered
        elif y_direction != 'centered' and y_error > min_y_error:
            if y_direction == 'forward':
                self.hardware.move_forward(duration=config.CENTERING_DRIVE_DURATION, speed=movement_speed)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"Centering: forward (y_error: {y_error:.0f}px, dist: {distance_to_target:.0f}px)")
            elif y_direction == 'backward':
                self.hardware.move_backward(duration=config.CENTERING_DRIVE_DURATION, speed=movement_speed)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"Centering: backward (y_error: {y_error:.0f}px, dist: {distance_to_target:.0f}px)")
            
            movement_made = True
            self.last_centering_direction = y_direction
        
        if movement_made:
            time.sleep(post_delay)
        else:
            if config.DEBUG_MOVEMENT:
                self.logger.info(f"Ball within tolerances (x:{x_error:.0f}, y:{y_error:.0f}, dist:{distance_to_target:.0f}px)")
            time.sleep(0.1)

    def handle_approaching_ball(self, balls, nav_command):
        """Handle approaching ball state (legacy - might not be used)"""
        # This state might not be used in current implementation
        # but keeping for compatibility
        self.state = RobotState.CENTERING_BALL

    def handle_collecting_ball(self):
        """Handle ball collection with PROPER sequence"""
        current_target = self.vision.current_target
        
        if current_target:
            confidence = current_target.confidence
            self.logger.info(f"Starting collection: white ball (confidence: {confidence:.2f})")
        else:
            self.logger.info("Starting collection: white ball")
        
        # Get the fixed drive time from vision system
        drive_time = self.vision.get_drive_time_to_collection()
        self.logger.info(f"Using collection sequence: servo up -> drive {drive_time:.2f}s -> servo down")
        
        # STEP 1: PREPARE SERVOS FOR COLLECTION (SERVO UP)
        self.logger.info("Step 1: Preparing servos for collection (UP)")
        success = self._prepare_servos_for_collection()
        if not success:
            self.logger.warning("Failed to prepare servos - aborting collection")
            self.state = RobotState.SEARCHING
            return
        
        # STEP 2: DRIVE FORWARD TO BALL
        self.logger.info("Step 2: Driving forward to ball")
        self.hardware.move_forward(duration=drive_time, speed=config.COLLECTION_SPEED)
        time.sleep(0.1)
        
        # STEP 3: COMPLETE COLLECTION (SERVO DOWN/GRAB)
        self.logger.info("Step 3: Completing collection (DOWN)")
        success = self._complete_servo_collection()
        
        if success:
            total_balls = self.hardware.get_ball_count()
            self.logger.info(f"✅ White ball collected with proper sequence! Total: {total_balls}")
        else:
            self.logger.warning(f"❌ White ball collection failed")
        
        # Return to searching
        self.state = RobotState.SEARCHING

    def handle_avoiding_boundary(self, near_boundary):
        """Handle boundary avoidance with simplified approach"""
        if near_boundary:
            self.logger.warning("⚠️ Executing boundary avoidance maneuver")
            
            # Get specific avoidance command
            avoidance_command = self.vision.boundary_system.get_avoidance_command(self.vision.last_frame)
            
            # Stop and execute avoidance
            self.hardware.stop_motors()
            time.sleep(0.1)
            
            if avoidance_command == 'move_backward':
                self.hardware.move_backward(duration=0.3)
            elif avoidance_command == 'turn_right':
                self.hardware.turn_right(duration=0.4)
            elif avoidance_command == 'turn_left':
                self.hardware.turn_left(duration=0.4)
            elif avoidance_command == 'backup_and_turn':
                self.logger.warning("CENTER wall detected - backing up and turning")
                self.hardware.move_backward(duration=0.3)
                time.sleep(0.1)
                self.hardware.turn_right(duration=0.6)
            else:
                # Default: back up and turn (fallback)
                self.hardware.move_backward(duration=0.3)
                time.sleep(0.1)
                self.hardware.turn_right(duration=0.4)
            
            time.sleep(0.1)
        else:
            # Clear of boundary - return to ball detection immediately
            if config.DEBUG_MOVEMENT:
                self.logger.info("✅ Clear of boundary - resuming ball detection")
            self.state = RobotState.SEARCHING

    def execute_search_pattern(self):
        """Execute search pattern with better timing"""
        pattern = config.SEARCH_PATTERN
        action = pattern[self.search_pattern_index % len(pattern)]
        
        if action == "forward":
            self.hardware.move_forward(duration=config.FORWARD_TIME_SHORT)
        elif action == "turn_right":
            self.hardware.turn_right(duration=config.TURN_TIME_90_DEGREES)
        elif action == "turn_left":
            self.hardware.turn_left(duration=config.TURN_TIME_90_DEGREES)
        
        self.search_pattern_index += 1
        time.sleep(0.2)

    def _prepare_servos_for_collection(self):
        """Prepare servos for collection - put them in position to catch ball"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Preparing servos for collection...")
            
            self.hardware.servo_sf_to_ready()
            time.sleep(0.2)
            self.hardware.servo_ss_to_pre_collect()
            time.sleep(0.2)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Servos prepared for collection")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare servos for collection: {e}")
            self.hardware.servo_ss_to_driving()
            self.hardware.servo_sf_to_ready()
            return False

    def _complete_servo_collection(self):
        """Complete the servo collection sequence after driving"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🤏 Completing servo collection sequence...")
            
            self.hardware.servo_ss_to_collect()
            time.sleep(0.15)
            self.hardware.servo_sf_to_catch()
            time.sleep(0.3)
            
            self.hardware.servo_ss_to_store()
            time.sleep(0.3)
            
            self.hardware.servo_ss_to_driving()
            time.sleep(0.1)
            self.hardware.servo_sf_to_ready()
            time.sleep(0.2)
            
            # Record collection
            self.hardware.collected_balls.append(time.time())
            
            if config.DEBUG_COLLECTION:
                ss_state = self.hardware.get_servo_ss_state()
                self.logger.info(f"✅ Collection sequence complete! SS state: {ss_state.upper()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Collection sequence failed: {e}")
            self.hardware.stop_motors()
            self.hardware.servo_ss_to_driving()
            self.hardware.servo_sf_to_ready()
            return False

    def _reset_centering_state(self):
        """Reset centering state tracking"""
        self.centering_history = []
        self.centering_stable_count = 0
        self.last_centering_direction = None
        self.direction_change_count = 0
        self.centering_start_time = None
        self.centering_attempts = 0
        self.last_significant_progress = None

    def _is_boundary_critical(self) -> bool:
        """Check if boundary is critically close (imminent collision)"""
        try:
            if hasattr(self.vision.boundary_system, 'get_closest_boundary_distance'):
                min_distance = self.vision.boundary_system.get_closest_boundary_distance()
                return min_distance < 15
            
            if hasattr(self.vision.boundary_system, 'detected_walls'):
                triggered_walls = [w for w in self.vision.boundary_system.detected_walls 
                                 if w.get('triggered', False)]
                return len(triggered_walls) >= 2
                
        except Exception as e:
            self.logger.warning(f"Boundary critical check failed: {e}")
        
        return False

    def emergency_stop(self):
        """Emergency stop the state machine"""
        self.state = RobotState.EMERGENCY_STOP
        self.hardware.emergency_stop()
        self.logger.warning("🛑 State machine emergency stop activated")