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
    FAILED_COLLECTION_RECOVERY = "failed_collection_recovery"
    DELIVERY_MODE = "delivery_mode"
    DELIVERY_ZONE_SEARCH = "delivery_zone_search"
    DELIVERY_ZONE_CENTERING = "delivery_zone_centering"
    DELIVERY_RELEASING = "delivery_releasing"
    POST_DELIVERY_TURN = "post_delivery_turn"
    EMERGENCY_STOP = "emergency_stop"

class RobotStateMachine:
    """Handles all robot state transitions and behavior logic with delivery cycle + failed collection recovery"""
    
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
        
        # Failed collection recovery tracking
        self.failed_collection_attempts = 0
        self.recovery_turn_start_time = None
        self.collection_interrupted_by_boundary = False
        
        # Delivery cycle tracking
        self.delivery_system = None
        self.post_delivery_start_time = None
        self.delivery_release_start_time = None
        self.delivery_search_start_time = None
        
    def execute_state_machine(self, balls, near_boundary, nav_command, delivery_zones=None):
        """Execute state logic with failed collection recovery and boundary avoidance delegation"""
        
        # === HIGHEST PRIORITY: Emergency boundary avoidance (except during non-interruptible states) ===
        non_interruptible_states = [
            RobotState.COLLECTING_BALL, 
            RobotState.DELIVERY_RELEASING,
            RobotState.FAILED_COLLECTION_RECOVERY
        ]
        
        if self.state not in non_interruptible_states:
            # === BOUNDARY AVOIDANCE (using boundary_avoidance.py) ===
            if near_boundary:
                # Get detailed boundary status from the dedicated boundary system
                boundary_status = self.vision.boundary_system.get_status()
                
                if not boundary_status['safe']:
                    self.logger.warning(f"🚨 BOUNDARY AVOIDANCE TRIGGERED during {self.state.value}")
                    self.logger.warning(f"   Danger zones: {boundary_status['danger_zones']}")
                    
                    # Check if this interrupts a collection attempt (centering or collecting)
                    if self.state in [RobotState.COLLECTING_BALL, RobotState.CENTERING_BALL]:
                        self.collection_interrupted_by_boundary = True
                        self.logger.warning("   Collection attempt interrupted by boundary detection!")
                    
                    # Interrupt delivery operations if necessary for safety
                    if self.state in [RobotState.DELIVERY_MODE, RobotState.DELIVERY_ZONE_SEARCH, 
                                    RobotState.DELIVERY_ZONE_CENTERING, RobotState.POST_DELIVERY_TURN]:
                        self.logger.warning("   Interrupting delivery operation for safety!")
                        # Reset delivery state tracking
                        self.delivery_search_start_time = None
                        self.post_delivery_start_time = None
                    
                    # Interrupt centering if necessary for safety
                    if self.state == RobotState.CENTERING_BALL:
                        self.logger.warning("   Interrupting ball centering for safety!")
                        self._reset_centering_state()
                    
                    self.state = RobotState.AVOIDING_BOUNDARY
                    # Execute boundary avoidance immediately
                    self.handle_avoiding_boundary(near_boundary)
                    return
        
        # === STATE-SPECIFIC HANDLING ===
        
        # Check for delivery trigger (except during non-interruptible states)
        if self.state not in non_interruptible_states:
            ball_count = self.hardware.get_ball_count()
            if ball_count >= config.BALLS_BEFORE_DELIVERY and self.state not in [
                RobotState.DELIVERY_MODE, RobotState.DELIVERY_ZONE_SEARCH, 
                RobotState.DELIVERY_ZONE_CENTERING, RobotState.DELIVERY_RELEASING, 
                RobotState.POST_DELIVERY_TURN, RobotState.FAILED_COLLECTION_RECOVERY
            ]:
                self.logger.info(f"🚚 DELIVERY TRIGGERED: {ball_count}/{config.BALLS_BEFORE_DELIVERY} balls collected")
                self.state = RobotState.DELIVERY_MODE
                return
        
        # Execute current state
        if self.state == RobotState.SEARCHING:
            self.handle_searching(balls, nav_command)
            
        elif self.state == RobotState.CENTERING_BALL:
            # Extra safety check - abort centering if boundary detected
            if near_boundary:
                self.logger.warning("🚨 Boundary detected during centering - aborting centering!")
                self._reset_centering_state()
                self.state = RobotState.AVOIDING_BOUNDARY
                self.handle_avoiding_boundary(near_boundary)
                return
                
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
            
        elif self.state == RobotState.COLLECTING_BALL:
            # This state is non-interruptible - no boundary checking
            self.handle_collecting_ball()
            
        elif self.state == RobotState.AVOIDING_BOUNDARY:
            # Handle boundary avoidance using the dedicated system
            self.handle_avoiding_boundary(near_boundary)
            
        elif self.state == RobotState.FAILED_COLLECTION_RECOVERY:
            # Handle failed collection recovery turn
            self.handle_failed_collection_recovery()
            
        elif self.state == RobotState.DELIVERY_MODE:
            self.handle_delivery_mode(delivery_zones)
            
        elif self.state == RobotState.DELIVERY_ZONE_SEARCH:
            self.handle_delivery_zone_search(delivery_zones)
            
        elif self.state == RobotState.DELIVERY_ZONE_CENTERING:
            self.handle_delivery_zone_centering(delivery_zones)
            
        elif self.state == RobotState.DELIVERY_RELEASING:
            # This state is non-interruptible - no boundary checking
            self.handle_delivery_releasing()
            
        elif self.state == RobotState.POST_DELIVERY_TURN:
            self.handle_post_delivery_turn()
            
        elif self.state == RobotState.EMERGENCY_STOP:
            self.hardware.emergency_stop()

    def handle_avoiding_boundary(self, near_boundary):
        """UPDATED: Use the dedicated boundary system with failed collection recovery"""
        
        # Get avoidance command from the dedicated boundary system
        avoidance_command = self.vision.boundary_system.get_avoidance_command(self.vision.last_frame)
        
        if avoidance_command:
            self.logger.warning(f"🚨 EXECUTING BOUNDARY AVOIDANCE: {avoidance_command}")
            
            # Stop current movement
            self.hardware.stop_motors()
            time.sleep(0.1)
            
            # Execute the command from boundary system using config values
            try:
                if avoidance_command == 'turn_right':
                    self.logger.warning("↗️ Boundary avoidance: Turn right")
                    self.hardware.turn_right(
                        duration=config.BOUNDARY_TURN_DURATION, 
                        speed=config.BOUNDARY_TURN_SPEED
                    )
                elif avoidance_command == 'turn_left':
                    self.logger.warning("↖️ Boundary avoidance: Turn left")
                    self.hardware.turn_left(
                        duration=config.BOUNDARY_TURN_DURATION, 
                        speed=config.BOUNDARY_TURN_SPEED
                    )
                elif avoidance_command == 'move_backward':
                    self.logger.warning("⬇️ Boundary avoidance: Move backward")
                    self.hardware.move_backward(
                        duration=config.BOUNDARY_BACKUP_DURATION, 
                        speed=config.BOUNDARY_BACKUP_SPEED
                    )
                elif avoidance_command == 'backup_and_turn':
                    self.logger.warning("🚨 Boundary avoidance: Backup and turn")
                    self.hardware.move_backward(
                        duration=config.BOUNDARY_BACKUP_DURATION, 
                        speed=config.BOUNDARY_BACKUP_SPEED
                    )
                    time.sleep(0.1)
                    self.hardware.turn_right(
                        duration=config.BOUNDARY_COMPOUND_TURN_DURATION, 
                        speed=config.BOUNDARY_TURN_SPEED
                    )
                elif avoidance_command == 'emergency_backup':
                    # NEW: Emergency backup for immediate danger
                    self.logger.warning("🚨🚨 EMERGENCY BACKUP: Immediate front danger!")
                    self.hardware.move_backward(
                        duration=config.BOUNDARY_BACKUP_DURATION * 1.5,
                        speed=config.BOUNDARY_BACKUP_SPEED
                    )
                    time.sleep(0.1)
                    # Follow with a turn to get away from the wall
                    self.hardware.turn_right(
                        duration=config.BOUNDARY_COMPOUND_TURN_DURATION * 1.2,
                        speed=config.BOUNDARY_TURN_SPEED
                    )
                else:
                    # Default fallback
                    self.logger.warning("🚨 Boundary avoidance: Default backup and turn")
                    self.hardware.move_backward(
                        duration=config.BOUNDARY_BACKUP_DURATION, 
                        speed=config.BOUNDARY_BACKUP_SPEED
                    )
                    time.sleep(0.1)
                    self.hardware.turn_right(
                        duration=config.BOUNDARY_TURN_DURATION, 
                        speed=config.BOUNDARY_TURN_SPEED
                    )
                
                # Allow movement to complete
                time.sleep(0.2)
                
            except Exception as e:
                self.logger.error(f"❌ Boundary avoidance execution failed: {e}")
                # Emergency fallback using config
                try:
                    self.hardware.move_backward(duration=config.BOUNDARY_BACKUP_DURATION, speed=0.4)
                    self.hardware.turn_right(duration=config.BOUNDARY_TURN_DURATION, speed=0.4)
                except:
                    self.logger.error("❌ Emergency fallback also failed!")
        
        # Check if we're clear using the boundary system
        ret, frame = self.vision.get_frame()
        if ret:
            # Fresh boundary check using the dedicated system
            still_near_boundary = self.vision.boundary_system.detect_boundaries(frame)
            boundary_status = self.vision.boundary_system.get_status()
            
            if boundary_status['safe'] and not still_near_boundary:
                self.logger.info("✅ Boundary avoidance successful")
                
                # Check if this was due to failed collection attempt
                if self.collection_interrupted_by_boundary:
                    self._handle_failed_collection_attempt()
                    self.collection_interrupted_by_boundary = False
                else:
                    # Normal boundary avoidance - return to search
                    self.state = RobotState.SEARCHING
            else:
                # Still detecting boundaries - log details and continue avoidance
                self.logger.warning("⚠️ Still detecting boundaries after avoidance")
                self.logger.warning(f"⚠️ Remaining danger zones: {boundary_status.get('danger_zones', [])}")
                # Stay in AVOIDING_BOUNDARY state for next iteration
        else:
            # Can't get frame - assume we're clear
            self.logger.warning("⚠️ Can't verify boundary status - assuming clear")
            if self.collection_interrupted_by_boundary:
                self._handle_failed_collection_attempt()
                self.collection_interrupted_by_boundary = False
            else:
                self.state = RobotState.SEARCHING

    def _handle_failed_collection_attempt(self):
        """Handle failed collection attempt and track for recovery"""
        # Simply increment counter for any failed collection
        self.failed_collection_attempts += 1
        self.logger.warning(f"🔄 Failed collection attempt #{self.failed_collection_attempts}")
        
        # Check if we need recovery turn
        if self.failed_collection_attempts >= config.FAILED_COLLECTION_MAX_ATTEMPTS:
            self.logger.warning(f"🔄 MAX FAILED ATTEMPTS REACHED ({self.failed_collection_attempts}) - Starting recovery turn")
            self.state = RobotState.FAILED_COLLECTION_RECOVERY
            self.recovery_turn_start_time = None  # Will be set when recovery actually starts
            return
        
        # Normal return to search if not triggering recovery
        self.state = RobotState.SEARCHING

    def handle_failed_collection_recovery(self):
        """Handle recovery turn after failed collection attempts - FIXED"""
        if self.recovery_turn_start_time is None:
            # Start recovery turn - NON-BLOCKING
            self.logger.warning(f"🔄 STARTING RECOVERY TURN: Right for {config.FAILED_COLLECTION_RECOVERY_TURN_DURATION}s")
            self.hardware.turn_right(speed=0.4)  # Start turning right (no duration = non-blocking)
            self.recovery_turn_start_time = time.time()
            return
        
        # Check if recovery turn is complete
        elapsed = time.time() - self.recovery_turn_start_time
        if elapsed >= config.FAILED_COLLECTION_RECOVERY_TURN_DURATION:
            # Stop the turn and complete recovery
            self.hardware.stop_motors()
            self.logger.info("✅ Recovery turn complete - resetting failed collection tracking")
            
            # Reset failed collection tracking
            self.failed_collection_attempts = 0
            self.recovery_turn_start_time = None
            
            # Return to normal search
            self.state = RobotState.SEARCHING

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
        """Handle ball collection using the proper collection system - SS only"""
        current_target = self.vision.current_target
        
        if current_target:
            confidence = current_target.confidence
            self.logger.info(f"Starting collection: white ball (confidence: {confidence:.2f})")
        else:
            self.logger.info("Starting collection: white ball")
        
        ball_count_before = self.hardware.get_ball_count()
        self.logger.info(f"Ball count before collection: {ball_count_before}")
        
        success = self.hardware.enhanced_collection_sequence()
        
        ball_count_after = self.hardware.get_ball_count()
        self.logger.info(f"Ball count after collection: {ball_count_after}")
        
        if success:
            self.logger.info(f"✅ White ball collected! Total: {ball_count_after}/{config.BALLS_BEFORE_DELIVERY}")
            # Reset failed collection tracking on successful collection
            self.failed_collection_attempts = 0
        else:
            self.logger.warning(f"❌ White ball collection failed")
        
        # Check delivery trigger
        current_ball_count = self.hardware.get_ball_count()
        delivery_target = config.BALLS_BEFORE_DELIVERY
        self.logger.info(f"Checking delivery trigger: {current_ball_count} >= {delivery_target}?")
        
        if current_ball_count >= delivery_target:
            self.logger.info(f"🚚 DELIVERY THRESHOLD REACHED: {current_ball_count}/{delivery_target}")
            self.state = RobotState.DELIVERY_MODE
        else:
            # Continue searching for more balls
            self.logger.info(f"Need more balls: {current_ball_count}/{delivery_target} - continuing search")
            self.state = RobotState.SEARCHING

    def handle_delivery_mode(self, delivery_zones=None):
        """Start delivery sequence - look for green zones"""
        self.logger.info("🚚 DELIVERY MODE: Looking for green delivery zones")
        self.state = RobotState.DELIVERY_ZONE_SEARCH
        self.delivery_search_start_time = time.time()
    
    def handle_delivery_zone_search(self, delivery_zones=None):
        """Search for green delivery zones - NOW WITH BOUNDARY SAFETY"""
        if delivery_zones and len(delivery_zones) > 0:
            # Found delivery zone(s)
            target_zone = self.vision.get_target_delivery_zone(delivery_zones)
            if target_zone:
                self.logger.info(f"🎯 Found delivery zone at {target_zone.center}, centering on it")
                self.state = RobotState.DELIVERY_ZONE_CENTERING
                return
        
        # No zones found - keep searching
        elapsed_search = time.time() - self.delivery_search_start_time if self.delivery_search_start_time else 0
        
        if elapsed_search > 60.0:  # Search timeout
            self.logger.warning("⏰ Delivery zone search timeout - delivering in place")
            self.state = RobotState.DELIVERY_RELEASING
            self.delivery_release_start_time = time.time()
            return
        
        # Continue searching pattern - BUT NOW WITH BOUNDARY AWARENESS!
        if config.DEBUG_MOVEMENT:
            self.logger.info("🔍 Searching for green delivery zones... (with boundary safety)")
        
        # Simple search pattern - boundary avoidance will interrupt if needed
        self.hardware.turn_right(duration=1, speed=0.4)
        time.sleep(0.2)
    
    def handle_delivery_zone_centering(self, delivery_zones=None):
        """UPDATED: Simple horizontal centering on delivery zone"""
        if not delivery_zones:
            self.logger.warning("Lost delivery zone during centering - searching again")
            self.state = RobotState.DELIVERY_ZONE_SEARCH
            self.delivery_search_start_time = time.time()
            return
        
        target_zone = self.vision.get_target_delivery_zone(delivery_zones)
        if not target_zone:
            self.logger.warning("No target delivery zone - searching again")
            self.state = RobotState.DELIVERY_ZONE_SEARCH
            self.delivery_search_start_time = time.time()
            return
        
        # Get horizontal centering command from vision system
        centering_command = self.vision.get_delivery_zone_centering_command(target_zone)
        
        # Calculate position info for logging
        zone_x, zone_y = target_zone.center
        x_offset = zone_x - self.vision.frame_center_x
        
        self.logger.info(f"🎯 Delivery zone horizontal centering: X_offset={x_offset:.0f}px, Command={centering_command}")
        
        # Check if horizontally centered
        if centering_command == "centered":
            self.logger.info("✅ Delivery zone HORIZONTALLY CENTERED - starting ball release!")
            self.state = RobotState.DELIVERY_RELEASING
            self.delivery_release_start_time = time.time()
            # Clean up centering timer if it exists
            if hasattr(self, 'delivery_centering_start_time'):
                delattr(self, 'delivery_centering_start_time')
            return
        
        # Execute centering movement
        movement_made = False
        
        if centering_command == "turn_right":
            self.hardware.turn_right(duration=0.3, speed=0.4)
            self.logger.info("↗️ Centering: Turn right toward delivery zone")
            movement_made = True
        elif centering_command == "turn_left":
            self.hardware.turn_left(duration=0.3, speed=0.4)
            self.logger.info("↖️ Centering: Turn left toward delivery zone")
            movement_made = True
        
        # Add timeout protection
        if not hasattr(self, 'delivery_centering_start_time'):
            self.delivery_centering_start_time = time.time()
        
        centering_elapsed = time.time() - self.delivery_centering_start_time
        if centering_elapsed > 30.0:  # 30 second timeout
            self.logger.warning("⏰ Delivery centering timeout - proceeding with release anyway")
            self.state = RobotState.DELIVERY_RELEASING
            self.delivery_release_start_time = time.time()
            delattr(self, 'delivery_centering_start_time')
            return
        
        if movement_made:
            time.sleep(0.1)  # Brief pause after movement
        else:
            # Shouldn't reach here, but safety fallback
            self.logger.info("✅ Delivery zone centered (fallback) - starting release")
            self.state = RobotState.DELIVERY_RELEASING
            self.delivery_release_start_time = time.time()
            if hasattr(self, 'delivery_centering_start_time'):
                delattr(self, 'delivery_centering_start_time')
    
    def handle_delivery_releasing(self):
        """Handle ball release sequence with timing"""
        if not hasattr(self, '_delivery_release_started'):
            self.logger.info("📦 STARTING BALL RELEASE SEQUENCE")
            
            # Set SS to STORE position and open SF
            self.hardware.servo_ss_to_store()
            time.sleep(0.3)
            self.hardware.servo_sf_to_open()
            
            self.logger.info(f"🔓 SF door opened for {config.DELIVERY_DOOR_OPEN_TIME} seconds")
            self._delivery_release_started = True
            self.delivery_release_start_time = time.time()
        
        # Check if door open time has elapsed
        elapsed = time.time() - self.delivery_release_start_time
        if elapsed >= config.DELIVERY_DOOR_OPEN_TIME:
            # Close door and finish release
            self.hardware.servo_sf_to_closed()
            self.logger.info("🔒 SF door closed - delivery complete")
            
            # Clear ball count
            balls_released = len(self.hardware.collected_balls)
            self.hardware.collected_balls.clear()
            self.logger.info(f"✅ Released {balls_released} balls")
            
            # Move to post-delivery turn
            self.state = RobotState.POST_DELIVERY_TURN
            self.post_delivery_start_time = time.time()
            del self._delivery_release_started
    
    def handle_post_delivery_turn(self):
        """Handle post-delivery turn and restart collection cycle"""
        if self.post_delivery_start_time is None:
            self.post_delivery_start_time = time.time()
            self.logger.info(f"🔄 POST-DELIVERY: Turning right for {config.POST_DELIVERY_TURN_DURATION}s")
            self.hardware.turn_right(duration=config.POST_DELIVERY_TURN_DURATION, speed=0.5)
        
        elapsed = time.time() - self.post_delivery_start_time
        if elapsed >= config.POST_DELIVERY_TURN_DURATION + 0.5:
            # Set servos back to competition positions
            self.hardware.servo_ss_to_driving()
            self.hardware.servo_sf_to_closed()
            
            self.logger.info("🔄 POST-DELIVERY COMPLETE: Restarting collection cycle")
            self.state = RobotState.SEARCHING
            self.post_delivery_start_time = None
            self.search_pattern_index = 0  # Reset search pattern

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

    def _reset_centering_state(self):
        """Reset centering state tracking"""
        self.centering_history = []
        self.centering_stable_count = 0
        self.last_centering_direction = None
        self.direction_change_count = 0
        self.centering_start_time = None
        self.centering_attempts = 0
        self.last_significant_progress = None

    def emergency_stop(self):
        """Emergency stop the state machine"""
        self.state = RobotState.EMERGENCY_STOP
        self.hardware.emergency_stop()
        self.logger.warning("🛑 State machine emergency stop activated")