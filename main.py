import time
import logging
import cv2
import signal
import sys
import os
from enum import Enum
from typing import Optional
import config
from hardware import GolfBotHardware
from vision import VisionSystem
from boundary_avoidance import BoundaryAvoidanceSystem
from telemetry import TelemetryLogger
from hardware_test import run_hardware_test

# Import the new dashboard (optional)
try:
    from dashboard import GolfBotDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    logging.warning("Dashboard not available - using legacy overlay mode")

class RobotState(Enum):
    SEARCHING = "searching"
    CENTERING_BALL = "centering_ball"  # Enhanced: Center ball X+Y before collection
    APPROACHING_BALL = "approaching_ball"
    COLLECTING_BALL = "collecting_ball"  # Enhanced: New servo sequence
    AVOIDING_BOUNDARY = "avoiding_boundary"
    EMERGENCY_STOP = "emergency_stop"

class GolfBot:
    def __init__(self, use_dashboard=True):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Check if display is available
        self.display_available = self.check_display_available()
        if not self.display_available:
            self.logger.info("No display detected - running in headless mode")
        
        # Dashboard mode
        self.use_dashboard = use_dashboard and DASHBOARD_AVAILABLE and self.display_available
        if self.use_dashboard:
            self.dashboard = GolfBotDashboard()
            self.logger.info("Using new dashboard interface")
        else:
            self.dashboard = None
            self.logger.info("Using legacy overlay interface")
        
        # Initialize systems
        # self.telemetry = TelemetryLogger()  # DISABLED for now
        self.telemetry = None
        self.hardware = GolfBotHardware()
        self.vision = VisionSystem()
        
        # Competition state
        self.start_time = None
        self.competition_active = False
        self.state = RobotState.SEARCHING
        self.search_pattern_index = 0
        self.last_ball_seen_time = None
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_skip_counter = 0  # Skip frames for performance
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def check_display_available(self):
        """Check if display/X11 is available"""
        try:
            if os.environ.get('DISPLAY') is None:
                return False
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('test')
            return True
        except Exception as e:
            return False
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('golfbot.log'),
                logging.StreamHandler()
            ]
        )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.emergency_stop()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Initialize all systems"""
        self.logger.info("Initializing GolfBot with enhanced collection system (WHITE BALLS ONLY)...")
        
        try:
            # Start vision system
            if not self.vision.start():
                self.logger.error("Failed to initialize vision system")
                return False
            
            # Let vision system detect arena boundaries on startup
            self.logger.info("Detecting arena boundaries...")
            ret, frame = self.vision.get_frame()
            if ret:
                self.vision.boundary_system.detect_arena_boundaries(frame)
                if self.vision.boundary_system.arena_detected:
                    self.logger.info("✅ Arena boundaries detected successfully")
                else:
                    self.logger.info("⚠️  Using fallback arena boundaries")
            
            self.logger.info("All systems initialized successfully - WHITE BALLS ONLY")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def start_competition(self):
        """Start the competition timer and main loop"""
        self.start_time = time.time()
        self.competition_active = True
        self.state = RobotState.SEARCHING
        
        self.logger.info("COMPETITION STARTED - WHITE BALLS ONLY!")
        self.logger.info(f"Time limit: {config.COMPETITION_TIME} seconds")
        self.logger.info("Using enhanced collection: Ball centering (X+Y) + Enhanced sequence")
        
        try:
            self.main_loop()
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.emergency_stop()
    
    def get_time_remaining(self) -> float:
        """Get remaining competition time"""
        if not self.start_time:
            return config.COMPETITION_TIME
        elapsed = time.time() - self.start_time
        return max(0, config.COMPETITION_TIME - elapsed)
    
    def is_time_up(self) -> bool:
        """Check if competition time is up"""
        return self.get_time_remaining() <= 0
    
    def main_loop(self):
        """Main competition control loop with enhanced collection - WHITE BALLS ONLY"""
        while self.competition_active and not self.is_time_up():
            try:
                frame_start = time.time()
                
                # Skip frames for performance (process every 2nd frame)
                self.frame_skip_counter += 1
                if self.frame_skip_counter % 2 != 0:
                    time.sleep(0.05)
                    continue
                
                # Get current vision data
                balls, _, near_boundary, nav_command, debug_frame = self.vision.process_frame(dashboard_mode=self.use_dashboard)
                
                if balls is None:  # Frame capture failed
                    continue
                
                # Store detected balls for dashboard access
                self.vision._last_detected_balls = balls if balls else []
                
                # Update ball tracking
                if balls:
                    self.last_ball_seen_time = time.time()
                    high_confidence_balls = [b for b in balls if b.confidence > 0.5]
                    if high_confidence_balls:
                        self.logger.debug(f"High confidence white balls: {len(high_confidence_balls)}")
                
                # Performance tracking
                frame_time = time.time() - frame_start
                fps = 1.0 / (time.time() - self.last_frame_time) if self.last_frame_time else 0
                self.last_frame_time = time.time()
                
                # Show display based on mode
                if config.SHOW_CAMERA_FEED and self.display_available:
                    try:
                        if self.use_dashboard and self.dashboard:
                            # NEW DASHBOARD MODE
                            dashboard_frame = self.dashboard.create_dashboard(
                                debug_frame, self.state, self.vision, self.hardware, None  # No telemetry
                            )
                            key = self.dashboard.show("GolfBot Dashboard - White Ball Collection")
                        else:
                            # LEGACY OVERLAY MODE  
                            if debug_frame is not None and debug_frame.size > 0:
                                self.add_status_overlay(debug_frame)
                                cv2.imshow('GolfBot Debug - White Ball Collection', debug_frame)
                                key = cv2.waitKey(1) & 0xFF
                            else:
                                key = -1
                        
                        if key == ord('q'):
                            break
                            
                    except Exception as e:
                        self.logger.warning(f"Display error: {e}")
                        self.display_available = False
                
                # State machine execution
                old_state = self.state
                self.execute_state_machine(balls, near_boundary, nav_command)
                
                # Adaptive sleep based on detection results and state
                if self.state == RobotState.CENTERING_BALL:
                    time.sleep(0.03)  # Faster when centering
                elif balls and len(balls) > 0:
                    time.sleep(0.05)  # Faster when balls detected
                else:
                    time.sleep(0.1)   # Slower when searching
                
            except Exception as e:
                self.logger.error(f"Main loop iteration error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
        
        self.end_competition()
    
    def execute_state_machine(self, balls, near_boundary, nav_command):
        """Execute state logic with ball detection priority - WHITE BALLS ONLY"""
        
        # HIGH PRIORITY: Never interrupt active collection
        if self.state == RobotState.COLLECTING_BALL:
            self.handle_collecting_ball()
            return
        
        # MEDIUM PRIORITY: Protect ball operations if we have good targets
        confident_balls = [ball for ball in balls if ball.confidence > 0.4] if balls else []
        has_good_target = bool(confident_balls)
        
        # SMART BOUNDARY AVOIDANCE: Only interrupt ball operations if:
        # 1. No confident balls detected, OR
        # 2. Currently just searching (not actively pursuing), OR
        # 3. Boundary is critically close (imminent collision)
        should_avoid_boundary = (
            near_boundary and (
                not has_good_target or 
                self.state == RobotState.SEARCHING or
                self._is_boundary_critical()
            )
        )
        
        if should_avoid_boundary:
            # Log why we're switching to boundary avoidance
            reason = "no_balls" if not has_good_target else "searching" if self.state == RobotState.SEARCHING else "critical_proximity"
            if config.DEBUG_MOVEMENT:
                self.logger.info(f"⚠️ Boundary avoidance triggered: {reason}")
            self.state = RobotState.AVOIDING_BOUNDARY
        
        # Execute current state
        if self.state == RobotState.SEARCHING:
            self.handle_searching(balls, nav_command)
            
        elif self.state == RobotState.CENTERING_BALL:  # Enhanced with X+Y centering
            self.handle_centering_ball(balls, nav_command)
            
        elif self.state == RobotState.APPROACHING_BALL:
            self.handle_approaching_ball(balls, nav_command)
            
        elif self.state == RobotState.AVOIDING_BOUNDARY:
            self.handle_avoiding_boundary(near_boundary)
            
        elif self.state == RobotState.EMERGENCY_STOP:
            self.hardware.emergency_stop()

    def _is_boundary_critical(self) -> bool:
        """Check if boundary is critically close (imminent collision)"""
        try:
            # Check if boundary system has distance measurement
            if hasattr(self.vision.boundary_system, 'get_closest_boundary_distance'):
                min_distance = self.vision.boundary_system.get_closest_boundary_distance()
                return min_distance < 15  # Very close - immediate danger
            
            # Fallback: check if multiple walls are triggered
            if hasattr(self.vision.boundary_system, 'detected_walls'):
                triggered_walls = [w for w in self.vision.boundary_system.detected_walls 
                                 if w.get('triggered', False)]
                return len(triggered_walls) >= 2  # Multiple walls = corner/tight spot
                
        except Exception as e:
            self.logger.warning(f"Boundary critical check failed: {e}")
        
        return False  # Default to not critical
    
    def handle_searching(self, balls, nav_command):
        """Handle searching with centering requirement - WHITE BALLS ONLY"""
        if balls:
            # Filter for high confidence balls only
            confident_balls = [ball for ball in balls if ball.confidence > 0.4]
            
            if confident_balls:
                ball_count = len(confident_balls)
                avg_confidence = sum(ball.confidence for ball in confident_balls) / ball_count
                
                self.logger.info(f"Found {ball_count} confident white ball(s) (avg conf: {avg_confidence:.2f})")
                self.state = RobotState.CENTERING_BALL  # NEW: Go to centering first
                return
        
        # No confident balls found
        self.execute_search_pattern()
    
    def handle_centering_ball(self, balls, nav_command):
        """SIMPLIFIED: Get ball into precise target zone in center of screen"""
        if not balls:
            self.logger.info("Lost sight of ball during centering - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        # Filter for confident balls
        confident_balls = [ball for ball in balls if ball.confidence > 0.3]
        
        if not confident_balls:
            self.logger.info("No confident ball detections during centering - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        # Target the closest confident ball
        target_ball = confident_balls[0]
        
        # Check if ball is in the precise target zone
        if self.vision.is_ball_centered_for_collection(target_ball):
            # Ball is in target zone - start collection!
            self.logger.info(f"White ball in TARGET ZONE! Starting collection with fixed drive")
            self.state = RobotState.COLLECTING_BALL
            return
        
        # Ball not in target zone - get movement to position it
        x_direction, y_direction = self.vision.get_centering_adjustment_v2(target_ball)
        
        # Move ball toward target zone - prioritize the axis that's furthest off
        if x_direction != 'centered':
            if x_direction == 'right':
                self.hardware.turn_right(duration=config.CENTERING_TURN_DURATION, 
                                        speed=config.CENTERING_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"Positioning to target zone: turning right")
            elif x_direction == 'left':
                self.hardware.turn_left(duration=config.CENTERING_TURN_DURATION, 
                                    speed=config.CENTERING_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"Positioning to target zone: turning left")
            
            time.sleep(0.03)
            return
        
        # X is centered, now work on Y
        if y_direction != 'centered':
            if y_direction == 'forward':
                self.hardware.move_forward(duration=config.CENTERING_DRIVE_DURATION, 
                                        speed=config.CENTERING_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"Positioning to target zone: moving forward")
            elif y_direction == 'backward':
                self.hardware.move_backward(duration=config.CENTERING_DRIVE_DURATION, 
                                        speed=config.CENTERING_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"Positioning to target zone: moving backward")
            
            time.sleep(0.03)
            return
        
        if config.DEBUG_MOVEMENT:
            self.logger.info("Ball should be in target zone now!")
    
    def handle_approaching_ball(self, balls, nav_command):
        """Handle approaching with confidence tracking (legacy mode) - WHITE BALLS ONLY"""
        if not balls:
            self.logger.info("Lost sight of all white balls - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        # Filter for confident balls
        confident_balls = [ball for ball in balls if ball.confidence > 0.4]
        
        if not confident_balls:
            self.logger.info("No confident white ball detections - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        # Target the closest confident ball
        target_ball = confident_balls[0]
        
        if target_ball.in_collection_zone:
            self.logger.info(f"White ball in collection zone - attempting collection (confidence: {target_ball.confidence:.2f})")
            self.state = RobotState.COLLECTING_BALL
            return
        
        # Navigate toward ball
        self.execute_navigation_command(nav_command)
    
    def handle_collecting_ball(self):
        """Handle ball collection with PROPER sequence: servo up -> drive -> servo down"""
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
        success = self.prepare_servos_for_collection()
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
        success = self.complete_servo_collection()
        
        if success:
            total_balls = self.hardware.get_ball_count()
            self.logger.info(f"✅ White ball collected with proper sequence! Total: {total_balls}")
        else:
            self.logger.warning(f"❌ White ball collection failed")
        
        # Return to searching
        self.state = RobotState.SEARCHING

    def execute_servo_collection_only(self):
        """Execute just the servo collection sequence without driving"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Executing servo-only collection sequence...")
            
            # Step 1: Prepare SF for catching
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 1: Preparing SF for catching")
            self.hardware.servo_sf_to_ready()
            time.sleep(0.2)
            
            # Step 2: Move SS from driving to pre-collect position
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 2: Moving SS from DRIVING to PRE-COLLECT")
            self.hardware.servo_ss_to_pre_collect()
            time.sleep(0.2)
            
            # Step 3: Coordinate collection - SS captures, SF assists
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 3: Coordinated collection - SS collect, SF catch")
            self.hardware.servo_ss_to_collect()
            time.sleep(0.15)
            self.hardware.servo_sf_to_catch()
            time.sleep(0.3)
            
            # Step 4: Move SS to store position (secure ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 4: Moving SS to STORE position (secure)")
            self.hardware.servo_ss_to_store()
            time.sleep(0.3)
            
            # Step 5: Return both servos to ready positions
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 5: Returning servos to ready positions")
            self.hardware.servo_ss_to_driving()
            time.sleep(0.1)
            self.hardware.servo_sf_to_ready()
            time.sleep(0.2)
            
            # Record collection
            self.hardware.collected_balls.append(time.time())
            
            if config.DEBUG_COLLECTION:
                ss_state = self.hardware.get_servo_ss_state()
                self.logger.info(f"✅ Servo collection complete! SS state: {ss_state.upper()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Servo collection sequence failed: {e}")
            self.hardware.stop_motors()
            # Ensure we return to ready positions on error
            self.hardware.servo_ss_to_driving()
            self.hardware.servo_sf_to_ready()
            return False
        
    def prepare_servos_for_collection(self):
        """Prepare servos for collection - put them in position to catch ball"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Preparing servos for collection...")
            
            # Step 1: Prepare SF for catching
            if config.DEBUG_COLLECTION:
                self.logger.info("Preparing SF for catching")
            self.hardware.servo_sf_to_ready()
            time.sleep(0.2)
            
            # Step 2: Move SS from driving to pre-collect position
            if config.DEBUG_COLLECTION:
                self.logger.info("Moving SS from DRIVING to PRE-COLLECT")
            self.hardware.servo_ss_to_pre_collect()
            time.sleep(0.2)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Servos prepared for collection")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare servos for collection: {e}")
            # Ensure we return to safe positions on error
            self.hardware.servo_ss_to_driving()
            self.hardware.servo_sf_to_ready()
            return False
        
    def complete_servo_collection(self):
        """Complete the servo collection sequence after driving"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🤏 Completing servo collection sequence...")
            
            # Step 1: Coordinate collection - SS captures, SF assists
            if config.DEBUG_COLLECTION:
                self.logger.info("Coordinated collection - SS collect, SF catch")
            self.hardware.servo_ss_to_collect()
            time.sleep(0.15)
            self.hardware.servo_sf_to_catch()
            time.sleep(0.3)
            
            # Step 2: Move SS to store position (secure ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Moving SS to STORE position (secure)")
            self.hardware.servo_ss_to_store()
            time.sleep(0.3)
            
            # Step 3: Return both servos to ready positions
            if config.DEBUG_COLLECTION:
                self.logger.info("Returning servos to ready positions")
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
            # Ensure we return to ready positions on error
            self.hardware.servo_ss_to_driving()
            self.hardware.servo_sf_to_ready()
            return False
    
    def handle_avoiding_boundary(self, near_boundary):
        """Handle boundary avoidance with faster return to ball detection"""
        if near_boundary:
            self.logger.warning("⚠️ Executing boundary avoidance maneuver")
            
            # Get specific avoidance command
            avoidance_command = self.vision.boundary_system.get_avoidance_command(self.vision.last_frame)
            
            # Stop and execute avoidance
            self.hardware.stop_motors()
            time.sleep(0.1)
            
            if avoidance_command == 'move_backward':
                self.hardware.move_backward(duration=0.3)  # Shorter duration
            elif avoidance_command == 'turn_right':
                self.hardware.turn_right(duration=0.4)     # Shorter duration
            elif avoidance_command == 'turn_left':
                self.hardware.turn_left(duration=0.4)      # Shorter duration
            else:
                # Default: back up and turn
                self.hardware.move_backward(duration=0.2)
                self.hardware.turn_right(duration=0.4)
            
            time.sleep(0.1)  # Shorter pause
        else:
            # Clear of boundary - return to ball detection immediately
            if config.DEBUG_MOVEMENT:
                self.logger.info("✅ Clear of boundary - resuming ball detection")
            self.state = RobotState.SEARCHING
    
    def execute_navigation_command(self, command):
        """Execute navigation with improved timing"""
        if command == "forward":
            self.hardware.move_forward(duration=0.50)  # Shorter movements
        elif command == "turn_left":
            self.hardware.turn_left(duration=0.15)
        elif command == "turn_right":
            self.hardware.turn_right(duration=0.15)
        elif command == "collect_ball":
            self.state = RobotState.COLLECTING_BALL
        else:
            self.execute_search_pattern()
    
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
        time.sleep(0.2)  # Shorter pause
    
    def add_status_overlay(self, frame):
        """LEGACY: Enhanced status overlay with new collection states - WHITE BALLS ONLY"""
        y = 30
        line_height = 25
        
        # Time remaining
        time_remaining = self.get_time_remaining()
        cv2.putText(frame, f"Time: {time_remaining:.0f}s", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Current state with enhanced info
        state_text = f"State: {self.state.value}"
        if self.state == RobotState.COLLECTING_BALL:
            state_text += " (Enhanced)"
        elif self.state == RobotState.CENTERING_BALL and self.vision.current_target:
            x_dir, y_dir = self.vision.get_centering_adjustment(self.vision.current_target)
            centered = self.vision.is_ball_centered(self.vision.current_target)
            state_text += f" ({'✓' if centered else f'{x_dir[:1].upper()}{y_dir[:1].upper()}'})"
        
        cv2.putText(frame, state_text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Ball count
        ball_count = self.hardware.get_ball_count()
        cv2.putText(frame, f"White Balls Collected: {ball_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Vision status
        arena_status = "Detected" if self.vision.arena_detected else "Fallback"
        cv2.putText(frame, f"Arena: {arena_status}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += line_height
        
        # Current target info with centering status
        if self.vision.current_target:
            target = self.vision.current_target
            centered = self.vision.is_ball_centered(target)
            center_status = "CENTERED" if centered else "CENTERING"
            target_info = f"Target: WHITE ({center_status})"
            
            color = (0, 255, 0) if centered else (0, 255, 255)
            cv2.putText(frame, target_info, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y += line_height - 5
            
            # Show drive time if centered
            if centered:
                drive_time = self.vision.calculate_drive_time_to_ball(target)
                cv2.putText(frame, f"Drive Time: {drive_time:.2f}s", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def end_competition(self):
        """End competition with enhanced results - WHITE BALLS ONLY"""
        self.competition_active = False
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("=" * 60)
        self.logger.info("COMPETITION ENDED - WHITE BALLS ONLY!")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {elapsed_time:.1f} seconds")
        self.logger.info(f"White balls collected: {self.hardware.get_ball_count()}")
        self.logger.info(f"Final state: {self.state.value}")
        self.logger.info(f"Collection system: Enhanced (X+Y Centering + Servo Sequence)")
        self.logger.info(f"Arena detection: {'Success' if self.vision.arena_detected else 'Fallback'}")
        self.logger.info(f"Boundary avoidance: Modular system")
        self.logger.info("=" * 60)
        
        # Enhanced competition results
        competition_result = {
            "elapsed_time": elapsed_time,
            "balls_collected": self.hardware.get_ball_count(),
            "final_state": self.state.value,
            "vision_system": "hough_circles_hybrid_white_only",
            "collection_system": "enhanced_xy_centering_servo_sequence",
            "boundary_system": "modular_avoidance_system",
            "arena_detected": self.vision.arena_detected
        }
        
        self.emergency_stop()
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        self.competition_active = False
        self.state = RobotState.EMERGENCY_STOP
        
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        try:
            self.hardware.emergency_stop()
            self.vision.cleanup()
            self.hardware.cleanup()
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")

def show_startup_menu():
    """Show startup menu with options"""
    print("\n" + "="*60)
    print("🤖 GOLFBOT CONTROL SYSTEM - WHITE BALL COLLECTION ONLY")
    print("="*60)
    print("1. Start Competition (Dashboard Mode)")
    print("2. Start Competition (Legacy Overlay Mode)")
    print("3. Hardware Testing") 
    print("4. Exit")
    print("="*60)
    print("FEATURES:")
    print("• White ball detection and collection only")
    print("• Ball centering before collection (X+Y axis)")
    print("• Enhanced servo collection sequence")
    print("• Faster centering adjustments (2x speed)")
    print("• Modular boundary avoidance system")
    print("• Clean dashboard interface (option 1) - Camera + Side panels")
    print("• Legacy overlay mode (option 2) - All info on camera")
    print("="*60)
    
    while True:
        try:
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                return 'competition_dashboard'
            elif choice == '2':
                return 'competition_legacy'
            elif choice == '3':
                return 'testing'
            elif choice == '4':
                return 'exit'
            else:
                print("Invalid choice. Enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return 'exit'
        except EOFError:
            print("\nExiting...")
            return 'exit'
        except Exception as e:
            print(f"Input error: {e}")
            return 'exit'

def main():
    """Main entry point"""
    mode = show_startup_menu()
    
    if mode == 'exit':
        print("Goodbye!")
        return 0
        
    elif mode == 'testing':
        print("\n🔧 Entering Hardware Testing Mode...")
        try:
            if run_hardware_test():
                print("✅ Testing completed successfully!")
            else:
                print("❌ Testing failed!")
        except Exception as e:
            print(f"Testing error: {e}")
        return 0
        
    elif mode in ['competition_dashboard', 'competition_legacy']:
        use_dashboard = (mode == 'competition_dashboard')
        interface_mode = "Dashboard" if use_dashboard else "Legacy Overlay"
        print(f"\n🏁 Entering Competition Mode with {interface_mode} Interface...")
        
        try:
            robot = GolfBot(use_dashboard=use_dashboard)
            
            if not robot.initialize():
                print("❌ Failed to initialize robot - exiting")
                return 1
            
            print("\n🚀 Robot ready with enhanced white ball collection system!")
            print("   - White ball detection and collection only")
            print("   - Ball centering for precision targeting (X+Y axis)")
            print("   - Enhanced servo collection sequence") 
            print("   - HoughCircles + Arena boundary detection")
            print("   - Enhanced servo control with gradual movement")
            print("   - Modular boundary avoidance system")
            print(f"   - {interface_mode} interface for monitoring")
            print(f"\n⚙️  Configuration:")
            print(f"   - X-centering tolerance: ±{config.CENTERING_TOLERANCE} pixels")
            print(f"   - Y-centering tolerance: ±{config.CENTERING_DISTANCE_TOLERANCE} pixels")
            print(f"   - Collection speed: {config.COLLECTION_SPEED}")
            print(f"   - Centering turn speed: {config.CENTERING_TURN_DURATION}s (FASTER)")
            print(f"   - Centering drive speed: {config.CENTERING_DRIVE_DURATION}s")
            print(f"   - Interface mode: {interface_mode}")
            print(f"   - Target: WHITE BALLS ONLY")
            print(f"   - Boundary system: Modular avoidance")
            print("\nPress Enter to start competition...")
            input()
            
            robot.start_competition()
            
        except KeyboardInterrupt:
            print("\n⚠️  Competition interrupted by user")
            if 'robot' in locals():
                robot.logger.info("Competition interrupted by user")
        except Exception as e:
            print(f"❌ Competition error: {e}")
            if 'robot' in locals():
                robot.logger.error(f"Unexpected error: {e}")
        finally:
            if 'robot' in locals():
                robot.emergency_stop()
        
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)