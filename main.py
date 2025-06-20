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
    CENTERING_1 = "centering_1"  # Initial X+Y centering
    CENTERING_2 = "centering_2"  # Collection zone positioning
    COLLECTING_BALL = "collecting_ball"
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
        self.frame_skip_counter = 0
        
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
        self.logger.info("Initializing GolfBot with two-phase collection system...")
        
        try:
            # Start vision system
            if not self.vision.start():
                self.logger.error("Failed to initialize vision system")
                return False
            
            # Let vision system detect arena boundaries on startup
            self.logger.info("Detecting arena boundaries...")
            ret, frame = self.vision.get_frame()
            if ret:
                self.vision.detect_arena_boundaries(frame)
                if self.vision.arena_detected:
                    self.logger.info("✅ Arena boundaries detected successfully")
                else:
                    self.logger.info("⚠️  Using fallback arena boundaries")
            
            self.logger.info("All systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def start_competition(self):
        """Start the competition timer and main loop"""
        self.start_time = time.time()
        self.competition_active = True
        self.state = RobotState.SEARCHING
        
        self.logger.info("COMPETITION STARTED!")
        self.logger.info(f"Time limit: {config.COMPETITION_TIME} seconds")
        self.logger.info("Using two-phase collection: Centering_1 + Centering_2 + Collection")
        
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
        """Main competition control loop with two-phase collection"""
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
                
                if balls is None:
                    continue
                
                # Store detected balls for dashboard access
                self.vision._last_detected_balls = balls if balls else []
                
                # Update ball tracking
                if balls:
                    self.last_ball_seen_time = time.time()
                    high_confidence_balls = [b for b in balls if b.confidence > 0.5]
                    if high_confidence_balls:
                        self.logger.debug(f"High confidence balls: {len(high_confidence_balls)}")
                
                # Performance tracking
                frame_time = time.time() - frame_start
                fps = 1.0 / (time.time() - self.last_frame_time) if self.last_frame_time else 0
                self.last_frame_time = time.time()
                
                # Show display based on mode
                if config.SHOW_CAMERA_FEED and self.display_available:
                    try:
                        if self.use_dashboard and self.dashboard:
                            dashboard_frame = self.dashboard.create_dashboard(
                                debug_frame, self.state, self.vision, self.hardware
                            )
                            key = self.dashboard.show("GolfBot Dashboard - Two-Phase Collection")
                        else:
                            if debug_frame is not None and debug_frame.size > 0:
                                self.add_status_overlay(debug_frame)
                                cv2.imshow('GolfBot Debug - Two-Phase Collection', debug_frame)
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
                if self.state in [RobotState.CENTERING_1, RobotState.CENTERING_2]:
                    time.sleep(0.03)
                elif balls and len(balls) > 0:
                    time.sleep(0.05)
                else:
                    time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Main loop iteration error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
        
        self.end_competition()
    
    def execute_state_machine(self, balls, near_boundary, nav_command):
        """Execute current state logic with two-phase collection"""
        
        # Always check for boundary first (but allow collection to complete)
        if near_boundary and self.state not in [RobotState.CENTERING_2, RobotState.COLLECTING_BALL]:
            self.state = RobotState.AVOIDING_BOUNDARY
        
        if self.state == RobotState.SEARCHING:
            self.handle_searching(balls, nav_command)
            
        elif self.state == RobotState.CENTERING_1:
            self.handle_centering_1(balls, nav_command)
            
        elif self.state == RobotState.CENTERING_2:
            self.handle_centering_2(balls, nav_command)
            
        elif self.state == RobotState.COLLECTING_BALL:
            self.handle_collecting_ball()
            
        elif self.state == RobotState.AVOIDING_BOUNDARY:
            self.handle_avoiding_boundary(near_boundary)
            
        elif self.state == RobotState.EMERGENCY_STOP:
            self.hardware.emergency_stop()
    
    def handle_searching(self, balls, nav_command):
        """Handle searching with centering requirement"""
        if balls:
            confident_balls = [ball for ball in balls if ball.confidence > 0.4]
            
            if confident_balls:
                ball_count = len(confident_balls)
                orange_count = sum(1 for ball in confident_balls if ball.object_type == 'orange_ball')
                white_count = ball_count - orange_count
                
                avg_confidence = sum(ball.confidence for ball in confident_balls) / ball_count
                
                self.logger.info(f"Found {ball_count} confident ball(s) - {white_count} white, {orange_count} orange (avg conf: {avg_confidence:.2f})")
                self.logger.info("Starting CENTERING_1 (Initial X+Y alignment)")
                self.state = RobotState.CENTERING_1
                return
        
        # No confident balls found
        self.execute_search_pattern()
    
    def handle_centering_1(self, balls, nav_command):
        """Phase 1: Center the ball in both X and Y axes for initial alignment"""
        if not balls:
            self.logger.info("Lost sight of ball during CENTERING_1 - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        confident_balls = [ball for ball in balls if ball.confidence > 0.4]
        
        if not confident_balls:
            self.logger.info("No confident ball detections during CENTERING_1 - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        target_ball = confident_balls[0]
        
        # Check if ball is fully centered for Phase 1 (using Phase 1 tolerances)
        x_offset = abs(target_ball.center[0] - self.vision.frame_center_x)
        y_offset = abs(target_ball.center[1] - self.vision.frame_center_y)
        
        x_centered = x_offset <= config.CENTERING_1_TOLERANCE
        y_centered = y_offset <= config.CENTERING_1_DISTANCE_TOLERANCE
        
        if x_centered and y_centered:
            ball_type = "orange" if target_ball.object_type == "orange_ball" else "white"
            self.logger.info(f"CENTERING_1 complete! Ball aligned. Starting CENTERING_2 (Collection zone positioning) for {ball_type} ball")
            self.state = RobotState.CENTERING_2
            return
        
        # Ball not fully centered - get centering adjustments for both axes
        # X-axis centering (left/right)
        if not x_centered:
            x_offset_signed = target_ball.center[0] - self.vision.frame_center_x
            if x_offset_signed > 0:
                self.hardware.turn_right(duration=config.CENTERING_1_TURN_DURATION, 
                                        speed=config.CENTERING_1_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"CENTERING_1 X: turning right (offset: {x_offset_signed})")
            else:
                self.hardware.turn_left(duration=config.CENTERING_1_TURN_DURATION, 
                                       speed=config.CENTERING_1_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"CENTERING_1 X: turning left (offset: {x_offset_signed})")
            
            time.sleep(0.03)
            return
        
        # Y-axis centering (distance - forward/backward)
        if not y_centered:
            y_offset_signed = target_ball.center[1] - self.vision.frame_center_y
            if y_offset_signed > 0:
                self.hardware.move_backward(duration=config.CENTERING_1_DRIVE_DURATION, 
                                          speed=config.CENTERING_1_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"CENTERING_1 Y: moving backward (offset: {y_offset_signed})")
            else:
                self.hardware.move_forward(duration=config.CENTERING_1_DRIVE_DURATION, 
                                         speed=config.CENTERING_1_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.logger.info(f"CENTERING_1 Y: moving forward (offset: {y_offset_signed})")
            
            time.sleep(0.03)
            return
    
    def handle_centering_2(self, balls, nav_command):
        """Phase 2: Move to pre-collection position and approach collection zone"""
        if not balls:
            self.logger.info("Lost sight of ball during CENTERING_2 - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        confident_balls = [ball for ball in balls if ball.confidence > 0.4]
        
        if not confident_balls:
            self.logger.info("No confident ball detections during CENTERING_2 - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        target_ball = confident_balls[0]
        
        # Check if ball is in the green collection zone
        if self.vision.is_in_collection_zone(target_ball.center):
            ball_type = "orange" if target_ball.object_type == "orange_ball" else "white"
            self.logger.info(f"CENTERING_2 complete! {ball_type.title()} ball is in collection zone. Starting final collection.")
            self.state = RobotState.COLLECTING_BALL
            return
        
        # Ball not yet in collection zone
        ball_type = "orange" if target_ball.object_type == "orange_ball" else "white"
        
        # Set servo to pre-collection position (only on first entry to this state)
        current_servo_state = self.hardware.get_servo_ss_state()
        if current_servo_state != "pre-collect":
            self.logger.info(f"CENTERING_2: Setting servo SS to PRE-COLLECT position for {ball_type} ball")
            self.hardware.servo_ss_to_pre_collect()
            time.sleep(0.2)
        
        # Drive forward slowly to bring ball into collection zone
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"CENTERING_2: Approaching collection zone (speed: {config.CENTERING_2_APPROACH_SPEED})")
        
        self.hardware.move_forward(duration=config.CENTERING_2_APPROACH_TIME, 
                                 speed=config.CENTERING_2_APPROACH_SPEED)
        
        time.sleep(0.1)
    
    def handle_collecting_ball(self):
        """Handle ball collection with optimized sequence for collection zone"""
        current_target = self.vision.current_target
        
        if current_target:
            ball_type = "orange" if current_target.object_type == "orange_ball" else "regular"
            confidence = current_target.confidence
            self.logger.info(f"Starting optimized collection of {ball_type} ball in collection zone (confidence: {confidence:.2f})...")
        else:
            ball_type = "unknown"
            self.logger.info("Starting optimized ball collection in collection zone...")
        
        # Use optimized collection sequence with collection zone settings
        success = self.hardware.optimized_collection_sequence()
        
        if success:
            total_balls = self.hardware.get_ball_count()
            self.logger.info(f"✅ {ball_type.title()} ball collected with optimized sequence! Total: {total_balls}")
        else:
            self.logger.warning(f"❌ {ball_type.title()} optimized collection failed")
        
        self.state = RobotState.SEARCHING
    
    def handle_avoiding_boundary(self, near_boundary):
        """Handle boundary avoidance"""
        if near_boundary:
            self.logger.warning("⚠️  Near arena boundary - executing avoidance")
            
            self.hardware.stop_motors()
            time.sleep(0.1)
            
            self.hardware.move_backward(duration=0.25)
            
            self.hardware.turn_right(duration=0.25)
            
            time.sleep(0.15)
        else:
            self.state = RobotState.SEARCHING
    
    def execute_navigation_command(self, command):
        """Execute navigation with improved timing"""
        if command == "forward":
            self.hardware.move_forward(duration=0.50)
        elif command == "turn_left":
            self.hardware.turn_left(duration=0.15)
        elif command == "turn_right":
            self.hardware.turn_right(duration=0.15)
        elif command == "collect_ball":
            self.state = RobotState.CENTERING_1  # Start with Phase 1 centering
        else:
            self.execute_search_pattern()
    
    def execute_search_pattern(self):
        """Execute search pattern"""
        pattern = config.SEARCH_PATTERN
        action = pattern[self.search_pattern_index % len(pattern)]
        
        if action == "forward":
            self.hardware.forward_step()
        elif action == "turn_right":
            self.hardware.turn_90_right()
        elif action == "turn_left":
            self.hardware.turn_90_left()
        
        self.search_pattern_index += 1
        time.sleep(0.2)
    
    def add_status_overlay(self, frame):
        """Enhanced status overlay with two-phase collection info"""
        y = 30
        line_height = 25
        
        time_remaining = self.get_time_remaining()
        cv2.putText(frame, f"Time: {time_remaining:.0f}s", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Enhanced state display with phase info
        state_text = f"State: {self.state.value.replace('_', ' ').title()}"
        if self.state == RobotState.CENTERING_1:
            state_text += " (X+Y Align)"
        elif self.state == RobotState.CENTERING_2:
            state_text += " (Zone Position)"
        elif self.state == RobotState.COLLECTING_BALL:
            state_text += " (Optimized)"
        
        # Add centering info if in centering states
        if self.state in [RobotState.CENTERING_1, RobotState.CENTERING_2] and self.vision.current_target:
            if self.state == RobotState.CENTERING_1:
                x_offset = abs(self.vision.current_target.center[0] - self.vision.frame_center_x)
                y_offset = abs(self.vision.current_target.center[1] - self.vision.frame_center_y)
                x_ok = x_offset <= config.CENTERING_1_TOLERANCE
                y_ok = y_offset <= config.CENTERING_1_DISTANCE_TOLERANCE
                status_char = f"{'✓' if x_ok else 'X'}{'✓' if y_ok else 'Y'}"
                state_text += f" ({status_char})"
            elif self.state == RobotState.CENTERING_2:
                in_zone = self.vision.is_in_collection_zone(self.vision.current_target.center)
                state_text += f" ({'✓' if in_zone else '→'})"
        
        cv2.putText(frame, state_text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        ball_count = self.hardware.get_ball_count()
        cv2.putText(frame, f"Balls Collected: {ball_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        arena_status = "Detected" if self.vision.arena_detected else "Fallback"
        cv2.putText(frame, f"Arena: {arena_status}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += line_height
        
        # Two-phase collection info
        cv2.putText(frame, f"Collection: Two-Phase (C1→C2→Collect)", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height - 5
        
        if self.vision.current_target:
            target = self.vision.current_target
            ball_type = "ORANGE" if target.object_type == 'orange_ball' else "WHITE"
            
            # Phase-specific info
            if self.state == RobotState.CENTERING_1:
                x_offset = abs(target.center[0] - self.vision.frame_center_x)
                y_offset = abs(target.center[1] - self.vision.frame_center_y)
                target_info = f"Target: {ball_type} | Phase1: X±{x_offset} Y±{y_offset}"
            elif self.state == RobotState.CENTERING_2:
                in_zone = self.vision.is_in_collection_zone(target.center)
                target_info = f"Target: {ball_type} | Phase2: {'IN ZONE' if in_zone else 'APPROACHING'}"
            else:
                target_info = f"Target: {ball_type} | Conf: {target.confidence:.2f}"
            
            color = (0, 255, 0) if self.state == RobotState.COLLECTING_BALL else (0, 255, 255)
            cv2.putText(frame, target_info, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def end_competition(self):
        """End competition"""
        self.competition_active = False
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("=" * 60)
        self.logger.info("COMPETITION ENDED!")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {elapsed_time:.1f} seconds")
        self.logger.info(f"Balls collected: {self.hardware.get_ball_count()}")
        self.logger.info(f"Final state: {self.state.value}")
        self.logger.info(f"Collection system: Two-Phase (Centering_1 + Centering_2 + Optimized Collection)")
        self.logger.info(f"Arena detection: {'Success' if self.vision.arena_detected else 'Fallback'}")
        self.logger.info("=" * 60)
        
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
    print("🤖 GOLFBOT CONTROL SYSTEM - TWO-PHASE COLLECTION")
    print("="*60)
    print("1. Start Competition (Dashboard Mode)")
    print("2. Start Competition (Legacy Overlay Mode)")
    print("3. Hardware Testing") 
    print("4. Exit")
    print("="*60)
    print("FEATURES:")
    print("• Phase 1: X+Y centering for ball alignment")
    print("• Phase 2: Collection zone positioning with servo pre-collect")
    print("• Optimized collection sequence in green zone")
    print("• Clean dashboard interface (option 1)")
    print("• Legacy overlay mode (option 2)")
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
            
            print("\n🚀 Robot ready with two-phase collection system!")
            print("   - Phase 1: X+Y centering for ball alignment")
            print("   - Phase 2: Collection zone positioning with servo pre-collect") 
            print("   - Optimized collection sequence in green zone")
            print(f"   - {interface_mode} interface for monitoring")
            print(f"\n⚙️  Configuration:")
            print(f"   - Phase 1 tolerances: ±{config.CENTERING_1_TOLERANCE}px X, ±{config.CENTERING_1_DISTANCE_TOLERANCE}px Y")
            print(f"   - Phase 2 approach: {config.CENTERING_2_APPROACH_SPEED} speed, {config.CENTERING_2_APPROACH_TIME}s time")
            print(f"   - Collection: {config.CENTERING_2_COLLECTION_SPEED} speed, {config.CENTERING_2_COLLECTION_TIME}s time")
            print(f"   - Servo pre-collect: {config.SERVO_SS_PRE_COLLECT}°")
            print(f"   - Interface mode: {interface_mode}")
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