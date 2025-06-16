print("DEBUG: Script starting...")
print("DEBUG: About to import time...")
import time
print("DEBUG: About to import logging...")
import logging
print("DEBUG: About to import cv2...")
import cv2
print("DEBUG: About to import signal...")
import signal
print("DEBUG: About to import sys...")
import sys
print("DEBUG: About to import os...")
import os
print("DEBUG: About to import Enum...")
from enum import Enum
print("DEBUG: About to import typing...")
from typing import Optional
print("DEBUG: About to import config...")
import config
print("DEBUG: About to import hardware...")
from hardware import GolfBotHardware
print("DEBUG: About to import vision...")
from vision import VisionSystem
print("DEBUG: About to import telemetry...")
from telemetry import TelemetryLogger
print("DEBUG: About to import hardware_test...")
from hardware_test import run_hardware_test
print("DEBUG: All imports completed successfully!")

class RobotState(Enum):
    SEARCHING = "searching"
    APPROACHING_BALL = "approaching_ball"
    COLLECTING_BALL = "collecting_ball"
    APPROACHING_GOAL = "approaching_goal"
    DELIVERING_BALLS = "delivering_balls"
    AVOIDING_BOUNDARY = "avoiding_boundary"
    EMERGENCY_STOP = "emergency_stop"

class GolfBot:
    def __init__(self):
        print("DEBUG: Entering GolfBot.__init__")
        print("DEBUG: About to call setup_logging...")
        self.setup_logging()
        print("DEBUG: setup_logging completed")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("DEBUG: Logger created")
        
        # Check if display is available
        print("DEBUG: About to check display...")
        self.display_available = self.check_display_available()
        print(f"DEBUG: Display available: {self.display_available}")
        if not self.display_available:
            self.logger.info("No display detected - running in headless mode")
        
        # Initialize telemetry
        print("DEBUG: About to initialize telemetry...")
        self.telemetry = TelemetryLogger()
        print("DEBUG: Telemetry initialized")
        
        # Initialize systems - THIS IS LIKELY WHERE IT HANGS
        print("DEBUG: About to initialize hardware...")
        try:
            self.hardware = GolfBotHardware()
            print("DEBUG: Hardware initialized successfully")
        except Exception as e:
            print(f"DEBUG: Hardware initialization FAILED: {e}")
            raise
        
        print("DEBUG: About to initialize vision...")
        try:
            self.vision = VisionSystem()
            print("DEBUG: Vision initialized successfully")
        except Exception as e:
            print(f"DEBUG: Vision initialization FAILED: {e}")
            raise
        
        # Competition state
        print("DEBUG: Setting up competition state...")
        self.start_time = None
        self.competition_active = False
        self.state = RobotState.SEARCHING
        self.search_pattern_index = 0
        self.last_ball_seen_time = None
        self.delivery_attempts = 0
        
        # Performance tracking
        self.last_frame_time = time.time()
        
        # Setup signal handlers
        print("DEBUG: Setting up signal handlers...")
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        print("DEBUG: GolfBot.__init__ completed successfully!")
    
    def check_display_available(self):
        """Check if display/X11 is available"""
        print("DEBUG: Checking display availability...")
        try:
            # Check for DISPLAY environment variable
            if os.environ.get('DISPLAY') is None:
                print("DEBUG: No DISPLAY environment variable")
                return False
            
            # Try to initialize a test window
            test_img = cv2.imread('/dev/null')  # This won't work but won't crash
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('test')
            print("DEBUG: Display test successful")
            return True
        except Exception as e:
            print(f"DEBUG: Display test failed: {e}")
            return False
        
    def setup_logging(self):
        """Configure logging"""
        print("DEBUG: Setting up logging...")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('golfbot.log'),
                logging.StreamHandler()
            ]
        )
        print("DEBUG: Logging setup completed")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"DEBUG: Signal {signum} received")
        self.logger.info("Shutdown signal received")
        self.emergency_stop()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Initialize all systems"""
        print("DEBUG: Entering initialize()")
        self.logger.info("Initializing GolfBot...")
        
        try:
            # Start vision system
            print("DEBUG: About to start vision system...")
            if not self.vision.start():
                self.logger.error("Failed to initialize vision system")
                return False
            print("DEBUG: Vision system started successfully")
            
            self.logger.info("All systems initialized successfully")
            print("DEBUG: initialize() completed successfully")
            return True
            
        except Exception as e:
            print(f"DEBUG: initialize() FAILED: {e}")
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def start_competition(self):
        """Start the competition timer and main loop"""
        self.start_time = time.time()
        self.competition_active = True
        self.state = RobotState.SEARCHING
        
        self.logger.info("COMPETITION STARTED!")
        self.logger.info(f"Time limit: {config.COMPETITION_TIME} seconds")
        
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
        """Main competition control loop"""
        while self.competition_active and not self.is_time_up():
            try:
                # Track frame performance
                frame_start = time.time()
                
                # Get current vision data
                balls, orange_ball, goals, near_boundary, nav_command, debug_frame = self.vision.process_frame()
                
                if balls is None:  # Frame capture failed
                    self.telemetry.log_error("Frame capture failed", "vision")
                    continue
                
                # Log vision detection results
                self.telemetry.log_ball_detection(balls, orange_ball, goals)
                
                # Update ball tracking
                if balls or orange_ball:
                    self.last_ball_seen_time = time.time()
                
                # Log hardware state periodically
                if self.telemetry.frame_count % 10 == 0:
                    self.telemetry.log_hardware_state(self.hardware)
                
                # Calculate and log performance
                frame_time = time.time() - frame_start
                fps = 1.0 / (time.time() - self.last_frame_time) if self.last_frame_time else 0
                self.last_frame_time = time.time()
                self.telemetry.log_performance_metrics(fps, frame_time)
                
                # Show debug frame if enabled AND display is available
                if (config.SHOW_CAMERA_FEED and self.display_available and 
                    debug_frame is not None and debug_frame.size > 0):
                    try:
                        self.add_status_overlay(debug_frame)
                        cv2.imshow('GolfBot Debug', debug_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        self.logger.warning(f"Display error: {e}")
                        self.display_available = False  # Disable further attempts
                
                # State machine
                old_state = self.state
                self.execute_state_machine(balls, orange_ball, goals, near_boundary, nav_command)
                
                # Log state transitions
                if old_state != self.state:
                    self.telemetry.log_state_transition(old_state, self.state, nav_command or "automatic")
                
                # Brief pause to prevent overwhelming
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Main loop iteration error: {e}")
                self.telemetry.log_error(f"Main loop error: {str(e)}", "main_loop")
                self.hardware.stop_motors()
                time.sleep(0.5)
        
        # Competition ended
        self.end_competition()
    
    def execute_state_machine(self, balls, orange_ball, goals, near_boundary, nav_command):
        """Execute current state logic"""
        
        # Always check for boundary first
        if near_boundary:
            self.state = RobotState.AVOIDING_BOUNDARY
        
        if self.state == RobotState.SEARCHING:
            self.handle_searching(balls, orange_ball, nav_command)
            
        elif self.state == RobotState.APPROACHING_BALL:
            self.handle_approaching_ball(balls, orange_ball, nav_command)
            
        elif self.state == RobotState.COLLECTING_BALL:
            self.handle_collecting_ball()
            
        elif self.state == RobotState.APPROACHING_GOAL:
            self.handle_approaching_goal(goals)
            
        elif self.state == RobotState.DELIVERING_BALLS:
            self.handle_delivering_balls()
            
        elif self.state == RobotState.AVOIDING_BOUNDARY:
            self.handle_avoiding_boundary(near_boundary)
            
        elif self.state == RobotState.EMERGENCY_STOP:
            self.hardware.emergency_stop()
    
    def handle_searching(self, balls, orange_ball, nav_command):
        """Handle searching for balls"""
        # Look for any balls (treat all balls equally)
        if balls:
            self.logger.info(f"Found {len(balls)} ball(s)")
            self.state = RobotState.APPROACHING_BALL
            return
        
        # No balls found - execute search pattern
        self.execute_search_pattern()
    
    def handle_approaching_ball(self, balls, orange_ball, nav_command):
        """Handle approaching detected ball"""
        # Get closest ball (any type)
        all_balls = []
        if balls:
            all_balls.extend(balls)
        if orange_ball:
            all_balls.append(orange_ball)
        
        if not all_balls:
            self.logger.info("Lost sight of ball - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        # Choose closest ball
        target_ball = min(all_balls, key=lambda b: b.distance_from_center)
        
        # Check if close enough to collect
        if target_ball.distance_from_center < config.COLLECTION_DISTANCE_THRESHOLD:
            self.logger.info("Ball in collection range")
            self.state = RobotState.COLLECTING_BALL
            return
        
        # Navigate toward ball
        self.execute_navigation_command(nav_command)
    
    def handle_collecting_ball(self):
        """Handle ball collection sequence"""
        self.logger.info("Attempting ball collection...")
        
        success = self.hardware.attempt_ball_collection()
        
        # Log collection attempt
        self.telemetry.log_collection_attempt(success, "regular")
        
        if success:
            self.logger.info("Ball collected successfully!")
            
            # Check if we should deliver or keep collecting
            if self.hardware.get_ball_count() >= 3 or self.get_time_remaining() < 120:
                self.logger.info("Ready to deliver balls")
                self.state = RobotState.APPROACHING_GOAL
            else:
                self.state = RobotState.SEARCHING
        else:
            self.logger.warning("Ball collection failed - returning to search")
            self.telemetry.log_error("Ball collection failed", "collection")
            self.state = RobotState.SEARCHING
    
    def handle_approaching_goal(self, goals):
        """Handle approaching goal for delivery"""
        if not self.hardware.has_balls():
            self.logger.info("No balls to deliver - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        if not goals:
            # No goals visible - search for goal
            self.execute_search_pattern()
            return
        
        # Choose goal strategy
        prefer_goal_a = self.should_prefer_goal_a()
        goal_direction = self.vision.find_goal_direction(goals, prefer_goal_a)
        
        if goal_direction:
            # Check if close enough to deliver
            target_goals = [g for g in goals if 
                          (g.object_type == 'goal_a' if prefer_goal_a else g.object_type == 'goal_b')]
            
            if target_goals:
                closest_goal = min(target_goals, key=lambda x: x.distance_from_center)
                if closest_goal.distance_from_center < 50:  # Close enough to deliver
                    self.state = RobotState.DELIVERING_BALLS
                    return
            
            # Navigate toward goal
            self.execute_navigation_command(goal_direction)
        else:
            self.execute_search_pattern()
    
    def handle_delivering_balls(self):
        """Handle ball delivery sequence"""
        self.logger.info("Delivering balls...")
        
        # Determine goal type for scoring
        goal_type = "A" if self.should_prefer_goal_a() else "B"
        balls_delivered = self.hardware.delivery_sequence(goal_type)
        
        # Log delivery attempt
        self.telemetry.log_delivery_attempt(balls_delivered, goal_type)
        
        if balls_delivered > 0:
            points = balls_delivered * (config.GOAL_A_POINTS if goal_type == "A" else config.GOAL_B_POINTS)
            self.logger.info(f"Delivered {balls_delivered} balls to Goal {goal_type} for {points} points!")
            
            self.delivery_attempts += 1
        
        # Return to searching
        self.state = RobotState.SEARCHING
    
    def handle_avoiding_boundary(self, near_boundary):
        """Handle boundary avoidance"""
        if near_boundary:
            self.logger.warning("Near boundary - avoiding")
            
            # Stop and back up
            self.hardware.stop_motors()
            time.sleep(0.2)
            self.hardware.move_backward(duration=0.5)
            
            # Turn away from boundary
            self.hardware.turn_90_right()  # Simple avoidance
            
        else:
            # No longer near boundary
            self.state = RobotState.SEARCHING
    
    def execute_navigation_command(self, command):
        """Execute navigation command from vision system"""
        if command == "forward":
            self.hardware.move_forward(duration=0.3)
        elif command == "turn_left":
            self.hardware.turn_left(duration=0.2)
        elif command == "turn_right":
            self.hardware.turn_right(duration=0.2)
        elif command == "collect_ball" or command == "collect_orange":
            self.state = RobotState.COLLECTING_BALL
        else:
            self.execute_search_pattern()
    
    def execute_search_pattern(self):
        """Execute systematic search pattern"""
        pattern = config.SEARCH_PATTERN
        action = pattern[self.search_pattern_index % len(pattern)]
        
        if action == "forward":
            self.hardware.forward_step()
        elif action == "turn_right":
            self.hardware.turn_90_right()
        elif action == "turn_left":
            self.hardware.turn_90_left()
        
        self.search_pattern_index += 1
        
        # Pause between search moves
        time.sleep(0.3)
    
    def should_prefer_goal_a(self) -> bool:
        """Determine if we should prefer Goal A (150 pts) over Goal B (100 pts)"""
        # Prefer Goal A if we have fewer balls or time is running out
        ball_count = self.hardware.get_ball_count()
        time_remaining = self.get_time_remaining()
        
        # Go for higher points if we have few balls or little time
        return ball_count <= 2 or time_remaining < 60
    
    def add_status_overlay(self, frame):
        """Add status information to debug frame"""
        y = 30
        line_height = 25
        
        # Time remaining
        time_remaining = self.get_time_remaining()
        cv2.putText(frame, f"Time: {time_remaining:.0f}s", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Current state
        cv2.putText(frame, f"State: {self.state.value}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Ball count
        ball_count = self.hardware.get_ball_count()
        cv2.putText(frame, f"Balls: {ball_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def end_competition(self):
        """End competition and calculate final score"""
        self.competition_active = False
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("COMPETITION ENDED!")
        self.logger.info(f"Total time: {elapsed_time:.1f} seconds")
        self.logger.info(f"Balls collected: {self.hardware.get_ball_count()}")
        self.logger.info(f"Deliveries made: {self.delivery_attempts}")
        
        # Create competition results for telemetry
        competition_result = {
            "elapsed_time": elapsed_time,
            "balls_collected": self.hardware.get_ball_count(),
            "delivery_attempts": self.delivery_attempts,
            "final_state": self.state.value
        }
        
        # Final delivery if we have balls
        if self.hardware.has_balls():
            self.logger.info("Final ball delivery...")
            balls_delivered = self.hardware.delivery_sequence()
            competition_result["final_delivery"] = balls_delivered
        
        # Create session summary with results
        summary = self.telemetry.create_session_summary(competition_result)
        self.logger.info(f"Session data saved to: {summary['session_metadata']['session_dir']}")
        
        self.emergency_stop()
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        self.competition_active = False
        self.state = RobotState.EMERGENCY_STOP
        
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        try:
            self.hardware.emergency_stop()
            self.vision.cleanup()
            
            # Export telemetry data for analysis
            export_file = self.telemetry.export_for_analysis()
            self.logger.info(f"📊 Telemetry exported: {export_file}")
            
            self.telemetry.cleanup()
            self.hardware.cleanup()
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")

def show_startup_menu():
    """Show startup menu with options"""
    print("DEBUG: Entering show_startup_menu")
    print("\n" + "="*50)
    print("🤖 GOLFBOT CONTROL SYSTEM")
    print("="*50)
    print("1. Start Competition")
    print("2. Hardware Testing")
    print("3. Exit")
    print("="*50)
    print("DEBUG: About to start while loop in menu")
    
    while True:
        print("DEBUG: About to call input()")
        try:
            choice = input("Select option (1-3): ").strip()
            print(f"DEBUG: Got choice: '{choice}'")
            
            if choice == '1':
                print("DEBUG: Choice 1 selected - returning 'competition'")
                return 'competition'
            elif choice == '2':
                print("DEBUG: Choice 2 selected - returning 'testing'")
                return 'testing'
            elif choice == '3':
                print("DEBUG: Choice 3 selected - returning 'exit'")
                return 'exit'
            else:
                print(f"DEBUG: Invalid choice '{choice}' - asking again")
                print("Invalid choice. Enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nDEBUG: KeyboardInterrupt in menu")
            return 'exit'
        except EOFError:
            print("\nDEBUG: EOFError in menu")
            return 'exit'
        except Exception as e:
            print(f"DEBUG: Exception in menu: {e}")
            return 'exit'

def main():
    """Main entry point"""
    print("DEBUG: Entering main()")
    
    # Show startup menu
    print("DEBUG: About to call show_startup_menu()")
    mode = show_startup_menu()
    print(f"DEBUG: show_startup_menu returned: {mode}")
    
    if mode == 'exit':
        print("DEBUG: Exit mode - goodbye!")
        print("Goodbye!")
        return 0
    elif mode == 'testing':
        print("DEBUG: Testing mode selected")
        print("\n🔧 Entering Hardware Testing Mode...")
        try:
            if run_hardware_test():
                print("Testing completed successfully!")
            else:
                print("Testing failed!")
        except Exception as e:
            print(f"DEBUG: Testing mode exception: {e}")
        return 0
    elif mode == 'competition':
        print("DEBUG: Competition mode selected")
        print("\n🏁 Entering Competition Mode...")
        
        print("DEBUG: About to create GolfBot instance...")
        try:
            robot = GolfBot()
            print("DEBUG: GolfBot instance created successfully!")
        except Exception as e:
            print(f"DEBUG: GolfBot creation FAILED: {e}")
            print("Failed to create robot - exiting")
            return 1
        
        print("DEBUG: About to initialize robot...")
        if not robot.initialize():
            print("DEBUG: Robot initialization failed")
            print("Failed to initialize robot - exiting")
            return 1
        print("DEBUG: Robot initialized successfully!")
        
        try:
            # Wait for user to start competition
            print("DEBUG: About to wait for user input to start...")
            input("Press Enter to start competition...")
            print("DEBUG: User pressed Enter - starting competition...")
            
            robot.start_competition()
            
        except KeyboardInterrupt:
            print("DEBUG: KeyboardInterrupt in competition")
            robot.logger.info("Interrupted by user")
        except Exception as e:
            print(f"DEBUG: Competition exception: {e}")
            robot.logger.error(f"Unexpected error: {e}")
        finally:
            print("DEBUG: Calling emergency_stop...")
            robot.emergency_stop()
        
        return 0

if __name__ == "__main__":
    print("DEBUG: Script being run directly")
    exit_code = main()
    print(f"DEBUG: main() returned exit code: {exit_code}")
    sys.exit(exit_code)