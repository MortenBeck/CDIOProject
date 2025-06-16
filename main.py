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
from telemetry import TelemetryLogger
from hardware_test import run_hardware_test

class RobotState(Enum):
    SEARCHING = "searching"
    APPROACHING_BALL = "approaching_ball"
    COLLECTING_BALL = "collecting_ball"
    AVOIDING_BOUNDARY = "avoiding_boundary"
    EMERGENCY_STOP = "emergency_stop"

class GolfBot:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Check if display is available
        self.display_available = self.check_display_available()
        if not self.display_available:
            self.logger.info("No display detected - running in headless mode")
        
        # Initialize systems
        self.telemetry = TelemetryLogger()
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
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def check_display_available(self):
        """Check if display/X11 is available"""
        try:
            # Check for DISPLAY environment variable
            if os.environ.get('DISPLAY') is None:
                return False
            
            # Try to initialize a test window
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
        self.logger.info("Initializing GolfBot...")
        
        try:
            # Start vision system
            if not self.vision.start():
                self.logger.error("Failed to initialize vision system")
                return False
            
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
                
                # Get current vision data (no goals anymore)
                balls, orange_ball, near_boundary, nav_command, debug_frame = self.vision.process_frame()
                
                if balls is None:  # Frame capture failed
                    self.telemetry.log_error("Frame capture failed", "vision")
                    continue
                
                # Log vision detection results
                self.telemetry.log_ball_detection(balls, orange_ball, None)
                
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
                self.execute_state_machine(balls, orange_ball, near_boundary, nav_command)
                
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
    
    def execute_state_machine(self, balls, orange_ball, near_boundary, nav_command):
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
            
        elif self.state == RobotState.AVOIDING_BOUNDARY:
            self.handle_avoiding_boundary(near_boundary)
            
        elif self.state == RobotState.EMERGENCY_STOP:
            self.hardware.emergency_stop()
    
    def handle_searching(self, balls, orange_ball, nav_command):
        """Handle searching for balls"""
        # Look for any balls
        if balls or orange_ball:
            ball_count = len(balls) if balls else 0
            orange_count = 1 if orange_ball else 0
            self.logger.info(f"Found {ball_count + orange_count} ball(s) - {ball_count} regular, {orange_count} orange")
            self.state = RobotState.APPROACHING_BALL
            return
        
        # No balls found - execute search pattern
        self.execute_search_pattern()
    
    def handle_approaching_ball(self, balls, orange_ball, nav_command):
        """Handle approaching detected ball"""
        # Get all available balls
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
        
        # Check if ball is in collection zone
        if target_ball.in_collection_zone:
            self.logger.info(f"Ball in collection zone - attempting collection")
            self.state = RobotState.COLLECTING_BALL
            return
        
        # Navigate toward ball
        self.execute_navigation_command(nav_command)
    
    def handle_collecting_ball(self):
        """Handle ball collection sequence"""
        self.logger.info("Attempting ball collection...")
        
        success = self.hardware.attempt_ball_collection()
        
        # Determine ball type for logging
        ball_type = "regular"
        if self.vision.current_target and self.vision.current_target.object_type == "orange_ball":
            ball_type = "orange"
        
        # Log collection attempt
        self.telemetry.log_collection_attempt(success, ball_type)
        
        if success:
            total_balls = self.hardware.get_ball_count()
            self.logger.info(f"Ball collected successfully! Total balls: {total_balls}")
        else:
            self.logger.warning("Ball collection failed")
            self.telemetry.log_error("Ball collection failed", "collection")
        
        # Return to searching for more balls
        self.state = RobotState.SEARCHING
    
    def handle_avoiding_boundary(self, near_boundary):
        """Handle boundary avoidance"""
        if near_boundary:
            self.logger.warning("Near red wall - executing avoidance maneuver")
            
            # Stop immediately
            self.hardware.stop_motors()
            time.sleep(0.2)
            
            # Back up to create distance
            self.hardware.move_backward(duration=0.8)
            
            # Turn to avoid wall
            self.hardware.turn_90_right()
            
            # Extra delay to ensure clear of wall
            time.sleep(0.5)
            
        else:
            # No longer near boundary - return to searching
            self.state = RobotState.SEARCHING
    
    def execute_navigation_command(self, command):
        """Execute navigation command from vision system"""
        if command == "forward":
            self.hardware.move_forward(duration=0.3)
        elif command == "turn_left":
            self.hardware.turn_left(duration=0.2)
        elif command == "turn_right":
            self.hardware.turn_right(duration=0.2)
        elif command == "collect_ball":
            self.state = RobotState.COLLECTING_BALL
        else:
            # Default to search pattern
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
        cv2.putText(frame, f"Balls Collected: {ball_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def end_competition(self):
        """End competition and show final results"""
        self.competition_active = False
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("=" * 50)
        self.logger.info("COMPETITION ENDED!")
        self.logger.info("=" * 50)
        self.logger.info(f"Total time: {elapsed_time:.1f} seconds")
        self.logger.info(f"Balls collected: {self.hardware.get_ball_count()}")
        self.logger.info(f"Final state: {self.state.value}")
        self.logger.info("=" * 50)
        
        # Create competition results for telemetry
        competition_result = {
            "elapsed_time": elapsed_time,
            "balls_collected": self.hardware.get_ball_count(),
            "final_state": self.state.value
        }
        
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
    print("\n" + "="*50)
    print("🤖 GOLFBOT CONTROL SYSTEM")
    print("="*50)
    print("1. Start Competition")
    print("2. Hardware Testing") 
    print("3. Exit")
    print("="*50)
    
    while True:
        try:
            choice = input("Select option (1-3): ").strip()
            
            if choice == '1':
                return 'competition'
            elif choice == '2':
                return 'testing'
            elif choice == '3':
                return 'exit'
            else:
                print("Invalid choice. Enter 1, 2, or 3.")
                
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
    # Show startup menu
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
        
    elif mode == 'competition':
        print("\n🏁 Entering Competition Mode...")
        
        try:
            # Create robot instance
            robot = GolfBot()
            
            # Initialize systems
            if not robot.initialize():
                print("❌ Failed to initialize robot - exiting")
                return 1
            
            # Wait for user to start competition
            print("\n🚀 Robot ready! Press Enter to start competition...")
            input()
            
            # Start competition
            robot.start_competition()
            
        except KeyboardInterrupt:
            print("\n⚠️  Competition interrupted by user")
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