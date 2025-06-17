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
        self.frame_skip_counter = 0  # NEW: Skip frames for performance
        
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
        self.logger.info("Initializing GolfBot with improved vision...")
        
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
        self.logger.info("Using improved vision: HoughCircles + Arena detection")
        
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
        """Main competition control loop with improved performance"""
        while self.competition_active and not self.is_time_up():
            try:
                frame_start = time.time()
                
                # Skip frames for performance (process every 2nd frame)
                self.frame_skip_counter += 1
                if self.frame_skip_counter % 2 != 0:
                    time.sleep(0.05)
                    continue
                
                # Get current vision data
                balls, _, near_boundary, nav_command, debug_frame = self.vision.process_frame()
                
                if balls is None:  # Frame capture failed
                    self.telemetry.log_error("Frame capture failed", "vision")
                    continue
                
                # Enhanced logging with detection method info
                detection_info = {
                    "detection_method": "hough_circles_hybrid",
                    "arena_detected": self.vision.arena_detected,
                    "balls_found": len(balls) if balls else 0
                }
                
                # Log ball detections with enhanced info
                if balls:
                    for i, ball in enumerate(balls):
                        detection_info[f"ball_{i}_confidence"] = ball.confidence
                        detection_info[f"ball_{i}_type"] = ball.object_type
                        detection_info[f"ball_{i}_in_zone"] = ball.in_collection_zone
                
                self.telemetry.log_ball_detection(balls, None, None)
                self.telemetry.log_frame_data(extra_data=detection_info)
                
                # Update ball tracking
                if balls:
                    self.last_ball_seen_time = time.time()
                    high_confidence_balls = [b for b in balls if b.confidence > 0.5]
                    if high_confidence_balls:
                        self.logger.debug(f"High confidence balls: {len(high_confidence_balls)}")
                
                # Log hardware state periodically
                if self.telemetry.frame_count % 20 == 0:  # Less frequent logging
                    self.telemetry.log_hardware_state(self.hardware)
                
                # Performance tracking
                frame_time = time.time() - frame_start
                fps = 1.0 / (time.time() - self.last_frame_time) if self.last_frame_time else 0
                self.last_frame_time = time.time()
                self.telemetry.log_performance_metrics(fps, frame_time)
                
                # Show debug frame if enabled
                if (config.SHOW_CAMERA_FEED and self.display_available and 
                    debug_frame is not None and debug_frame.size > 0):
                    try:
                        self.add_status_overlay(debug_frame)
                        cv2.imshow('GolfBot Debug - Improved Vision', debug_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        self.logger.warning(f"Display error: {e}")
                        self.display_available = False
                
                # State machine execution
                old_state = self.state
                self.execute_state_machine(balls, near_boundary, nav_command)
                
                # Log state transitions with enhanced info
                if old_state != self.state:
                    reason = f"{nav_command} | balls={len(balls) if balls else 0} | boundary={near_boundary}"
                    self.telemetry.log_state_transition(old_state, self.state, reason)
                
                # Adaptive sleep based on detection results
                if balls and len(balls) > 0:
                    time.sleep(0.05)  # Faster when balls detected
                else:
                    time.sleep(0.1)   # Slower when searching
                
            except Exception as e:
                self.logger.error(f"Main loop iteration error: {e}")
                self.telemetry.log_error(f"Main loop error: {str(e)}", "main_loop")
                self.hardware.stop_motors()
                time.sleep(0.5)
        
        self.end_competition()
    
    def execute_state_machine(self, balls, near_boundary, nav_command):
        """Execute current state logic with improved ball handling"""
        
        # Always check for boundary first
        if near_boundary:
            self.state = RobotState.AVOIDING_BOUNDARY
        
        if self.state == RobotState.SEARCHING:
            self.handle_searching(balls, nav_command)
            
        elif self.state == RobotState.APPROACHING_BALL:
            self.handle_approaching_ball(balls, nav_command)
            
        elif self.state == RobotState.COLLECTING_BALL:
            self.handle_collecting_ball()
            
        elif self.state == RobotState.AVOIDING_BOUNDARY:
            self.handle_avoiding_boundary(near_boundary)
            
        elif self.state == RobotState.EMERGENCY_STOP:
            self.hardware.emergency_stop()
    
    def handle_searching(self, balls, nav_command):
        """Handle searching with confidence filtering"""
        if balls:
            # Filter for high confidence balls only
            confident_balls = [ball for ball in balls if ball.confidence > 0.4]
            
            if confident_balls:
                ball_count = len(confident_balls)
                orange_count = sum(1 for ball in confident_balls if ball.object_type == 'orange_ball')
                white_count = ball_count - orange_count
                
                avg_confidence = sum(ball.confidence for ball in confident_balls) / ball_count
                
                self.logger.info(f"Found {ball_count} confident ball(s) - {white_count} white, {orange_count} orange (avg conf: {avg_confidence:.2f})")
                self.state = RobotState.APPROACHING_BALL
                return
        
        # No confident balls found
        self.execute_search_pattern()
    
    def handle_approaching_ball(self, balls, nav_command):
        """Handle approaching with confidence tracking"""
        if not balls:
            self.logger.info("Lost sight of all balls - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        # Filter for confident balls
        confident_balls = [ball for ball in balls if ball.confidence > 0.4]
        
        if not confident_balls:
            self.logger.info("No confident ball detections - returning to search")
            self.state = RobotState.SEARCHING
            return
        
        # Target the closest confident ball
        target_ball = confident_balls[0]
        
        if target_ball.in_collection_zone:
            self.logger.info(f"Ball in collection zone - attempting collection (confidence: {target_ball.confidence:.2f})")
            self.state = RobotState.COLLECTING_BALL
            return
        
        # Navigate toward ball
        self.execute_navigation_command(nav_command)
    
    def handle_collecting_ball(self):
        """Handle ball collection with improved logging"""
        current_target = self.vision.current_target
        
        if current_target:
            ball_type = "orange" if current_target.object_type == "orange_ball" else "regular"
            confidence = current_target.confidence
            self.logger.info(f"Attempting {ball_type} ball collection (confidence: {confidence:.2f})...")
        else:
            ball_type = "unknown"
            self.logger.info("Attempting ball collection...")
        
        success = self.hardware.attempt_ball_collection()
        
        # Enhanced logging
        self.telemetry.log_collection_attempt(success, ball_type)
        
        if success:
            total_balls = self.hardware.get_ball_count()
            self.logger.info(f"✅ {ball_type.title()} ball collected! Total: {total_balls}")
            
            # Log collection success with details
            collection_data = {
                "ball_type": ball_type,
                "confidence": confidence if current_target else 0.0,
                "total_collected": total_balls
            }
            self.telemetry.log_frame_data(action="successful_collection", extra_data=collection_data)
        else:
            self.logger.warning(f"❌ {ball_type.title()} ball collection failed")
            self.telemetry.log_error(f"Ball collection failed - {ball_type}", "collection")
        
        # Return to searching
        self.state = RobotState.SEARCHING
    
    def handle_avoiding_boundary(self, near_boundary):
        """Handle boundary avoidance with improved timing"""
        if near_boundary:
            self.logger.warning("⚠️  Near arena boundary - executing avoidance")
            
            # Immediate stop
            self.hardware.stop_motors()
            time.sleep(0.1)
            
            # Quick backup
            self.hardware.move_backward(duration=0.25)
            
            # Small turn to avoid
            self.hardware.turn_right(duration=0.25)
            
            time.sleep(0.15)
        else:
            # Clear of boundary
            self.state = RobotState.SEARCHING
    
    def execute_navigation_command(self, command):
        """Execute navigation with improved timing"""
        if command == "forward":
            self.hardware.move_forward(duration=0.25)  # Shorter movements
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
            self.hardware.forward_step()
        elif action == "turn_right":
            self.hardware.turn_90_right()
        elif action == "turn_left":
            self.hardware.turn_90_left()
        
        self.search_pattern_index += 1
        time.sleep(0.2)  # Shorter pause
    
    def add_status_overlay(self, frame):
        """Enhanced status overlay with vision info"""
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
        y += line_height
        
        # Vision status
        arena_status = "Detected" if self.vision.arena_detected else "Fallback"
        cv2.putText(frame, f"Arena: {arena_status}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += line_height
        
        # Current target info
        if self.vision.current_target:
            target = self.vision.current_target
            target_info = f"Target: {target.object_type} (Conf: {target.confidence:.2f})"
            cv2.putText(frame, target_info, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def end_competition(self):
        """End competition with enhanced results"""
        self.competition_active = False
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("=" * 50)
        self.logger.info("COMPETITION ENDED!")
        self.logger.info("=" * 50)
        self.logger.info(f"Total time: {elapsed_time:.1f} seconds")
        self.logger.info(f"Balls collected: {self.hardware.get_ball_count()}")
        self.logger.info(f"Final state: {self.state.value}")
        self.logger.info(f"Arena detection: {'Success' if self.vision.arena_detected else 'Fallback'}")
        self.logger.info("=" * 50)
        
        # Enhanced competition results
        competition_result = {
            "elapsed_time": elapsed_time,
            "balls_collected": self.hardware.get_ball_count(),
            "final_state": self.state.value,
            "vision_system": "hough_circles_hybrid",
            "arena_detected": self.vision.arena_detected
        }
        
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
            
            export_file = self.telemetry.export_for_analysis()
            self.logger.info(f"📊 Telemetry exported: {export_file}")
            
            self.telemetry.cleanup()
            self.hardware.cleanup()
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")

def show_startup_menu():
    """Show startup menu with options"""
    print("\n" + "="*50)
    print("🤖 GOLFBOT CONTROL SYSTEM - IMPROVED VISION")
    print("="*50)
    print("1. Start Competition (HoughCircles + Arena Detection)")
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
        print("\n🏁 Entering Competition Mode with Improved Vision...")
        
        try:
            robot = GolfBot()
            
            if not robot.initialize():
                print("❌ Failed to initialize robot - exiting")
                return 1
            
            print("\n🚀 Robot ready with improved vision system!")
            print("   - HoughCircles shape detection")
            print("   - Arena boundary detection") 
            print("   - Enhanced ball confidence scoring")
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