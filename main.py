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
    CENTERING_BALL = "centering_ball"  # Center ball before collection
    APPROACHING_BALL = "approaching_ball"
    COLLECTING_BALL = "collecting_ball"
    BLIND_COLLECTION = "blind_collection"  # Drive to ball without vision
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
        
        # Enhanced collection tracking
        self.blind_collection_drive_time = 0.0
        self.centering_start_time = None
        self.centering_attempt_count = 0
        self.last_centering_log_time = 0
        
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
        self.logger.info("Initializing GolfBot with enhanced collection system...")
        
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
        
        self.logger.info("🏁 COMPETITION STARTED!")
        self.logger.info(f"Time limit: {config.COMPETITION_TIME} seconds")
        self.logger.info("🎯 Enhanced Collection System Active:")
        self.logger.info(f"   - Centering tolerance: ±{config.CENTERING_TOLERANCE} pixels")
        self.logger.info(f"   - Max centering time: {config.MAX_CENTERING_TIME} seconds")
        self.logger.info(f"   - Collection speed: {config.COLLECTION_SPEED}")
        self.logger.info(f"   - Drive time calc: {config.COLLECTION_DRIVE_TIME_PER_PIXEL:.3f}s/pixel")
        
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
        """Main competition control loop with enhanced collection"""
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
                    "balls_found": len(balls) if balls else 0,
                    "current_state": self.state.value
                }
                
                # Log ball detections with enhanced info
                if balls:
                    for i, ball in enumerate(balls):
                        detection_info[f"ball_{i}_confidence"] = ball.confidence
                        detection_info[f"ball_{i}_type"] = ball.object_type
                        detection_info[f"ball_{i}_centered"] = self.vision.is_ball_centered(ball)
                        detection_info[f"ball_{i}_in_zone"] = ball.in_collection_zone
                
                self.telemetry.log_ball_detection(balls, None, None)
                self.telemetry.log_frame_data(extra_data=detection_info)
                
                # Update ball tracking
                if balls:
                    self.last_ball_seen_time = time.time()
                    high_confidence_balls = [b for b in balls if b.confidence > 0.5]
                    if high_confidence_balls and config.DEBUG_VISION:
                        self.logger.debug(f"High confidence balls: {len(high_confidence_balls)}")
                
                # Log hardware state periodically
                if self.telemetry.frame_count % 30 == 0:  # Less frequent logging
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
                        cv2.imshow('GolfBot - Enhanced Collection System', debug_frame)
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
                    if self.vision.current_target:
                        centered = self.vision.is_ball_centered(self.vision.current_target)
                        reason += f" | centered={centered}"
                    self.telemetry.log_state_transition(old_state, self.state, reason)
                    self.logger.info(f"🔄 State: {old_state.value} → {self.state.value} | {reason}")
                
                # Adaptive sleep based on state
                if self.state == RobotState.CENTERING_BALL:
                    time.sleep(0.03)  # Faster when centering
                elif self.state == RobotState.BLIND_COLLECTION:
                    time.sleep(0.02)  # Fastest during blind collection
                elif balls and len(balls) > 0:
                    time.sleep(0.05)  # Normal when balls detected
                else:
                    time.sleep(0.1)   # Slower when searching
                
            except Exception as e:
                self.logger.error(f"Main loop iteration error: {e}")
                self.telemetry.log_error(f"Main loop error: {str(e)}", "main_loop")
                self.hardware.stop_motors()
                time.sleep(0.5)
        
        self.end_competition()
    
    def execute_state_machine(self, balls, near_boundary, nav_command):
        """Execute current state logic with enhanced ball centering and blind collection"""
        
        # Always check for boundary first (except during blind collection)
        if near_boundary and self.state not in [RobotState.BLIND_COLLECTION]:
            if self.state == RobotState.CENTERING_BALL:
                self.reset_centering_state()
            self.state = RobotState.AVOIDING_BOUNDARY
        
        if self.state == RobotState.SEARCHING:
            self.handle_searching(balls, nav_command)
            
        elif self.state == RobotState.CENTERING_BALL:
            self.handle_centering_ball(balls, nav_command)
            
        elif self.state == RobotState.APPROACHING_BALL:
            self.handle_approaching_ball(balls, nav_command)
            
        elif self.state == RobotState.COLLECTING_BALL:
            self.handle_collecting_ball()
            
        elif self.state == RobotState.BLIND_COLLECTION:
            self.handle_blind_collection()
            
        elif self.state == RobotState.AVOIDING_BOUNDARY:
            self.handle_avoiding_boundary(near_boundary)
            
        elif self.state == RobotState.EMERGENCY_STOP:
            self.hardware.emergency_stop()
    
    def reset_centering_state(self):
        """Reset centering state variables"""
        self.centering_start_time = None
        self.centering_attempt_count = 0
        self.last_centering_log_time = 0
    
    def handle_searching(self, balls, nav_command):
        """Handle searching with centering requirement"""
        if balls:
            # Filter for high confidence balls only
            confident_balls = [ball for ball in balls if ball.confidence > 0.4]
            
            if confident_balls:
                ball_count = len(confident_balls)
                orange_count = sum(1 for ball in confident_balls if ball.object_type == 'orange_ball')
                white_count = ball_count - orange_count
                
                avg_confidence = sum(ball.confidence for ball in confident_balls) / ball_count
                
                self.logger.info(f"🎯 Found {ball_count} confident ball(s) - {white_count} white, {orange_count} orange (avg conf: {avg_confidence:.2f})")
                self.state = RobotState.CENTERING_BALL
                return
        
        # No confident balls found
        self.execute_search_pattern()
    
    def handle_centering_ball(self, balls, nav_command):
        """Enhanced ball centering with timeout and adaptive adjustments"""
        
        # Initialize centering if just started
        if self.centering_start_time is None:
            self.centering_start_time = time.time()
            self.centering_attempt_count = 0
            self.last_centering_log_time = time.time()
            self.logger.info("🎯 Starting enhanced ball centering sequence...")
        
        # Check for centering timeout
        centering_elapsed = time.time() - self.centering_start_time
        if centering_elapsed > config.MAX_CENTERING_TIME:
            self.logger.warning(f"⏰ Centering timeout after {centering_elapsed:.1f}s - falling back to legacy collection")
            self.reset_centering_state()
            self.state = RobotState.COLLECTING_BALL
            return
        
        # Check for balls
        if not balls:
            self.logger.info("❌ Lost sight of ball during centering - returning to search")
            self.reset_centering_state()
            self.state = RobotState.SEARCHING
            return
        
        # Filter for confident balls
        confident_balls = [ball for ball in balls if ball.confidence > 0.4]
        
        if not confident_balls:
            self.logger.info("❌ No confident ball detections during centering - returning to search")
            self.reset_centering_state()
            self.state = RobotState.SEARCHING
            return
        
        # Target the closest confident ball
        target_ball = confident_balls[0]
        
        # Calculate centering metrics
        x_offset = target_ball.center[0] - self.frame_center_x
        abs_offset = abs(x_offset)
        centered = self.vision.is_ball_centered(target_ball)
        
        # Periodic logging (every 1 second)
        current_time = time.time()
        if current_time - self.last_centering_log_time >= 1.0:
            self.logger.info(f"🎯 CENTERING [{centering_elapsed:.1f}s]: Offset={x_offset:+d}px, Tolerance=±{config.CENTERING_TOLERANCE}px, Centered={centered}")
            self.last_centering_log_time = current_time
        
        # Check if ball is centered
        if centered:
            # Ball is centered - calculate drive time and start blind collection
            drive_time = self.vision.calculate_drive_time_to_ball(target_ball)
            self.blind_collection_drive_time = drive_time
            
            ball_type = "orange" if target_ball.object_type == "orange_ball" else "white"
            self.logger.info(f"✅ Ball centered in {centering_elapsed:.1f}s! Starting blind collection")
            self.logger.info(f"   Ball type: {ball_type}")
            self.logger.info(f"   Drive time: {drive_time:.2f}s")
            self.logger.info(f"   Confidence: {target_ball.confidence:.2f}")
            
            self.reset_centering_state()
            self.state = RobotState.BLIND_COLLECTION
            return
        
        # Ball not centered - make adaptive adjustment
        if abs_offset > config.CENTERING_TOLERANCE:
            # Adaptive turning - smaller adjustments for smaller offsets
            turn_base = config.CENTERING_TURN_BASE
            turn_scale = config.CENTERING_TURN_SCALE
            turn_duration = turn_base + (abs_offset * turn_scale)
            
            # Clamp turn duration to reasonable bounds
            turn_duration = min(0.20, max(0.04, turn_duration))
            
            self.centering_attempt_count += 1
            
            if x_offset > 0:
                if config.DEBUG_CENTERING:
                    self.logger.debug(f"🔄 RIGHT turn #{self.centering_attempt_count}: {turn_duration:.3f}s (offset: +{x_offset})")
                self.hardware.turn_right(duration=turn_duration)
            else:
                if config.DEBUG_CENTERING:
                    self.logger.debug(f"🔄 LEFT turn #{self.centering_attempt_count}: {turn_duration:.3f}s (offset: {x_offset})")
                self.hardware.turn_left(duration=turn_duration)
            
            # Brief pause for stability
            time.sleep(0.1)
        else:
            # This shouldn't happen if is_ball_centered is working correctly
            self.logger.warning(f"⚠️ Ball appears centered but check failed! Offset: {x_offset}, Tolerance: {config.CENTERING_TOLERANCE}")
            # Force it to be centered
            self.reset_centering_state()
            self.state = RobotState.BLIND_COLLECTION
    
    def handle_approaching_ball(self, balls, nav_command):
        """Handle approaching with confidence tracking (legacy mode)"""
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
            self.logger.info(f"Ball in collection zone - attempting legacy collection (confidence: {target_ball.confidence:.2f})")
            self.state = RobotState.COLLECTING_BALL
            return
        
        # Navigate toward ball
        self.execute_navigation_command(nav_command)
    
    def handle_collecting_ball(self):
        """Handle ball collection with improved logging (legacy mode)"""
        current_target = self.vision.current_target
        
        if current_target:
            ball_type = "orange" if current_target.object_type == "orange_ball" else "regular"
            confidence = current_target.confidence
            self.logger.info(f"🤖 Attempting legacy {ball_type} ball collection (confidence: {confidence:.2f})...")
        else:
            ball_type = "unknown"
            self.logger.info("🤖 Attempting legacy ball collection...")
        
        success = self.hardware.attempt_ball_collection()
        
        # Enhanced logging
        self.telemetry.log_collection_attempt(success, ball_type)
        
        if success:
            total_balls = self.hardware.get_ball_count()
            self.logger.info(f"✅ Legacy {ball_type} ball collected! Total: {total_balls}")
            
            # Log collection success with details
            collection_data = {
                "ball_type": ball_type,
                "confidence": confidence if current_target else 0.0,
                "total_collected": total_balls,
                "collection_method": "legacy_collection"
            }
            self.telemetry.log_frame_data(action="successful_legacy_collection", extra_data=collection_data)
        else:
            self.logger.warning(f"❌ Legacy {ball_type} ball collection failed")
            self.telemetry.log_error(f"Legacy ball collection failed - {ball_type}", "collection")
        
        # Return to searching
        self.state = RobotState.SEARCHING
    
    def handle_blind_collection(self):
        """Execute enhanced blind collection sequence"""
        self.logger.info("🚀 Executing blind collection sequence...")
        
        # Get ball type for logging
        ball_type = "unknown"
        if self.vision.current_target:
            ball_type = "orange" if self.vision.current_target.object_type == "orange_ball" else "regular"
        
        # Execute the blind collection
        success = self.hardware.blind_collection_sequence(self.blind_collection_drive_time)
        
        # Enhanced logging
        self.telemetry.log_collection_attempt(success, ball_type)
        
        if success:
            total_balls = self.hardware.get_ball_count()
            self.logger.info(f"✅ BLIND COLLECTION SUCCESS! Total balls: {total_balls}")
            
            # Log collection success with details
            collection_data = {
                "ball_type": ball_type,
                "collection_method": "blind_collection",
                "drive_time": self.blind_collection_drive_time,
                "total_collected": total_balls
            }
            self.telemetry.log_frame_data(action="successful_blind_collection", extra_data=collection_data)
        else:
            self.logger.warning(f"❌ BLIND COLLECTION FAILED")
            self.telemetry.log_error(f"Blind collection failed - {ball_type}", "collection")
        
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
            self.hardware.move_backward(duration=0.3)
            
            # Turn to avoid
            self.hardware.turn_right(duration=0.3)
            
            time.sleep(0.2)
        else:
            # Clear of boundary
            self.state = RobotState.SEARCHING
    
    def execute_navigation_command(self, command):
        """Execute navigation with improved timing"""
        if command == "forward":
            self.hardware.move_forward(duration=0.5)
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
        time.sleep(0.2)
    
    def add_status_overlay(self, frame):
        """Enhanced status overlay with centering information"""
        y = 30
        line_height = 25
        
        # Time remaining
        time_remaining = self.get_time_remaining()
        cv2.putText(frame, f"Time: {time_remaining:.0f}s", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Current state with enhanced info
        state_text = f"State: {self.state.value}"
        if self.state == RobotState.BLIND_COLLECTION and hasattr(self, 'blind_collection_drive_time'):
            state_text += f" ({self.blind_collection_drive_time:.1f}s)"
        elif self.state == RobotState.CENTERING_BALL and self.centering_start_time:
            elapsed = time.time() - self.centering_start_time
            state_text += f" ({elapsed:.1f}s)"
            if self.vision.current_target:
                centered = self.vision.is_ball_centered(self.vision.current_target)
                state_text += f" {'✓' if centered else '⊙'}"
        
        cv2.putText(frame, state_text, (10, y), 
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
        
        # Current target info with centering status
        if self.vision.current_target:
            target = self.vision.current_target
            centered = self.vision.is_ball_centered(target)
            center_status = "CENTERED" if centered else "CENTERING"
            ball_type = "ORANGE" if target.object_type == 'orange_ball' else "WHITE"
            target_info = f"Target: {ball_type} ({center_status})"
            
            color = (0, 255, 0) if centered else (0, 255, 255)
            cv2.putText(frame, target_info, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y += line_height - 5
            
            # Show centering details
            if self.state == RobotState.CENTERING_BALL:
                x_offset = target.center[0] - self.vision.frame_center_x
                cv2.putText(frame, f"Offset: {x_offset:+d}px (±{config.CENTERING_TOLERANCE})", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                y += line_height - 10
            
            # Show drive time if centered
            if centered:
                drive_time = self.vision.calculate_drive_time_to_ball(target)
                cv2.putText(frame, f"Drive Time: {drive_time:.2f}s", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def end_competition(self):
        """End competition with enhanced results"""
        self.competition_active = False
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("=" * 60)
        self.logger.info("🏁 COMPETITION ENDED!")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {elapsed_time:.1f} seconds")
        self.logger.info(f"Balls collected: {self.hardware.get_ball_count()}")
        self.logger.info(f"Final state: {self.state.value}")
        self.logger.info(f"Collection system: Enhanced (Centering + Blind Collection)")
        self.logger.info(f"Arena detection: {'Success' if self.vision.arena_detected else 'Fallback'}")
        self.logger.info(f"Centering tolerance: ±{config.CENTERING_TOLERANCE} pixels")
        self.logger.info("=" * 60)
        
        # Enhanced competition results
        competition_result = {
            "elapsed_time": elapsed_time,
            "balls_collected": self.hardware.get_ball_count(),
            "final_state": self.state.value,
            "vision_system": "hough_circles_hybrid",
            "collection_system": "enhanced_centering_blind",
            "arena_detected": self.vision.arena_detected,
            "centering_tolerance": config.CENTERING_TOLERANCE
        }
        
        summary = self.telemetry.create_session_summary(competition_result)
        self.logger.info(f"📊 Session data saved to: {summary['session_metadata']['session_dir']}")
        
        self.emergency_stop()
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        self.competition_active = False
        self.state = RobotState.EMERGENCY_STOP
        
        self.logger.warning("🛑 EMERGENCY STOP ACTIVATED")
        
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
    print("\n" + "="*70)
    print("🤖 GOLFBOT ENHANCED COLLECTION SYSTEM")
    print("="*70)
    print("1. Start Competition (Ball Centering + Blind Collection)")
    print("2. Hardware Testing") 
    print("3. Exit")
    print("="*70)
    print("🎯 ENHANCED FEATURES:")
    print(f"   • Ball centering (±{config.CENTERING_TOLERANCE}px tolerance)")
    print(f"   • Blind drive collection ({config.COLLECTION_DRIVE_TIME_PER_PIXEL:.3f}s/pixel)")
    print(f"   • Adaptive servo control (gradual movement)")
    print(f"   • Timeout protection ({config.MAX_CENTERING_TIME}s max centering)")
    print(f"   • Enhanced debugging and telemetry")
    print("="*70)
    
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
        print("👋 Goodbye!")
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
        print("\n🏁 Entering Competition Mode with Enhanced Collection...")
        
        try:
            robot = GolfBot()
            
            if not robot.initialize():
                print("❌ Failed to initialize robot - exiting")
                return 1
            
            print("\n🚀 Robot ready with enhanced collection system!")
            print("   ✓ Ball centering for precision targeting")
            print("   ✓ Blind collection to avoid vision occlusion") 
            print("   ✓ HoughCircles + Arena boundary detection")
            print("   ✓ Enhanced servo control with gradual movement")
            print("   ✓ Adaptive turning based on ball offset")
            print("   ✓ Timeout protection and fallback modes")
            print(f"\n⚙️  CONFIGURATION:")
            print(f"   • Centering tolerance: ±{config.CENTERING_TOLERANCE} pixels")
            print(f"   • Collection speed: {config.COLLECTION_SPEED}")
            print(f"   • Drive time calculation: {config.COLLECTION_DRIVE_TIME_PER_PIXEL:.4f}s/pixel")
            print(f"   • Max centering time: {config.MAX_CENTERING_TIME}s")
            print(f"   • Base turn duration: {config.CENTERING_TURN_BASE}s")
            print(f"   • Turn scale factor: {config.CENTERING_TURN_SCALE}")
            print("\n🎯 COLLECTION FLOW:")
            print("   1. SEARCHING → Find confident ball")
            print("   2. CENTERING_BALL → Align ball to center (±40px)")
            print("   3. BLIND_COLLECTION → Drive forward + close cage")
            print("   4. Return to SEARCHING")
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