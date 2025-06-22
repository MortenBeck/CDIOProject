import time
import logging
import cv2
import numpy as np
from typing import Optional
import config
from robot_state_machine import RobotStateMachine, RobotState

class CompetitionManager:
    """Manages the competition loop, timing, and display with delivery cycle"""
    
    def __init__(self, hardware, vision, use_dashboard=True):
        self.logger = logging.getLogger(__name__)
        self.hardware = hardware
        self.vision = vision
        self.use_dashboard = use_dashboard
        
        # Import dashboard if available
        try:
            from dashboard import GolfBotDashboard
            if use_dashboard:
                self.dashboard = GolfBotDashboard()
                self.logger.info("Using new dashboard interface")
            else:
                self.dashboard = None
        except ImportError:
            self.dashboard = None
            self.logger.warning("Dashboard not available - using legacy overlay mode")
        
        # Initialize state machine
        self.state_machine = RobotStateMachine(hardware, vision)
        
        # Competition state
        self.start_time = None
        self.competition_active = False
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_skip_counter = 0
        
        # Display availability
        self.display_available = self._check_display_available()
        if not self.display_available:
            self.logger.info("No display detected - running in headless mode")

    def _check_display_available(self):
        """Check if display/X11 is available"""
        try:
            import os
            if os.environ.get('DISPLAY') is None:
                return False
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('test')
            return True
        except Exception:
            return False

    def start_competition(self):
        """Start the competition timer and main loop"""
        self.start_time = time.time()
        self.competition_active = True
        self.state_machine.state = RobotState.SEARCHING
        
        self.logger.info("COMPETITION STARTED - WHITE BALLS COLLECTION + DELIVERY CYCLE!")
        self.logger.info(f"Time limit: {config.COMPETITION_TIME} seconds")
        self.logger.info(f"Delivery trigger: {config.BALLS_BEFORE_DELIVERY} balls")
        self.logger.info("Using enhanced collection: Ball centering (X+Y) + Enhanced sequence")
        
        try:
            self.main_loop()
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.emergency_stop()

    def main_loop(self):
        """Main competition control loop with delivery cycle - WHITE BALLS ONLY"""
        while self.competition_active and not self.is_time_up():
            try:
                frame_start = time.time()
                
                # Skip frames for performance (process every 2nd frame)
                self.frame_skip_counter += 1
                if self.frame_skip_counter % 2 != 0:
                    time.sleep(0.05)
                    continue
                
                # Get current vision data
                balls, _, near_boundary, nav_command, debug_frame = self.vision.process_frame(
                    dashboard_mode=self.use_dashboard and self.dashboard is not None
                )
                
                if balls is None:  # Frame capture failed
                    continue
                
                # Store detected balls for dashboard access
                self.vision._last_detected_balls = balls if balls else []
                
                # Update ball tracking
                if balls:
                    self.state_machine.last_ball_seen_time = time.time()
                    high_confidence_balls = [b for b in balls if b.confidence > 0.5]
                    if high_confidence_balls:
                        self.logger.debug(f"High confidence white balls: {len(high_confidence_balls)}")
                
                # Performance tracking
                frame_time = time.time() - frame_start
                fps = 1.0 / (time.time() - self.last_frame_time) if self.last_frame_time else 0
                self.last_frame_time = time.time()
                
                # Show display based on mode
                if config.SHOW_CAMERA_FEED and self.display_available:
                    key = self._handle_display(debug_frame)
                    if key == ord('q'):
                        break
                
                # State machine execution
                old_state = self.state_machine.state
                self.state_machine.execute_state_machine(balls, near_boundary, nav_command)
                
                # Adaptive sleep based on detection results and state
                if self.state_machine.state == RobotState.CENTERING_BALL:
                    time.sleep(0.03)  # Faster when centering
                elif self.state_machine.state in [RobotState.DELIVERY_MODE, RobotState.POST_DELIVERY_TURN]:
                    time.sleep(0.1)   # Normal when in delivery cycle
                elif balls and len(balls) > 0:
                    time.sleep(0.05)  # Faster when balls detected
                else:
                    time.sleep(0.1)   # Slower when searching
                
            except Exception as e:
                self.logger.error(f"Main loop iteration error: {e}")
                self.hardware.stop_motors()
                time.sleep(0.5)
        
        self.end_competition()

    def _handle_display(self, debug_frame):
        """Handle display rendering and user input"""
        try:
            if self.use_dashboard and self.dashboard:
                # NEW DASHBOARD MODE
                dashboard_frame = self.dashboard.create_dashboard(
                    debug_frame, self.state_machine.state, self.vision, self.hardware, None
                )
                key = self.dashboard.show("GolfBot Dashboard - White Ball Collection + Delivery")
            else:
                # LEGACY OVERLAY MODE  
                if debug_frame is not None and debug_frame.size > 0:
                    self._add_status_overlay(debug_frame)
                    cv2.imshow('GolfBot Debug - White Ball Collection + Delivery', debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = -1
            
            return key
            
        except Exception as e:
            self.logger.warning(f"Display error: {e}")
            self.display_available = False
            return -1

    def _add_status_overlay(self, frame):
        """LEGACY: Enhanced status overlay with delivery cycle info - WHITE BALLS ONLY"""
        y = 30
        line_height = 25
        
        # Time remaining
        time_remaining = self.get_time_remaining()
        cv2.putText(frame, f"Time: {time_remaining:.0f}s", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Current state with enhanced info
        state_text = f"State: {self.state_machine.state.value.replace('_', ' ').title()}"
        if self.state_machine.state == RobotState.COLLECTING_BALL:
            state_text += " (Enhanced)"
        elif self.state_machine.state == RobotState.CENTERING_BALL and self.vision.current_target:
            x_dir, y_dir = self.vision.get_centering_adjustment(self.vision.current_target)
            centered = self.vision.is_ball_centered(self.vision.current_target)
            state_text += f" ({'✓' if centered else f'{x_dir[:1].upper()}{y_dir[:1].upper()}'})"
        elif self.state_machine.state == RobotState.DELIVERY_MODE:
            state_text += " (Releasing Balls)"
        elif self.state_machine.state == RobotState.POST_DELIVERY_TURN:
            state_text += " (Turning Right)"
        
        cv2.putText(frame, state_text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Ball count with delivery progress
        ball_count = self.hardware.get_ball_count()
        delivery_target = config.BALLS_BEFORE_DELIVERY
        progress_text = f"White Balls: {ball_count}/{delivery_target}"
        
        # Color based on progress and state
        if self.state_machine.state in [RobotState.DELIVERY_MODE, RobotState.POST_DELIVERY_TURN]:
            progress_color = (255, 0, 255)  # Magenta during delivery cycle
            if self.state_machine.state == RobotState.DELIVERY_MODE:
                progress_text += " (DELIVERING)"
            else:
                progress_text += " (TURNING)"
        elif ball_count >= delivery_target:
            progress_color = (0, 255, 0)  # Green when ready for delivery
            progress_text += " (DELIVERY READY)"
        elif ball_count >= delivery_target - 1:
            progress_color = (0, 255, 255)  # Yellow when close
        else:
            progress_color = (255, 255, 255)  # White normally
        
        cv2.putText(frame, progress_text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, progress_color, 2)
        y += line_height
        
        # Delivery cycle info
        cycle_text = f"Delivery cycle: {config.BALLS_BEFORE_DELIVERY} balls -> Release -> Turn -> Repeat"
        cv2.putText(frame, cycle_text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
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

    def get_time_remaining(self) -> float:
        """Get remaining competition time"""
        if not self.start_time:
            return config.COMPETITION_TIME
        elapsed = time.time() - self.start_time
        return max(0, config.COMPETITION_TIME - elapsed)

    def is_time_up(self) -> bool:
        """Check if competition time is up"""
        return self.get_time_remaining() <= 0

    def end_competition(self):
        """End competition with enhanced results - WHITE BALLS ONLY"""
        self.competition_active = False
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("=" * 60)
        self.logger.info("COMPETITION ENDED - WHITE BALLS + DELIVERY CYCLE!")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {elapsed_time:.1f} seconds")
        self.logger.info(f"White balls collected: {self.hardware.get_ball_count()}")
        self.logger.info(f"Delivery target: {config.BALLS_BEFORE_DELIVERY} balls per cycle")
        self.logger.info(f"Final state: {self.state_machine.state.value}")
        self.logger.info(f"Collection system: Enhanced (X+Y Centering + Servo Sequence)")
        self.logger.info(f"Arena detection: {'Success' if self.vision.arena_detected else 'Fallback'}")
        self.logger.info(f"Boundary avoidance: Modular system")
        self.logger.info("=" * 60)
        
        # Enhanced competition results
        competition_result = {
            "elapsed_time": elapsed_time,
            "balls_collected": self.hardware.get_ball_count(),
            "delivery_target": config.BALLS_BEFORE_DELIVERY,
            "final_state": self.state_machine.state.value,
            "vision_system": "hough_circles_hybrid_white_only",
            "collection_system": "enhanced_xy_centering_servo_sequence",
            "boundary_system": "modular_avoidance_system",
            "delivery_system": "cycle_based_collection_delivery",
            "arena_detected": self.vision.arena_detected
        }
        
        self.emergency_stop()

    def emergency_stop(self):
        """Emergency stop all systems"""
        self.competition_active = False
        self.state_machine.emergency_stop()
        
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        try:
            self.hardware.emergency_stop()
            self.vision.cleanup()
            self.hardware.cleanup()
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")

    # Delegate state machine properties for compatibility
    @property
    def state(self):
        """Get current robot state"""
        return self.state_machine.state
    
    @state.setter
    def state(self, value):
        """Set robot state"""
        self.state_machine.state = value