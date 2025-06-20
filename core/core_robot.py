import time
import logging
import cv2
import signal
import sys
import os
from typing import Optional

import config
from hardware import GolfBotHardware
from vision import VisionSystem, GolfBotDashboard, BoundaryAvoidanceSystem
from .core_state_machine import StateMachine
from .core_competition import CompetitionManager
from states import RobotState

class GolfBot:
    """Main robot coordinator - brings together all subsystems"""
    
    def __init__(self, use_dashboard=True):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Check if display is available
        self.display_available = self.check_display_available()
        if not self.display_available:
            self.logger.info("No display detected - running in headless mode")
        
        # Dashboard mode
        self.use_dashboard = use_dashboard and self.display_available
        if self.use_dashboard:
            try:
                self.dashboard = GolfBotDashboard()
                self.logger.info("Using new dashboard interface")
            except ImportError:
                self.dashboard = None
                self.use_dashboard = False
                self.logger.warning("Dashboard not available - using legacy overlay interface")
        else:
            self.dashboard = None
            self.logger.info("Using legacy overlay interface")
        
        # Initialize subsystems
        self.hardware = GolfBotHardware()
        self.vision = VisionSystem()
        self.boundary_avoidance = BoundaryAvoidanceSystem()
        self.state_machine = StateMachine()
        self.competition = CompetitionManager()
        
        # Context for state execution
        self.context = {
            'hardware': self.hardware,
            'vision': self.vision,
            'boundary_avoidance': self.boundary_avoidance,
            'competition': self.competition,
            'current_state': RobotState.SEARCHING,
            'locked_target': None,
            'balls': [],
            'near_boundary': False,
            'current_frame': None,
        }
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_skip_counter = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
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
            
            # Initialize servos for competition
            self.hardware.initialize_servos_for_competition()
            
            self.logger.info("All systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def show_ready_message(self):
        """Show ready message with configuration details"""
        print("\n🚀 Robot ready with two-phase collection system!")
        print("   - Phase 1: X+Y centering for ball alignment")
        print("   - Phase 2: Collection zone positioning with servo pre-collect") 
        print("   - Optimized collection sequence in green zone")
        
        interface_mode = "Dashboard" if self.use_dashboard else "Legacy Overlay"
        print(f"   - {interface_mode} interface for monitoring")
        
        print(f"\n⚙️  Configuration:")
        print(f"   - Phase 1 tolerances: ±{config.CENTERING_1_TOLERANCE}px X, ±{config.CENTERING_1_DISTANCE_TOLERANCE}px Y")
        print(f"   - Phase 2 centering: ±{config.CENTERING_2_TOLERANCE}px X, ±{config.CENTERING_2_DISTANCE_TOLERANCE}px Y")
        print(f"   - Phase 2 speed: {config.CENTERING_2_SPEED}")
        print(f"   - Collection: {config.CENTERING_2_COLLECTION_SPEED} speed, {config.CENTERING_2_COLLECTION_TIME}s time")
        print(f"   - Servo pre-collect: {config.SERVO_SS_PRE_COLLECT}°")
        print(f"   - Interface mode: {interface_mode}")
    
    def start_competition(self):
        """Start the competition and main loop"""
        self.competition.start_competition()
        
        try:
            self.main_loop()
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            self.emergency_stop()
    
    def main_loop(self):
        """Main competition control loop"""
        last_status_log = time.time()
        status_log_interval = 30  # Log status every 30 seconds
        
        while self.competition.is_active and not self.competition.is_time_up():
            try:
                frame_start = time.time()
                
                # Skip frames for performance (process every 2nd frame)
                self.frame_skip_counter += 1
                if self.frame_skip_counter % 2 != 0:
                    time.sleep(0.05)
                    continue
                
                # Get current raw frame for wall avoidance
                ret, raw_frame = self.vision.get_frame()
                if not ret:
                    continue
                
                # Get current vision data
                balls, _, near_boundary, nav_command, debug_frame = self.vision.process_frame(
                    dashboard_mode=self.use_dashboard
                )
                
                if balls is None:
                    continue
                
                # Check for wall avoidance (higher priority than boundary detection)
                wall_danger = self.boundary_avoidance.detect_boundaries(raw_frame)
                if wall_danger and self.state_machine.get_current_state() != RobotState.AVOIDING_BOUNDARY:
                    self.logger.info("Wall danger detected - switching to boundary avoidance")
                    self.state_machine.transition_to(RobotState.AVOIDING_BOUNDARY)
                
                # Update context with all data including current frame
                self.context.update({
                    'balls': balls,
                    'near_boundary': near_boundary,
                    'current_state': self.state_machine.get_current_state(),
                    'current_frame': raw_frame,
                    'wall_danger': wall_danger
                })
                
                # Execute state machine
                self.state_machine.execute_state(self.context)
                
                # Update performance tracking
                frame_time = time.time() - frame_start
                fps = 1.0 / (time.time() - self.last_frame_time) if self.last_frame_time else 0
                self.last_frame_time = time.time()
                
                # Show display based on mode
                if config.SHOW_CAMERA_FEED and self.display_available:
                    try:
                        # Add wall avoidance visualization to debug frame
                        if debug_frame is not None and debug_frame.size > 0:
                            debug_frame = self.boundary_avoidance.draw_boundary_visualization(debug_frame)
                        
                        if self.use_dashboard and self.dashboard:
                            # Get wall status for dashboard
                            wall_status = self.boundary_avoidance.get_status()
                            dashboard_frame = self.dashboard.create_dashboard(
                                debug_frame, self.state_machine.get_current_state(), 
                                self.vision, self.hardware, wall_status
                            )
                            key = self.dashboard.show("GolfBot Dashboard - Two-Phase Collection")
                        else:
                            if debug_frame is not None and debug_frame.size > 0:
                                self._add_legacy_status_overlay(debug_frame)
                                cv2.imshow('GolfBot Debug - Two-Phase Collection', debug_frame)
                                key = cv2.waitKey(1) & 0xFF
                            else:
                                key = -1
                        
                        if key == ord('q'):
                            break
                            
                    except Exception as e:
                        self.logger.warning(f"Display error: {e}")
                        self.display_available = False
                
                # Periodic status logging
                current_time = time.time()
                if current_time - last_status_log >= status_log_interval:
                    self.competition.log_status_update(self.hardware)
                    self.state_machine.log_state_summary()
                    last_status_log = current_time
                
                # Check time warnings
                self.competition.check_time_warnings()
                
                # Adaptive sleep based on current state
                current_state = self.state_machine.get_current_state()
                if current_state in [RobotState.CENTERING_1, RobotState.CENTERING_2]:
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
    
    def _add_legacy_status_overlay(self, frame):
        """Enhanced status overlay for legacy mode"""
        y = 30
        line_height = 25
        
        # Competition status
        remaining = self.competition.get_time_remaining()
        cv2.putText(frame, f"Time: {remaining:.0f}s", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Current state with phase info
        current_state = self.state_machine.get_current_state()
        state_text = f"State: {current_state.value.replace('_', ' ').title()}"
        
        if current_state == RobotState.CENTERING_1:
            state_text += " (X+Y Align)"
        elif current_state == RobotState.CENTERING_2:
            state_text += " (Zone Position)"
        elif current_state == RobotState.COLLECTING_BALL:
            state_text += " (Optimized)"
        
        # Add centering info if in centering states
        if (current_state in [RobotState.CENTERING_1, RobotState.CENTERING_2] and 
            self.vision.current_target):
            if current_state == RobotState.CENTERING_1:
                x_offset = abs(self.vision.current_target.center[0] - self.vision.frame_center_x)
                y_offset = abs(self.vision.current_target.center[1] - self.vision.frame_center_y)
                x_ok = x_offset <= config.CENTERING_1_TOLERANCE
                y_ok = y_offset <= config.CENTERING_1_DISTANCE_TOLERANCE
                status_char = f"{'✓' if x_ok else 'X'}{'✓' if y_ok else 'Y'}"
                state_text += f" ({status_char})"
            elif current_state == RobotState.CENTERING_2:
                in_zone = self.vision.is_in_collection_zone(self.vision.current_target.center)
                state_text += f" ({'✓' if in_zone else '→'})"
        
        cv2.putText(frame, state_text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Ball count
        ball_count = self.hardware.get_ball_count()
        cv2.putText(frame, f"Balls Collected: {ball_count}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Arena status
        arena_status = "Detected" if self.vision.arena_detected else "Fallback"
        cv2.putText(frame, f"Arena: {arena_status}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += line_height
        
        # Two-phase collection info
        cv2.putText(frame, f"Collection: Two-Phase (C1→C2→Collect)", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def end_competition(self):
        """End competition and cleanup"""
        self.competition.end_competition(self.hardware)
        self.emergency_stop()
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        try:
            self.state_machine.emergency_stop(self.context)
            self.hardware.emergency_stop()
            self.vision.cleanup()
            self.hardware.cleanup()
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            'competition': self.competition.get_competition_status(),
            'state_machine': self.state_machine.get_state_summary(),
            'hardware': self.hardware.get_status(),
            'vision': {
                'arena_detected': self.vision.arena_detected,
                'current_target': self.vision.current_target is not None,
                'detection_method': getattr(self.vision, 'detection_method', 'hybrid')
            },
            'display': {
                'available': self.display_available,
                'dashboard_mode': self.use_dashboard
            }
        }
