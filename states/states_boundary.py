import time
from typing import Optional
from .states_base import BaseState, RobotState
from vision import BoundaryAvoidanceSystem

class BoundaryAvoidanceState(BaseState):
    """State for avoiding arena boundaries and walls"""
    
    def __init__(self):
        super().__init__("AVOIDING_BOUNDARY")
        self.boundary_system = BoundaryAvoidanceSystem()
        self.avoidance_start_time = None
        self.preferred_direction = None  # 'left' or 'right' - stick to one direction
    
    def enter(self, context):
        """Enter boundary avoidance state"""
        self.log_state_info("⚠️  Wall/boundary detected - executing smart avoidance")
        self.avoidance_start_time = time.time()
        self.boundary_system.reset()
        self.preferred_direction = None  # Reset direction preference
    
    def execute(self, context) -> Optional[RobotState]:
        """Execute smart boundary/wall avoidance using vision system"""
        hardware = context['hardware']
        frame = context.get('current_frame')
        
        if frame is None:
            self.log_state_info("No camera frame available - basic avoidance")
            hardware.move_backward(duration=0.4, speed=0.3)  # Slower basic backing
            hardware.turn_right(duration=0.6, speed=0.4)  # Slower basic turn
            return RobotState.SEARCHING
        
        # Get avoidance command from boundary system
        raw_avoidance_command = self.boundary_system.get_avoidance_command(frame)
        
        # Apply direction preference to avoid oscillation
        avoidance_command = self._get_consistent_avoidance_command(raw_avoidance_command)
        
        if avoidance_command:
            self.log_state_info(f"Executing avoidance command: {avoidance_command} (raw: {raw_avoidance_command})")
            
            # Stop current movement first
            hardware.stop_motors()
            time.sleep(0.1)
            
            # Execute the appropriate avoidance maneuver
            if avoidance_command == 'turn_left':
                hardware.turn_left(duration=1.5)
                self.log_state_info("Turned left away from right wall")
            elif avoidance_command == 'turn_right':
                hardware.turn_right(duration=1.5)
                self.log_state_info("Turned right away from left wall")
            elif avoidance_command == 'move_backward':
                hardware.move_backward(duration=0.5, speed=0.3)  # Slower backing
                hardware.turn_right(duration=0.8, speed=0.4)  # Slower turn after backing
                self.log_state_info("Moved backward slowly and turned away from wall")
            
            # Brief pause to stabilize
            time.sleep(0.2)
        
        # Check if we've been avoiding for too long
        elapsed_time = time.time() - self.avoidance_start_time
        if elapsed_time > 2.0:  # Max 2 seconds in avoidance
            self.log_state_info("Avoidance timeout - forcing exit to search")
            return RobotState.SEARCHING
        
        # Check if walls are still detected
        still_dangerous = self.boundary_system.detect_boundaries(frame)
        if not still_dangerous:
            self.log_state_info("No more walls detected - avoidance complete")
            return RobotState.SEARCHING
        
        # Continue avoiding if walls still detected
        return None
    
    def _get_consistent_avoidance_command(self, raw_command):
        """Apply direction preference to avoid oscillation between left/right"""
        if raw_command is None:
            return None
            
        # If we don't have a preferred direction yet, set it based on the first command
        if self.preferred_direction is None:
            if raw_command in ['turn_left', 'turn_right']:
                self.preferred_direction = raw_command.replace('turn_', '')
                self.log_state_info(f"Setting preferred avoidance direction: {self.preferred_direction}")
            return raw_command
        
        # If the raw command wants to turn in the opposite direction, stick to our preference
        if raw_command == 'turn_left' and self.preferred_direction == 'right':
            self.log_state_info("Overriding turn_left with preferred direction: turn_right")
            return 'turn_right'
        elif raw_command == 'turn_right' and self.preferred_direction == 'left':
            self.log_state_info("Overriding turn_right with preferred direction: turn_left")
            return 'turn_left'
        
        # For backward movement or same direction, use the raw command
        return raw_command
    
    def exit(self, context):
        """Exit boundary avoidance state"""
        self.log_state_info("Exiting boundary avoidance - resuming normal operation")
        # Clear any locked target since we may have moved significantly
        context['locked_target'] = None
        if 'vision' in context:
            context['vision'].current_target = None
        # Reset direction preference for next avoidance session
        self.preferred_direction = None
    
    def should_avoid_boundary(self, context) -> bool:
        """Check if boundary avoidance is needed using vision system"""
        frame = context.get('current_frame')
        if frame is None:
            return False
        
        return self.boundary_system.detect_boundaries(frame)
    
    def get_boundary_status(self, context) -> dict:
        """Get current boundary detection status for debugging"""
        frame = context.get('current_frame')
        if frame is None:
            return {'error': 'No camera frame'}
        
        return self.boundary_system.get_status()
