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
    
    def enter(self, context):
        """Enter boundary avoidance state"""
        self.log_state_info("⚠️  Wall/boundary detected - executing smart avoidance")
        self.avoidance_start_time = time.time()
        self.boundary_system.reset()
    
    def execute(self, context) -> Optional[RobotState]:
        """Execute smart boundary/wall avoidance using vision system"""
        hardware = context['hardware']
        frame = context.get('current_frame')
        
        if frame is None:
            self.log_state_info("No camera frame available - basic avoidance")
            hardware.move_backward(duration=0.3)
            hardware.turn_right(duration=0.3)
            return RobotState.SEARCHING
        
        # Get avoidance command from boundary system
        avoidance_command = self.boundary_system.get_avoidance_command(frame)
        
        if avoidance_command:
            self.log_state_info(f"Executing avoidance command: {avoidance_command}")
            
            # Stop current movement first
            hardware.stop_motors()
            time.sleep(0.1)
            
            # Execute the appropriate avoidance maneuver
            if avoidance_command == 'turn_left':
                hardware.turn_left(duration=0.4)
                self.log_state_info("Turned left away from right wall")
            elif avoidance_command == 'turn_right':
                hardware.turn_right(duration=0.4)
                self.log_state_info("Turned right away from left wall")
            elif avoidance_command == 'move_backward':
                hardware.move_backward(duration=0.4)
                hardware.turn_right(duration=0.3)  # Add turn after backing up
                self.log_state_info("Moved backward and turned away from wall")
            
            # Brief pause to stabilize
            time.sleep(0.2)
        
        # Check if we've been avoiding for too long
        elapsed_time = time.time() - self.avoidance_start_time
        if elapsed_time > 3.0:  # Max 3 seconds in avoidance
            self.log_state_info("Avoidance timeout - returning to search")
            return RobotState.SEARCHING
        
        # Check if walls are still detected
        still_dangerous = self.boundary_system.detect_boundaries(frame)
        if not still_dangerous:
            self.log_state_info("No more walls detected - avoidance complete")
            return RobotState.SEARCHING
        
        # Continue avoiding if walls still detected
        return None
    
    def exit(self, context):
        """Exit boundary avoidance state"""
        self.log_state_info("Exiting boundary avoidance - resuming normal operation")
        # Clear any locked target since we may have moved significantly
        context['locked_target'] = None
        if 'vision' in context:
            context['vision'].current_target = None
    
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
