import time
from typing import Optional
from .base import BaseState, RobotState

class BoundaryAvoidanceState(BaseState):
    """State for avoiding arena boundaries and walls"""
    
    def __init__(self):
        super().__init__("AVOIDING_BOUNDARY")
    
    def enter(self, context):
        """Enter boundary avoidance state"""
        self.log_state_info("⚠️  Near arena boundary - executing avoidance maneuver")
    
    def execute(self, context) -> Optional[RobotState]:
        """Execute boundary avoidance maneuver"""
        near_boundary = context.get('near_boundary', False)
        hardware = context['hardware']
        
        if near_boundary:
            self.log_state_info("Executing boundary avoidance sequence")
            
            # Stop current movement
            hardware.stop_motors()
            time.sleep(0.1)
            
            # Back away from boundary
            hardware.move_backward(duration=0.25)
            self.log_state_info("Moved backward to avoid boundary")
            
            # Turn to change direction
            hardware.turn_right(duration=0.25)
            self.log_state_info("Turned right to change direction")
            
            # Brief pause to stabilize
            time.sleep(0.15)
            
            self.log_state_info("Boundary avoidance complete")
        
        # Check if still near boundary
        if not near_boundary:
            self.log_state_info("Boundary cleared - returning to search")
            return RobotState.SEARCHING
        
        return None  # Stay in boundary avoidance if still near boundary
    
    def exit(self, context):
        """Exit boundary avoidance state"""
        self.log_state_info("Exiting boundary avoidance - resuming normal operation")
        # Clear any locked target since we may have moved significantly
        context['locked_target'] = None
        if 'vision' in context:
            context['vision'].current_target = None
    
    def should_avoid_boundary(self, context) -> bool:
        """Boundary avoidance state always handles boundaries"""
        return True  # We're already handling boundaries
