import time
from typing import Optional
from .states_base import BaseState, RobotState
import config

class SearchingState(BaseState):
    """State for searching for balls when none are targeted"""
    
    def __init__(self):
        super().__init__("SEARCHING")
        self.search_pattern_index = 0
        self.last_ball_seen_time = None
    
    def enter(self, context):
        """Enter searching state"""
        self.log_state_info("Entering search mode - looking for balls")
        # Clear any previous target
        if 'vision' in context:
            context['vision'].current_target = None
        context['locked_target'] = None
    
    def execute(self, context) -> Optional[RobotState]:
        """Search for balls and transition to centering if found"""
        balls = context.get('balls', [])
        near_boundary = context.get('near_boundary', False)
        hardware = context['hardware']
        vision = context['vision']
        
        # Check for boundary avoidance first
        if near_boundary:
            return RobotState.AVOIDING_BOUNDARY
        
        # Look for balls
        if balls:
            # Use lower confidence threshold for initial ball detection
            confident_balls = [ball for ball in balls if ball.confidence > 0.5]
            
            if confident_balls:
                ball_count = len(confident_balls)
                orange_count = sum(1 for ball in confident_balls if ball.object_type == 'orange_ball')
                white_count = ball_count - orange_count
                
                avg_confidence = sum(ball.confidence for ball in confident_balls) / ball_count
                
                # Lock onto the closest ball to prevent switching
                closest_ball = min(confident_balls, key=lambda x: x.distance_from_center)
                context['locked_target'] = closest_ball
                vision.current_target = closest_ball
                
                self.log_state_info(f"Found {ball_count} confident ball(s) - {white_count} white, {orange_count} orange (avg conf: {avg_confidence:.2f})")
                self.log_state_info(f"Locked onto {'orange' if closest_ball.object_type == 'orange_ball' else 'white'} ball for centering sequence")
                
                return RobotState.CENTERING_1
        
        # No confident balls found - execute search pattern
        self._execute_search_pattern(hardware)
        return None  # Stay in searching state
    
    def exit(self, context):
        """Exit searching state"""
        self.log_state_info("Exiting search mode - target acquired")
    
    def _execute_search_pattern(self, hardware):
        """Execute the configured search pattern"""
        pattern = config.SEARCH_PATTERN
        action = pattern[self.search_pattern_index % len(pattern)]
        
        if action == "forward":
            hardware.forward_step()
            if config.DEBUG_MOVEMENT:
                self.log_state_info("Search pattern: forward step")
        elif action == "turn_right":
            hardware.turn_90_right()
            if config.DEBUG_MOVEMENT:
                self.log_state_info("Search pattern: turn right 90°")
        elif action == "turn_left":
            hardware.turn_90_left()
            if config.DEBUG_MOVEMENT:
                self.log_state_info("Search pattern: turn left 90°")
        
        self.search_pattern_index += 1
        time.sleep(0.2)
    
    def should_avoid_boundary(self, context) -> bool:
        """Searching state should avoid boundaries"""
        return context.get('near_boundary', False)
