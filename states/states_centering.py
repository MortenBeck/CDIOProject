import time
from typing import Optional
from .states_base import BaseState, RobotState
import config

class CenteringState1(BaseState):
    """Phase 1: Center the locked target ball in both X and Y axes for initial alignment"""
    
    def __init__(self):
        super().__init__("CENTERING_1")
    
    def enter(self, context):
        """Enter centering phase 1"""
        locked_target = context.get('locked_target')
        if locked_target:
            ball_type = "orange" if locked_target.object_type == "orange_ball" else "white"
            self.log_state_info(f"Starting Phase 1 centering for locked {ball_type} ball")
        else:
            self.log_state_info("Starting Phase 1 centering")
    
    def execute(self, context) -> Optional[RobotState]:
        """Execute centering phase 1 logic"""
        balls = context.get('balls', [])
        hardware = context['hardware']
        vision = context['vision']
        
        if not balls:
            self.log_state_info("Lost sight of ball during CENTERING_1 - returning to search")
            context['locked_target'] = None
            vision.current_target = None
            return RobotState.SEARCHING
        
        # Use locked target instead of recalculating
        target_ball = self._find_locked_target_in_balls(balls, context)
        
        if not target_ball:
            self.log_state_info("Lost locked target during CENTERING_1 - returning to search")
            context['locked_target'] = None
            vision.current_target = None
            return RobotState.SEARCHING
        
        # Update vision current target
        vision.current_target = target_ball
        
        # Check if ball is fully centered for Phase 1 (using Phase 1 tolerances)
        x_offset = abs(target_ball.center[0] - vision.frame_center_x)
        y_offset = abs(target_ball.center[1] - vision.frame_center_y)
        
        x_centered = x_offset <= config.CENTERING_1_TOLERANCE
        y_centered = y_offset <= config.CENTERING_1_DISTANCE_TOLERANCE
        
        if x_centered and y_centered:
            ball_type = "orange" if target_ball.object_type == "orange_ball" else "white"
            self.log_state_info(f"CENTERING_1 complete! Locked {ball_type} ball aligned. Starting CENTERING_2")
            return RobotState.CENTERING_2
        
        # Ball not fully centered - get centering adjustments for both axes
        # X-axis centering (left/right)
        if not x_centered:
            x_offset_signed = target_ball.center[0] - vision.frame_center_x
            if x_offset_signed > 0:
                hardware.turn_right(duration=config.CENTERING_1_TURN_DURATION, 
                                   speed=config.CENTERING_1_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.log_state_info(f"CENTERING_1 X: turning right (offset: {x_offset_signed})")
            else:
                hardware.turn_left(duration=config.CENTERING_1_TURN_DURATION, 
                                  speed=config.CENTERING_1_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.log_state_info(f"CENTERING_1 X: turning left (offset: {x_offset_signed})")
            
            time.sleep(0.03)
            return None
        
        # Y-axis centering (distance - forward/backward)
        if not y_centered:
            y_offset_signed = target_ball.center[1] - vision.frame_center_y
            if y_offset_signed > 0:
                hardware.move_backward(duration=config.CENTERING_1_DRIVE_DURATION, 
                                     speed=config.CENTERING_1_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.log_state_info(f"CENTERING_1 Y: moving backward (offset: {y_offset_signed})")
            else:
                hardware.move_forward(duration=config.CENTERING_1_DRIVE_DURATION, 
                                    speed=config.CENTERING_1_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.log_state_info(f"CENTERING_1 Y: moving forward (offset: {y_offset_signed})")
            
            time.sleep(0.03)
            return None
        
        return None  # Stay in centering_1
    
    def exit(self, context):
        """Exit centering phase 1"""
        self.log_state_info("Phase 1 centering complete")
    
    def _find_locked_target_in_balls(self, balls, context):
        """Find the locked target ball in current detections"""
        locked_target = context.get('locked_target')
        if not locked_target or not balls:
            return None
        
        # Find ball closest to locked target position
        target_pos = locked_target.center
        for ball in balls:
            if ball.confidence > 0.3:  # Lower threshold for locked target
                distance = ((ball.center[0] - target_pos[0])**2 + (ball.center[1] - target_pos[1])**2)**0.5
                if distance < 50:  # Same ball if within 50 pixels
                    return ball
        return None
    
    def should_avoid_boundary(self, context) -> bool:
        """Centering phase 1 should avoid boundaries"""
        return context.get('near_boundary', False)

class CenteringState2(BaseState):
    """Phase 2: IDENTICAL to Phase 1 centering, but targets upper green zone using locked target"""
    
    def __init__(self):
        super().__init__("CENTERING_2")
    
    def enter(self, context):
        """Enter centering phase 2"""
        locked_target = context.get('locked_target')
        if locked_target:
            ball_type = "orange" if locked_target.object_type == "orange_ball" else "white"
            self.log_state_info(f"Starting Phase 2 centering for locked {ball_type} ball - targeting collection zone")
        else:
            self.log_state_info("Starting Phase 2 centering - targeting collection zone")
    
    def execute(self, context) -> Optional[RobotState]:
        """Execute centering phase 2 logic"""
        balls = context.get('balls', [])
        hardware = context['hardware']
        vision = context['vision']
        
        if not balls:
            self.log_state_info("Lost sight of ball during CENTERING_2 - returning to search")
            context['locked_target'] = None
            vision.current_target = None
            return RobotState.SEARCHING
        
        # Use locked target instead of recalculating
        target_ball = self._find_locked_target_in_balls(balls, context)
        
        if not target_ball:
            self.log_state_info("Lost locked target during CENTERING_2 - returning to search")
            context['locked_target'] = None
            vision.current_target = None
            return RobotState.SEARCHING
        
        # Update vision current target
        vision.current_target = target_ball
        
        # Set servo to pre-collection position (only on first entry to this state)
        current_servo_state = hardware.get_servo_ss_state()
        if current_servo_state != "pre-collect":
            ball_type = "orange" if target_ball.object_type == "orange_ball" else "white"
            self.log_state_info(f"CENTERING_2: Setting servo SS to PRE-COLLECT for locked {ball_type} ball")
            hardware.servo_ss_to_pre_collect()
            time.sleep(0.2)
        
        # Calculate target position (upper green zone instead of frame center)
        target_x = vision.frame_center_x  # Same X target as Phase 1
        target_y = vision.frame_center_y + config.CENTERING_2_Y_TARGET_OFFSET  # Offset into green zone
        
        # Check if ball is fully centered for Phase 2 (using Phase 2 tolerances)
        x_offset = abs(target_ball.center[0] - target_x)
        y_offset = abs(target_ball.center[1] - target_y)
        
        x_centered = x_offset <= config.CENTERING_2_TOLERANCE
        y_centered = y_offset <= config.CENTERING_2_DISTANCE_TOLERANCE
        
        if x_centered and y_centered:
            ball_type = "orange" if target_ball.object_type == "orange_ball" else "white"
            self.log_state_info(f"CENTERING_2 complete! Locked {ball_type} ball centered in upper green zone. Starting collection")
            return RobotState.COLLECTING_BALL
        
        # Ball not fully centered - IDENTICAL centering logic to Phase 1
        # X-axis centering (left/right)
        if not x_centered:
            x_offset_signed = target_ball.center[0] - target_x
            if x_offset_signed > 0:
                hardware.turn_right(duration=config.CENTERING_2_TURN_DURATION, 
                                   speed=config.CENTERING_2_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.log_state_info(f"CENTERING_2 X: turning right (offset: {x_offset_signed})")
            else:
                hardware.turn_left(duration=config.CENTERING_2_TURN_DURATION, 
                                  speed=config.CENTERING_2_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.log_state_info(f"CENTERING_2 X: turning left (offset: {x_offset_signed})")
            
            time.sleep(0.03)
            return None
        
        # Y-axis centering (distance - forward/backward) 
        if not y_centered:
            y_offset_signed = target_ball.center[1] - target_y
            if y_offset_signed > 0:
                hardware.move_backward(duration=config.CENTERING_2_DRIVE_DURATION, 
                                     speed=config.CENTERING_2_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.log_state_info(f"CENTERING_2 Y: moving backward (offset: {y_offset_signed})")
            else:
                hardware.move_forward(duration=config.CENTERING_2_DRIVE_DURATION, 
                                    speed=config.CENTERING_2_SPEED)
                if config.DEBUG_MOVEMENT:
                    self.log_state_info(f"CENTERING_2 Y: moving forward (offset: {y_offset_signed})")
            
            time.sleep(0.03)
            return None
        
        return None  # Stay in centering_2
    
    def exit(self, context):
        """Exit centering phase 2"""
        self.log_state_info("Phase 2 centering complete - ready for collection")
    
    def _find_locked_target_in_balls(self, balls, context):
        """Find the locked target ball in current detections"""
        locked_target = context.get('locked_target')
        if not locked_target or not balls:
            return None
        
        # Find ball closest to locked target position
        target_pos = locked_target.center
        for ball in balls:
            if ball.confidence > 0.3:  # Lower threshold for locked target
                distance = ((ball.center[0] - target_pos[0])**2 + (ball.center[1] - target_pos[1])**2)**0.5
                if distance < 50:  # Same ball if within 50 pixels
                    return ball
        return None
    
    def should_avoid_boundary(self, context) -> bool:
        """Centering phase 2 should NOT avoid boundaries - allow collection to complete"""
        # During centering phase 2, we're likely close to the collection zone
        # Don't interrupt for boundary avoidance unless critical
        return False
