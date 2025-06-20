from typing import Optional
from .states_base import BaseState, RobotState

class CollectingState(BaseState):
    """State for executing ball collection sequence"""
    
    def __init__(self):
        super().__init__("COLLECTING_BALL")
    
    def enter(self, context):
        """Enter collection state"""
        current_target = context['vision'].current_target
        
        if current_target:
            ball_type = "orange" if current_target.object_type == "orange_ball" else "regular"
            confidence = current_target.confidence
            self.log_state_info(f"Starting optimized collection of {ball_type} ball in collection zone (confidence: {confidence:.2f})...")
        else:
            self.log_state_info("Starting optimized ball collection in collection zone...")
    
    def execute(self, context) -> Optional[RobotState]:
        """Execute ball collection with optimized sequence for collection zone"""
        hardware = context['hardware']
        vision = context['vision']
        current_target = vision.current_target
        
        # Determine ball type for logging
        if current_target:
            ball_type = "orange" if current_target.object_type == "orange_ball" else "regular"
        else:
            ball_type = "unknown"
        
        # Use optimized collection sequence with collection zone settings
        success = hardware.optimized_collection_sequence()
        
        if success:
            total_balls = hardware.get_ball_count()
            self.log_state_info(f"✅ {ball_type.title()} ball collected with optimized sequence! Total: {total_balls}")
        else:
            self.log_state_info(f"❌ {ball_type.title()} optimized collection failed")
        
        # Clear the locked target and return to searching
        context['locked_target'] = None
        vision.current_target = None
        
        return RobotState.SEARCHING
    
    def exit(self, context):
        """Exit collection state"""
        self.log_state_info("Collection sequence complete - returning to search")
    
    def should_avoid_boundary(self, context) -> bool:
        """Collection state should NOT avoid boundaries - let collection complete"""
        # During collection, we don't want to interrupt the sequence
        # The collection is quick and we're likely in the safe collection zone
        return False
