import logging
from typing import Dict, Optional
from states import (
    BaseState, RobotState, SearchingState, CenteringState1, 
    CenteringState2, CollectingState, BoundaryAvoidanceState
)

class StateMachine:
    """Manages robot state transitions and execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_state = RobotState.SEARCHING
        self.previous_state = None
        
        # Initialize all state handlers
        self.states: Dict[RobotState, BaseState] = {
            RobotState.SEARCHING: SearchingState(),
            RobotState.CENTERING_1: CenteringState1(),
            RobotState.CENTERING_2: CenteringState2(),
            RobotState.COLLECTING_BALL: CollectingState(),
            RobotState.AVOIDING_BOUNDARY: BoundaryAvoidanceState(),
        }
        
        # Track state entry for proper enter/exit calls
        self._state_entered = False
        
        self.logger.info("State machine initialized with two-phase collection system")
    
    def get_current_state(self) -> RobotState:
        """Get current state"""
        return self.current_state
    
    def get_current_state_handler(self) -> BaseState:
        """Get current state handler"""
        return self.states[self.current_state]
    
    def execute_state(self, context: dict) -> Optional[RobotState]:
        """Execute current state and handle transitions"""
        try:
            # Get current state handler
            state_handler = self.states[self.current_state]
            
            # Call enter() if we just transitioned to this state
            if not self._state_entered:
                state_handler.enter(context)
                self._state_entered = True
            
            # Emergency stop check
            if self.current_state == RobotState.EMERGENCY_STOP:
                return self.current_state
            
            # Check for boundary avoidance (unless already avoiding or in critical states)
            if (self.current_state not in [RobotState.AVOIDING_BOUNDARY, RobotState.CENTERING_2, RobotState.COLLECTING_BALL] and
                state_handler.should_avoid_boundary(context)):
                self._transition_to_state(RobotState.AVOIDING_BOUNDARY, context)
                return self.current_state
            
            # Execute current state logic
            next_state = state_handler.execute(context)
            
            # Handle state transition if needed
            if next_state and next_state != self.current_state:
                self._transition_to_state(next_state, context)
            
            return self.current_state
            
        except Exception as e:
            self.logger.error(f"State execution error in {self.current_state.value}: {e}")
            # Transition to emergency stop on critical errors
            self._transition_to_state(RobotState.EMERGENCY_STOP, context)
            return self.current_state
    
    def _transition_to_state(self, new_state: RobotState, context: dict):
        """Handle state transition with proper enter/exit calls"""
        if new_state == self.current_state:
            return
        
        old_state = self.current_state
        old_handler = self.states[old_state]
        
        # Call exit on current state
        if self._state_entered:
            try:
                old_handler.exit(context)
            except Exception as e:
                self.logger.error(f"Error exiting state {old_state.value}: {e}")
        
        # Update state
        self.previous_state = self.current_state
        self.current_state = new_state
        self._state_entered = False  # Will trigger enter() on next execute
        
        # Log transition
        self.logger.info(f"State transition: {old_state.value} → {new_state.value}")
        
        # Update context with new state
        context['current_state'] = new_state
    
    def transition_to(self, new_state: RobotState):
        """Transition to a new state (used by main loop)"""
        # This method is called from main loop without context, 
        # so we need to handle it differently
        self.logger.info(f"Requested transition to {new_state.value}")
        old_state = self.current_state
        self.previous_state = self.current_state
        self.current_state = new_state
        self._state_entered = False  # Will trigger enter() on next execute
        self.logger.info(f"State transition: {old_state.value} → {new_state.value}")
    
    def force_state(self, new_state: RobotState, context: dict):
        """Force transition to a specific state (for emergency situations)"""
        self.logger.warning(f"Forcing state transition to {new_state.value}")
        self._transition_to_state(new_state, context)
    
    def emergency_stop(self, context: dict):
        """Force emergency stop state"""
        self.logger.warning("Emergency stop activated")
        self.force_state(RobotState.EMERGENCY_STOP, context)
    
    def reset_to_searching(self, context: dict):
        """Reset state machine to searching state"""
        self.logger.info("Resetting state machine to SEARCHING")
        # Clear any locked targets
        context['locked_target'] = None
        if 'vision' in context:
            context['vision'].current_target = None
        self.force_state(RobotState.SEARCHING, context)
    
    def get_state_summary(self) -> dict:
        """Get summary of current state machine status"""
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'state_entered': self._state_entered,
            'available_states': [state.value for state in self.states.keys()]
        }
    
    def log_state_summary(self):
        """Log current state machine status"""
        summary = self.get_state_summary()
        self.logger.info(f"🔧 STATE MACHINE SUMMARY:")
        self.logger.info(f"   Current: {summary['current_state']}")
        self.logger.info(f"   Previous: {summary['previous_state']}")
        self.logger.info(f"   State entered: {summary['state_entered']}")
