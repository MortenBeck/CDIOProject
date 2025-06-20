from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List
import logging

class RobotState(Enum):
    SEARCHING = "searching"
    CENTERING_1 = "centering_1"  # Initial X+Y centering
    CENTERING_2 = "centering_2"  # Collection zone positioning
    COLLECTING_BALL = "collecting_ball"
    AVOIDING_BOUNDARY = "avoiding_boundary"
    EMERGENCY_STOP = "emergency_stop"

class BaseState(ABC):
    """Abstract base class for robot behavior states"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def enter(self, context):
        """Called when entering this state"""
        pass
    
    @abstractmethod
    def execute(self, context) -> Optional[RobotState]:
        """Execute state logic and return next state if transition needed"""
        pass
    
    @abstractmethod
    def exit(self, context):
        """Called when exiting this state"""
        pass
    
    def handle_emergency(self, context) -> RobotState:
        """Handle emergency situations - default implementation"""
        return RobotState.EMERGENCY_STOP
    
    def should_avoid_boundary(self, context) -> bool:
        """Check if should switch to boundary avoidance"""
        # Default implementation - can be overridden
        return context.get('near_boundary', False)
    
    def log_state_info(self, message: str, level=logging.INFO):
        """Log state-specific information"""
        self.logger.log(level, f"[{self.name}] {message}")
