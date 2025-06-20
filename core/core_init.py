"""
Core Module
Main robot coordination and state management
"""

from .robot import GolfBot
from .competition import CompetitionManager
from .state_machine import StateMachine

__all__ = ['GolfBot', 'CompetitionManager', 'StateMachine']
