#!/usr/bin/env python3
"""
Robot State Definitions for GolfBot Autonomous System
"""

class RobotState:
    """Main robot behavior states"""
    SEARCHING = 0    # Looking for balls by rotating
    MOVING = 1       # Moving towards a detected ball
    COLLECTING = 2   # Collecting the ball
    AVOIDING = 3     # Emergency wall avoidance

class MotorState:
    """Motor states for display and control"""
    STOPPED = 0
    FORWARD = 1
    REVERSE = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
