#!/usr/bin/env python3
"""
GolfBot Configuration Settings
"""

# === PERFORMANCE SETTINGS ===
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_WIDTH = 320      # Process at half resolution for speed
PROCESS_HEIGHT = 240
TARGET_FPS = 15
DISPLAY_FRAME_SKIP = 1   # Display every Nth frame (1 = all frames)

# Detection settings
MIN_BALL_AREA = 150      # White ball minimum area
MIN_WALL_AREA = 100      # Red wall minimum area
BALL_CIRCULARITY_THRESHOLD = 0.25  # Ball circularity
WALL_MIN_LENGTH = 50     # Minimum wall segment length
ENABLE_PERFORMANCE_STATS = True

# Motor control settings
MOTOR_SPEED_SLOW = 0.3    # 30% speed
MOTOR_SPEED_MEDIUM = 0.5  # 50% speed
MOTOR_SPEED_FAST = 0.8    # 80% speed
DEFAULT_SPEED = MOTOR_SPEED_SLOW  # Default speed for autonomous operation

# Autonomous behavior settings
SEARCH_ROTATION_SPEED = 0.4      # Speed when searching for balls
MOVE_SPEED = 0.5                 # Speed when moving to ball
COLLECTION_TIME = 3.0            # Time to spend collecting (seconds)
SEARCH_TIMEOUT = 10.0            # Max time to search before giving up (seconds)
BALL_REACHED_DISTANCE = 30       # Pixels - when ball is "reached"
WALL_DANGER_DISTANCE = 80        # Pixels from bottom - closer than before for more aggressive behavior
WALL_AVOIDANCE_TURN_TIME = 1.5   # Time to turn when avoiding walls
BALL_CONFIRMATION_TIME = 2.0     # Time to observe ball before moving (seconds)
BALL_POSITION_TOLERANCE = 30     # Pixels - how much ball can move and still be "same" ball
AUTO_ENABLED = True              # Enable autonomous mode
HEADLESS_MODE = False            # Run without display (for remote operation)

# GPIO Pin assignments
MOTOR_IN1_PIN = 19  # GPIO 19 (Pin 35) - Motor A direction 1
MOTOR_IN2_PIN = 26  # GPIO 26 (Pin 37) - Motor A direction 2
MOTOR_IN3_PIN = 20  # GPIO 20 (Pin 38) - Motor B direction 1
MOTOR_IN4_PIN = 21  # GPIO 21 (Pin 40) - Motor B direction 2

# Camera settings
LIBCAMERA_TEMP_FILE = "/tmp/golfbot_frame_optimized.jpg"
