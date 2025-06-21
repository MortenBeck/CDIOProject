import numpy as np

# === COMPETITION SETTINGS ===
COMPETITION_TIME = 8 * 60  # 8 minutes in seconds
BALL_COUNT = 11
VIP_BALL_COLOR = "orange"

# === SCORING ===
GOAL_A_POINTS = 150  # Smaller goal
GOAL_B_POINTS = 100  # Larger goal
VIP_FIRST_BONUS = 200
TIME_BONUS_PER_SECOND = 3
BOUNDARY_PENALTY = -50
OBSTACLE_MOVE_PENALTY = -100
EGG_MOVE_PENALTY = -300

# === PCA9685 SERVO SETTINGS ===
PCA9685_ADDRESS = 0x40
PCA9685_FREQUENCY = 50
SERVO_SS_CHANNEL = 0  # PCA9685 Channel 0 - "ss" servo
SERVO_SF_CHANNEL = 1  # PCA9685 Channel 1 - "SF" servo

# === DC MOTOR GPIO PINS ===
MOTOR_IN1 = 19  # GPIO 19 (Pin 35) - Motor A
MOTOR_IN2 = 26  # GPIO 26 (Pin 37) - Motor A
MOTOR_IN3 = 20  # GPIO 20 (Pin 38) - Motor B
MOTOR_IN4 = 21  # GPIO 21 (Pin 40) - Motor B

# === MOTOR SETTINGS ===
MOTOR_SPEED_SLOW = 0.3
MOTOR_SPEED_MEDIUM = 0.5
MOTOR_SPEED_FAST = 0.8
DEFAULT_SPEED = MOTOR_SPEED_MEDIUM

# === CAMERA SETTINGS ===
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FRAMERATE = 20

# === VISION PARAMETERS ===
# Ping pong ball detection (white/light colored)
BALL_HSV_LOWER = np.array([0, 0, 180])    # White/light detection
BALL_HSV_UPPER = np.array([180, 30, 255])
BALL_MIN_AREA = 100
BALL_MAX_AREA = 10000
BALL_MIN_RADIUS = 10
BALL_MAX_RADIUS = 70

# Orange VIP ball detection
##ORANGE_HSV_LOWER = np.array([10, 100, 100])
##ORANGE_HSV_UPPER = np.array([25, 255, 255])

# Goal detection (red tape - smaller=A, larger=B)
GOAL_HSV_LOWER = np.array([0, 100, 100])
GOAL_HSV_UPPER = np.array([10, 255, 255])

# Obstacle detection (cross/boundaries)
BOUNDARY_HSV_LOWER = np.array([0, 0, 0])    # Dark boundaries
BOUNDARY_HSV_UPPER = np.array([180, 255, 50])

# === SERVO POSITIONS (ANGLES IN DEGREES) ===
SERVO_CENTER = 90
SERVO_COLLECT_OPEN = 45   # Open position for collection
SERVO_COLLECT_CLOSE = 135 # Close position to hold ball
SERVO_RELEASE = 0         # Release position

# === SERVO SS (SERVO 1) FOUR-STATE SYSTEM ===
SERVO_SS_STORE = 110        # Store position
SERVO_SS_PRE_COLLECT = 50   # Pre-collect position
SERVO_SS_DRIVING = 25       # Driving position (default/start position)
SERVO_SS_COLLECT = 14        # Collect position
SERVO_SS_STEP_SIZE = 5      # Incremental movement step size

# === SERVO SF (SERVO 2) POSITIONS ===
SERVO_SF_READY = 90         # Ready position
SERVO_SF_CATCH = 135        # Catch position
SERVO_SF_RELEASE = 0        # Release position

# === ENHANCED COLLECTION POSITIONS ===
SERVO_READY_POSITION = 90  # Servos up and ready to catch
SERVO_CATCH_POSITION = 135  # Close position to secure ball

# === SERVO MOVEMENT SETTINGS ===
SERVO_GRADUAL_MOVEMENT = True  # Enable gradual servo movement to reduce current draw
SERVO_STEP_DELAY = 0.02  # Seconds between angle steps for gradual movement
SERVO_SMOOTH_DURATION = 0.5  # Duration for smooth movements (seconds)
SERVO_SEQUENTIAL_DELAY = 0.1  # Delay between multiple servo movements (seconds)

# === MOVEMENT PARAMETERS ===
TURN_TIME_90_DEGREES = 0.6  # Time to turn 90 degrees
FORWARD_TIME_SHORT = 0.2    # Short forward movement
BOUNDARY_DETECTION_THRESHOLD = 50  # Pixels from edge to consider boundary

# === COLLECTION BEHAVIOR ===
CENTERING_TOLERANCE = 25  # Pixels - more lenient X centering (was 15)
CENTERING_DISTANCE_TOLERANCE = 30  # Pixels - more lenient Y centering (was 20)
COLLECTION_DRIVE_TIME_PER_PIXEL = 0.003  # Seconds per pixel distance to ball
MIN_COLLECTION_DRIVE_TIME = 0.5  # Minimum drive time
MAX_COLLECTION_DRIVE_TIME = 2.0  # Maximum drive time for safety
COLLECTION_SPEED = 0.4  # Slower speed for precise collection

# === ENHANCED CENTERING BEHAVIOR (FASTER) ===
CENTERING_TURN_DURATION = 0.25  # Faster horizontal centering (was 0.16)
CENTERING_DRIVE_DURATION = 0.15  # Duration for forward/backward centering adjustments
CENTERING_SPEED = 0.4  # Speed for centering movements

# === NAVIGATION STRATEGY ===
SEARCH_PATTERN = [
    "forward", "turn_right", "forward", "turn_right", 
    "forward", "turn_right", "forward", "turn_right"
]

# === BALL COLLECTION ===
COLLECTION_DISTANCE_THRESHOLD = 30  # Pixels - how close before attempting collection
BALL_LOST_TIMEOUT = 2.0  # Seconds before giving up on a ball
MAX_COLLECTION_ATTEMPTS = 3

# === DEBUGGING ===
DEBUG_VISION = True
DEBUG_MOVEMENT = True
DEBUG_COLLECTION = True
SHOW_CAMERA_FEED = True

# === ERROR HANDLING ===
MOTOR_TIMEOUT = 5.0  # Max time for any single movement
VISION_TIMEOUT = 1.0  # Max time to wait for camera frame
RESTART_THRESHOLD = 5  # Number of consecutive errors before restart

# === PRECISE TARGET ZONE (NEW APPROACH) ===
TARGET_ZONE_WIDTH = 80              # Target zone width in pixels (ping pong ball sized)
TARGET_ZONE_HEIGHT = 60             # Target zone height in pixels
FIXED_COLLECTION_DRIVE_TIME = 1.05  # Fixed time to drive from target zone to collection point

# Enhanced centering - ball must be in target zone AND X-centered before collection
CENTERING_TOLERANCE = 25  # X-axis centering tolerance (pixels)
REQUIRE_TARGET_ZONE_FOR_COLLECTION = True  # Ball must be in target zone, not just centered
