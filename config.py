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
SERVO_SS_CHANNEL = 0  # PCA9685 Channel 0 (main collection servo)
SERVO_SF_CHANNEL = 1  # PCA9685 Channel 1 (secondary servo)

# === DC MOTOR GPIO PINS ===
MOTOR_IN1 = 19  # GPIO 19 (Pin 35) - Motor A
MOTOR_IN2 = 26  # GPIO 26 (Pin 37) - Motor A
MOTOR_IN3 = 20  # GPIO 20 (Pin 38) - Motor B
MOTOR_IN4 = 21  # GPIO 21 (Pin 40) - Motor B

# === MOTOR SETTINGS ===
MOTOR_SPEED_SLOW = 0.5
MOTOR_SPEED_MEDIUM = 0.6
MOTOR_SPEED_FAST = 0.8
DEFAULT_SPEED = MOTOR_SPEED_SLOW

# === MOTOR CALIBRATION ===
# Motor balance adjustment for slow speeds (to compensate for motor differences)
MOTOR_LEFT_COMPENSATION = 1.0   # Multiplier for left motors (adjust if drifting right)
MOTOR_RIGHT_COMPENSATION = 1.05  # Multiplier for right motors (adjust if drifting left)
SLOW_SPEED_THRESHOLD = 0.3       # Below this speed, use compensation

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
BALL_MAX_RADIUS = 30

# Orange VIP ball detection
ORANGE_HSV_LOWER = np.array([10, 100, 100])
ORANGE_HSV_UPPER = np.array([25, 255, 255])

# Goal detection (red tape - smaller=A, larger=B)
GOAL_HSV_LOWER = np.array([0, 100, 100])
GOAL_HSV_UPPER = np.array([10, 255, 255])

# Obstacle detection (cross/boundaries)
BOUNDARY_HSV_LOWER = np.array([0, 0, 0])    # Dark boundaries
BOUNDARY_HSV_UPPER = np.array([180, 255, 50])

# === SERVO SS (MAIN COLLECTION) POSITIONS ===
SERVO_SS_STORE = 110          # Store position
SERVO_SS_PRE_COLLECT = 130    # Pre-collect position (UPDATED to 130)
SERVO_SS_DRIVING = 28         # Driving position (default/start position)
SERVO_SS_COLLECT = 18         # Collect position
SERVO_SS_STEP_SIZE = 5        # Incremental movement step size

# === SERVO SF (SECONDARY) POSITIONS ===
SERVO_SF_READY_POSITION = 90  # Ready to catch
SERVO_SF_CATCH_POSITION = 135  # Close position to secure ball

# === RELEASE POSITION ===
SERVO_RELEASE = 0  # Release position for ball delivery

# === SERVO MOVEMENT SETTINGS ===
SERVO_GRADUAL_MOVEMENT = True  # Enable gradual servo movement to reduce current draw
SERVO_STEP_DELAY = 0.02  # Seconds between angle steps for gradual movement
SERVO_SMOOTH_DURATION = 0.5  # Duration for smooth movements (seconds)
SERVO_SEQUENTIAL_DELAY = 0.1  # Delay between multiple servo movements (seconds)

# === MOVEMENT PARAMETERS ===
TURN_TIME_90_DEGREES = 0.8  # Time to turn 90 degrees
FORWARD_TIME_SHORT = 0.4    # Short forward movement
BOUNDARY_DETECTION_THRESHOLD = 50  # Pixels from edge to consider boundary

# === TWO-PHASE COLLECTION SETTINGS ===
# Phase 1 - Initial Ball Centering (X+Y alignment to frame center)
CENTERING_1_TOLERANCE = 25  # Pixels - X-axis centering tolerance
CENTERING_1_DISTANCE_TOLERANCE = 30  # Pixels - Y-axis centering tolerance
CENTERING_1_TURN_DURATION = 0.4  # Duration for horizontal centering turns
CENTERING_1_DRIVE_DURATION = 0.4  # Duration for forward/backward centering adjustments
CENTERING_1_SPEED = 0.5  # Speed for initial centering movements

# Phase 2 - Collection Zone Centering (identical system, different Y target)
CENTERING_2_TOLERANCE = 25  # Pixels - X-axis centering tolerance (same as Phase 1)
CENTERING_2_DISTANCE_TOLERANCE = 30  # Pixels - Y-axis centering tolerance (same as Phase 1)
CENTERING_2_TURN_DURATION = 0.6  # Duration for horizontal centering turns (same)
CENTERING_2_DRIVE_DURATION = 0.6  # Duration for forward/backward centering adjustments (same)
CENTERING_2_SPEED = 0.5  # Speed for centering movements (same)
CENTERING_2_Y_TARGET_OFFSET = 100  # Pixels below frame center to target upper green zone

# Final Collection Settings
CENTERING_2_COLLECTION_SPEED = 0.5  # Speed during final collection sequence
CENTERING_2_COLLECTION_TIME = 0.8   # Reduced time for final collection sequence

# === LEGACY SETTINGS (maintained for compatibility) ===
CENTERING_TOLERANCE = CENTERING_1_TOLERANCE  # Backward compatibility
CENTERING_DISTANCE_TOLERANCE = CENTERING_1_DISTANCE_TOLERANCE  # Backward compatibility
CENTERING_TURN_DURATION = CENTERING_1_TURN_DURATION
CENTERING_DRIVE_DURATION = CENTERING_1_DRIVE_DURATION
CENTERING_SPEED = CENTERING_1_SPEED
ENHANCED_COLLECTION_DRIVE_TIME = CENTERING_2_COLLECTION_TIME  # Backward compatibility
COLLECTION_SPEED = CENTERING_2_COLLECTION_SPEED

# === NAVIGATION STRATEGY ===
SEARCH_PATTERN = [
    "forward", "turn_right", "forward", "turn_right", 
    "forward", "turn_right", "forward", "turn_right"
]

# === DEBUGGING ===
DEBUG_VISION = True
DEBUG_MOVEMENT = True
DEBUG_COLLECTION = True
SHOW_CAMERA_FEED = True

# === ERROR HANDLING ===
MOTOR_TIMEOUT = 5.0  # Max time for any single movement
VISION_TIMEOUT = 1.0  # Max time to wait for camera frame
RESTART_THRESHOLD = 5  # Number of consecutive errors before restart