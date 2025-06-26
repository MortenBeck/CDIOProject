# GolfBot - Autonomous Ball Collection & Delivery Robot

Advanced competition robot that autonomously collects white ping pong balls and delivers them to target zones using precision navigation and computer vision.

## 🏆 Competition Overview

- **Field**: 180x120cm arena with boundaries and obstacles
- **Mission**: Collect white ping pong balls and deliver to green zones
- **Time Limit**: 8 minutes maximum
- **Collection Strategy**: Configurable balls-per-delivery cycle (default: 1 ball)
- **Scoring**: Based on successful deliveries and time efficiency
- **Penalties**: Boundary contact (-50), obstacle movement (-100/-300)

## 🤖 Architecture

```
robot_project/
├── main.py                    # Main entry point + startup menu
├── competition_manager.py     # Competition control loop + timing
├── robot_state_machine.py     # State machine logic + delivery cycles
├── hardware.py               # Hardware interface (motors + servos)
├── vision.py                 # Computer vision + ball detection
├── boundary_avoidance.py     # Wall detection + avoidance
├── ball_collection.py        # Collection sequences
├── delivery_system.py        # Delivery zone targeting
├── triangle_delivery_system.py # Precision triangle targeting
├── dashboard.py              # Real-time dashboard interface
├── config.py                 # All configuration settings
├── servo_controller.py       # PCA9685 servo control
├── motor_controller.py       # DC motor control
├── startup_menu.py           # Interactive startup interface
├── hardware_test.py          # Hardware testing utilities
└── testing/                  # Additional test utilities
```

## 🔧 Hardware Configuration

### **Raspberry Pi 5 Setup**
- **Servos**: PCA9685 I2C controller (channels 0, 1)
  - **SS Servo**: Channel 0 - 4-state collection mechanism
  - **SF Servo**: Channel 1 - 2-state assist/delivery door
- **Motors**: DC motors via GPIO H-bridge control
  - Motor A: GPIO 19, 26 (PWM speed control)
  - Motor B: GPIO 20, 21 (PWM speed control)
- **Camera**: Pi Camera v2 via libcamera
- **Power**: 25W USB-C + separate battery packs recommended

### **Servo System (Enhanced Two-Servo Design)**
```python
# SS Servo (Primary Collection) - 4 States:
SERVO_SS_DRIVING = 97      # Default navigation position
SERVO_SS_PRE_COLLECT = 110 # Pre-collection positioning  
SERVO_SS_COLLECT = 80      # Active ball capture
SERVO_SS_STORE = 180       # Ball storage position

# SF Servo (Assist/Delivery) - 2 States:
SERVO_SF_CLOSED = 180      # Default closed state
SERVO_SF_OPEN = 70         # Open for ball delivery
```

## 🚀 Installation & Setup

### **Dependencies**
```bash
# System packages
sudo apt update
sudo apt install python3-pip libopencv-dev libcamera-apps

# Python packages  
pip3 install opencv-python numpy gpiozero adafruit-pca9685 adafruit-circuitpython-motor
```

### **Hardware Connections**
```bash
# Verify I2C and camera are enabled
sudo raspi-config
# Enable: Interface Options → I2C, Camera

# Test camera
libcamera-hello --list-cameras
```

### **Project Setup**
```bash
git clone <your-repo-url>
cd robot_project
python3 main.py
```

## 🎮 Usage Modes

### **1. Competition Mode**
```bash
python3 main.py
# Select: 1 (Dashboard) or 2 (Legacy Overlay)
```

**Competition Features:**
- **White ball detection** using HoughCircles + color verification
- **Precision centering** with X+Y axis alignment
- **Enhanced collection** with 4-state servo sequence
- **Delivery cycles** (collect N balls → deliver → repeat)
- **Wall avoidance** using modular boundary detection
- **Real-time dashboard** with status panels and visualizations

### **2. Hardware Testing**
```bash
python3 main.py
# Select: 3 (Hardware Testing)
```

Interactive testing interface:
- Individual servo control and positioning
- Motor movement testing (forward, backward, turns)
- Collection sequence testing
- Full system demonstrations

### **3. Delivery System Testing**
```bash
python3 main.py  
# Select: 4 (Delivery System)
```

Advanced delivery targeting:
- **Precision Triangle Targeting**: Dead-straight approach to triangle tips
- **Rectangular Zone Delivery**: Standard green zone targeting
- **Real-time alignment** with visual feedback

## 🧠 Core Systems

### **Vision System**
- **Primary**: HoughCircles detection for robust ball finding
- **Color Verification**: HSV-based white ball confirmation
- **Arena Detection**: Automatic boundary recognition
- **Target Zone**: Precise collection area with centering tolerance
- **Delivery Zones**: Green target detection for ball delivery

```python
# Detection Parameters (config.py)
BALL_HSV_LOWER = np.array([0, 0, 180])    # White ball detection
BALL_HSV_UPPER = np.array([180, 30, 255])
CENTERING_TOLERANCE = 25                   # X-axis centering (pixels)
CENTERING_DISTANCE_TOLERANCE = 30          # Y-axis centering (pixels)
```

### **State Machine**
Advanced behavior system with delivery cycles:

```
SEARCHING → CENTERING_BALL → COLLECTING_BALL → [Repeat until target count]
    ↓
DELIVERY_MODE → DELIVERY_ZONE_SEARCH → DELIVERY_ZONE_CENTERING → 
DELIVERY_RELEASING → POST_DELIVERY_TURN → [Back to SEARCHING]
```

**States:**
- **SEARCHING**: Rotating scan for balls
- **CENTERING_BALL**: Precision X+Y alignment 
- **COLLECTING_BALL**: Enhanced servo collection sequence
- **DELIVERY_MODE**: Target-based delivery cycle
- **AVOIDING_BOUNDARY**: Emergency wall avoidance

### **Collection System**
Enhanced ball collection with servo optimization:

```python
def enhanced_collection_sequence():
    """SS-only collection with pre-positioning"""
    servo_ss_to_pre_collect()    # Position for capture
    drive_forward()              # Approach ball
    servo_ss_to_collect()        # Capture ball
    servo_ss_to_store()          # Secure ball
    servo_ss_to_collect()        # Second capture
    servo_ss_to_store()          # Secure second ball  
    servo_ss_to_driving()        # Return to nav position
```

### **Boundary Avoidance**
Modular wall detection system:
- **Detection Zones**: Bottom center + forward center areas
- **Color Range**: Dual-range red detection (0-15°, 160-180° hue)
- **Response**: Backup + turn combinations
- **Safety**: Non-interruptible collection states

## ⚙️ Configuration

### **Key Settings** (`config.py`)
```python
# Competition
BALLS_BEFORE_DELIVERY = 1              # Balls per delivery cycle
POST_DELIVERY_TURN_DURATION = 3.0     # Seconds to turn after delivery

# Movement
CENTERING_TURN_DURATION = 0.25        # Fast centering turns
CENTERING_SPEED = 0.4                 # Centering movement speed
COLLECTION_SPEED = 0.3                # Collection approach speed

# Vision
TARGET_ZONE_VERTICAL_POSITION = 0.65  # Collection zone position
TARGET_ZONE_WIDTH = 60                # Collection zone width (pixels)
TARGET_ZONE_HEIGHT = 45               # Collection zone height (pixels)

# Servos  
SERVO_GRADUAL_MOVEMENT = True         # Smooth servo transitions
SERVO_STEP_DELAY = 0.02               # Gradual movement timing
```

### **Debug Options**
```python
DEBUG_VISION = True          # Show detection overlays
DEBUG_MOVEMENT = True        # Log movement commands
DEBUG_COLLECTION = True      # Log collection sequences
SHOW_CAMERA_FEED = True      # Display camera window
```

## 🎯 Competition Strategy

### **Phase 1: Ball Collection**
1. **Search Pattern**: Systematic rotation scanning
2. **Target Selection**: Closest confident white ball detection
3. **Precision Centering**: X+Y axis alignment within tolerance
4. **Enhanced Collection**: 4-state servo sequence for reliability
5. **Cycle Management**: Collect configured number before delivery

### **Phase 2: Delivery**
1. **Zone Detection**: Scan for green delivery zones
2. **Target Selection**: Highest confidence zone
3. **Horizontal Centering**: Align robot with delivery target
4. **Ball Release**: Open SF servo for controlled delivery
5. **Return**: Turn and resume collection cycle

### **Phase 3: Optimization**
- **Boundary Avoidance**: Interrupt any operation for safety
- **Failed Collection Recovery**: Turn after consecutive failures  
- **Time Management**: Configurable competition timer
- **Performance Monitoring**: Real-time FPS and error tracking

## 🔍 Testing & Calibration

### **Vision Calibration**
```bash
# Test ball detection
python3 testing/cv_test.py

# Test wall detection  
python3 testing/wall_detection_test.py

# Autonomous behavior test
python3 testing/auto_drive_test.py
```

### **Hardware Calibration**
```bash
# Interactive hardware testing
python3 hardware_test.py

# Basic component test
python3 testing/basic_test.py
```

### **Competition Day Checklist**
- [ ] **Power**: Charge all batteries (Pi + motors + servos)
- [ ] **Camera**: Test detection with competition lighting  
- [ ] **Movement**: Calibrate turn timing on competition surface
- [ ] **Collection**: Verify servo sequence with actual balls
- [ ] **Arena**: Test boundary detection on competition walls
- [ ] **Config**: Set `DEBUG_VISION = False` for competition
- [ ] **Start**: Position robot and run: `python3 main.py`

## 🐛 Troubleshooting

### **Common Issues**

**Camera not working on Pi 5:**
```bash
sudo apt install libcamera-apps
# Check camera in raspi-config → Interface Options → Camera
libcamera-hello --list-cameras
```

**Poor ball detection:**
- Adjust `BALL_HSV_LOWER/UPPER` in config.py for lighting
- Use debug mode to see detection overlays: `DEBUG_VISION = True`
- Check camera focus and positioning

**Servo not responding:**
- Verify I2C connection: `sudo i2cdetect -y 1`
- Check PCA9685 address (default: 0x40)
- Test with hardware testing interface

**Motors not responding:**
- Verify GPIO connections match config.py pins
- Check power supply (separate battery recommended)
- Test individual motors in hardware test mode

**Robot hitting boundaries:**
- Tune `BOUNDARY_DETECTION_THRESHOLD` for your walls
- Adjust red HSV ranges for your tape color
- Test boundary system in wall detection test

### **Performance Optimization**
- **FPS**: Target 15+ FPS for smooth operation
- **Latency**: Process at 320x240 for speed, display at full resolution
- **Memory**: Monitor with `htop` during long runs
- **Power**: Use separate battery packs for motors vs electronics

## 📊 Dashboard Interface

**Dashboard Mode** provides real-time monitoring:
- **Camera View**: Live detection overlays
- **Vision Status**: Arena detection, target info, ball counts
- **Robot Status**: Speed, servo positions, ball count, state
- **Delivery Cycle**: Progress bar, cycle status, ready indicators
- **Wall Danger**: Real-time boundary detection status
- **Detection Details**: Confidence levels, centering status

**Controls:**
- Press 'Q' to quit
- Dashboard automatically updates at camera framerate

## 🏁 Competition Performance

**Optimized for:**
- **Reliability**: Robust detection and collection mechanisms
- **Speed**: Fast centering and efficient movement patterns  
- **Precision**: Accurate ball placement and delivery targeting
- **Safety**: Comprehensive boundary avoidance and error recovery
- **Flexibility**: Configurable collection cycles and strategies

**Typical Performance:**
- **Detection Range**: 10-50 pixel radius balls
- **Centering Accuracy**: ±25 pixels X, ±30 pixels Y
- **Collection Success**: >90% with enhanced sequence
- **FPS**: 15+ with full processing pipeline
- **Battery Life**: 30+ minutes continuous operation

---

**Built for competitive robotics with Pi 5 + modern computer vision**  
**Ready for autonomous ball collection competitions! 🏆**