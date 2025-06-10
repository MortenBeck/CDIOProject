# GolfBot - Autonomous Golf Ball Collection Robot

Competition robot that autonomously collects ping pong balls and delivers them to goals within an 8-minute time limit.

## Competition Overview

- **Field**: 180x120cm with obstacles and boundaries
- **Balls**: 11 ping pong balls (10 white + 1 orange VIP)
- **Goals**: Goal A (150 pts) and Goal B (100 pts) marked with red tape
- **Scoring**: VIP first bonus (200 pts), time bonus (3 pts/sec remaining)
- **Penalties**: Boundary contact (-50), obstacle movement (-100/-300)

## Architecture

```
robot_project/
├── main.py          # Main control loop + competition state machine
├── hardware.py      # Motor/servo control + ball collection/delivery
├── vision.py        # Pi5 camera + ball/goal/boundary detection  
├── telemetry.py     # Data logging + troubleshooting exports
├── config.py        # All settings (pins, colors, timing, scoring)
└── logs/            # Auto-generated session data
```

## Hardware Setup

**Raspberry Pi 5 Configuration:**
- Servos: GPIO 18, 12, 13 (Hardware PWM)
- Motors: GPIO 19, 26, 20, 21 (H-bridge control)
- Camera: Pi Camera v2 via libcamera
- Power: 25W USB-C + battery packs for motors/servos

**Wiring matches your existing setup from the documentation.**

## Installation

```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip libopencv-dev
pip3 install opencv-python numpy gpiozero

# Clone and setup
git clone <https://github.com/MortenBeck/CDIOProject>
cd robot_project
```

## Configuration

Edit `config.py` for your setup:

```python
# GPIO pins (match your wiring)
SERVO_1_PIN = 18
MOTOR_IN1 = 19  # etc.

# Vision tuning (adjust for lighting)
BALL_HSV_LOWER = np.array([0, 0, 180])    # White balls
ORANGE_HSV_LOWER = np.array([10, 100, 100])  # VIP ball

# Movement timing (calibrate on your robot)
TURN_TIME_90_DEGREES = 0.6
DEFAULT_SPEED = 0.3
```

## Usage

### Competition Run
```bash
python3 main.py
```
Press Enter when positioned and ready to start the 8-minute timer.

### Debug Mode
Enable visual debugging in `config.py`:
```python
DEBUG_VISION = True
SHOW_CAMERA_FEED = True
```

## Strategy

1. **Orange VIP Priority**: Seeks orange ball first (200 bonus points)
2. **Efficient Collection**: Collects 2-3 balls before delivery
3. **Goal Selection**: Prefers Goal A (150 pts) when feasible
4. **Time Management**: Switches to delivery mode with <2 minutes remaining
5. **Boundary Avoidance**: Backs up and turns when near edges

## Telemetry System

### Automatic Logging
Every run generates detailed telemetry in `logs/golfbot_YYYYMMDD_HHMMSS/`:
- `telemetry.jsonl` - Frame-by-frame data
- `session_summary.json` - Session overview  
- `analysis_export.json` - Complete export for troubleshooting

### Data Logged
- Ball detection (positions, confidence, distance from center)
- Robot states and transitions
- Hardware status (servo positions, motor speeds, collection count)
- Performance metrics (FPS, processing time, error rates)
- Events (collections, deliveries, errors)

### Troubleshooting with Claude
1. Find export file: `logs/golfbot_*/analysis_export.json`
2. Copy JSON contents
3. Share with Claude:
```
Here's my robot's telemetry data for troubleshooting:

[paste JSON content]

Issues I'm seeing:
- Balls not being detected properly
- Robot keeps hitting boundaries
```

### Manual Export
```python
from telemetry import TelemetryLogger
logger = TelemetryLogger()
export_file = logger.export_for_analysis()
```

## Testing Components

### Camera Test
```python
python3 -c "
from vision import VisionSystem
vision = VisionSystem()
vision.start()
ret, frame = vision.get_frame()
print('Camera working:', ret)
"
```

### Hardware Test
```python
python3 -c "
from hardware import GolfBotHardware
hw = GolfBotHardware()
hw.center_servos()
hw.forward_step()
hw.cleanup()
"
```

## Common Issues

**Camera not working on Pi 5:**
- Ensure libcamera installed: `sudo apt install libcamera-apps`
- Check camera connection and enable in raspi-config

**Poor ball detection:**
- Adjust `BALL_HSV_LOWER/UPPER` in config.py for your lighting
- Use debug mode to see detection overlays
- Check camera focus and positioning

**Motors not responding:**
- Verify GPIO connections match config.py pins
- Check power supply (separate battery for motors recommended)
- Test with basic gpiozero commands

**Robot hitting boundaries:**
- Tune `BOUNDARY_DETECTION_THRESHOLD` in config.py
- Adjust search pattern timing
- Check camera angle covers floor edges

## Calibration

1. **Movement timing**: Adjust `TURN_TIME_90_DEGREES` for accurate turns
2. **Speed settings**: Tune `MOTOR_SPEED_*` values for your motors
3. **Collection distance**: Modify `COLLECTION_DISTANCE_THRESHOLD` for reliable pickup
4. **Color detection**: Use debug mode to tune HSV ranges for your lighting

## Competition Day Checklist

- [ ] Charge all batteries (Pi + motors + servos)
- [ ] Test camera detection with competition lighting
- [ ] Calibrate movement timing on competition surface
- [ ] Verify servo collection mechanism works
- [ ] Set `DEBUG_VISION = False` for competition
- [ ] Position robot and run: `python3 main.py`

## File Structure Details

- **main.py**: State machine handling search→approach→collect→deliver cycle
- **hardware.py**: Low-level control with safety checks and emergency stops  
- **vision.py**: Pi5-compatible camera with ball/goal/boundary detection
- **telemetry.py**: Comprehensive logging for performance analysis
- **config.py**: Centralized settings for easy tuning without code changes