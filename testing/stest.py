#!/usr/bin/env python3
"""
GolfBot Control System with Pi Camera 2 Integration
Servo/Motor control + Camera testing for Pi 5
WITH SERVO ALIGNMENT CORRECTION AND HARDWARE CALIBRATION
"""

from gpiozero import Servo, OutputDevice
import time
import subprocess
import os

# === SERVO CALIBRATION CONSTANTS ===
SERVO1_MIN_VALUE = -0.8    # Actual minimum gpiozero value for servo 1
SERVO1_MAX_VALUE = 0.8     # Actual maximum gpiozero value for servo 1

# === SERVO SETUP (ONLY SERVO 1 ACTIVE) ===
servo1 = Servo(13)  # GPIO 18 (Pin 12) - Hardware PWM0
# servo2 and servo3 disabled to save power

# === DC MOTOR SETUP ===
motor_in1 = OutputDevice(19)  # GPIO 19 (Pin 35)
motor_in2 = OutputDevice(26)  # GPIO 26 (Pin 37)
motor_in3 = OutputDevice(20)  # GPIO 20 (Pin 38)
motor_in4 = OutputDevice(21)  # GPIO 21 (Pin 40)

# === CAMERA SETUP ===
def test_camera_detection():
    """Test if Pi Camera 2 is detected"""
    print("Testing Pi Camera 2 detection...")
    try:
        result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Camera detected:")
            print(result.stdout)
            return True
        else:
            print("✗ Camera detection failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Camera test error: {e}")
        return False

def camera_preview_test():
    """Test camera preview for 5 seconds"""
    print("Starting 5-second camera preview...")
    try:
        subprocess.run(['libcamera-hello', '-t', '5000'], timeout=10)
        print("✓ Camera preview test complete")
        return True
    except Exception as e:
        print(f"✗ Camera preview failed: {e}")
        return False

def camera_photo_test():
    """Take a test photo"""
    print("Taking test photo...")
    filename = f"test_photo_{int(time.time())}.jpg"
    try:
        subprocess.run(['libcamera-still', '-o', filename, '-t', '2000'], timeout=10)
        if os.path.exists(filename):
            print(f"✓ Photo saved: {filename}")
            return True
        else:
            print("✗ Photo not saved")
            return False
    except Exception as e:
        print(f"✗ Photo capture failed: {e}")
        return False

def camera_video_test():
    """Record 3-second test video"""
    print("Recording 3-second test video...")
    filename = f"test_video_{int(time.time())}.h264"
    try:
        subprocess.run(['libcamera-vid', '-o', filename, '-t', '3000'], timeout=10)
        if os.path.exists(filename):
            print(f"✓ Video saved: {filename}")
            return True
        else:
            print("✗ Video not saved")
            return False
    except Exception as e:
        print(f"✗ Video capture failed: {e}")
        return False

def camera_full_test():
    """Complete camera system test"""
    print("=== COMPLETE CAMERA TEST ===")
    
    tests = [
        ("Camera Detection", test_camera_detection),
        ("Preview Test", camera_preview_test),
        ("Photo Capture", camera_photo_test),
        ("Video Recording", camera_video_test)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        results[test_name] = test_func()
        time.sleep(1)
    
    print("\n=== CAMERA TEST RESULTS ===")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    if all(results.values()):
        print("🎉 All camera tests passed!")
    else:
        print("⚠️  Some camera tests failed - check connections")

# === CORRECTED SERVO FUNCTIONS ===
def set_servo_angle_raw(servo, angle):
    """Set servo to raw angle (0-180 degrees) - no offset correction"""
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    value = (angle - 90) / 90
    servo.value = value

def set_servo_angle_corrected(servo, target_angle, offset):
    """Set servo to target angle with offset correction"""
    corrected_angle = target_angle + offset
    
    # Clamp to valid range
    if corrected_angle < 0:
        corrected_angle = 0
        print(f"Warning: Angle clamped to 0° (requested {target_angle}° with {offset}° offset)")
    elif corrected_angle > 180:
        corrected_angle = 180
        print(f"Warning: Angle clamped to 180° (requested {target_angle}° with {offset}° offset)")
    
    set_servo_angle_raw(servo, corrected_angle)
    return corrected_angle

def set_servo1(angle):
    actual_angle = set_servo_angle_raw(servo1, angle)
    print(f"Servo 1 → {angle}°")

def center_servo():
    """Reset servo to 90° center position"""
    set_servo1(90)

def test_alignment():
    """Test servo 1 at different angles"""
    print("=== SERVO 1 TEST ===")
    print("Testing servo 1 at different angles...")
    
    test_angles = [0, 45, 90, 135, 180]
    for angle in test_angles:
        print(f"\nSetting servo 1 to {angle}°...")
        set_servo1(angle)
        input("Press Enter to continue...")
    
    print("Servo test complete!")

# === RAW SERVO FUNCTIONS ===
def set_servo1_raw(angle):
    set_servo_angle_raw(servo1, angle)
    print(f"Servo 1 RAW → {angle}°")

def calibrate_servo():
    """Interactive servo 1 calibration"""
    print("=== SERVO 1 CALIBRATION MODE ===")
    print("Use raw commands:")
    print("  s1r-X  : Set servo 1 to raw angle X")
    print("  done   : Exit calibration")
    
    while True:
        try:
            cmd = input("Calibrate> ").strip().lower()
            
            if cmd.startswith('s1r-'):
                angle = int(cmd[4:])
                if 0 <= angle <= 180:
                    set_servo1_raw(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd == 'done':
                break
            else:
                print("Unknown command")
                
        except (ValueError, IndexError):
            print("Invalid format. Use: s1r-90")

# === HARDWARE CALIBRATION FUNCTIONS ===
def set_servo_calibrated(servo, angle, min_val, max_val):
    """Set servo with hardware-calibrated range"""
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    # Map 0-180° to actual hardware range
    normalized = angle / 180.0  # 0.0 to 1.0
    value = min_val + (normalized * (max_val - min_val))
    servo.value = value
    return value

def set_servo1_calibrated(angle):
    actual = set_servo_calibrated(servo1, angle, SERVO1_MIN_VALUE, SERVO1_MAX_VALUE)
    print(f"Servo 1 → {angle}° (gpiozero: {actual:.2f})")
    return actual

def find_servo_limits():
    """Interactive calibration to find servo 1 limits"""
    print("=== SERVO 1 RANGE CALIBRATION ===")
    print("Testing gpiozero values...")
    
    for test_val in [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        servo1.value = test_val
        response = input(f"Value {test_val}: Does this work? (y/n): ")
        if response.lower() == 'n':
            print(f"Found limit at {test_val}")
    
    min_val = float(input("Enter minimum working value: "))
    max_val = float(input("Enter maximum working value: "))
    print(f"Servo 1: MIN={min_val}, MAX={max_val}")
    print(f"Update constant: SERVO1_MIN_VALUE = {min_val}")
    print(f"Update constant: SERVO1_MAX_VALUE = {max_val}")

# === DC MOTOR FUNCTIONS ===
def stop_motors():
    motor_in1.off()
    motor_in2.off()
    motor_in3.off()
    motor_in4.off()
    print("Motors stopped")

def motor_a_forward():
    motor_in1.on()
    motor_in2.off()
    print("Motor A forward")

def motor_a_reverse():
    motor_in1.off()
    motor_in2.on()
    print("Motor A reverse")

def motor_b_forward():
    motor_in3.on()
    motor_in4.off()
    print("Motor B forward")

def motor_b_reverse():
    motor_in3.off()
    motor_in4.on()
    print("Motor B reverse")

# CORRECTED FUNCTIONS - Motor B is reversed for straight movement
def both_motors_forward():
    motor_a_forward()
    motor_b_reverse()  # Reversed because motor B is mirrored
    print("Both motors forward (straight)")

def both_motors_reverse():
    motor_a_reverse()
    motor_b_forward()  # Reversed because motor B is mirrored
    print("Both motors reverse (straight)")

# NEW FUNCTIONS - For actual turning
def turn_right():
    motor_a_forward()
    motor_b_forward()  # Both same direction = turn right
    print("Turning right")

def turn_left():
    motor_a_reverse()
    motor_b_reverse()  # Both same direction = turn left
    print("Turning left")

# === INTEGRATED DEMO ===
def demo_sequence():
    """Demo with servo, motor, and camera integration"""
    print("=== COMPREHENSIVE SYSTEM DEMO ===")
    
    # 1. Initialize
    print("1. Initializing systems...")
    center_servo()
    stop_motors()
    time.sleep(1)
    
    # 2. Camera test
    print("2. Testing camera...")
    camera_preview_test()
    
    # 3. Servo test
    print("3. Testing servo...")
    for angle in [0, 90, 180]:
        print(f"Setting servo to {angle}°...")
        set_servo1(angle)
        time.sleep(1)
    
    # 4. Motor tests
    print("4. Testing motors...")
    both_motors_forward()
    time.sleep(1)
    both_motors_reverse()
    time.sleep(1)
    stop_motors()
    
    # 5. Coordinated test with photo
    print("5. Coordinated movement with photo capture...")
    set_servo1(45)
    both_motors_forward()
    
    # Take photo during movement
    camera_photo_test()
    
    stop_motors()
    center_servo()
    
    print("🎉 INTEGRATED DEMO COMPLETE! 🎉")

def emergency_stop():
    """Stop everything immediately and reset servo to 90°"""
    stop_motors()
    center_servo()
    print("EMERGENCY STOP - All systems stopped")

def test_servo_pins():
    """Test servo pin responsiveness"""
    from gpiozero import LED
    
    print("Testing servo 1 pin with LED mode...")
    led = LED(18)
    led.on()
    time.sleep(0.5)
    led.off()
    time.sleep(0.5)
    led.close()
    print("GPIO 18 responded")

def competition_ready():
    """Full system check for competition"""
    print("=== COMPETITION READINESS CHECK ===")
    
    # Test all systems
    print("Testing servo...")
    center_servo()  # Reset to 90°
    set_servo1(45)
    time.sleep(0.5)
    set_servo1(135)
    time.sleep(0.5)
    set_servo1(90)
    print("✓ Servo responsive")
    
    print("Testing motors...")
    motor_a_forward()
    time.sleep(0.5)
    motor_b_forward()
    time.sleep(0.5)
    stop_motors()
    print("✓ Motors responsive")
    
    print("Testing camera...")
    camera_ok = test_camera_detection()
    
    if camera_ok:
        print("🏆 SYSTEM READY FOR COMPETITION!")
    else:
        print("⚠️  Camera issues detected - fix before competition")

# === INITIALIZE ===
print("=== GOLFBOT CONTROL SYSTEM - SERVO 1 ONLY ===")
print("Initializing...")
stop_motors()
center_servo()
time.sleep(1)
print("Ready! Type 'help' for commands")

# === MAIN CONTROL LOOP ===
try:
    while True:
        try:
            cmd = input("Control> ").strip().lower()
            if 'inputsubmission' in cmd:
                cmd = cmd.split("'")[1].replace('\\n', '').strip()
            
            # === SERVO COMMANDS ===
            if cmd.startswith('s1-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_servo1(angle)
                else:
                    print("Angle must be 0-180")
            
            # === RAW SERVO COMMANDS ===
            elif cmd.startswith('s1r-'):
                angle = int(cmd[4:])
                if 0 <= angle <= 180:
                    set_servo1_raw(angle)
                else:
                    print("Angle must be 0-180")
            
            # === CALIBRATED SERVO COMMANDS ===
            elif cmd.startswith('s1c-'):  # Calibrated servo 1
                angle = int(cmd[4:])
                if 0 <= angle <= 180:
                    set_servo1_calibrated(angle)
                else:
                    print("Angle must be 0-180")
            
            # === MOTOR COMMANDS ===
            elif cmd == 'maf':
                motor_a_forward()
            elif cmd == 'mar':
                motor_a_reverse()
            elif cmd == 'mbf':
                motor_b_forward()
            elif cmd == 'mbr':
                motor_b_reverse()
            elif cmd == 'mf':
                both_motors_forward()
            elif cmd == 'mr':
                both_motors_reverse()
            elif cmd == 'ms':
                stop_motors()
            elif cmd == 'tr':
                turn_right()
            elif cmd == 'tl':
                turn_left()
            
            # === CAMERA COMMANDS ===
            elif cmd == 'cam':
                test_camera_detection()
            elif cmd == 'preview':
                camera_preview_test()
            elif cmd == 'photo':
                camera_photo_test()
            elif cmd == 'video':
                camera_video_test()
            elif cmd == 'camtest':
                camera_full_test()
            
            # === SYSTEM COMMANDS ===
            elif cmd == 'sc':
                center_servo()
            elif cmd == 'demo':
                demo_sequence()
            elif cmd == 'stop':
                emergency_stop()
            elif cmd == 'test':
                test_servo_pins()
            elif cmd == 'ready':
                competition_ready()
            elif cmd == 'align':
                test_alignment()
            elif cmd == 'calibrate':
                calibrate_servo()
            elif cmd == 'findlimits':
                find_servo_limits()
            elif cmd == 'testrange':
                # Test full calibrated range for servo 1
                print("Testing Servo 1 calibrated range...")
                for angle in [0, 45, 90, 135, 180]:
                    set_servo1_calibrated(angle)
                    time.sleep(1)
            elif cmd == 'quit' or cmd == 'q':
                emergency_stop()
                print("All systems stopped")
                break
                
            elif cmd == 'help':
                print("\n=== SERVO COMMANDS ===")
                print("  s1-90    - Set servo 1 to 90°")
                print("  sc       - Reset servo to 90°")
                print("\n=== RAW SERVO COMMANDS ===")
                print("  s1r-90   - Set servo 1 to raw 90°")
                print("\n=== CALIBRATED SERVO COMMANDS ===")
                print("  s1c-90   - Set servo 1 to calibrated 90°")
                print("\n=== MOTOR COMMANDS ===")
                print("  maf      - Motor A forward")
                print("  mar      - Motor A reverse")
                print("  mbf      - Motor B forward")
                print("  mbr      - Motor B reverse")
                print("  mf       - Both motors forward")
                print("  mr       - Both motors reverse")
                print("  ms       - Stop motors")
                print("  tr       - Turn right")
                print("  tl       - Turn left")
                print("\n=== CAMERA COMMANDS ===")
                print("  cam      - Detect camera")
                print("  preview  - 5-second preview")
                print("  photo    - Take test photo")
                print("  video    - Record 3s video")
                print("  camtest  - Full camera test")
                print("\n=== SYSTEM COMMANDS ===")
                print("  demo     - Integrated system demo")
                print("  test     - Test servo pin")
                print("  ready    - Competition readiness check")
                print("  align    - Test servo positions")
                print("  calibrate- Enter calibration mode")
                print("  findlimits- Find servo hardware limits")
                print("  testrange- Test calibrated servo range")
                print("  stop     - Emergency stop")
                print("  help     - Show this help")
                print("  q        - Quit")
            else:
                print("Unknown command. Type 'help' for available commands")
                
        except (ValueError, IndexError):
            print("Invalid format. Use: s1-90, cam, etc.")
            
except KeyboardInterrupt:
    emergency_stop()
    print("\nAll systems stopped")
finally:
    servo1.close()
    motor_in1.close()
    motor_in2.close()
    motor_in3.close()
    motor_in4.close()
