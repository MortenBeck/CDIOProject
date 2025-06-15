#!/usr/bin/env python3
"""
GolfBot Control System - FIXED SERVO VERSION
Focus on servo2 and servo3 with proper angle mapping
"""

from gpiozero import Servo, OutputDevice
import time
import subprocess
import os

# === SERVO SETUP - ONLY SERVO2 AND SERVO3 ===
# Using custom min/max pulse widths for full 180° range
servo2 = Servo(12, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)  # GPIO 12 (Pin 32)
servo3 = Servo(13, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)  # GPIO 13 (Pin 33)

# === DC MOTOR SETUP ===
motor_in1 = OutputDevice(19)  # GPIO 19 (Pin 35)
motor_in2 = OutputDevice(26)  # GPIO 26 (Pin 37)
motor_in3 = OutputDevice(20)  # GPIO 20 (Pin 38)
motor_in4 = OutputDevice(21)  # GPIO 21 (Pin 40)

# === IMPROVED SERVO FUNCTIONS ===
def set_servo_angle_precise(servo, angle):
    """
    Set servo to precise angle with proper 0-180° mapping
    0° = -1 (leftmost position)
    90° = 0 (center position)  
    180° = +1 (rightmost position)
    """
    # Clamp angle to valid range
    angle = max(0, min(180, angle))
    
    # Convert 0-180° to -1 to +1 range
    # Formula: value = (angle - 90) / 90
    value = (angle - 90) / 90
    
    servo.value = value
    return angle

def set_servo2(angle):
    """Set servo2 to specific angle"""
    actual_angle = set_servo_angle_precise(servo2, angle)
    print(f"Servo 2 → {actual_angle}° (value: {servo2.value:.3f})")
    return actual_angle

def set_servo3(angle):
    """Set servo3 to specific angle"""
    actual_angle = set_servo_angle_precise(servo3, angle)
    print(f"Servo 3 → {actual_angle}° (value: {servo3.value:.3f})")
    return actual_angle

def set_both_servos(angle):
    """Set both servos to same angle"""
    angle2 = set_servo_angle_precise(servo2, angle)
    angle3 = set_servo_angle_precise(servo3, angle)
    print(f"Both servos → {angle}°")
    return angle

def center_servos():
    """Center both servos at 90°"""
    set_both_servos(90)
    print("Servos centered")

def servo_test_sequence():
    """Test servo range and precision"""
    print("=== SERVO TEST SEQUENCE ===")
    
    test_angles = [0, 45, 90, 135, 180]
    
    print("Testing Servo 2...")
    for angle in test_angles:
        set_servo2(angle)
        time.sleep(1)
    
    print("Testing Servo 3...")
    for angle in test_angles:
        set_servo3(angle)
        time.sleep(1)
    
    print("Testing both servos together...")
    for angle in test_angles:
        set_both_servos(angle)
        time.sleep(1)
    
    center_servos()
    print("✓ Servo test complete")

def servo_sweep_test():
    """Smooth sweep test to verify full range"""
    print("=== SERVO SWEEP TEST ===")
    
    print("Sweeping Servo 2...")
    for angle in range(0, 181, 5):
        set_servo2(angle)
        time.sleep(0.1)
    
    print("Sweeping Servo 3...")
    for angle in range(0, 181, 5):
        set_servo3(angle)
        time.sleep(0.1)
    
    center_servos()
    print("✓ Sweep test complete")

def servo_precision_test():
    """Test precise angle control"""
    print("=== SERVO PRECISION TEST ===")
    
    precise_angles = [0, 30, 60, 90, 120, 150, 180]
    
    for angle in precise_angles:
        print(f"Setting both to {angle}°...")
        set_both_servos(angle)
        time.sleep(1.5)  # Give time to observe position
    
    center_servos()
    print("✓ Precision test complete")

# === DC MOTOR FUNCTIONS (UNCHANGED) ===
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

def both_motors_forward():
    motor_a_forward()
    motor_b_reverse()  # Reversed because motor B is mirrored
    print("Both motors forward (straight)")

def both_motors_reverse():
    motor_a_reverse()
    motor_b_forward()  # Reversed because motor B is mirrored
    print("Both motors reverse (straight)")

def turn_right():
    motor_a_forward()
    motor_b_forward()  # Both same direction = turn right
    print("Turning right")

def turn_left():
    motor_a_reverse()
    motor_b_reverse()  # Both same direction = turn left
    print("Turning left")

# === CAMERA FUNCTIONS (SIMPLIFIED) ===
def test_camera_detection():
    """Test if Pi Camera 2 is detected"""
    print("Testing Pi Camera 2 detection...")
    try:
        result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Camera detected")
            return True
        else:
            print("✗ Camera detection failed")
            return False
    except Exception as e:
        print(f"✗ Camera test error: {e}")
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

# === DEMO FUNCTIONS ===
def golf_demo():
    """Golf-specific demo with servo and motor coordination"""
    print("=== GOLF ROBOT DEMO ===")
    
    # 1. Initialize
    print("1. Initializing...")
    center_servos()
    stop_motors()
    time.sleep(1)
    
    # 2. Setup position
    print("2. Moving to golf setup position...")
    set_servo2(45)   # Angle servo to 45°
    set_servo3(135)  # Lift servo to 135°
    time.sleep(2)
    
    # 3. Approach ball
    print("3. Approaching ball...")
    both_motors_forward()
    time.sleep(1)
    stop_motors()
    
    # 4. Align for shot
    print("4. Aligning for shot...")
    set_servo2(90)   # Center angle
    set_servo3(90)   # Lower to ball level
    time.sleep(1)
    
    # 5. Take photo of ball
    print("5. Taking photo of setup...")
    camera_photo_test()
    
    # 6. Execute swing motion
    print("6. Executing swing...")
    set_servo3(45)   # Pull back
    time.sleep(0.5)
    set_servo3(135)  # Swing forward
    time.sleep(0.5)
    
    # 7. Return to center
    print("7. Returning to ready position...")
    center_servos()
    
    print("🏌️ GOLF DEMO COMPLETE! 🏌️")

def emergency_stop():
    """Stop everything immediately"""
    stop_motors()
    center_servos()
    print("EMERGENCY STOP - All systems stopped")

# === INITIALIZE ===
print("=== GOLFBOT CONTROL SYSTEM - SERVO FIXED ===")
print("Using Servo 2 (GPIO 12) and Servo 3 (GPIO 13)")
print("Initializing...")
stop_motors()
center_servos()
time.sleep(1)
print("Ready! Type 'help' for commands")

# === MAIN CONTROL LOOP ===
try:
    while True:
        try:
            cmd = input("Golf> ").strip().lower()
            
            # === SERVO COMMANDS ===
            if cmd.startswith('s2-'):
                angle = int(cmd[3:])
                set_servo2(angle)
            elif cmd.startswith('s3-'):
                angle = int(cmd[3:])
                set_servo3(angle)
            elif cmd.startswith('sb-'):  # Both servos
                angle = int(cmd[3:])
                set_both_servos(angle)
            
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
            elif cmd == 'photo':
                camera_photo_test()
            
            # === SERVO TEST COMMANDS ===
            elif cmd == 'sc':
                center_servos()
            elif cmd == 'stest':
                servo_test_sequence()
            elif cmd == 'ssweep':
                servo_sweep_test()
            elif cmd == 'sprec':
                servo_precision_test()
            
            # === DEMO COMMANDS ===
            elif cmd == 'golf':
                golf_demo()
            elif cmd == 'stop':
                emergency_stop()
            elif cmd == 'quit' or cmd == 'q':
                emergency_stop()
                print("All systems stopped")
                break
                
            elif cmd == 'help':
                print("\n=== SERVO COMMANDS (0-180°) ===")
                print("  s2-90    - Set servo 2 to 90°")
                print("  s3-45    - Set servo 3 to 45°") 
                print("  sb-90    - Set both servos to 90°")
                print("  sc       - Center both servos (90°)")
                print("\n=== SERVO TESTS ===")
                print("  stest    - Test servo positions")
                print("  ssweep   - Smooth sweep test")
                print("  sprec    - Precision angle test")
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
                print("  photo    - Take test photo")
                print("\n=== DEMO COMMANDS ===")
                print("  golf     - Golf robot demo")
                print("  stop     - Emergency stop")
                print("  help     - Show this help")
                print("  q        - Quit")
            else:
                print("Unknown command. Type 'help' for available commands")
                
        except (ValueError, IndexError):
            print("Invalid format. Use: s2-90, s3-45, etc.")
            
except KeyboardInterrupt:
    emergency_stop()
    print("\nAll systems stopped")
finally:
    servo2.close()
    servo3.close()
    motor_in1.close()
    motor_in2.close()
    motor_in3.close()
    motor_in4.close()