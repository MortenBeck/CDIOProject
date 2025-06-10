#!/usr/bin/env python3
"""
GolfBot Control System with Pi Camera 2 Integration
Servo/Motor control + Camera testing for Pi 5
WITH SERVO ALIGNMENT CORRECTION
"""

from gpiozero import Servo, OutputDevice
import time
import subprocess
import os

# === SERVO ALIGNMENT OFFSETS ===
SERVO1_OFFSET = 0     # Servo 1 baseline (no offset)
SERVO2_OFFSET = 0     # Servo 2 (adjust if needed)
SERVO3_OFFSET = 40    # Servo 3 needs +40° to align with servo 1

# === SERVO SETUP ===
servo1 = Servo(18)  # GPIO 18 (Pin 12) - Hardware PWM0
servo2 = Servo(12)  # GPIO 12 (Pin 32) - Hardware PWM0 ALT  
servo3 = Servo(13)  # GPIO 13 (Pin 33) - Hardware PWM1 ALT

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
    actual_angle = set_servo_angle_corrected(servo1, angle, SERVO1_OFFSET)
    print(f"Servo 1 → {angle}° (actual: {actual_angle}°)")

def set_servo2(angle):
    actual_angle = set_servo_angle_corrected(servo2, angle, SERVO2_OFFSET)
    print(f"Servo 2 → {angle}° (actual: {actual_angle}°)")

def set_servo3(angle):
    actual_angle = set_servo_angle_corrected(servo3, angle, SERVO3_OFFSET)
    print(f"Servo 3 → {angle}° (actual: {actual_angle}°)")

def set_all_servos(angle):
    """Set all servos to the same LOGICAL angle (with automatic offset correction)"""
    actual1 = set_servo_angle_corrected(servo1, angle, SERVO1_OFFSET)
    actual2 = set_servo_angle_corrected(servo2, angle, SERVO2_OFFSET)
    actual3 = set_servo_angle_corrected(servo3, angle, SERVO3_OFFSET)
    print(f"All servos → {angle}° (actual: S1:{actual1}°, S2:{actual2}°, S3:{actual3}°)")

def center_servos():
    """Reset all servos to 0° logical position"""
    set_all_servos(0)

def test_alignment():
    """Test servo alignment - should be physically aligned at same logical angle"""
    print("=== SERVO ALIGNMENT TEST ===")
    print("Testing alignment at different angles...")
    
    test_angles = [0, 45, 90, 135, 180]
    for angle in test_angles:
        print(f"\nSetting all servos to {angle}°...")
        set_all_servos(angle)
        input("Press Enter to continue (check if servos are aligned)...")
    
    print("Alignment test complete!")

# === RAW SERVO FUNCTIONS (for manual correction) ===
def set_servo1_raw(angle):
    set_servo_angle_raw(servo1, angle)
    print(f"Servo 1 RAW → {angle}°")

def set_servo2_raw(angle):
    set_servo_angle_raw(servo2, angle)
    print(f"Servo 2 RAW → {angle}°")

def set_servo3_raw(angle):
    set_servo_angle_raw(servo3, angle)
    print(f"Servo 3 RAW → {angle}°")

def calibrate_servos():
    """Interactive servo calibration"""
    print("=== SERVO CALIBRATION MODE ===")
    print("Use raw commands to find alignment:")
    print("  s1r-X  : Set servo 1 to raw angle X")
    print("  s2r-X  : Set servo 2 to raw angle X") 
    print("  s3r-X  : Set servo 3 to raw angle X")
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
            elif cmd.startswith('s2r-'):
                angle = int(cmd[4:])
                if 0 <= angle <= 180:
                    set_servo2_raw(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd.startswith('s3r-'):
                angle = int(cmd[4:])
                if 0 <= angle <= 180:
                    set_servo3_raw(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd == 'done':
                break
            else:
                print("Unknown command")
                
        except (ValueError, IndexError):
            print("Invalid format. Use: s1r-90, s2r-45, etc.")

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
    center_servos()
    stop_motors()
    time.sleep(1)
    
    # 2. Camera test
    print("2. Testing camera...")
    camera_preview_test()
    
    # 3. Servo tests (now with alignment correction)
    print("3. Testing aligned servos...")
    for angle in [0, 90, 180]:
        print(f"Setting all servos to {angle}° (aligned)...")
        set_all_servos(angle)
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
    set_servo2(90)
    set_servo3(135)
    both_motors_forward()
    
    # Take photo during movement
    camera_photo_test()
    
    stop_motors()
    center_servos()
    
    print("🎉 INTEGRATED DEMO COMPLETE! 🎉")

def emergency_stop():
    """Stop everything immediately and reset servos to 0°"""
    stop_motors()
    center_servos()
    print("EMERGENCY STOP - All systems stopped")

def test_servo_pins():
    """Test servo pin responsiveness"""
    from gpiozero import LED
    test_pins = [18, 12, 13]
    
    print("Testing servo pins with LED mode...")
    for pin in test_pins:
        print(f"Testing GPIO {pin}...")
        led = LED(pin)
        led.on()
        time.sleep(0.5)
        led.off()
        time.sleep(0.5)
        led.close()
        print(f"GPIO {pin} responded")
    print("Pin test complete!")

def competition_ready():
    """Full system check for competition"""
    print("=== COMPETITION READINESS CHECK ===")
    
    # Test all systems
    print("Testing aligned servos...")
    center_servos()  # Reset to 0°
    set_all_servos(45)
    time.sleep(0.5)
    set_all_servos(135)
    time.sleep(0.5)
    set_all_servos(90)
    print("✓ Servos responsive and aligned")
    
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
        print(f"📐 Servo offsets: S1:{SERVO1_OFFSET}°, S2:{SERVO2_OFFSET}°, S3:{SERVO3_OFFSET}°")
    else:
        print("⚠️  Camera issues detected - fix before competition")

# === INITIALIZE ===
print("=== GOLFBOT CONTROL SYSTEM WITH ALIGNED SERVOS ===")
print(f"Servo offsets: S1:{SERVO1_OFFSET}°, S2:{SERVO2_OFFSET}°, S3:{SERVO3_OFFSET}°")
print("Initializing...")
stop_motors()
center_servos()
time.sleep(1)
print("Ready! Type 'help' for commands")

# === MAIN CONTROL LOOP ===
try:
    while True:
        try:
            cmd = input("Control> ").strip().lower()
            if 'inputsubmission' in cmd:
                cmd = cmd.split("'")[1].replace('\\n', '').strip()
            
            # === SERVO COMMANDS (ALIGNED) ===
            if cmd.startswith('s1-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_servo1(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd.startswith('s2-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_servo2(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd.startswith('s3-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_servo3(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd.startswith('sa-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_all_servos(angle)
                else:
                    print("Angle must be 0-180")
            
            # === RAW SERVO COMMANDS ===
            elif cmd.startswith('s1r-'):
                angle = int(cmd[4:])
                if 0 <= angle <= 180:
                    set_servo1_raw(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd.startswith('s2r-'):
                angle = int(cmd[4:])
                if 0 <= angle <= 180:
                    set_servo2_raw(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd.startswith('s3r-'):
                angle = int(cmd[4:])
                if 0 <= angle <= 180:
                    set_servo3_raw(angle)
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
                center_servos()
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
                calibrate_servos()
            elif cmd == 'quit' or cmd == 'q':
                emergency_stop()
                print("All systems stopped")
                break
                
            elif cmd == 'help':
                print("\n=== ALIGNED SERVO COMMANDS ===")
                print("  s1-90    - Set servo 1 to 90° (corrected)")
                print("  s2-45    - Set servo 2 to 45° (corrected)") 
                print("  s3-180   - Set servo 3 to 180° (corrected)")
                print("  sa-90    - Set all servos to 90° (aligned)")
                print("  sc       - Reset all servos to 0°")
                print("\n=== RAW SERVO COMMANDS ===")
                print("  s1r-90   - Set servo 1 to raw 90°")
                print("  s2r-45   - Set servo 2 to raw 45°")
                print("  s3r-180  - Set servo 3 to raw 180°")
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
                print("  test     - Test servo pins")
                print("  ready    - Competition readiness check")
                print("  align    - Test servo alignment")
                print("  calibrate- Enter calibration mode")
                print("  stop     - Emergency stop")
                print("  help     - Show this help")
                print("  q        - Quit")
                print(f"\n📐 Current offsets: S1:{SERVO1_OFFSET}°, S2:{SERVO2_OFFSET}°, S3:{SERVO3_OFFSET}°")
            else:
                print("Unknown command. Type 'help' for available commands")
                
        except (ValueError, IndexError):
            print("Invalid format. Use: s1-90, cam, etc.")
            
except KeyboardInterrupt:
    emergency_stop()
    print("\nAll systems stopped")
finally:
    servo1.close()
    servo2.close()
    servo3.close()
    motor_in1.close()
    motor_in2.close()
    motor_in3.close()
    motor_in4.close()
