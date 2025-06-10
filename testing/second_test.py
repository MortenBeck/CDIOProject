#!/usr/bin/env python3
"""
GolfBot Control System with Pi Camera 2 Integration and Speed Control
Servo/Motor control + Camera testing for Pi 5
"""

from gpiozero import Servo, PWMOutputDevice
import time
import subprocess
import os

# === SERVO SETUP ===
servo1 = Servo(18)  # GPIO 18 (Pin 12) - Hardware PWM0
servo2 = Servo(12)  # GPIO 12 (Pin 32) - Hardware PWM0 ALT  
servo3 = Servo(13)  # GPIO 13 (Pin 33) - Hardware PWM1 ALT

# === DC MOTOR SETUP WITH PWM ===
motor_in1 = PWMOutputDevice(19)  # GPIO 19 (Pin 35) - Motor A direction 1
motor_in2 = PWMOutputDevice(26)  # GPIO 26 (Pin 37) - Motor A direction 2
motor_in3 = PWMOutputDevice(20)  # GPIO 20 (Pin 38) - Motor B direction 1
motor_in4 = PWMOutputDevice(21)  # GPIO 21 (Pin 40) - Motor B direction 2

# === SPEED SETTINGS ===
MOTOR_SPEED_SLOW = 0.3    # 30% speed
MOTOR_SPEED_MEDIUM = 0.5  # 50% speed
MOTOR_SPEED_FAST = 0.8    # 80% speed
MOTOR_SPEED_MAX = 1.0     # 100% speed

# Current speed setting (change this to adjust default speed)
current_speed = MOTOR_SPEED_SLOW

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

# === SERVO FUNCTIONS ===
def set_servo_angle(servo, angle):
    """Set servo to angle (0-180 degrees)"""
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    value = (angle - 90) / 90
    servo.value = value

def set_servo1(angle):
    set_servo_angle(servo1, angle)
    print(f"Servo 1 → {angle}°")

def set_servo2(angle):
    set_servo_angle(servo2, angle)
    print(f"Servo 2 → {angle}°")

def set_servo3(angle):
    set_servo_angle(servo3, angle)
    print(f"Servo 3 → {angle}°")

def set_all_servos(angle):
    set_servo_angle(servo1, angle)
    set_servo_angle(servo2, angle)
    set_servo_angle(servo3, angle)
    print(f"All servos → {angle}°")

def center_servos():
    set_all_servos(90)

# === DC MOTOR FUNCTIONS WITH SPEED CONTROL ===
def stop_motors():
    motor_in1.off()
    motor_in2.off()
    motor_in3.off()
    motor_in4.off()
    print("Motors stopped")

def motor_a_forward(speed=None):
    if speed is None:
        speed = current_speed
    motor_in1.value = speed
    motor_in2.off()
    print(f"Motor A forward at {int(speed*100)}% speed")

def motor_a_reverse(speed=None):
    if speed is None:
        speed = current_speed
    motor_in1.off()
    motor_in2.value = speed
    print(f"Motor A reverse at {int(speed*100)}% speed")

def motor_b_forward(speed=None):
    if speed is None:
        speed = current_speed
    motor_in3.value = speed
    motor_in4.off()
    print(f"Motor B forward at {int(speed*100)}% speed")

def motor_b_reverse(speed=None):
    if speed is None:
        speed = current_speed
    motor_in3.off()
    motor_in4.value = speed
    print(f"Motor B reverse at {int(speed*100)}% speed")

# CORRECTED FUNCTIONS - Motor B is reversed for straight movement
def both_motors_forward(speed=None):
    if speed is None:
        speed = current_speed
    motor_a_forward(speed)
    motor_b_reverse(speed)  # Reversed because motor B is mirrored
    print(f"Moving forward at {int(speed*100)}% speed")

def both_motors_reverse(speed=None):
    if speed is None:
        speed = current_speed
    motor_a_reverse(speed)
    motor_b_forward(speed)  # Reversed because motor B is mirrored
    print(f"Moving reverse at {int(speed*100)}% speed")

def turn_right(speed=None):
    if speed is None:
        speed = current_speed
    motor_a_forward(speed)
    motor_b_forward(speed)  # Both same direction = turn right
    print(f"Turning right at {int(speed*100)}% speed")

def turn_left(speed=None):
    if speed is None:
        speed = current_speed
    motor_a_reverse(speed)
    motor_b_reverse(speed)  # Both same direction = turn left
    print(f"Turning left at {int(speed*100)}% speed")

# === SPEED CONTROL FUNCTIONS ===
def set_speed(speed):
    """Set motor speed (0.0 to 1.0)"""
    global current_speed
    if 0.0 <= speed <= 1.0:
        current_speed = speed
        print(f"Motor speed set to {int(speed*100)}%")
    else:
        print("Speed must be between 0.0 and 1.0")

def set_speed_slow():
    set_speed(MOTOR_SPEED_SLOW)

def set_speed_medium():
    set_speed(MOTOR_SPEED_MEDIUM)

def set_speed_fast():
    set_speed(MOTOR_SPEED_FAST)

def set_speed_max():
    set_speed(MOTOR_SPEED_MAX)

def get_current_speed():
    print(f"Current speed: {int(current_speed*100)}%")
    return current_speed

# === GRADUAL MOVEMENT FUNCTIONS ===
def gradual_forward(duration=2, target_speed=None):
    """Gradually accelerate forward then stop"""
    if target_speed is None:
        target_speed = current_speed
    
    print(f"Gradual forward movement for {duration}s...")
    steps = 20
    for i in range(steps):
        speed = (target_speed * i) / steps
        both_motors_forward(speed)
        time.sleep(duration / (steps * 2))
    
    # Run at full speed for half the time
    both_motors_forward(target_speed)
    time.sleep(duration / 2)
    
    # Gradual stop
    for i in range(steps, 0, -1):
        speed = (target_speed * i) / steps
        both_motors_forward(speed)
        time.sleep(duration / (steps * 4))
    
    stop_motors()
    print("Gradual movement complete")

# === TEST FUNCTIONS ===
def speed_test():
    """Test all speed levels"""
    print("=== SPEED TEST ===")
    speeds = [
        ("Slow", MOTOR_SPEED_SLOW),
        ("Medium", MOTOR_SPEED_MEDIUM), 
        ("Fast", MOTOR_SPEED_FAST),
        ("Max", MOTOR_SPEED_MAX)
    ]
    
    for name, speed in speeds:
        print(f"Testing {name} speed ({int(speed*100)}%)...")
        both_motors_forward(speed)
        time.sleep(1.5)
        stop_motors()
        time.sleep(0.5)
    
    print("Speed test complete!")

def precision_test():
    """Test precise movements at slow speed"""
    print("=== PRECISION MOVEMENT TEST ===")
    old_speed = current_speed
    set_speed(0.2)  # Very slow for precision
    
    print("Forward...")
    both_motors_forward()
    time.sleep(1)
    
    print("Right turn...")
    turn_right()
    time.sleep(0.5)
    
    print("Forward...")
    both_motors_forward()
    time.sleep(1)
    
    print("Left turn...")
    turn_left()
    time.sleep(0.5)
    
    stop_motors()
    set_speed(old_speed)  # Restore original speed
    print("Precision test complete!")

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
    
    # 3. Servo tests
    print("3. Testing servos...")
    for i, servo_func in enumerate([set_servo1, set_servo2, set_servo3], 1):
        servo_func(0)
        time.sleep(0.5)
        servo_func(180)
        time.sleep(0.5)
        servo_func(90)
        time.sleep(0.5)
    
    # 4. Motor tests at slow speed
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
    """Stop everything immediately"""
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
    print("Testing servos...")
    center_servos()
    for i in [1, 2, 3]:
        getattr(globals(), f'set_servo{i}')(45)
        time.sleep(0.2)
        getattr(globals(), f'set_servo{i}')(135)
        time.sleep(0.2)
        getattr(globals(), f'set_servo{i}')(90)
    print("✓ Servos responsive")
    
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
print("=== GOLFBOT CONTROL SYSTEM WITH CAMERA AND SPEED CONTROL ===")
print("Initializing...")
stop_motors()
center_servos()
time.sleep(1)
print(f"Ready! Default speed: {int(current_speed*100)}%. Type 'help' for commands")

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
            elif cmd == 'tr':
                turn_right()
            elif cmd == 'tl':
                turn_left()
            elif cmd == 'ms':
                stop_motors()
            
            # === SPEED COMMANDS ===
            elif cmd.startswith('speed-'):
                try:
                    speed = float(cmd[6:])
                    set_speed(speed)
                except ValueError:
                    print("Invalid speed. Use: speed-0.3")
            elif cmd == 'slow':
                set_speed_slow()
            elif cmd == 'medium':
                set_speed_medium()
            elif cmd == 'fast':
                set_speed_fast()
            elif cmd == 'max':
                set_speed_max()
            elif cmd == 'speed':
                get_current_speed()
            
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
            elif cmd == 'speedtest':
                speed_test()
            elif cmd == 'precision':
                precision_test()
            elif cmd == 'gradual':
                gradual_forward()
            elif cmd == 'ready':
                competition_ready()
            elif cmd == 'quit' or cmd == 'q':
                emergency_stop()
                print("All systems stopped")
                break
                
            elif cmd == 'help':
                print("\n=== SERVO COMMANDS ===")
                print("  s1-90    - Set servo 1 to 90°")
                print("  s2-45    - Set servo 2 to 45°") 
                print("  s3-180   - Set servo 3 to 180°")
                print("  sa-90    - Set all servos to 90°")
                print("  sc       - Center all servos")
                print("\n=== MOTOR COMMANDS ===")
                print("  maf      - Motor A forward")
                print("  mar      - Motor A reverse")
                print("  mbf      - Motor B forward")
                print("  mbr      - Motor B reverse")
                print("  mf       - Move forward (straight)")
                print("  mr       - Move reverse (straight)")
                print("  tr       - Turn right")
                print("  tl       - Turn left")
                print("  ms       - Stop motors")
                print("\n=== SPEED CONTROL ===")
                print("  slow     - Set slow speed (30%)")
                print("  medium   - Set medium speed (50%)")
                print("  fast     - Set fast speed (80%)")
                print("  max      - Set max speed (100%)")
                print("  speed-0.3 - Set custom speed (0.0-1.0)")
                print("  speed    - Show current speed")
                print("\n=== CAMERA COMMANDS ===")
                print("  cam      - Detect camera")
                print("  preview  - 5-second preview")
                print("  photo    - Take test photo")
                print("  video    - Record 3s video")
                print("  camtest  - Full camera test")
                print("\n=== SYSTEM COMMANDS ===")
                print("  demo     - Integrated system demo")
                print("  test     - Test servo pins")
                print("  speedtest - Test all speed levels")
                print("  precision - Precision movement test")
                print("  gradual  - Gradual acceleration test")
                print("  ready    - Competition readiness check")
                print("  stop     - Emergency stop")
                print("  help     - Show this help")
                print("  q        - Quit")
            else:
                print("Unknown command. Type 'help' for available commands")
                
        except (ValueError, IndexError):
            print("Invalid format. Use: s1-90, cam, speed-0.3, etc.")
            
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
