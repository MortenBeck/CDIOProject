#!/usr/bin/env python3
"""
GolfBot Control System with Pi Camera 2 Integration
Servo controller (PCA9685) + Motor control + Camera testing for Pi 5
"""

from gpiozero import OutputDevice
import time
import subprocess
import os

# === SERVO CONTROLLER SETUP (PCA9685) ===
try:
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=16)  # PCA9685 with 16 channels
    
    # Define which channels your servos are connected to
    SERVO2_CHANNEL = 0  # Channel 1 on the controller
    SERVO3_CHANNEL = 2  # Channel 2 on the controller (if you have 3 servos)
    
    servo_controller_available = True
    print("✓ PCA9685 servo controller initialized")
except ImportError:
    print("⚠️  Adafruit ServoKit not installed. Install with: pip install adafruit-circuitpython-servokit")
    servo_controller_available = False
except Exception as e:
    print(f"⚠️  Servo controller initialization failed: {e}")
    servo_controller_available = False

# === DC MOTOR SETUP (unchanged) ===
motor_in1 = OutputDevice(19)  # GPIO 19 (Pin 35)
motor_in2 = OutputDevice(26)  # GPIO 26 (Pin 37)
motor_in3 = OutputDevice(20)  # GPIO 20 (Pin 38)
motor_in4 = OutputDevice(21)  # GPIO 21 (Pin 40)

# === CAMERA SETUP (unchanged) ===
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

# === MODIFIED SERVO FUNCTIONS ===
def set_servo_angle(channel, angle):
    """Set servo to specific angle using servo controller"""
    if not servo_controller_available:
        print("⚠️  Servo controller not available")
        return
    
    try:
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        kit.servo[channel].angle = angle
    except Exception as e:
        print(f"Error setting servo {channel} to {angle}°: {e}")

def set_servo1(angle):
    set_servo_angle(SERVO1_CHANNEL, angle)
    print(f"Servo 1 → {angle}°")

def set_servo2(angle):
    set_servo_angle(SERVO2_CHANNEL, angle)
    print(f"Servo 2 → {angle}°")

def set_servo3(angle):
    set_servo_angle(SERVO3_CHANNEL, angle)
    print(f"Servo 3 → {angle}°")

def set_all_servos(angle):
    set_servo_angle(SERVO1_CHANNEL, angle)
    set_servo_angle(SERVO2_CHANNEL, angle)
    set_servo_angle(SERVO3_CHANNEL, angle)
    print(f"All servos → {angle}°")

def center_servos():
    set_all_servos(90)

def servo_off(channel):
    """Turn off PWM signal to servo (servo will go limp)"""
    if not servo_controller_available:
        return
    try:
        kit.servo[channel].angle = None  # This turns off the PWM signal
    except Exception as e:
        print(f"Error turning off servo {channel}: {e}")

def all_servos_off():
    """Turn off all servos"""
    servo_off(SERVO1_CHANNEL)
    servo_off(SERVO2_CHANNEL)
    servo_off(SERVO3_CHANNEL)
    print("All servos turned off")

# === DC MOTOR FUNCTIONS (unchanged) ===
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

# === INTEGRATED DEMO ===
def demo_sequence():
    """Demo with servo, motor, and camera integration"""
    print("=== COMPREHENSIVE SYSTEM DEMO ===")
    
    if not servo_controller_available:
        print("⚠️  Servo controller not available - skipping servo tests")
    
    # 1. Initialize
    print("1. Initializing systems...")
    if servo_controller_available:
        center_servos()
    stop_motors()
    time.sleep(1)
    
    # 2. Camera test
    print("2. Testing camera...")
    camera_preview_test()
    
    # 3. Servo tests
    if servo_controller_available:
        print("3. Testing servos...")
        for i, servo_func in enumerate([set_servo1, set_servo2], 1):  # Only 2 servos now
            servo_func(0)
            time.sleep(0.5)
            servo_func(180)
            time.sleep(0.5)
            servo_func(90)
            time.sleep(0.5)
    
    # 4. Motor tests
    print("4. Testing motors...")
    both_motors_forward()
    time.sleep(1)
    both_motors_reverse()
    time.sleep(1)
    stop_motors()
    
    # 5. Coordinated test with photo
    print("5. Coordinated movement with photo capture...")
    if servo_controller_available:
        set_servo1(45)
        set_servo2(135)
    both_motors_forward()
    
    # Take photo during movement
    camera_photo_test()
    
    stop_motors()
    if servo_controller_available:
        center_servos()
    
    print("🎉 INTEGRATED DEMO COMPLETE! 🎉")

def emergency_stop():
    """Stop everything immediately"""
    stop_motors()
    if servo_controller_available:
        all_servos_off()  # Turn off servos completely
    print("EMERGENCY STOP - All systems stopped")

def competition_ready():
    """Full system check for competition"""
    print("=== COMPETITION READINESS CHECK ===")
    
    # Test servos if available
    if servo_controller_available:
        print("Testing servos...")
        center_servos()
        for i in [1, 2]:  # Only 2 servos
            servo_func = globals()[f'set_servo{i}']
            servo_func(45)
            time.sleep(0.2)
            servo_func(135)
            time.sleep(0.2)
            servo_func(90)
        print("✓ Servos responsive")
    else:
        print("⚠️  Servo controller not available")
    
    print("Testing motors...")
    motor_a_forward()
    time.sleep(0.5)
    motor_b_forward()
    time.sleep(0.5)
    stop_motors()
    print("✓ Motors responsive")
    
    print("Testing camera...")
    camera_ok = test_camera_detection()
    
    if camera_ok and servo_controller_available:
        print("🏆 SYSTEM READY FOR COMPETITION!")
    else:
        print("⚠️  Some systems have issues - check before competition")

# === INITIALIZE ===
print("=== GOLFBOT CONTROL SYSTEM WITH SERVO CONTROLLER ===")
print("Initializing...")
stop_motors()
if servo_controller_available:
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
            elif cmd.startswith('sa-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_all_servos(angle)
                else:
                    print("Angle must be 0-180")
            
            # === MOTOR COMMANDS (unchanged) ===
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
            
            # === CAMERA COMMANDS (unchanged) ===
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
            elif cmd == 'soff':
                all_servos_off()
            elif cmd == 'demo':
                demo_sequence()
            elif cmd == 'stop':
                emergency_stop()
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
                print("  sa-90    - Set both servos to 90°")
                print("  sc       - Center both servos")
                print("  soff     - Turn off all servos")
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
                print("  ready    - Competition readiness check")
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
    # Clean up motor pins
    motor_in1.close()
    motor_in2.close()
    motor_in3.close()
    motor_in4.close()
    print("GPIO cleanup complete")