#!/usr/bin/env python3
"""
GolfBot Control System with Pi Camera 2 Integration
Servo/Motor control + Camera testing for Pi 5
Using PCA9685 for servo control
"""

from gpiozero import Servo, OutputDevice
import time
import subprocess
import os

# === SERVO SETUP (Direct GPIO) ===
ss = Servo(2)   # GPIO 2 (Pin 3)
sf = Servo(3)   # GPIO 3 (Pin 5)

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

# === SERVO FUNCTIONS ===
def set_servo_angle(servo_obj, angle):
    """Set servo to angle (0-180 degrees)"""
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    value = (angle - 90) / 90
    servo_obj.value = value

def set_ss(angle):
    set_servo_angle(ss, angle)
    print(f"SS servo → {angle}°")

def set_sf(angle):
    set_servo_angle(sf, angle)
    print(f"SF servo → {angle}°")

def set_both_servos(angle):
    set_servo_angle(ss, angle)
    set_servo_angle(sf, angle)
    print(f"Both servos → {angle}°")

def center_servos():
    set_both_servos(90)

# === DC MOTOR FUNCTIONS ===
# === CORRECTED DC MOTOR FUNCTIONS ===
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
    
    # 3. Servo tests
    print("3. Testing servos...")
    for i, servo_func in enumerate([set_ss, set_sf], 1):
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
    set_ss(45)
    set_sf(135)
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
    test_pins = [2, 3]
    
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
    set_ss(45)
    time.sleep(0.2)
    set_ss(135)
    time.sleep(0.2)
    set_ss(90)
    set_sf(45)
    time.sleep(0.2)
    set_sf(135)
    time.sleep(0.2)
    set_sf(90)
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
print("=== GOLFBOT CONTROL SYSTEM - GPIO SERVOS ===")
print("SS servo: GPIO 2 (Pin 3)")
print("SF servo: GPIO 3 (Pin 5)")
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
            
            # === SERVO COMMANDS ===
            if cmd.startswith('ss-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_ss(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd.startswith('sf-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_sf(angle)
                else:
                    print("Angle must be 0-180")
            elif cmd.startswith('sb-'):
                angle = int(cmd[3:])
                if 0 <= angle <= 180:
                    set_both_servos(angle)
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
            elif cmd == 'quit' or cmd == 'q':
                emergency_stop()
                print("All systems stopped")
                break
                
            elif cmd == 'help':
                print("\n=== SERVO COMMANDS ===")
                print("  ss-90    - Set SS servo to 90°")
                print("  sf-45    - Set SF servo to 45°") 
                print("  sb-90    - Set both servos to 90°")
                print("  sc       - Center both servos")
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
    ss.close()
    sf.close()
    motor_in1.close()
    motor_in2.close()
    motor_in3.close()
    motor_in4.close()