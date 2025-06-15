#!/usr/bin/env python3
"""
GolfBot Control System - Motors and Camera Only
No servo functionality
"""

from gpiozero import OutputDevice
import time
import subprocess
import os

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
    """Demo with motor and camera integration"""
    print("=== MOTOR & CAMERA DEMO ===")
    
    # 1. Initialize
    print("1. Initializing systems...")
    stop_motors()
    time.sleep(1)
    
    # 2. Camera test
    print("2. Testing camera...")
    camera_preview_test()
    
    # 3. Motor tests
    print("3. Testing motors...")
    both_motors_forward()
    time.sleep(1)
    both_motors_reverse()
    time.sleep(1)
    turn_right()
    time.sleep(1)
    turn_left()
    time.sleep(1)
    stop_motors()
    
    # 4. Photo during movement
    print("4. Photo capture during movement...")
    both_motors_forward()
    camera_photo_test()
    stop_motors()
    
    print("🎉 DEMO COMPLETE! 🎉")

def emergency_stop():
    """Stop everything immediately"""
    stop_motors()
    print("EMERGENCY STOP - Motors stopped")

def test_motor_pins():
    """Test motor pin responsiveness"""
    from gpiozero import LED
    test_pins = [19, 26, 20, 21]
    
    print("Testing motor pins with LED mode...")
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
print("=== GOLFBOT CONTROL SYSTEM - MOTORS & CAMERA ===")
print("Initializing...")
stop_motors()
time.sleep(1)
print("Ready! Type 'help' for commands")

# === MAIN CONTROL LOOP ===
try:
    while True:
        try:
            cmd = input("Control> ").strip().lower()
            if 'inputsubmission' in cmd:
                cmd = cmd.split("'")[1].replace('\\n', '').strip()
            
            # === MOTOR COMMANDS ===
            if cmd == 'maf':
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
            elif cmd == 'demo':
                demo_sequence()
            elif cmd == 'stop':
                emergency_stop()
            elif cmd == 'test':
                test_motor_pins()
            elif cmd == 'ready':
                competition_ready()
            elif cmd == 'quit' or cmd == 'q':
                emergency_stop()
                print("All systems stopped")
                break
                
            elif cmd == 'help':
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
                print("  demo     - Motor & camera demo")
                print("  test     - Test motor pins")
                print("  ready    - Competition readiness check")
                print("  stop     - Emergency stop")
                print("  help     - Show this help")
                print("  q        - Quit")
            else:
                print("Unknown command. Type 'help' for available commands")
                
        except (ValueError, IndexError):
            print("Invalid format. Use: mf, cam, etc.")
            
except KeyboardInterrupt:
    emergency_stop()
    print("\nAll systems stopped")
finally:
    motor_in1.close()
    motor_in2.close()
    motor_in3.close()
    motor_in4.close()
