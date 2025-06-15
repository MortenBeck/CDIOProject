#!/usr/bin/env python3
"""
Simple Servo Control with PCA9685
Interactive servo movement
"""

import time
import busio
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# === PCA9685 SETUP ===
print("=== SIMPLE SERVO CONTROL ===")
print("Initializing PCA9685...")

try:
    # Initialize I2C and PCA9685
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=0x40)
    pca.frequency = 50
    
    # Create servo objects
    servo0 = servo.Servo(pca.channels[0])  # Channel 0
    servo1 = servo.Servo(pca.channels[1])  # Channel 1
    
    print("✓ PCA9685 ready!")
    print("✓ Servo 0 on channel 0")
    print("✓ Servo 1 on channel 1")
    
    # Center servos
    servo0.angle = 90
    servo1.angle = 90
    print("Servos centered to 90°")
    
except Exception as e:
    print(f"✗ Setup failed: {e}")
    exit(1)

# === CONTROL FUNCTIONS ===
def set_servo0(angle):
    if 0 <= angle <= 180:
        servo0.angle = angle
        print(f"Servo 0 → {angle}°")
    else:
        print("Angle must be 0-180")

def set_servo1(angle):
    if 0 <= angle <= 180:
        servo1.angle = angle
        print(f"Servo 1 → {angle}°")
    else:
        print("Angle must be 0-180")

def set_both(angle):
    if 0 <= angle <= 180:
        servo0.angle = angle
        servo1.angle = angle
        print(f"Both servos → {angle}°")
    else:
        print("Angle must be 0-180")

def demo():
    print("Running demo...")
    angles = [0, 45, 90, 135, 180, 90]
    for angle in angles:
        print(f"Moving to {angle}°...")
        servo0.angle = angle
        servo1.angle = angle
        time.sleep(1)
    print("Demo complete!")

# === MAIN CONTROL LOOP ===
print("\nReady! Type commands:")
print("Commands:")
print("  s0-90    - Move servo 0 to 90°")
print("  s1-45    - Move servo 1 to 45°")
print("  both-90  - Move both servos to 90°")
print("  center   - Center both servos (90°)")
print("  demo     - Run movement demo")
print("  help     - Show commands")
print("  quit     - Exit")

try:
    while True:
        cmd = input("\nServo> ").strip().lower()
        
        if cmd.startswith('s0-'):
            try:
                angle = int(cmd[3:])
                set_servo0(angle)
            except ValueError:
                print("Invalid format. Use: s0-90")
                
        elif cmd.startswith('s1-'):
            try:
                angle = int(cmd[3:])
                set_servo1(angle)
            except ValueError:
                print("Invalid format. Use: s1-90")
                
        elif cmd.startswith('both-'):
            try:
                angle = int(cmd[5:])
                set_both(angle)
            except ValueError:
                print("Invalid format. Use: both-90")
                
        elif cmd == 'center':
            set_both(90)
            
        elif cmd == 'demo':
            demo()
            
        elif cmd == 'help':
            print("\nCommands:")
            print("  s0-90    - Move servo 0 to 90°")
            print("  s1-45    - Move servo 1 to 45°")
            print("  both-90  - Move both servos to 90°")
            print("  center   - Center both servos (90°)")
            print("  demo     - Run movement demo")
            print("  help     - Show commands")
            print("  quit     - Exit")
            
        elif cmd == 'quit' or cmd == 'q':
            break
            
        else:
            print("Unknown command. Type 'help' for commands")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # Center servos and cleanup
    try:
        servo0.angle = 90
        servo1.angle = 90
        print("Servos centered")
        pca.deinit()
        print("PCA9685 stopped")
    except:
        pass
    
print("Goodbye!")