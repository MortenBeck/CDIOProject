#!/usr/bin/env python3
"""
Simple servo position reader - shows current servo positions without moving them
"""

from gpiozero import Servo
import time

# === SERVO SETUP ===
servo1 = Servo(18)  # GPIO 18
servo2 = Servo(12)  # GPIO 12  
servo3 = Servo(13)  # GPIO 13

def gpiozero_to_degrees(value):
    """Convert gpiozero value (-1.0 to 1.0) back to degrees (0-180)"""
    if value is None:
        return "Not Set"
    degrees = (value * 90) + 90
    return round(degrees, 1)

def read_servo_positions():
    """Read and display current servo positions"""
    print("=== CURRENT SERVO POSITIONS ===")
    
    servos = [
        ("Servo 1 (GPIO 18)", servo1),
        ("Servo 2 (GPIO 12)", servo2),
        ("Servo 3 (GPIO 13)", servo3)
    ]
    
    for name, servo in servos:
        raw_value = servo.value
        degrees = gpiozero_to_degrees(raw_value)
        print(f"{name}:")
        print(f"  Raw gpiozero value: {raw_value}")
        print(f"  Equivalent degrees: {degrees}°")
        print()

if __name__ == "__main__":
    try:
        print("Reading servo positions (no movement)...")
        read_servo_positions()
        
        print("Press Ctrl+C to exit, or Enter to read again...")
        while True:
            input()
            read_servo_positions()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        servo1.close()
        servo2.close()
        servo3.close()
