#!/usr/bin/env python3
import time
try:
    from adafruit_servokit import ServoKit
except ImportError:
    print("Install required library: pip3 install adafruit-circuitpython-servokit")
    exit(1)

# Initialize servo controller (assumes PCA9685 at default address 0x40)
kit = ServoKit(channels=16)

def test_servo():
    print("Testing servo on channel 0...")
    
    # Test sequence
    positions = [0, 90, 180, 90, 0]
    
    for pos in positions:
        print(f"Moving to {pos} degrees")
        kit.servo[0].angle = pos
        time.sleep(1)
    
    print("Test complete!")

if __name__ == "__main__":
    try:
        test_servo()
    except Exception as e:
        print(f"Error: {e}")
        print("Check connections and run: sudo i2cdetect -y 1")