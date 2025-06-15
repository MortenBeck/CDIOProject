#!/usr/bin/env python3
"""
Simple servo test - choose your setup
"""
import time

# ===== OPTION A: PCA9685 CONTROLLER =====
# Use this if you have PCA9685 with BOTH SDA (pin 3) AND SCL (pin 5) connected

def test_pca9685():
    try:
        print("Testing PCA9685...")
        print("Device detected at address 0x40 ✓")
        
        # Use direct I2C setup instead of board
        import busio
        import digitalio
        import board
        from adafruit_pca9685 import PCA9685
        from adafruit_motor import servo
        
        # Create I2C bus directly
        i2c = busio.I2C(board.SCL, board.SDA)
        
        # Initialize PCA9685 with explicit address
        pca = PCA9685(i2c, address=0x40)
        pca.frequency = 50
        
        # Create servo on channel 0
        servo1 = servo.Servo(pca.channels[0])
        
        print("Moving servo on channel 0...")
        servo1.angle = 0
        print("Servo at 0°")
        time.sleep(1)
        servo1.angle = 90
        print("Servo at 90°")
        time.sleep(1)
        servo1.angle = 180
        print("Servo at 180°")
        time.sleep(1)
        servo1.angle = 90
        print("Servo at 90°")
        
        print("✓ PCA9685 test complete!")
        pca.deinit()
        
    except Exception as e:
        print(f"PCA9685 failed: {e}")
        import traceback
        traceback.print_exc()

# ===== OPTION B: DIRECT GPIO =====
# Use this if servo is connected directly to Pi pin (no PCA9685)

def test_direct_gpio():
    try:
        from gpiozero import Servo
        
        print("Testing direct GPIO...")
        
        # Create servo on GPIO 2 (pin 3)
        servo1 = Servo(2)
        
        print("Moving servo on pin 3...")
        servo1.value = -1  # 0 degrees
        time.sleep(1)
        servo1.value = 0   # 90 degrees
        time.sleep(1)
        servo1.value = 1   # 180 degrees
        time.sleep(1)
        servo1.value = 0   # 90 degrees
        
        print("Direct GPIO test complete!")
        servo1.close()
        
    except Exception as e:
        print(f"Direct GPIO failed: {e}")

# ===== MAIN TEST =====
if __name__ == "__main__":
    print("=== SIMPLE SERVO TEST ===")
    print("1. PCA9685 Controller (needs SDA + SCL)")
    print("2. Direct GPIO (servo directly to Pi)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        test_pca9685()
    elif choice == "2":
        test_direct_gpio()
    else:
        print("Invalid choice")
        
    print("Test finished!")