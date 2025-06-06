from machine import Pin, PWM
import time

# === SERVO SETUP ===
# Servos on pins 12, 32, and 33 (GPIO18, GPIO12, GPIO13)
servo1 = PWM(Pin(18))  # Pin 12 = GPIO18
servo2 = PWM(Pin(12))  # Pin 32 = GPIO12  
servo3 = PWM(Pin(13))  # Pin 33 = GPIO13

# Set all servos to 50Hz
servo1.freq(50)
servo2.freq(50)
servo3.freq(50)

# === SERVO FUNCTIONS ===
def set_angle(servo, angle):
    """Set a specific servo to angle (0-180 degrees)"""
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    # Calculate duty cycle (0-65535 range)
    duty = int(1640 + (angle * 36.4))
    servo.duty_u16(duty)

def set_servo1(angle):
    set_angle(servo1, angle)
    print(f"Servo 1 → {angle}°")

def set_servo2(angle):
    set_angle(servo2, angle)
    print(f"Servo 2 → {angle}°")

def set_servo3(angle):
    set_angle(servo3, angle)
    print(f"Servo 3 → {angle}°")

def set_all_servos(angle):
    set_angle(servo1, angle)
    set_angle(servo2, angle)
    set_angle(servo3, angle)
    print(f"All servos → {angle}°")

def center_servos():
    set_all_servos(90)

def demo_sequence():
    """Demo showing servo movements"""
    print("=== SERVO DEMO SEQUENCE ===")
    
    # Center servos
    print("1. Centering servos...")
    center_servos()
    time.sleep(1)
    
    # Test individual servos
    print("2. Testing individual servos...")
    set_servo1(45)
    time.sleep(1)
    set_servo2(135)
    time.sleep(1)
    set_servo3(0)
    time.sleep(1)
    
    # Move all servos together
    print("3. Moving all servos together...")
    set_all_servos(180)
    time.sleep(1)
    set_all_servos(0)
    time.sleep(1)
    
    # Return to center
    print("4. Reset to center...")
    center_servos()
    time.sleep(1)
    print("Demo complete!")

def emergency_stop():
    """Center all servos"""
    center_servos()
    print("EMERGENCY STOP - All servos centered")

# === INITIALIZE ===
print("=== SERVO CONTROL ===")
print("Initializing...")
center_servos()
time.sleep(1)
print("Ready! Type 'help' for commands")

# === MAIN CONTROL LOOP ===
while True:
    try:
        cmd = input("Servo> ").strip().lower()
        
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
        
        # === QUICK COMMANDS ===
        elif cmd == 'sc':
            center_servos()
        elif cmd == 'demo':
            demo_sequence()
        elif cmd == 'stop':
            emergency_stop()
        elif cmd == 'quit' or cmd == 'q':
            emergency_stop()
            servo1.deinit()
            servo2.deinit()
            servo3.deinit()
            print("All servos stopped and disconnected")
            break
            
        elif cmd == 'help':
            print("\n=== SERVO COMMANDS ===")
            print("  s1-90    - Set servo 1 to 90°")
            print("  s2-45    - Set servo 2 to 45°") 
            print("  s3-180   - Set servo 3 to 180°")
            print("  sa-90    - Set all servos to 90°")
            print("  sc       - Center all servos (90°)")
            print("\n=== SPECIAL COMMANDS ===")
            print("  demo     - Run servo demo sequence")
            print("  stop     - Emergency stop (center all)")
            print("  help     - Show this help")
            print("  q        - Quit")
        else:
            print("Unknown command. Type 'help' for available commands")
            
    except KeyboardInterrupt:
        emergency_stop()
        servo1.deinit()
        servo2.deinit()
        servo3.deinit()
        print("\nAll servos stopped")
        break
    except (ValueError, IndexError):
        print("Invalid format. Use: s1-90, s2-45, etc. Type 'help' for commands")
