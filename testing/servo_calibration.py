# === SERVO CALIBRATION CONSTANTS (find these values) ===
SERVO1_MIN_VALUE = -0.8    # Actual minimum gpiozero value for servo 1
SERVO1_MAX_VALUE = 0.8     # Actual maximum gpiozero value for servo 1
SERVO3_MIN_VALUE = -0.7    # Actual minimum gpiozero value for servo 3  
SERVO3_MAX_VALUE = 0.7     # Actual maximum gpiozero value for servo 3

def set_servo_calibrated(servo, angle, min_val, max_val):
    """Set servo with hardware-calibrated range"""
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    # Map 0-180° to actual hardware range
    normalized = angle / 180.0  # 0.0 to 1.0
    value = min_val + (normalized * (max_val - min_val))
    servo.value = value
    return value

def set_servo1_calibrated(angle):
    actual = set_servo_calibrated(servo1, angle, SERVO1_MIN_VALUE, SERVO1_MAX_VALUE)
    print(f"Servo 1 → {angle}° (gpiozero: {actual:.2f})")
    return actual

def set_servo3_calibrated(angle):
    actual = set_servo_calibrated(servo3, angle, SERVO3_MIN_VALUE, SERVO3_MAX_VALUE)
    print(f"Servo 3 → {angle}° (gpiozero: {actual:.2f})")
    return actual

def find_servo_limits():
    """Interactive calibration to find actual servo limits"""
    print("=== SERVO 1 & 3 RANGE CALIBRATION ===")
    
    for servo_name, servo in [("Servo 1", servo1), ("Servo 3", servo3)]:
        print(f"\nCalibrating {servo_name}:")
        print("Testing gpiozero values...")
        
        for test_val in [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            servo.value = test_val
            response = input(f"Value {test_val}: Does this work? (y/n): ")
            if response.lower() == 'n':
                print(f"Found limit at {test_val}")
        
        min_val = float(input(f"Enter minimum working value for {servo_name}: "))
        max_val = float(input(f"Enter maximum working value for {servo_name}: "))
        print(f"{servo_name}: MIN={min_val}, MAX={max_val}")
        print(f"Update constants: {servo_name.upper().replace(' ', '')}_MIN_VALUE = {min_val}")
        print(f"Update constants: {servo_name.upper().replace(' ', '')}_MAX_VALUE = {max_val}")

# Add these commands to your main control loop:
def handle_command(cmd):
    if cmd.startswith('s1c-'):  # Calibrated servo 1
        angle = int(cmd[4:])
        if 0 <= angle <= 180:
            set_servo1_calibrated(angle)
        else:
            print("Angle must be 0-180")

    elif cmd.startswith('s3c-'):  # Calibrated servo 3
        angle = int(cmd[4:])
        if 0 <= angle <= 180:
            set_servo3_calibrated(angle)
        else:
            print("Angle must be 0-180")

    elif cmd == 'findlimits':
        find_servo_limits()

    elif cmd == 'testrange':
        # Test full calibrated range
        print("Testing Servo 1 calibrated range...")
        for angle in [0, 45, 90, 135, 180]:
            set_servo1_calibrated(angle)
            time.sleep(1)
        
        print("Testing Servo 3 calibrated range...")
        for angle in [0, 45, 90, 135, 180]:
            set_servo3_calibrated(angle)
            time.sleep(1)