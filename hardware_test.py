#!/usr/bin/env python3
"""
GolfBot Hardware Testing Interface
Interactive testing for servos and DC motors
"""

import time
import logging
from hardware import GolfBotHardware
import config

class HardwareTester:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hardware = None
        
    def initialize(self):
        """Initialize hardware for testing"""
        try:
            print("=== GOLFBOT HARDWARE TESTER ===")
            print("Initializing hardware...")
            self.hardware = GolfBotHardware()
            print("✓ Hardware initialized successfully!")
            return True
        except Exception as e:
            print(f"✗ Hardware initialization failed: {e}")
            return False
    
    def servo_test_menu(self):
        """Interactive servo testing"""
        print("\n=== SERVO TESTING ===")
        print("Commands:")
        print("  s1-90    - Move servo 1 to 90°")
        print("  s2-45    - Move servo 2 to 45°") 
        print("  s3-135   - Move servo 3 to 135°")
        print("  all-90   - Move all servos to 90°")
        print("  center   - Center all servos")
        print("  open     - Collection open position")
        print("  close    - Collection close position")
        print("  release  - Release position")
        print("  demo     - Run servo demo")
        print("  status   - Show servo angles")
        print("  back     - Return to main menu")
        
        while True:
            cmd = input("\nServo> ").strip().lower()
            
            if cmd.startswith('s1-'):
                try:
                    angle = int(cmd[3:])
                    self.hardware.set_servo_angle(self.hardware.servo1, angle)
                    print(f"Servo 1 → {angle}°")
                except ValueError:
                    print("Invalid format. Use: s1-90")
                    
            elif cmd.startswith('s2-'):
                try:
                    angle = int(cmd[3:])
                    self.hardware.set_servo_angle(self.hardware.servo2, angle)
                    print(f"Servo 2 → {angle}°")
                except ValueError:
                    print("Invalid format. Use: s2-90")
                    
            elif cmd.startswith('s3-'):
                try:
                    angle = int(cmd[3:])
                    self.hardware.set_servo_angle(self.hardware.servo3, angle)
                    print(f"Servo 3 → {angle}°")
                except ValueError:
                    print("Invalid format. Use: s3-90")
                    
            elif cmd.startswith('all-'):
                try:
                    angle = int(cmd[4:])
                    self.hardware.set_servo_angle(self.hardware.servo1, angle)
                    self.hardware.set_servo_angle(self.hardware.servo2, angle)
                    self.hardware.set_servo_angle(self.hardware.servo3, angle)
                    print(f"All servos → {angle}°")
                except ValueError:
                    print("Invalid format. Use: all-90")
                    
            elif cmd == 'center':
                self.hardware.center_servos()
                print("All servos centered (90°)")
                
            elif cmd == 'open':
                self.hardware.collection_position()
                print("Collection open position")
                
            elif cmd == 'close':
                self.hardware.grab_ball()
                print("Collection close position")
                
            elif cmd == 'release':
                self.hardware.release_balls()
                print("Release position")
                
            elif cmd == 'demo':
                self.servo_demo()
                
            elif cmd == 'status':
                angles = self.hardware.get_servo_angles()
                print(f"Servo angles: {angles}")
                
            elif cmd == 'back':
                break
                
            else:
                print("Unknown command. Available: s1-90, s2-45, s3-135, all-90, center, open, close, release, demo, status, back")
    
    def motor_test_menu(self):
        """Interactive motor testing"""
        print("\n=== MOTOR TESTING ===")
        print("Commands:")
        print("  forward [time]  - Move forward (default 1s)")
        print("  backward [time] - Move backward") 
        print("  left [time]     - Turn left")
        print("  right [time]    - Turn right")
        print("  stop            - Stop all motors")
        print("  speed-30        - Set speed to 30%")
        print("  demo            - Run motor demo")
        print("  status          - Show motor status")
        print("  back            - Return to main menu")
        
        while True:
            cmd = input("\nMotor> ").strip().lower()
            parts = cmd.split()
            
            if parts[0] == 'forward':
                duration = float(parts[1]) if len(parts) > 1 else 1.0
                print(f"Moving forward for {duration}s...")
                self.hardware.move_forward(duration=duration)
                
            elif parts[0] == 'backward':
                duration = float(parts[1]) if len(parts) > 1 else 1.0
                print(f"Moving backward for {duration}s...")
                self.hardware.move_backward(duration=duration)
                
            elif parts[0] == 'left':
                duration = float(parts[1]) if len(parts) > 1 else 0.6
                print(f"Turning left for {duration}s...")
                self.hardware.turn_left(duration=duration)
                
            elif parts[0] == 'right':
                duration = float(parts[1]) if len(parts) > 1 else 0.6
                print(f"Turning right for {duration}s...")
                self.hardware.turn_right(duration=duration)
                
            elif parts[0] == 'stop':
                self.hardware.stop_motors()
                print("Motors stopped")
                
            elif parts[0].startswith('speed-'):
                try:
                    speed_percent = int(parts[0][6:])
                    speed = speed_percent / 100.0
                    self.hardware.set_speed(speed)
                    print(f"Speed set to {speed_percent}%")
                except ValueError:
                    print("Invalid format. Use: speed-50")
                    
            elif parts[0] == 'demo':
                self.motor_demo()
                
            elif parts[0] == 'status':
                status = self.hardware.get_status()
                print(f"Speed: {status['speed_percentage']}")
                print(f"Collected balls: {status['collected_balls']}")
                
            elif parts[0] == 'back':
                break
                
            else:
                print("Unknown command. Available: forward, backward, left, right, stop, speed-XX, demo, status, back")
    
    def collection_test_menu(self):
        """Test ball collection sequences"""
        print("\n=== COLLECTION TESTING ===")
        print("Commands:")
        print("  collect     - Full collection sequence")
        print("  deliver     - Full delivery sequence")
        print("  open        - Open collection mechanism")
        print("  grab        - Grab ball")
        print("  release     - Release balls")
        print("  count       - Show ball count")
        print("  reset       - Reset ball count")
        print("  back        - Return to main menu")
        
        while True:
            cmd = input("\nCollection> ").strip().lower()
            
            if cmd == 'collect':
                print("Running collection sequence...")
                success = self.hardware.attempt_ball_collection()
                print(f"Collection {'successful' if success else 'failed'}")
                
            elif cmd == 'deliver':
                print("Running delivery sequence...")
                balls = self.hardware.delivery_sequence("A")
                print(f"Delivered {balls} balls")
                
            elif cmd == 'open':
                self.hardware.collection_position()
                print("Collection mechanism opened")
                
            elif cmd == 'grab':
                self.hardware.grab_ball()
                print("Ball grabbed")
                
            elif cmd == 'release':
                balls = self.hardware.release_balls()
                print(f"Released {balls} balls")
                
            elif cmd == 'count':
                count = self.hardware.get_ball_count()
                print(f"Ball count: {count}")
                
            elif cmd == 'reset':
                self.hardware.collected_balls.clear()
                print("Ball count reset to 0")
                
            elif cmd == 'back':
                break
                
            else:
                print("Unknown command. Available: collect, deliver, open, grab, release, count, reset, back")
    
    def servo_demo(self):
        """Run servo demonstration"""
        print("Running servo demo...")
        angles = [0, 45, 90, 135, 180, 90]
        for angle in angles:
            print(f"  Moving to {angle}°...")
            self.hardware.set_servo_angle(self.hardware.servo1, angle)
            self.hardware.set_servo_angle(self.hardware.servo2, angle)
            self.hardware.set_servo_angle(self.hardware.servo3, angle)
            time.sleep(1)
        print("Servo demo complete!")
    
    def motor_demo(self):
        """Run motor demonstration"""
        print("Running motor demo...")
        print("  Forward...")
        self.hardware.move_forward(duration=1.0)
        time.sleep(0.5)
        
        print("  Backward...")
        self.hardware.move_backward(duration=1.0)
        time.sleep(0.5)
        
        print("  Turn right...")
        self.hardware.turn_right(duration=0.6)
        time.sleep(0.5)
        
        print("  Turn left...")
        self.hardware.turn_left(duration=0.6)
        time.sleep(0.5)
        
        print("Motor demo complete!")
    
    def main_menu(self):
        """Main testing menu"""
        print("\n=== MAIN MENU ===")
        print("1. Servo Testing")
        print("2. Motor Testing") 
        print("3. Collection Testing")
        print("4. Full System Demo")
        print("5. Emergency Stop")
        print("6. Exit")
        
        while True:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                self.servo_test_menu()
            elif choice == '2':
                self.motor_test_menu()
            elif choice == '3':
                self.collection_test_menu()
            elif choice == '4':
                self.full_system_demo()
            elif choice == '5':
                self.emergency_stop()
            elif choice == '6':
                break
            else:
                print("Invalid choice. Enter 1-6.")
    
    def full_system_demo(self):
        """Demonstrate full system capabilities"""
        print("\n=== FULL SYSTEM DEMO ===")
        print("This will run a complete demonstration of all systems.")
        
        if input("Continue? (y/N): ").lower() != 'y':
            return
            
        print("\n1. Servo demo...")
        self.servo_demo()
        time.sleep(1)
        
        print("\n2. Motor demo...")
        self.motor_demo()
        time.sleep(1)
        
        print("\n3. Collection sequence...")
        self.hardware.attempt_ball_collection()
        time.sleep(1)
        
        print("\n4. Delivery sequence...")
        self.hardware.delivery_sequence("A")
        
        print("\nFull system demo complete!")
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        print("\n🚨 EMERGENCY STOP ACTIVATED")
        if self.hardware:
            self.hardware.emergency_stop()
        print("All systems stopped!")
    
    def cleanup(self):
        """Clean shutdown - motors only"""
        if self.hardware:
            print("\nStopping motors...")
            # Only stop motors, don't center servos
            self.hardware.stop_motors()
            # Close GPIO connections only
            try:
                for component in [self.hardware.motor_in1, self.hardware.motor_in2, 
                                self.hardware.motor_in3, self.hardware.motor_in4]:
                    component.close()
                print("✓ Motor cleanup complete (servos left in position)")
            except Exception as e:
                print(f"Motor cleanup error: {e}")

def run_hardware_test():
    """Main entry point for hardware testing"""
    tester = HardwareTester()
    
    if not tester.initialize():
        return False
    
    try:
        print("\n🤖 GolfBot Hardware Tester Ready!")
        print("Use this interface to test servos, motors, and collection systems.")
        print("Type commands or use the menu to control hardware.")
        
        tester.main_menu()
        
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
    except Exception as e:
        print(f"\nTesting error: {e}")
    finally:
        tester.cleanup()
    
    return True

if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO)
    run_hardware_test()