#!/usr/bin/env python3
"""
GolfBot Hardware Testing Interface - FIXED
Interactive testing for servos and DC motors
Updated for two-servo system: SS and SF
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
            print("Two-Servo System: SS and SF")
            print("Initializing hardware...")
            self.hardware = GolfBotHardware()
            print("✓ Hardware initialized successfully!")
            return True
        except Exception as e:
            print(f"✗ Hardware initialization failed: {e}")
            return False
    
    def servo_test_menu(self):
        """Interactive servo testing"""
        print("\n=== SERVO TESTING (Two-Servo System) ===")
        print("Commands:")
        print("  ss-90    - Move servo SS to 90°")
        print("  sf-45    - Move servo SF to 45°") 
        print("  all-90   - Move both servos to 90°")
        print("  center   - Center both servos (SS driving, SF ready)")
        print("  open     - Collection open position")
        print("  close    - Collection close position")
        print("  release  - Release position")
        print("  demo     - Run servo demo")
        print("  ss-demo  - Demo SS four-state system")
        print("  sf-demo  - Demo SF positions")
        print("  status   - Show servo angles")
        print("  back     - Return to main menu")
        
        while True:
            cmd = input("\nServo> ").strip().lower()
            
            if cmd.startswith('ss-'):
                try:
                    angle = int(cmd[3:])
                    # FIXED: Use servo controller directly
                    self.hardware.servo_controller.set_servo_angle(self.hardware.servo_controller.servo_ss, angle)
                    print(f"Servo SS → {angle}°")
                except ValueError:
                    print("Invalid format. Use: ss-90")
                    
            elif cmd.startswith('sf-'):
                try:
                    angle = int(cmd[3:])
                    # FIXED: Use servo controller directly
                    self.hardware.servo_controller.set_servo_angle(self.hardware.servo_controller.servo_sf, angle)
                    print(f"Servo SF → {angle}°")
                except ValueError:
                    print("Invalid format. Use: sf-90")
                    
            elif cmd.startswith('all-'):
                try:
                    angle = int(cmd[4:])
                    # FIXED: Use servo controller directly
                    self.hardware.servo_controller.set_servo_angle(self.hardware.servo_controller.servo_ss, angle)
                    self.hardware.servo_controller.set_servo_angle(self.hardware.servo_controller.servo_sf, angle)
                    print(f"Both servos → {angle}°")
                except ValueError:
                    print("Invalid format. Use: all-90")
                    
            elif cmd == 'center':
                self.hardware.center_servos()
                print("Servos centered (SS driving, SF ready)")
                
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
                
            elif cmd == 'ss-demo':
                self.servo_ss_demo()
                
            elif cmd == 'sf-demo':
                self.servo_sf_demo()
                
            elif cmd == 'status':
                angles = self.hardware.get_servo_angles()
                ss_state = self.hardware.get_servo_ss_state()
                print(f"Servo angles: SS={angles['servo_ss']}° ({ss_state}), SF={angles['servo_sf']}°")
                
            elif cmd == 'back':
                break
                
            else:
                print("Unknown command. Available: ss-90, sf-45, all-90, center, open, close, release, demo, ss-demo, sf-demo, status, back")
    
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
        print("\n=== COLLECTION TESTING (Two-Servo System) ===")
        print("Commands:")
        print("  collect     - Enhanced collection sequence")
        print("  deliver     - Full delivery sequence")
        print("  open        - Open collection mechanism")
        print("  grab        - Grab ball")
        print("  release     - Release balls")
        print("  count       - Show ball count")
        print("  reset       - Reset ball count")
        print("  prepare     - Prepare for collection")
        print("  back        - Return to main menu")
        
        while True:
            cmd = input("\nCollection> ").strip().lower()
            
            if cmd == 'collect':
                print("Running enhanced collection sequence...")
                success = self.hardware.enhanced_collection_sequence()
                print(f"Collection {'successful' if success else 'failed'}")
                
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
                print("Unknown command. Available: collect, deliver, open, grab, release, count, reset, prepare, back")
    
    def servo_demo(self):
        """FIXED: Run servo demonstration for both servos"""
        print("Running two-servo demo...")
        angles = [0, 45, 90, 135, 180, 90]
        for angle in angles:
            print(f"  Moving both servos to {angle}°...")
            # FIXED: Use servo controller directly
            self.hardware.servo_controller.set_servo_angle(self.hardware.servo_controller.servo_ss, angle)
            self.hardware.servo_controller.set_servo_angle(self.hardware.servo_controller.servo_sf, angle)
            time.sleep(1)
        print("Two-servo demo complete!")
    
    def servo_ss_demo(self):
        """Demo servo SS four-state system"""
        print("Running servo SS four-state demo...")
        print("  1. Driving position...")
        self.hardware.servo_ss_to_driving()
        time.sleep(1.5)
        print("  2. Pre-collect position...")
        self.hardware.servo_ss_to_pre_collect()
        time.sleep(1.5)
        print("  3. Collect position...")
        self.hardware.servo_ss_to_collect()
        time.sleep(1.5)
        print("  4. Store position...")
        self.hardware.servo_ss_to_store()
        time.sleep(1.5)
        print("  5. Return to driving...")
        self.hardware.servo_ss_to_driving()
        time.sleep(1)
        print("Servo SS four-state demo complete!")
    
    def servo_sf_demo(self):
        """Demo servo SF positions"""
        print("Running servo SF demo...")
        print("  1. Ready position...")
        self.hardware.servo_sf_to_ready()
        time.sleep(1.5)
        print("  2. Catch position...")
        self.hardware.servo_sf_to_catch()
        time.sleep(1.5)
        print("  3. Release position...")
        self.hardware.servo_sf_to_release()
        time.sleep(1.5)
        print("  4. Return to ready...")
        self.hardware.servo_sf_to_ready()
        time.sleep(1)
        print("Servo SF demo complete!")
    
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
        print("1. Servo Testing (SS & SF)")
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
        print("\n=== FULL SYSTEM DEMO (Two-Servo System) ===")
        print("This will run a complete demonstration of all systems.")
        
        if input("Continue? (y/N): ").lower() != 'y':
            return
            
        print("\n1. Two-servo demo...")
        self.servo_demo()
        time.sleep(1)
        
        print("\n2. Servo SS four-state demo...")
        self.servo_ss_demo()
        time.sleep(1)
        
        print("\n3. Servo SF demo...")
        self.servo_sf_demo()
        time.sleep(1)
        
        print("\n4. Motor demo...")
        self.motor_demo()
        time.sleep(1)
        
        print("\n5. Enhanced collection sequence...")
        self.hardware.enhanced_collection_sequence()
        time.sleep(1)
        
        print("\nFull system demo complete!")
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        print("\n🚨 EMERGENCY STOP ACTIVATED")
        if self.hardware:
            self.hardware.emergency_stop()
        print("All systems stopped!")
    
    def cleanup(self):
        """FIXED: Clean shutdown - motors only"""
        if self.hardware:
            print("\nStopping motors...")
            # Only stop motors, don't center servos
            self.hardware.stop_motors()
            # FIXED: Access motor components through motor_controller
            try:
                motor_controller = self.hardware.motor_controller
                for component in [motor_controller.motor_in1, motor_controller.motor_in2, 
                                motor_controller.motor_in3, motor_controller.motor_in4]:
                    if hasattr(component, 'close'):
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
        print("Two-Servo System: SS (collection) and SF (assist)")
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