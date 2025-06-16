#!/usr/bin/env python3
"""
GolfBot Autonomous Wall Avoidance System - MAIN APPLICATION
Pi 5 Compatible with automatic ball collection and wall avoidance

FEATURES:
- Autonomous operation with state machine
- White ball detection and movement
- Red wall/boundary detection and avoidance
- Automatic ball collection (servo placeholder)
- State-based behavior: SEARCHING -> MOVING -> COLLECTING
- Emergency wall avoidance override
"""

import cv2
import time
import sys

# Import our modular components
from config import *
from camera_system import CameraManager
from motor_control import MotorController, ServoController
from detection_system import detect_white_balls_fast, detect_red_walls_fast
from autonomous_controller import AutonomousController
from display_system import (
    PerformanceMonitor, draw_autonomous_display, add_performance_stats,
    add_mode_indicator, print_headless_status
)

def autonomous_wall_avoidance_system():
    """Main autonomous system"""
    global AUTO_ENABLED, HEADLESS_MODE
    
    print("=== AUTONOMOUS WALL AVOIDANCE SYSTEM ===")
    print("Robot will run autonomously to find and collect balls")
    
    if not HEADLESS_MODE:
        print("Controls:")
        print("  'q' - Quit")
        print("  'm' - Toggle manual/autonomous mode")
        print("  's' - Show/hide performance stats")
        print("  SPACE - Emergency stop")
    else:
        print("Running in HEADLESS mode - use Ctrl+C to stop")
    
    # Initialize components
    perf_monitor = PerformanceMonitor()
    camera_manager = CameraManager()
    motor_controller = MotorController()
    servo_controller = ServoController()
    autonomous_controller = AutonomousController(motor_controller, servo_controller)
    
    show_stats = ENABLE_PERFORMANCE_STATS and not HEADLESS_MODE
    
    # Initialize camera
    if not camera_manager.initialize_camera():
        print("❌ Camera initialization failed!")
        return
    
    print(f"✓ Using: {camera_manager.camera_type}")
    print(f"✓ Motors: {'Available' if motor_controller.motors_available else 'Disabled'}")
    print(f"✓ Autonomous: {'Enabled' if AUTO_ENABLED else 'Disabled'}")
    print(f"✓ Target FPS: {TARGET_FPS}")
    
    # Initialize motors
    motor_controller.stop_motors()
    
    try:
        frame_count = 0
        start_time = time.time()
        
        print("\n🚀 Starting autonomous operation...")
        
        while True:
            # Capture frame
            ret, frame = camera_manager.capture_frame()
            if not ret or frame is None:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            perf_monitor.update()
            
            # Detect balls and walls
            balls = detect_white_balls_fast(frame)
            walls, wall_debug_mask, danger_detected = detect_red_walls_fast(frame)
            
            # Run autonomous behavior
            if AUTO_ENABLED:
                autonomous_controller.update_autonomous_behavior(balls, danger_detected)
            else:
                # Manual mode - stop motors
                if autonomous_controller.motor_state != MotorState.STOPPED:
                    motor_controller.stop_motors()
                    autonomous_controller.motor_state = MotorState.STOPPED
            
            # Display only if not headless
            if not HEADLESS_MODE:
                # Calculate confirmation progress for display
                confirmation_progress = autonomous_controller.get_confirmation_progress()
                
                # Check if debug mode is active
                if show_debug_masks:
                    # Show debug mask view
                    display_frame = create_combined_debug_view(frame, balls, walls)
                else:
                    # Show normal autonomous display
                    display_frame = draw_autonomous_display(
                        frame, balls, walls, 
                        autonomous_controller.state, 
                        autonomous_controller.motor_state, 
                        danger_detected, 
                        autonomous_controller.target_ball,
                        autonomous_controller.candidate_ball,
                        confirmation_progress
                    )
                
                # Add performance stats if enabled
                if show_stats:
                    display_frame = add_performance_stats(display_frame, perf_monitor)
                
                # Add mode indicator
                display_frame = add_mode_indicator(display_frame, AUTO_ENABLED)
                
                # Display frame
                cv2.imshow('Autonomous GolfBot System', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    AUTO_ENABLED = not AUTO_ENABLED
                    if not AUTO_ENABLED:
                        motor_controller.stop_motors()
                        autonomous_controller.motor_state = MotorState.STOPPED
                    print(f"Mode: {'AUTONOMOUS' if AUTO_ENABLED else 'MANUAL'}")
                elif key == ord('s'):
                    show_stats = not show_stats
                    print(f"Performance stats: {'ON' if show_stats else 'OFF'}")
                elif key == ord('o'):  # NEW DEBUG TOGGLE
                    show_debug_masks = not show_debug_masks
                    print(f"Debug masks: {'ON' if show_debug_masks else 'OFF'}")
                elif key == ord(' '):  # Emergency stop
                    autonomous_controller.emergency_stop()
            else:
                # Headless mode - print periodic status
                print_headless_status(frame_count, TARGET_FPS, autonomous_controller, 
                                    balls, walls, perf_monitor, danger_detected)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        motor_controller.stop_motors()
        servo_controller.deactivate_collection_servo()
        
        camera_manager.release()
        
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        
        # Close motor connections
        motor_controller.cleanup()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"✓ Cleanup complete")
        print(f"📊 Final stats:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Camera type: {camera_manager.camera_type}")
        print(f"   Motors: {'Available' if motor_controller.motors_available else 'Disabled'}")

def configure_system():
    """Handle system configuration before starting"""
    global MOVE_SPEED, SEARCH_ROTATION_SPEED, COLLECTION_TIME
    global BALL_CONFIRMATION_TIME, WALL_DANGER_DISTANCE, AUTO_ENABLED, HEADLESS_MODE
    
    print("=== AUTONOMOUS BEHAVIOR SETTINGS ===")
    print(f"Search rotation speed: {int(SEARCH_ROTATION_SPEED*100)}%")
    print(f"Movement speed: {int(MOVE_SPEED*100)}%")
    print(f"Collection time: {COLLECTION_TIME}s")
    print(f"Search timeout: {SEARCH_TIMEOUT}s")
    print(f"Ball confirmation time: {BALL_CONFIRMATION_TIME}s")
    print(f"Ball reach distance: {BALL_REACHED_DISTANCE} pixels")
    print(f"Wall danger distance: {WALL_DANGER_DISTANCE} pixels")
    print(f"Wall avoidance turn time: {WALL_AVOIDANCE_TURN_TIME}s")
    print()
    
    print("=== CONFIGURATION OPTIONS ===")
    print("  ENTER - Start with default settings")
    print("  'c' - Customize behavior settings")
    print("  'v' - Vision-only mode (no motor control)")
    print("  'h' - Headless mode (no display, for remote operation)")
    
    try:
        config_choice = input("Choose configuration: ").strip().lower()
        
        if config_choice == "c":
            print("\n=== CUSTOM CONFIGURATION ===")
            try:
                move_speed_input = input(f"Movement speed % (current: {int(MOVE_SPEED*100)}): ").strip()
                if move_speed_input:
                    MOVE_SPEED = int(move_speed_input) / 100.0
                    
                search_speed_input = input(f"Search rotation speed % (current: {int(SEARCH_ROTATION_SPEED*100)}): ").strip()
                if search_speed_input:
                    SEARCH_ROTATION_SPEED = int(search_speed_input) / 100.0
                    
                collection_time_input = input(f"Collection time seconds (current: {COLLECTION_TIME}): ").strip()
                if collection_time_input:
                    COLLECTION_TIME = float(collection_time_input)
                    
                confirmation_time_input = input(f"Ball confirmation time seconds (current: {BALL_CONFIRMATION_TIME}): ").strip()
                if confirmation_time_input:
                    BALL_CONFIRMATION_TIME = float(confirmation_time_input)
                    
                danger_distance_input = input(f"Wall danger distance pixels (current: {WALL_DANGER_DISTANCE}): ").strip()
                if danger_distance_input:
                    WALL_DANGER_DISTANCE = int(danger_distance_input)
                    
                print("✓ Custom configuration applied")
            except ValueError:
                print("⚠️  Invalid input, using defaults")
                
        elif config_choice == "v":
            print("✓ Vision-only mode selected")
            # This will be handled by MotorController initialization
            
        elif config_choice == "h":
            print("✓ Headless mode selected - no display will be shown")
            HEADLESS_MODE = True
            
        elif config_choice == "":
            print("✓ Using default configuration")
        else:
            print("✓ Invalid choice, using defaults")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit()

if __name__ == "__main__":
    print("🤖 Autonomous GolfBot Wall Avoidance System")
    print("===========================================")
    print("This system will autonomously:")
    print("  🔍 SEARCH for white balls by rotating")
    print("  🎯 MOVE towards the closest detected ball")
    print("  🔧 COLLECT balls using servo mechanism")
    print("  🚨 AVOID red walls/boundaries automatically")
    print("  🔄 REPEAT the cycle continuously")
    print()
    
    # Configure system
    configure_system()
    
    print(f"\n🚀 Starting autonomous system...")
    if HEADLESS_MODE:
        print("Running in headless mode - use Ctrl+C to stop")
    else:
        print("Press 'q' to quit, 'm' to toggle manual mode, SPACE for emergency stop")
    time.sleep(2)
    
    try:
        autonomous_wall_avoidance_system()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
