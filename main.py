#!/usr/bin/env python3
"""
GolfBot - Entry Point
Autonomous Golf Ball Collection Robot with Two-Phase Collection System
"""

import sys
import logging
from ui.ui_menu import show_startup_menu
from core.core_robot import GolfBot
from hardware import run_hardware_test


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('golfbot.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point"""
    setup_logging()
    mode = show_startup_menu()
    
    if mode == 'exit':
        print("Goodbye!")
        return 0
        
    elif mode == 'testing':
        print("\n🔧 Entering Hardware Testing Mode...")
        try:
            if run_hardware_test():
                print("✅ Testing completed successfully!")
            else:
                print("❌ Testing failed!")
        except Exception as e:
            print(f"Testing error: {e}")
        return 0
        
    elif mode in ['competition_dashboard', 'competition_legacy']:
        use_dashboard = (mode == 'competition_dashboard')
        interface_mode = "Dashboard" if use_dashboard else "Legacy Overlay"
        print(f"\n🏁 Entering Competition Mode with {interface_mode} Interface...")
        
        try:
            robot = GolfBot(use_dashboard=use_dashboard)
            
            if not robot.initialize():
                print("❌ Failed to initialize robot - exiting")
                return 1
            
            robot.show_ready_message()
            input("Press Enter to start competition...")
            robot.start_competition()
            
        except KeyboardInterrupt:
            print("\n⚠️  Competition interrupted by user")
        except Exception as e:
            print(f"❌ Competition error: {e}")
        finally:
            if 'robot' in locals():
                robot.emergency_stop()
        
        return 0


if __name__ == "__main__":
    sys.exit(main())