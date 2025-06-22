#!/usr/bin/env python3
"""
GolfBot Main Entry Point - Simplified and Modular
White ball collection system with enhanced features + Delivery System
"""

import logging
import signal
import sys
import os
from hardware import GolfBotHardware
from vision import VisionSystem
from competition_manager import CompetitionManager
from startup_menu import show_startup_menu, show_competition_info, show_delivery_info
from hardware_test import run_hardware_test
from delivery_system import run_enhanced_delivery_test

class GolfBot:
    """Main GolfBot class - simplified to coordinate components"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize systems
        self.hardware = None
        self.vision = None
        self.competition_manager = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('golfbot.log'),
                logging.StreamHandler()
            ]
        )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.emergency_stop()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Initialize all systems"""
        self.logger.info("Initializing GolfBot with enhanced collection system (WHITE BALLS ONLY)...")
        
        try:
            # Initialize hardware
            self.hardware = GolfBotHardware()
            
            # Initialize vision system
            self.vision = VisionSystem()
            if not self.vision.start():
                self.logger.error("Failed to initialize vision system")
                return False
            
            # Let vision system detect arena boundaries on startup
            self.logger.info("Detecting arena boundaries...")
            ret, frame = self.vision.get_frame()
            if ret:
                self.vision.boundary_system.detect_arena_boundaries(frame)
                if self.vision.boundary_system.arena_detected:
                    self.logger.info("✅ Arena boundaries detected successfully")
                else:
                    self.logger.info("⚠️  Using fallback arena boundaries")
            
            self.logger.info("All systems initialized successfully - WHITE BALLS ONLY")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def start_competition(self, use_dashboard=True):
        """Start competition with specified interface mode"""
        try:
            # Initialize competition manager
            self.competition_manager = CompetitionManager(
                self.hardware, self.vision, use_dashboard=use_dashboard
            )
            
            # Start the competition
            self.competition_manager.start_competition()
            
        except KeyboardInterrupt:
            self.logger.info("Competition interrupted by user")
        except Exception as e:
            self.logger.error(f"Competition error: {e}")
        finally:
            self.emergency_stop()
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        try:
            if self.competition_manager:
                self.competition_manager.emergency_stop()
            elif self.hardware:
                self.hardware.emergency_stop()
            
            if self.vision:
                self.vision.cleanup()
            if self.hardware:
                self.hardware.cleanup()
                
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")

def main():
    """Main entry point"""
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
        
    elif mode == 'delivery':
        print("\n🚚 Entering Delivery System Test Mode...")
        
        # Show delivery info
        show_delivery_info()
        
        try:
            if run_enhanced_delivery_test():
                print("✅ Delivery test completed successfully!")
            else:
                print("❌ Delivery test failed!")
        except Exception as e:
            print(f"Delivery test error: {e}")
        return 0
        
    elif mode in ['competition_dashboard', 'competition_legacy']:
        use_dashboard = (mode == 'competition_dashboard')
        
        # Show competition info
        show_competition_info(use_dashboard)
        
        try:
            robot = GolfBot()
            
            if not robot.initialize():
                print("❌ Failed to initialize robot - exiting")
                return 1
            
            robot.start_competition(use_dashboard=use_dashboard)
            
        except KeyboardInterrupt:
            print("\n⚠️  Competition interrupted by user")
        except Exception as e:
            print(f"❌ Competition error: {e}")
        
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)