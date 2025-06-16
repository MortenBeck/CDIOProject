print("DEBUG: Script starting...")
print("DEBUG: About to import time...")
import time
print("DEBUG: About to import logging...")
import logging
print("DEBUG: About to import cv2...")
import cv2
print("DEBUG: About to import signal...")
import signal
print("DEBUG: About to import sys...")
import sys
print("DEBUG: About to import os...")
import os
print("DEBUG: About to import Enum...")
from enum import Enum
print("DEBUG: About to import typing...")
from typing import Optional
print("DEBUG: About to import config...")
import config
print("DEBUG: About to import hardware...")
from hardware import GolfBotHardware
print("DEBUG: About to import vision...")
from vision import VisionSystem
print("DEBUG: About to import telemetry...")
from telemetry import TelemetryLogger
print("DEBUG: About to import hardware_test...")
from hardware_test import run_hardware_test
print("DEBUG: All imports completed successfully!")

class RobotState(Enum):
    SEARCHING = "searching"
    APPROACHING_BALL = "approaching_ball"
    COLLECTING_BALL = "collecting_ball"
    AVOIDING_BOUNDARY = "avoiding_boundary"
    EMERGENCY_STOP = "emergency_stop"

class GolfBot:
    def __init__(self):
        print("DEBUG: Entering GolfBot.__init__")
        print("DEBUG: About to call setup_logging...")
        self.setup_logging()
        print("DEBUG: setup_logging completed")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("DEBUG: Logger created")
        
        # Check if display is available
        print("DEBUG: About to check display...")
        self.display_available = self.check_display_available()
        print(f"DEBUG: Display available: {self.display_available}")
        if not self.display_available:
            self.logger.info("No display detected - running in headless mode")
        
        # Initialize telemetry
        print("DEBUG: About to initialize telemetry...")
        self.telemetry = TelemetryLogger()
        print("DEBUG: Telemetry initialized")
        
        # Initialize systems - THIS IS LIKELY WHERE IT HANGS
        print("DEBUG: About to initialize hardware...")
        try:
            self.hardware = GolfBotHardware()
            print("DEBUG: Hardware initialized successfully")
        except Exception as e:
            print(f"DEBUG: Hardware initialization FAILED: {e}")
            raise
        
        print("DEBUG: About to initialize vision...")
        try:
            self.vision = VisionSystem()
            print("DEBUG: Vision initialized successfully")
        except Exception as e:
            print(f"DEBUG: Vision initialization FAILED: {e}")
            raise
        
        # Competition state
        print("DEBUG: Setting up competition state...")
        self.start_time = None
        self.competition_active = False
        self.state = RobotState.SEARCHING
        self.search_pattern_index = 0
        self.last_ball_seen_time = None
        
        # Performance tracking
        self.last_frame_time = time.time()
        
        # Setup signal handlers
        print("DEBUG: Setting up signal handlers...")
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        print("DEBUG: GolfBot.__init__ completed successfully!")
    
    def check_display_available(self):
        """Check if display/X11 is available"""
        print("DEBUG: Checking display availability...")
        try:
            # Check for DISPLAY environment variable
            if os.environ.get('DISPLAY') is None:
                print("DEBUG: No DISPLAY environment variable")
                return False
            
            # Try to initialize a test window
            test_img = cv2.imread('/dev/null')  # This won't work but won't crash
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.destroyWindow('test')
            print("DEBUG: Display test successful")
            return True
        except Exception as e:
            print(f"DEBUG: Display test failed: {e}")