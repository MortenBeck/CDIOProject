#!/usr/bin/env python3
"""
Camera System for GolfBot
Handles different camera backends (picamera2, libcamera-still)
"""

import cv2
import time
import subprocess
import os
from config import CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS, LIBCAMERA_TEMP_FILE

# Try to import picamera2 (best performance)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    print("✓ picamera2 available - will use for best performance")
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  picamera2 not available - using fallback methods")

class FastPiCamera2:
    """Ultra-fast camera using picamera2 - BEST PERFORMANCE"""
    
    def __init__(self):
        self.picam2 = None
        self.running = False
        
    def start_capture(self):
        try:
            self.picam2 = Picamera2()
            
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT), 
                    "format": "BGR888"
                },
                controls={
                    "FrameRate": TARGET_FPS,
                    "ExposureTime": 20000,
                    "AnalogueGain": 1.0,
                    "AwbEnable": True,
                    "AwbMode": 0,
                    "AeEnable": True,
                    "Brightness": 0.0,
                    "Contrast": 1.0
                }
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            self.running = True
            time.sleep(2.0)
            
            print(f"✓ picamera2 initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {TARGET_FPS}FPS")
            return True
            
        except Exception as e:
            print(f"❌ picamera2 failed: {e}")
            return False
    
    def capture_frame(self):
        if not self.running or not self.picam2:
            return False, None
            
        try:
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame
        except Exception as e:
            print(f"Frame capture error: {e}")
            return False, None
    
    def release(self):
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        self.running = False

class FallbackLibCamera:
    """Fallback libcamera using optimized still capture"""
    
    def __init__(self):
        self.temp_file = LIBCAMERA_TEMP_FILE
        self.running = False
        
    def start_capture(self):
        try:
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '1',
                '--width', str(CAMERA_WIDTH),
                '--height', str(CAMERA_HEIGHT),
                '--quality', '70',
                '--immediate',
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=3)
            
            if result.returncode == 0 and os.path.exists(self.temp_file):
                self.running = True
                print(f"✓ libcamera-still optimized mode: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
                os.remove(self.temp_file)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ libcamera-still failed: {e}")
            return False
    
    def capture_frame(self):
        if not self.running:
            return False, None
            
        try:
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '1',
                '--width', str(CAMERA_WIDTH),
                '--height', str(CAMERA_HEIGHT),
                '--quality', '60',
                '--immediate',
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode == 0 and os.path.exists(self.temp_file):
                frame = cv2.imread(self.temp_file)
                return True, frame
            else:
                return False, None
                
        except Exception as e:
            return False, None
    
    def release(self):
        self.running = False
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

class CameraManager:
    """Manages camera initialization and selection"""
    
    def __init__(self):
        self.camera = None
        self.camera_type = "Unknown"
    
    def initialize_camera(self):
        """Initialize the best available camera"""
        print("\n🔍 Detecting camera...")
        
        if PICAMERA2_AVAILABLE:
            print("Trying picamera2...")
            self.camera = FastPiCamera2()
            if self.camera.start_capture():
                self.camera_type = "picamera2 (Ultra-Fast)"
                return True
            else:
                self.camera = None
        
        if self.camera is None:
            print("Trying libcamera-still...")
            self.camera = FallbackLibCamera()
            if self.camera.start_capture():
                self.camera_type = "libcamera-still (Fallback)"
                return True
            else:
                self.camera = None
        
        if self.camera is None:
            print("❌ No camera available!")
            return False
        
        print(f"✓ Using: {self.camera_type}")
        return True
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        if self.camera is None:
            return False, None
        return self.camera.capture_frame()
    
    def release(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
