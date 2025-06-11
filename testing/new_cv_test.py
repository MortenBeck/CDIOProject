#!/usr/bin/env python3
"""
GolfBot Camera Test - OPTIMIZED LOW LATENCY VERSION (Camera Only)
Pi 5 Compatible with significant lag reduction optimizations

PERFORMANCE IMPROVEMENTS:
- picamera2 for direct memory access (30-50ms latency vs 200ms+)
- Removed artificial delays
- Smart resolution scaling
- Optimized computer vision
- Frame skipping options
- Multiple fallback camera methods

EXPECTED: 20-30 FPS vs original 3-5 FPS
"""

import cv2
import numpy as np
import time
import subprocess
import os
import sys

# Try to import picamera2 (best performance)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    print("✓ picamera2 available - will use for best performance")
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  picamera2 not available - using fallback methods")

# === PERFORMANCE SETTINGS ===
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_WIDTH = 320      # Process at half resolution for speed
PROCESS_HEIGHT = 240
TARGET_FPS = 30
DISPLAY_FRAME_SKIP = 1   # Display every Nth frame (1 = all frames)

# Computer vision optimization settings
MIN_BALL_AREA = 150      # Reduced for faster processing
CIRCULARITY_THRESHOLD = 0.25  # Relaxed for speed
ENABLE_PERFORMANCE_STATS = True

# === PERFORMANCE MONITORING ===
class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.last_time = time.time()
        self.fps = 0
        self.avg_latency = 0
        
    def update(self):
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        
        # Keep only last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
        if len(self.frame_times) > 1:
            self.fps = 1.0 / np.mean(self.frame_times)
            self.avg_latency = np.mean(self.frame_times) * 1000  # ms
            
        self.last_time = current_time

# === OPTIMIZED CAMERA CLASSES ===
class FastPiCamera2:
    """Ultra-fast camera using picamera2 - BEST PERFORMANCE"""
    
    def __init__(self):
        self.picam2 = None
        self.running = False
        
    def start_capture(self):
        try:
            self.picam2 = Picamera2()
            
            # Use BGR888 format (what the camera actually outputs)
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT), 
                    "format": "BGR888"  # Camera natively outputs BGR
                },
                controls={
                    "FrameRate": TARGET_FPS,
                    "ExposureTime": 20000,  # Slightly longer exposure
                    "AnalogueGain": 1.0,
                    "AwbEnable": True,      # Auto white balance
                    "AwbMode": 0,           # Auto white balance mode
                    "AeEnable": True,       # Auto exposure
                    "Brightness": 0.0,      # Normal brightness
                    "Contrast": 1.0         # Normal contrast
                }
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            self.running = True
            
            # Longer warm-up for color adjustment
            time.sleep(2.0)  
            
            print(f"✓ picamera2 initialized with correct colors: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {TARGET_FPS}FPS")
            return True
            
        except Exception as e:
            print(f"❌ picamera2 failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture frame with minimal latency and proper color correction"""
        if not self.running or not self.picam2:
            return False, None
            
        try:
            # Direct array capture
            frame = self.picam2.capture_array()
            
            # Convert BGR to RGB (Mode 3 - the one that worked!)
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

class OptimizedLibCamera:
    """Optimized libcamera using video streaming - GOOD PERFORMANCE"""
    
    def __init__(self):
        self.process = None
        self.running = False
        self.frame_buffer = None
        
    def start_capture(self):
        try:
            # Use libcamera-vid for streaming (much faster than libcamera-still)
            cmd = [
                'libcamera-vid',
                '--output', '-',  # Output to stdout
                '--timeout', '0',  # Continuous
                '--width', str(CAMERA_WIDTH),
                '--height', str(CAMERA_HEIGHT),
                '--framerate', str(TARGET_FPS),
                '--codec', 'mjpeg',
                '--quality', '80',
                '--nopreview'
            ]
            
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
            
            self.running = True
            print(f"✓ libcamera-vid streaming started: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            return True
            
        except Exception as e:
            print(f"❌ libcamera-vid failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture frame from video stream"""
        if not self.running or not self.process:
            return False, None
            
        try:
            # Read JPEG frame from stream
            # This is a simplified version - real implementation would need proper MJPEG parsing
            return False, None  # Placeholder - complex to implement properly
            
        except Exception as e:
            return False, None
    
    def release(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()

class FallbackLibCamera:
    """Fallback libcamera using optimized still capture - MODERATE PERFORMANCE"""
    
    def __init__(self):
        self.temp_file = "/tmp/golfbot_frame_optimized.jpg"
        self.running = False
        
    def start_capture(self):
        try:
            # Test capture
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '1',
                '--width', str(CAMERA_WIDTH),
                '--height', str(CAMERA_HEIGHT),
                '--quality', '70',  # Reduced for speed
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
        """Optimized single frame capture"""
        if not self.running:
            return False, None
            
        try:
            # Faster capture with reduced quality
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '1',
                '--width', str(CAMERA_WIDTH),
                '--height', str(CAMERA_HEIGHT),
                '--quality', '60',  # Lower quality for speed
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

class FastOpenCVCamera:
    """Optimized OpenCV camera - FALLBACK PERFORMANCE"""
    
    def __init__(self):
        self.cap = None
        self.running = False
        
    def start_capture(self):
        try:
            # Try different backends and indices
            backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                for i in range(3):
                    try:
                        cap = cv2.VideoCapture(i, backend)
                        if cap.isOpened():
                            # Test frame capture
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                # Optimize settings
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
                                
                                self.cap = cap
                                self.running = True
                                print(f"✓ OpenCV camera: index {i}, backend {backend}")
                                return True
                        cap.release()
                    except:
                        continue
            
            return False
            
        except Exception as e:
            print(f"❌ OpenCV camera failed: {e}")
            return False
    
    def capture_frame(self):
        if not self.running or not self.cap:
            return False, None
            
        try:
            ret, frame = self.cap.read()
            return ret, frame
        except Exception as e:
            return False, None
    
    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()

# === OPTIMIZED COMPUTER VISION ===
def detect_white_balls_fast(frame):
    """
    Optimized white ball detection with speed improvements
    """
    if frame is None:
        return []
    
    # Resize for processing (major speed improvement)
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    
    # Convert to HSV
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # Optimized white detection range
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 25, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Simplified morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    scale_x = CAMERA_WIDTH / PROCESS_WIDTH
    scale_y = CAMERA_HEIGHT / PROCESS_HEIGHT
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_BALL_AREA:
            # Quick circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > CIRCULARITY_THRESHOLD:
                    # Get center and radius
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Scale back to original resolution
                    center_x = int(x * scale_x)
                    center_y = int(y * scale_y)
                    radius_scaled = int(radius * max(scale_x, scale_y))
                    
                    if 5 < radius_scaled < 150:
                        balls.append((center_x, center_y, radius_scaled))
    
    return balls

def draw_detections_fast(frame, balls):
    """Optimized detection drawing"""
    if frame is None:
        return frame
        
    for x, y, radius in balls:
        # Simple circle drawing
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    return frame

# === MAIN CAMERA TEST ===
def optimized_camera_test():
    """Main optimized camera test function"""
    print("=== OPTIMIZED CAMERA TEST (Low Latency) ===")
    print("Testing camera with maximum performance optimizations")
    print("Controls:")
    print("  'q' - Quit")
    print("  'f' - Toggle frame skipping")
    print("  's' - Show/hide performance stats")
    print("  'r' - Reset performance stats")
    
    # Performance monitoring
    perf_monitor = PerformanceMonitor()
    show_stats = ENABLE_PERFORMANCE_STATS
    frame_skip_counter = 0
    current_skip = DISPLAY_FRAME_SKIP
    
    # Try cameras in order of performance
    camera = None
    camera_type = "Unknown"
    
    print("\n🔍 Detecting best camera method...")
    
    # 1. Try picamera2 (fastest)
    if PICAMERA2_AVAILABLE:
        print("Trying picamera2...")
        camera = FastPiCamera2()
        if camera.start_capture():
            camera_type = "picamera2 (Ultra-Fast)"
        else:
            camera = None
    
    # 2. Try optimized libcamera fallback
    if camera is None:
        print("Trying optimized libcamera-still...")
        camera = FallbackLibCamera()
        if camera.start_capture():
            camera_type = "libcamera-still (Optimized)"
        else:
            camera = None
    
    # 3. Try OpenCV fallback
    if camera is None:
        print("Trying OpenCV...")
        camera = FastOpenCVCamera()
        if camera.start_capture():
            camera_type = "OpenCV (Fallback)"
        else:
            camera = None
    
    if camera is None:
        print("❌ No camera available!")
        return
    
    print(f"✓ Using: {camera_type}")
    print(f"✓ Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"✓ Processing: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    print(f"✓ Target FPS: {TARGET_FPS}")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Capture frame
            ret, frame = camera.capture_frame()
            if not ret or frame is None:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            perf_monitor.update()
            
            # Detect balls
            balls = detect_white_balls_fast(frame)
            
            # Draw detections
            frame = draw_detections_fast(frame, balls)
            
            # Add status overlay
            status_color = (0, 255, 0) if len(balls) > 0 else (0, 0, 255)
            cv2.putText(frame, f"Balls: {len(balls)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Performance stats
            if show_stats:
                fps_text = f"FPS: {perf_monitor.fps:.1f}"
                latency_text = f"Latency: {perf_monitor.avg_latency:.1f}ms"
                cv2.putText(frame, fps_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, latency_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Camera info
            cv2.putText(frame, f"Camera: {camera_type}", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Frame skip info
            if current_skip > 1:
                cv2.putText(frame, f"Skip: 1/{current_skip}", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display frame (with optional skipping)
            if frame_skip_counter % current_skip == 0:
                cv2.imshow('Optimized Camera Test - Low Latency', frame)
            frame_skip_counter += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                current_skip = 3 if current_skip == 1 else 1
                print(f"Frame skip: 1/{current_skip}")
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"Performance stats: {'ON' if show_stats else 'OFF'}")
            elif key == ord('r'):
                perf_monitor = PerformanceMonitor()
                print("Performance stats reset")
            
            # No artificial delays!
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"✓ Cleanup complete")
        print(f"📊 Final stats:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Camera type: {camera_type}")

def benchmark_camera():
    """Quick camera benchmark"""
    print("=== CAMERA BENCHMARK ===")
    print("Testing all available camera methods...")
    
    methods = []
    
    # Test picamera2
    if PICAMERA2_AVAILABLE:
        print("\nTesting picamera2...")
        camera = FastPiCamera2()
        if camera.start_capture():
            start_time = time.time()
            frames_captured = 0
            
            for _ in range(30):  # Capture 30 frames
                ret, frame = camera.capture_frame()
                if ret:
                    frames_captured += 1
            
            elapsed = time.time() - start_time
            fps = frames_captured / elapsed if elapsed > 0 else 0
            methods.append(("picamera2", fps, elapsed / frames_captured * 1000))
            camera.release()
        
    # Test libcamera fallback
    print("\nTesting libcamera-still...")
    camera = FallbackLibCamera()
    if camera.start_capture():
        start_time = time.time()
        frames_captured = 0
        
        for _ in range(10):  # Fewer frames due to slower method
            ret, frame = camera.capture_frame()
            if ret:
                frames_captured += 1
        
        elapsed = time.time() - start_time
        fps = frames_captured / elapsed if elapsed > 0 else 0
        methods.append(("libcamera-still", fps, elapsed / frames_captured * 1000))
        camera.release()
    
    # Test OpenCV
    print("\nTesting OpenCV...")
    camera = FastOpenCVCamera()
    if camera.start_capture():
        start_time = time.time()
        frames_captured = 0
        
        for _ in range(30):
            ret, frame = camera.capture_frame()
            if ret:
                frames_captured += 1
        
        elapsed = time.time() - start_time
        fps = frames_captured / elapsed if elapsed > 0 else 0
        methods.append(("OpenCV", fps, elapsed / frames_captured * 1000))
        camera.release()
    
    # Results
    print("\n📊 BENCHMARK RESULTS:")
    print("Method               | FPS    | Latency")
    print("--------------------|--------|----------")
    for method, fps, latency in sorted(methods, key=lambda x: x[1], reverse=True):
        print(f"{method:<20}| {fps:6.1f} | {latency:6.1f}ms")

def test_color_modes():
    """Test different color modes to fix blue tint"""
    print("=== COLOR MODE TEST ===")
    print("Testing different color conversions...")
    print("Controls:")
    print("  '1' - Original RGB->BGR")
    print("  '2' - No conversion")  
    print("  '3' - BGR->RGB")
    print("  '4' - RGB->GRAY->BGR")
    print("  'q' - Quit")
    
    if not PICAMERA2_AVAILABLE:
        print("❌ picamera2 not available")
        return
        
    camera = FastPiCamera2()
    if not camera.start_capture():
        print("❌ Camera failed to start")
        return
        
    conversion_mode = 1
    
    try:
        while True:
            ret, frame = camera.capture_frame()
            if not ret:
                break
                
            # Apply different color conversions based on mode
            display_frame = frame.copy()
            
            if conversion_mode == 1:
                # Already converted in capture_frame (RGB->BGR)
                mode_text = "RGB->BGR (should be normal)"
            elif conversion_mode == 2:
                # No additional conversion
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                mode_text = "No conversion"
            elif conversion_mode == 3:
                # BGR to RGB 
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mode_text = "BGR->RGB (may fix blue tint)"
            elif conversion_mode == 4:
                # Grayscale test
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                mode_text = "Grayscale test"
            
            # Add mode info
            cv2.putText(display_frame, f"Mode {conversion_mode}: {mode_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 1-4 to change modes, q to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Color Mode Test', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                conversion_mode = 1
                print("Mode 1: RGB->BGR")
            elif key == ord('2'):
                conversion_mode = 2  
                print("Mode 2: No conversion")
            elif key == ord('3'):
                conversion_mode = 3
                print("Mode 3: BGR->RGB")
            elif key == ord('4'):
                conversion_mode = 4
                print("Mode 4: Grayscale")
                
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("GolfBot Optimized Camera Test - Low Latency Version")
    print("Choose test mode:")
    print("1. Optimized camera test (recommended)")
    print("2. Camera benchmark comparison") 
    print("3. Quick test (30 seconds)")
    print("4. Color mode test (fix blue tint)")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            optimized_camera_test()
        elif choice == "2":
            benchmark_camera()
        elif choice == "3":
            print("Running 30-second quick test...")
            # Run main test but auto-quit after 30 seconds
            import threading
            def auto_quit():
                time.sleep(30)
                print("\n⏰ 30-second test completed")
                os._exit(0)
            
            timer = threading.Thread(target=auto_quit)
            timer.daemon = True
            timer.start()
            optimized_camera_test()
        elif choice == "4":
            test_color_modes()
        else:
            print("Invalid choice. Running optimized test...")
            optimized_camera_test()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
