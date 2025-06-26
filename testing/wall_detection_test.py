"""
GolfBot Ball and Wall Detection - OPTIMIZED LOW LATENCY VERSION
Pi 5 Compatible with white ball and red wall detection

FEATURES:
- White ball detection (from original code)
- Red wall/boundary detection for navigation
- Same optimized camera system with picamera2
- Real-time performance monitoring
- Multiple detection visualization modes

EXPECTED: 15 FPS with dual detection for stable performance
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
TARGET_FPS = 15
DISPLAY_FRAME_SKIP = 1   # Display every Nth frame (1 = all frames)

# Detection settings
MIN_BALL_AREA = 150      # White ball minimum area
MIN_WALL_AREA = 100      # Red wall minimum area
BALL_CIRCULARITY_THRESHOLD = 0.25  # Ball circularity
WALL_MIN_LENGTH = 50     # Minimum wall segment length
ENABLE_PERFORMANCE_STATS = True

# Visualization modes
class VisualizationMode:
    BOTH = 0          # Show both balls and walls
    BALLS_ONLY = 1    # Show only balls
    WALLS_ONLY = 2    # Show only walls
    DEBUG = 3         # Show debug masks

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

# === OPTIMIZED CAMERA CLASS ===
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
            
            print(f"✓ picamera2 initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {TARGET_FPS}FPS")
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

# === FALLBACK CAMERA CLASS ===
class FallbackLibCamera:
    """Fallback libcamera using optimized still capture"""
    
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
        """Optimized single frame capture"""
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

# === DETECTION FUNCTIONS ===
def detect_white_balls_fast(frame):
    """
    Optimized white ball detection
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
                
                if circularity > BALL_CIRCULARITY_THRESHOLD:
                    # Get center and radius
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Scale back to original resolution
                    center_x = int(x * scale_x)
                    center_y = int(y * scale_y)
                    radius_scaled = int(radius * max(scale_x, scale_y))
                    
                    if 5 < radius_scaled < 150:
                        balls.append((center_x, center_y, radius_scaled))
    
    return balls

def detect_red_walls_fast(frame):
    """
    Optimized red wall/boundary detection
    """
    if frame is None:
        return [], None
    
    # Resize for processing
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    
    # Convert to HSV
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # Red color detection (two ranges for red hue wrap-around)
    # Lower red range (0-10)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    
    # Upper red range (170-180)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    walls = []
    scale_x = CAMERA_WIDTH / PROCESS_WIDTH
    scale_y = CAMERA_HEIGHT / PROCESS_HEIGHT
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_WALL_AREA:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Scale back to original resolution
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            
            # Check if it's long enough to be a wall segment
            length = max(w_scaled, h_scaled)
            if length > WALL_MIN_LENGTH:
                walls.append((x_scaled, y_scaled, w_scaled, h_scaled))
    
    # Return both walls and debug mask (scaled up for visualization)
    debug_mask = cv2.resize(red_mask, (CAMERA_WIDTH, CAMERA_HEIGHT))
    return walls, debug_mask

def draw_detections(frame, balls, walls, mode=VisualizationMode.BOTH):
    """Draw detections based on visualization mode"""
    if frame is None:
        return frame
    
    display_frame = frame.copy()
    
    # Draw balls
    if mode in [VisualizationMode.BOTH, VisualizationMode.BALLS_ONLY]:
        for x, y, radius in balls:
            cv2.circle(display_frame, (x, y), radius, (0, 255, 0), 2)  # Green circle
            cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)      # Green center dot
            # Add label
            cv2.putText(display_frame, "BALL", (x-20, y-radius-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw walls
    if mode in [VisualizationMode.BOTH, VisualizationMode.WALLS_ONLY]:
        for x, y, w, h in walls:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle
            # Add label  
            cv2.putText(display_frame, "WALL", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return display_frame

def draw_debug_masks(frame, ball_mask, wall_mask):
    """Draw debug masks side by side"""
    if frame is None:
        return frame
    
    # Create 3-channel versions of masks
    ball_mask_color = cv2.cvtColor(ball_mask, cv2.COLOR_GRAY2BGR) if ball_mask is not None else np.zeros_like(frame)
    wall_mask_color = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR) if wall_mask is not None else np.zeros_like(frame)
    
    # Resize to half width for side-by-side display
    h, w = frame.shape[:2]
    half_w = w // 2
    
    frame_small = cv2.resize(frame, (half_w, h))
    ball_small = cv2.resize(ball_mask_color, (half_w, h))
    wall_small = cv2.resize(wall_mask_color, (half_w, h))
    
    # Create combined view: original | ball mask | wall mask
    # Top half: original and ball mask
    top = np.hstack([frame_small, ball_small])
    # Bottom half: wall mask and combined
    combined = cv2.bitwise_or(ball_small, wall_small)
    bottom = np.hstack([wall_small, combined])
    
    # Stack vertically
    debug_frame = np.vstack([top, bottom])
    
    # Add labels
    cv2.putText(debug_frame, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_frame, "BALL MASK", (half_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_frame, "WALL MASK", (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_frame, "COMBINED", (half_w + 10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return debug_frame

# === MAIN DETECTION FUNCTION ===
def ball_and_wall_detection():
    """Main ball and wall detection function"""
    print("=== BALL AND WALL DETECTION ===")
    print("Detecting white balls and red walls with optimized performance")
    print("Controls:")
    print("  'q' - Quit")
    print("  'v' - Cycle visualization modes")
    print("  's' - Show/hide performance stats")
    print("  'r' - Reset performance stats")
    print("  'd' - Toggle debug mode")
    
    # Performance monitoring
    perf_monitor = PerformanceMonitor()
    show_stats = ENABLE_PERFORMANCE_STATS
    vis_mode = VisualizationMode.BOTH
    debug_mode = False
    
    # Initialize camera
    camera = None
    camera_type = "Unknown"
    
    print("\n🔍 Detecting camera...")
    
    # Try picamera2 first
    if PICAMERA2_AVAILABLE:
        print("Trying picamera2...")
        camera = FastPiCamera2()
        if camera.start_capture():
            camera_type = "picamera2 (Ultra-Fast)"
        else:
            camera = None
    
    # Fallback to libcamera
    if camera is None:
        print("Trying libcamera-still...")
        camera = FallbackLibCamera()
        if camera.start_capture():
            camera_type = "libcamera-still (Fallback)"
        else:
            camera = None
    
    if camera is None:
        print("❌ No camera available!")
        return
    
    print(f"✓ Using: {camera_type}")
    print(f"✓ Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"✓ Processing: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    
    # Visualization mode names
    mode_names = ["BOTH", "BALLS ONLY", "WALLS ONLY", "DEBUG"]
    
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
            
            # Detect balls and walls
            balls = detect_white_balls_fast(frame)
            walls, wall_debug_mask = detect_red_walls_fast(frame)
            
            # Choose display based on mode
            if debug_mode:
                # Create a simple ball mask for debug view
                ball_debug_mask = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint8)
                for x, y, radius in balls:
                    cv2.circle(ball_debug_mask, (x, y), radius, 255, -1)
                
                display_frame = draw_debug_masks(frame, ball_debug_mask, wall_debug_mask)
            else:
                display_frame = draw_detections(frame, balls, walls, vis_mode)
            
            # Add status overlay
            ball_color = (0, 255, 0) if len(balls) > 0 else (100, 100, 100)
            wall_color = (0, 0, 255) if len(walls) > 0 else (100, 100, 100)
            
            cv2.putText(display_frame, f"Balls: {len(balls)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, ball_color, 2)
            cv2.putText(display_frame, f"Walls: {len(walls)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, wall_color, 2)
            
            # Visualization mode
            if not debug_mode:
                cv2.putText(display_frame, f"Mode: {mode_names[vis_mode]}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(display_frame, "DEBUG MODE", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Performance stats
            if show_stats and not debug_mode:
                fps_text = f"FPS: {perf_monitor.fps:.1f}"
                latency_text = f"Latency: {perf_monitor.avg_latency:.1f}ms"
                cv2.putText(display_frame, fps_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, latency_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Camera info
            if not debug_mode:
                cv2.putText(display_frame, f"Camera: {camera_type}", (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            window_name = 'Ball and Wall Detection - Low Latency'
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v') and not debug_mode:
                vis_mode = (vis_mode + 1) % 3  # Cycle through first 3 modes
                print(f"Visualization mode: {mode_names[vis_mode]}")
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"Performance stats: {'ON' if show_stats else 'OFF'}")
            elif key == ord('r'):
                perf_monitor = PerformanceMonitor()
                print("Performance stats reset")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            
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

if __name__ == "__main__":
    print("GolfBot Ball and Wall Detection - Low Latency Version")
    print("This will detect:")
    print("  🟢 White balls (like golf balls)")
    print("  🔴 Red walls/boundaries (like the tape in your image)")
    print()
    
    try:
        ball_and_wall_detection()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
