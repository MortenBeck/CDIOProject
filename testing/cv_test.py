"""
GolfBot Computer Vision Test - Pi 5 Compatible
Uses libcamera for Pi 5 camera support
"""

import cv2
import numpy as np
import time
import subprocess
import threading
from gpiozero import OutputDevice
import os

# === MOTOR SETUP ===
motor_in1 = OutputDevice(19)  # GPIO 19 - Motor A
motor_in2 = OutputDevice(26)  # GPIO 26 - Motor A
motor_in3 = OutputDevice(20)  # GPIO 20 - Motor B  
motor_in4 = OutputDevice(21)  # GPIO 21 - Motor B

# === MOTOR FUNCTIONS ===
def stop_motors():
    """Stop all motors"""
    motor_in1.off()
    motor_in2.off()
    motor_in3.off()
    motor_in4.off()

def motors_forward():
    """Both motors forward"""
    motor_in1.on()
    motor_in2.off()
    motor_in3.on()
    motor_in4.off()

def motors_reverse():
    """Both motors reverse"""
    motor_in1.off()
    motor_in2.on()
    motor_in3.off()
    motor_in4.on()

# === CAMERA FUNCTIONS FOR PI 5 ===
class Pi5Camera:
    """Camera class that works with Pi 5 libcamera"""
    
    def __init__(self):
        self.process = None
        self.temp_file = "/tmp/golfbot_frame.jpg"
        self.running = False
        
    def start_capture(self):
        """Start continuous capture process"""
        try:
            # Use libcamera to continuously capture frames
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '0',  # Continuous mode
                '--width', '640',
                '--height', '480',
                '--quality', '80',
                '--immediate'
            ]
            
            self.running = True
            print("✓ Pi 5 camera initialized with libcamera")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Pi 5 camera: {e}")
            return False
    
    def capture_frame(self):
        """Capture a single frame"""
        try:
            # Capture single frame
            cmd = [
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '1',
                '--width', '640',
                '--height', '480',
                '--quality', '80'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode == 0 and os.path.exists(self.temp_file):
                # Read the captured image
                frame = cv2.imread(self.temp_file)
                return True, frame
            else:
                return False, None
                
        except Exception as e:
            print(f"Frame capture error: {e}")
            return False, None
    
    def release(self):
        """Clean up camera resources"""
        self.running = False
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

def try_opencv_camera():
    """Try to open camera with OpenCV (fallback method)"""
    print("Trying OpenCV camera access...")
    
    # Try different camera indices and backends
    backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    
    for backend in backends:
        for i in range(3):  # Try camera indices 0, 1, 2
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ OpenCV camera working: index {i}, backend {backend}")
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap
                cap.release()
            except:
                continue
    
    return None

# === COMPUTER VISION FUNCTIONS ===
def detect_white_balls(frame):
    """
    Detect white/light colored balls in the frame
    Returns: list of (x, y, radius) for detected balls
    """
    if frame is None:
        return []
    
    # Convert to HSV color space for better color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for white/light colors
    lower_white = np.array([0, 0, 180])    # Very low saturation, high value
    upper_white = np.array([180, 30, 255]) # Any hue, low saturation, max value
    
    # Create mask for white areas
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Remove noise with morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    for contour in contours:
        # Filter by area (remove very small detections)
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            # Check if contour is roughly circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # If reasonably circular (ball-like)
                if circularity > 0.3:  # Relaxed circularity threshold
                    # Get center and radius
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    # Only consider if radius is reasonable for a ball
                    if 10 < radius < 100:
                        balls.append((center[0], center[1], radius))
    
    return balls

def draw_detections(frame, balls):
    """Draw detected balls on the frame"""
    if frame is None:
        return frame
        
    for x, y, radius in balls:
        # Draw circle around detected ball
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        # Draw center point
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        # Add text label
        cv2.putText(frame, f'Ball', (x-20, y-radius-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main_vision_loop():
    """Main computer vision and motor control loop"""
    print("=== GOLFBOT VISION TEST (Pi 5) ===")
    print("Starting camera and ball detection...")
    print("Press 'q' to quit, 's' to stop motors")
    
    # Try Pi 5 libcamera first
    pi5_camera = Pi5Camera()
    opencv_camera = None
    use_pi5_camera = False
    
    if pi5_camera.start_capture():
        use_pi5_camera = True
        print("✓ Using Pi 5 libcamera")
    else:
        print("⚠️  Pi 5 libcamera failed, trying OpenCV...")
        opencv_camera = try_opencv_camera()
        if opencv_camera is None:
            print("❌ No camera available!")
            return
        print("✓ Using OpenCV camera")
    
    # Motor control state
    motors_running = False
    last_detection_time = 0
    motor_timeout = 2.0  # Stop motors after 2 seconds without detection
    
    try:
        frame_count = 0
        while True:
            frame = None
            
            # Capture frame based on camera type
            if use_pi5_camera:
                ret, frame = pi5_camera.capture_frame()
                if not ret:
                    print("❌ Failed to capture frame with libcamera")
                    break
            else:
                ret, frame = opencv_camera.read()
                if not ret:
                    print("❌ Failed to grab frame with OpenCV")
                    break
            
            frame_count += 1
            
            # Detect white balls
            balls = detect_white_balls(frame)
            
            # Motor control logic
            current_time = time.time()
            
            if len(balls) > 0:
                # Ball detected!
                last_detection_time = current_time
                
                if not motors_running:
                    print(f"🏀 Ball detected! Starting motors... ({len(balls)} ball(s))")
                    motors_forward()
                    motors_running = True
                
                # Draw detections
                frame = draw_detections(frame, balls)
                
                # Add status text
                cv2.putText(frame, f"BALLS DETECTED: {len(balls)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "MOTORS: FORWARD", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            else:
                # No balls detected
                if motors_running and (current_time - last_detection_time) > motor_timeout:
                    print("⏹️  No balls detected for 2s - stopping motors")
                    stop_motors()
                    motors_running = False
                
                # Add status text
                cv2.putText(frame, "NO BALLS DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if motors_running:
                    cv2.putText(frame, "MOTORS: FORWARD (timeout soon)", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "MOTORS: STOPPED", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add camera type and frame info
            camera_type = "libcamera (Pi 5)" if use_pi5_camera else "OpenCV"
            cv2.putText(frame, f"Camera: {camera_type} | Frame: {frame_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to stop motors", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('GolfBot Vision Test - Pi 5', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("🛑 Manual stop requested")
                stop_motors()
                motors_running = False
                last_detection_time = 0
            
            # Small delay for Pi 5 libcamera
            if use_pi5_camera:
                time.sleep(0.1)  # Prevent overloading
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        print("Cleaning up...")
        stop_motors()
        
        if use_pi5_camera:
            pi5_camera.release()
        else:
            opencv_camera.release()
            
        cv2.destroyAllWindows()
        
        # Close GPIO
        motor_in1.close()
        motor_in2.close()
        motor_in3.close()
        motor_in4.close()
        
        print("✓ Cleanup complete")

def test_camera_only():
    """Test camera without motor control"""
    print("=== CAMERA-ONLY TEST (Pi 5) ===")
    print("Testing camera and ball detection without motors")
    print("Press 'q' to quit")
    
    # Try Pi 5 camera first
    pi5_camera = Pi5Camera()
    opencv_camera = None
    use_pi5_camera = False
    
    if pi5_camera.start_capture():
        use_pi5_camera = True
        print("✓ Using Pi 5 libcamera")
    else:
        print("⚠️  Pi 5 libcamera failed, trying OpenCV...")
        opencv_camera = try_opencv_camera()
        if opencv_camera is None:
            print("❌ No camera available!")
            return
        print("✓ Using OpenCV camera")
    
    try:
        frame_count = 0
        while True:
            frame = None
            
            if use_pi5_camera:
                ret, frame = pi5_camera.capture_frame()
                if not ret:
                    print("Frame capture failed")
                    continue
            else:
                ret, frame = opencv_camera.read()
                if not ret:
                    break
            
            frame_count += 1
            
            balls = detect_white_balls(frame)
            frame = draw_detections(frame, balls)
            
            cv2.putText(frame, f"Balls detected: {len(balls)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            camera_type = "libcamera (Pi 5)" if use_pi5_camera else "OpenCV"
            cv2.putText(frame, f"Camera: {camera_type} | Frame: {frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('Camera Test - Pi 5', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if use_pi5_camera:
                time.sleep(0.1)
                
    finally:
        if use_pi5_camera:
            pi5_camera.release()
        else:
            opencv_camera.release()
        cv2.destroyAllWindows()

def test_motors_only():
    """Test motors without camera"""
    print("=== MOTOR-ONLY TEST ===")
    print("Testing motor control")
    
    try:
        print("Motors forward for 2 seconds...")
        motors_forward()
        time.sleep(2)
        
        print("Motors stopped for 1 second...")
        stop_motors()
        time.sleep(1)
        
        print("Motors reverse for 2 seconds...")
        motors_reverse()
        time.sleep(2)
        
        print("Motors stopped")
        stop_motors()
        
        print("✓ Motor test complete")
        
    finally:
        stop_motors()
        motor_in1.close()
        motor_in2.close()
        motor_in3.close()
        motor_in4.close()

if __name__ == "__main__":
    print("GolfBot Vision Test - Pi 5 Compatible")
    print("Choose test mode:")
    print("1. Full vision + motor test")
    print("2. Camera only test")
    print("3. Motor only test")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            main_vision_loop()
        elif choice == "2":
            test_camera_only()
        elif choice == "3":
            test_motors_only()
        else:
            print("Invalid choice. Running camera test...")
            test_camera_only()
            
    except KeyboardInterrupt:
        print("\nExiting...")
        stop_motors()
