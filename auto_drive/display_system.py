#!/usr/bin/env python3
"""
Enhanced Display System with Live Mask Debugging
Press 'o' to toggle between normal view and detection masks
"""

import cv2
import time
import numpy as np
from robot_states import RobotState, MotorState
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, PROCESS_WIDTH, PROCESS_HEIGHT,
    WALL_DANGER_DISTANCE, BALL_CONFIRMATION_TIME, TARGET_FPS, 
    ENABLE_PERFORMANCE_STATS
)

class PerformanceMonitor:
    """Monitors system performance and FPS"""
    
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

def create_white_ball_mask(frame):
    """Create a more selective white ball detection mask for debugging"""
    if frame is None:
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint8)
    
    # Resize for processing
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    
    # Method 1: More restrictive HSV - target bright, low-saturation objects
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    lower_white_hsv = np.array([0, 0, 140])      # Higher brightness threshold
    upper_white_hsv = np.array([180, 40, 255])   # Lower saturation threshold
    mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
    
    # Method 2: More restrictive BGR - all channels must be high AND similar
    b, g, r = cv2.split(small_frame)
    
    # All channels must be above a high threshold
    high_threshold = 120
    mask_b = (b > high_threshold).astype(np.uint8)
    mask_g = (g > high_threshold).astype(np.uint8)
    mask_r = (r > high_threshold).astype(np.uint8)
    
    # All channels must be similar (white objects have balanced RGB)
    max_diff = 30  # Maximum difference between channels
    diff_bg = np.abs(b.astype(np.int16) - g.astype(np.int16))
    diff_br = np.abs(b.astype(np.int16) - r.astype(np.int16))
    diff_gr = np.abs(g.astype(np.int16) - r.astype(np.int16))
    
    mask_similar = ((diff_bg < max_diff) & (diff_br < max_diff) & (diff_gr < max_diff)).astype(np.uint8)
    
    # Combine: high values AND similar values
    mask_bgr = cv2.bitwise_and(
        cv2.bitwise_and(cv2.bitwise_and(mask_b, mask_g), mask_r),
        mask_similar
    ) * 255
    
    # Method 3: More restrictive grayscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    _, mask_gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Higher threshold
    
    # Combine methods - require at least 2 out of 3 methods to agree
    vote_count = (mask_hsv > 0).astype(np.uint8) + (mask_bgr > 0).astype(np.uint8) + (mask_gray > 0).astype(np.uint8)
    combined_mask = (vote_count >= 2).astype(np.uint8) * 255
    
    # More aggressive morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Scale back up to original size
    mask_full_size = cv2.resize(combined_mask, (CAMERA_WIDTH, CAMERA_HEIGHT))
    
    return mask_full_size

def create_red_wall_mask(frame):
    """Create red wall detection mask for debugging"""
    if frame is None:
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint8)
    
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # Red detection
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Scale back up
    mask_full_size = cv2.resize(red_mask, (CAMERA_WIDTH, CAMERA_HEIGHT))
    
    return mask_full_size

def create_combined_debug_view(frame, balls, walls, hsv_threshold=140, bgr_threshold=120, show_individual_masks=False):
    """Create a combined debug view showing detection masks and results"""
    if frame is None:
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    
    # Get masks
    white_mask = create_white_ball_mask(frame)
    red_mask = create_red_wall_mask(frame)
    
    if show_individual_masks:
        # Show individual method masks for debugging
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        # HSV mask
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 0, hsv_threshold]), np.array([180, 40, 255]))
        mask_hsv_full = cv2.resize(mask_hsv, (CAMERA_WIDTH, CAMERA_HEIGHT))
        
        # Create RGB view: R=HSV, G=BGR, B=Combined
        debug_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        debug_frame[:, :, 0] = mask_hsv_full  # Red channel = HSV mask
        debug_frame[:, :, 1] = white_mask     # Green channel = Combined mask
        debug_frame[:, :, 2] = red_mask       # Blue channel = Red walls
        
        # Add text overlay
        cv2.putText(debug_frame, "INDIVIDUAL MASKS DEBUG", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Red: HSV mask (thresh={hsv_threshold})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(debug_frame, "Green: Combined white mask", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(debug_frame, "Blue: Red walls", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(debug_frame, "Press 'i' to toggle individual/combined view", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    else:
        # Normal combined view
        debug_frame = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        
        # Add red mask in red channel
        debug_frame[:, :, 2] = cv2.bitwise_or(debug_frame[:, :, 2], red_mask)
        
        # Add text overlay
        cv2.putText(debug_frame, "COMBINED MASK VIEW", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"White: Ball detection", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Red: Wall detection", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(debug_frame, "Press 'i' to see individual masks", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw detected balls as green circles
    for x, y, radius in balls:
        cv2.circle(debug_frame, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(debug_frame, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(debug_frame, f"R:{radius}", (x-20, y-radius-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw detected walls as blue rectangles
    for x, y, w, h in walls:
        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Status info
    cv2.putText(debug_frame, f"Balls found: {len(balls)}", (10, 140), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(debug_frame, f"Walls found: {len(walls)}", (10, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(debug_frame, "Press 'o' to return to normal view", (10, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return debug_frame

def draw_autonomous_display(frame, balls, walls, robot_state, motor_state, danger_detected, 
                          target_ball=None, candidate_ball=None, confirmation_progress=0.0):
    """Draw autonomous robot status and detections with ball confirmation info"""
    if frame is None:
        return frame
    
    display_frame = frame.copy()
    
    # Draw danger zone line
    danger_y = CAMERA_HEIGHT - WALL_DANGER_DISTANCE
    color = (0, 0, 255) if danger_detected else (0, 255, 255)
    cv2.line(display_frame, (0, danger_y), (CAMERA_WIDTH, danger_y), color, 2)
    cv2.putText(display_frame, "DANGER ZONE", (CAMERA_WIDTH - 150, danger_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw all balls
    for i, (x, y, radius) in enumerate(balls):
        ball_color = (0, 255, 0)  # Default green
        ball_thickness = 2
        label = f"BALL {i+1}"
        
        # Highlight candidate ball being observed
        if candidate_ball and (x, y, radius) == candidate_ball:
            ball_color = (0, 255, 255)  # Yellow for candidate
            ball_thickness = 3
            label = f"OBSERVING {confirmation_progress:.1f}s"
            # Draw progress circle
            progress_radius = radius + 10
            progress_angle = int(360 * (confirmation_progress / BALL_CONFIRMATION_TIME))
            if progress_angle > 0:
                # Draw arc to show confirmation progress
                cv2.ellipse(display_frame, (x, y), (progress_radius, progress_radius), 
                           0, 0, progress_angle, (0, 255, 255), 3)
        
        # Highlight confirmed target ball
        if target_ball and (x, y, radius) == target_ball:
            cv2.circle(display_frame, (x, y), radius + 8, (255, 255, 0), 4)  # Yellow highlight
            ball_color = (0, 255, 0)  # Keep green but with yellow border
            label = "TARGET"
        
        cv2.circle(display_frame, (x, y), radius, ball_color, ball_thickness)
        cv2.circle(display_frame, (x, y), 2, ball_color, -1)
        cv2.putText(display_frame, label, (x-30, y-radius-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_color, 1)
    
    # Draw walls with danger highlighting
    for x, y, w, h in walls:
        wall_in_danger = (y + h) > (CAMERA_HEIGHT - WALL_DANGER_DISTANCE)
        wall_color = (0, 0, 255) if wall_in_danger else (255, 0, 0)
        thickness = 3 if wall_in_danger else 2
        
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), wall_color, thickness)
        label = "DANGER!" if wall_in_danger else "WALL"
        cv2.putText(display_frame, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, wall_color, 2)
    
    # Robot state indicator
    state_names = {
        RobotState.SEARCHING: "SEARCHING",
        RobotState.MOVING: "MOVING", 
        RobotState.COLLECTING: "COLLECTING",
        RobotState.AVOIDING: "AVOIDING"
    }
    
    state_colors = {
        RobotState.SEARCHING: (255, 255, 0),    # Yellow
        RobotState.MOVING: (0, 255, 0),         # Green
        RobotState.COLLECTING: (255, 0, 255),   # Magenta
        RobotState.AVOIDING: (0, 0, 255)        # Red
    }
    
    state_name = state_names.get(robot_state, "UNKNOWN")
    state_color = state_colors.get(robot_state, (255, 255, 255))
    
    # Draw state box
    cv2.rectangle(display_frame, (10, 10), (250, 50), state_color, -1)
    cv2.rectangle(display_frame, (10, 10), (250, 50), (255, 255, 255), 2)
    cv2.putText(display_frame, f"STATE: {state_name}", (15, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Ball confirmation status
    if candidate_ball and robot_state == RobotState.SEARCHING:
        remaining_time = BALL_CONFIRMATION_TIME - confirmation_progress
        cv2.putText(display_frame, f"Confirming: {remaining_time:.1f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Ball and wall counts
    y_offset = 90 if candidate_ball and robot_state == RobotState.SEARCHING else 70
    cv2.putText(display_frame, f"Balls: {len(balls)}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Walls: {len(walls)}", (10, y_offset + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Autonomous mode indicator
    cv2.putText(display_frame, "AUTONOMOUS MODE", (10, y_offset + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Debug mode hint
    cv2.putText(display_frame, "Press 'o' for debug masks", (10, y_offset + 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    if danger_detected:
        cv2.putText(display_frame, "⚠️ DANGER ZONE ⚠️", (CAMERA_WIDTH//2 - 100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    return display_frame

def add_performance_stats(display_frame, perf_monitor):
    """Add performance statistics to display"""
    if display_frame is None or not ENABLE_PERFORMANCE_STATS:
        return display_frame
    
    fps_text = f"FPS: {perf_monitor.fps:.1f}"
    latency_text = f"Latency: {perf_monitor.avg_latency:.1f}ms"
    cv2.putText(display_frame, fps_text, (10, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(display_frame, latency_text, (10, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return display_frame

def add_mode_indicator(display_frame, auto_enabled):
    """Add autonomous/manual mode indicator"""
    if display_frame is None:
        return display_frame
    
    mode_text = "AUTO" if auto_enabled else "MANUAL"
    mode_color = (0, 255, 0) if auto_enabled else (0, 0, 255)
    cv2.putText(display_frame, f"MODE: {mode_text}", (CAMERA_WIDTH - 150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    return display_frame

def print_headless_status(frame_count, target_fps, autonomous_controller, balls, walls, 
                         perf_monitor, danger_detected):
    """Print status information for headless mode"""
    if frame_count % (target_fps * 2) == 0:  # Every 2 seconds
        state_names = {
            RobotState.SEARCHING: "SEARCHING",
            RobotState.MOVING: "MOVING", 
            RobotState.COLLECTING: "COLLECTING",
            RobotState.AVOIDING: "AVOIDING"
        }
        state_name = state_names.get(autonomous_controller.state, "UNKNOWN")
        
        status_msg = f"Status: {state_name} | Balls: {len(balls)} | Walls: {len(walls)} | FPS: {perf_monitor.fps:.1f}"
        
        # Add confirmation info
        if autonomous_controller.candidate_ball:
            confirmation_progress = autonomous_controller.get_confirmation_progress()
            remaining = BALL_CONFIRMATION_TIME - confirmation_progress
            status_msg += f" | Confirming ball: {remaining:.1f}s"
        elif autonomous_controller.confirmed_ball:
            status_msg += " | Ball confirmed!"
        
        print(status_msg)
        
        if danger_detected:
            print("⚠️  DANGER: Wall detected!")