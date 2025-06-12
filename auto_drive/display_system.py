#!/usr/bin/env python3
"""
Display System for GolfBot
Handles all visual display and status information
"""

import cv2
import time
import numpy as np
from robot_states import RobotState, MotorState
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, WALL_DANGER_DISTANCE,
    BALL_CONFIRMATION_TIME, TARGET_FPS, ENABLE_PERFORMANCE_STATS
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
