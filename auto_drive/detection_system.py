#!/usr/bin/env python3
"""
Detection System for GolfBot
Handles white ball and red wall detection using computer vision
"""

import cv2
import numpy as np
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, PROCESS_WIDTH, PROCESS_HEIGHT,
    MIN_BALL_AREA, MIN_WALL_AREA, BALL_CIRCULARITY_THRESHOLD,
    WALL_MIN_LENGTH, WALL_DANGER_DISTANCE
)

def detect_white_balls_fast(frame):
    """Optimized white ball detection"""
    if frame is None:
        return []
    
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 25, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    scale_x = CAMERA_WIDTH / PROCESS_WIDTH
    scale_y = CAMERA_HEIGHT / PROCESS_HEIGHT
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_BALL_AREA:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > BALL_CIRCULARITY_THRESHOLD:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    center_x = int(x * scale_x)
                    center_y = int(y * scale_y)
                    radius_scaled = int(radius * max(scale_x, scale_y))
                    
                    if 5 < radius_scaled < 150:
                        balls.append((center_x, center_y, radius_scaled))
    
    return balls

def detect_red_walls_fast(frame):
    """Optimized red wall/boundary detection with danger zone analysis"""
    if frame is None:
        return [], None, False
    
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    walls = []
    danger_detected = False
    scale_x = CAMERA_WIDTH / PROCESS_WIDTH
    scale_y = CAMERA_HEIGHT / PROCESS_HEIGHT
    
    danger_y_threshold = PROCESS_HEIGHT - (WALL_DANGER_DISTANCE / scale_y)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_WALL_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            
            length = max(w_scaled, h_scaled)
            if length > WALL_MIN_LENGTH:
                walls.append((x_scaled, y_scaled, w_scaled, h_scaled))
                
                wall_bottom_y = y + h
                if wall_bottom_y > danger_y_threshold:
                    danger_detected = True
    
    debug_mask = cv2.resize(red_mask, (CAMERA_WIDTH, CAMERA_HEIGHT))
    return walls, debug_mask, danger_detected
