#!/usr/bin/env python3
"""
Motor Control System for GolfBot
Handles all motor movement and servo control
"""

from gpiozero import PWMOutputDevice
from config import (
    MOTOR_IN1_PIN, MOTOR_IN2_PIN, MOTOR_IN3_PIN, MOTOR_IN4_PIN,
    DEFAULT_SPEED
)

class MotorController:
    """Handles all motor control operations"""
    
    def __init__(self):
        self.motors_available = False
        self.motor_in1 = None
        self.motor_in2 = None
        self.motor_in3 = None
        self.motor_in4 = None
        
        self._initialize_motors()
    
    def _initialize_motors(self):
        """Initialize motor control GPIO pins"""
        print("Initializing motor control...")
        try:
            self.motor_in1 = PWMOutputDevice(MOTOR_IN1_PIN)
            self.motor_in2 = PWMOutputDevice(MOTOR_IN2_PIN)
            self.motor_in3 = PWMOutputDevice(MOTOR_IN3_PIN)
            self.motor_in4 = PWMOutputDevice(MOTOR_IN4_PIN)
            self.motors_available = True
            print("✓ Motors initialized successfully")
        except Exception as e:
            self.motors_available = False
            print(f"⚠️  Motor initialization failed: {e}")
    
    def stop_motors(self):
        """Stop all motors immediately"""
        if not self.motors_available:
            return
        self.motor_in1.off()
        self.motor_in2.off()
        self.motor_in3.off()
        self.motor_in4.off()
    
    def motor_a_forward(self, speed=DEFAULT_SPEED):
        """Motor A forward"""
        if not self.motors_available:
            return
        self.motor_in1.value = speed
        self.motor_in2.off()
    
    def motor_a_reverse(self, speed=DEFAULT_SPEED):
        """Motor A reverse"""
        if not self.motors_available:
            return
        self.motor_in1.off()
        self.motor_in2.value = speed
    
    def motor_b_forward(self, speed=DEFAULT_SPEED):
        """Motor B forward"""
        if not self.motors_available:
            return
        self.motor_in3.value = speed
        self.motor_in4.off()
    
    def motor_b_reverse(self, speed=DEFAULT_SPEED):
        """Motor B reverse"""
        if not self.motors_available:
            return
        self.motor_in3.off()
        self.motor_in4.value = speed
    
    def both_motors_forward(self, speed=DEFAULT_SPEED):
        """Move forward (straight) - both motors forward"""
        if not self.motors_available:
            return
        self.motor_a_forward(speed)
        self.motor_b_forward(speed)
    
    def both_motors_reverse(self, speed=DEFAULT_SPEED):
        """Move reverse (straight) - both motors reverse"""
        if not self.motors_available:
            return
        self.motor_a_reverse(speed)
        self.motor_b_reverse(speed)
    
    def turn_right(self, speed=DEFAULT_SPEED):
        """Turn right - one forward, one reverse"""
        if not self.motors_available:
            return
        self.motor_a_forward(speed)
        self.motor_b_reverse(speed)
    
    def turn_left(self, speed=DEFAULT_SPEED):
        """Turn left - one reverse, one forward"""
        if not self.motors_available:
            return
        self.motor_a_reverse(speed)
        self.motor_b_forward(speed)
    
    def cleanup(self):
        """Clean up motor resources"""
        self.stop_motors()
        if self.motors_available:
            self.motor_in1.close()
            self.motor_in2.close()
            self.motor_in3.close()
            self.motor_in4.close()

class ServoController:
    """Handles servo control for ball collection"""
    
    def __init__(self):
        # TODO: Initialize actual servo control
        pass
    
    def activate_collection_servo(self):
        """Activate ball collection mechanism - PLACEHOLDER"""
        print("🔧 Collection servo activated (placeholder)")
        # TODO: Add actual servo control here
        # Example: servo.angle = 90, wait, servo.angle = 0
    
    def deactivate_collection_servo(self):
        """Deactivate ball collection mechanism - PLACEHOLDER"""
        print("🔧 Collection servo deactivated (placeholder)")
        # TODO: Add actual servo control here
