import time
import logging
import busio
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from gpiozero import PWMOutputDevice
import config

class GolfBotHardware:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_hardware()
        self.collected_balls = []
        self.current_speed = config.DEFAULT_SPEED
        
    def setup_hardware(self):
        """Initialize all hardware components"""
        try:
            # Setup PCA9685 for servo control
            self.logger.info("Initializing PCA9685 for servo control...")
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c, address=config.PCA9685_ADDRESS)
            self.pca.frequency = config.PCA9685_FREQUENCY
            
            # Setup servos
            self.servo1 = servo.Servo(self.pca.channels[config.SERVO_1_CHANNEL])
            self.servo2 = servo.Servo(self.pca.channels[config.SERVO_2_CHANNEL]) 
            self.servo3 = servo.Servo(self.pca.channels[config.SERVO_3_CHANNEL])
            
            # Setup motors with PWM for speed control
            self.motor_in1 = PWMOutputDevice(config.MOTOR_IN1)
            self.motor_in2 = PWMOutputDevice(config.MOTOR_IN2)
            self.motor_in3 = PWMOutputDevice(config.MOTOR_IN3)
            self.motor_in4 = PWMOutputDevice(config.MOTOR_IN4)
            
            # Initialize positions
            self.center_servos()
            self.stop_motors()
            
            self.logger.info("Hardware initialized successfully")
            self.logger.info("✓ PCA9685 ready for servo control")
            self.logger.info(f"✓ Servo 1 on channel {config.SERVO_1_CHANNEL}")
            self.logger.info(f"✓ Servo 2 on channel {config.SERVO_2_CHANNEL}")
            self.logger.info(f"✓ Servo 3 on channel {config.SERVO_3_CHANNEL}")
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            raise
    
    # === SERVO CONTROL ===
    def set_servo_angle(self, servo_obj, angle):
        """Set servo to specific angle (0-180 degrees)"""
        angle = max(0, min(180, angle))  # Clamp to valid range
        try:
            servo_obj.angle = angle
        except Exception as e:
            self.logger.error(f"Failed to set servo angle: {e}")
        
    def center_servos(self):
        """Center all servos to 90 degrees"""
        try:
            self.set_servo_angle(self.servo1, config.SERVO_CENTER)
            self.set_servo_angle(self.servo2, config.SERVO_CENTER)
            self.set_servo_angle(self.servo3, config.SERVO_CENTER)
            time.sleep(0.5)  # Allow time to reach position
        except Exception as e:
            self.logger.error(f"Failed to center servos: {e}")
        
    def collection_position(self):
        """Move servos to ball collection position"""
        try:
            self.set_servo_angle(self.servo1, config.SERVO_COLLECT_OPEN)
            self.set_servo_angle(self.servo2, config.SERVO_COLLECT_OPEN)
            self.set_servo_angle(self.servo3, config.SERVO_COLLECT_OPEN)
            time.sleep(0.5)
            if config.DEBUG_COLLECTION:
                self.logger.info("Servos in collection position")
        except Exception as e:
            self.logger.error(f"Failed to set collection position: {e}")
    
    def grab_ball(self):
        """Close servos to grab a ball"""
        try:
            self.set_servo_angle(self.servo1, config.SERVO_COLLECT_CLOSE)
            self.set_servo_angle(self.servo2, config.SERVO_COLLECT_CLOSE) 
            self.set_servo_angle(self.servo3, config.SERVO_COLLECT_CLOSE)
            time.sleep(0.8)  # Give time to secure ball
            self.collected_balls.append(time.time())  # Track collection time
            if config.DEBUG_COLLECTION:
                self.logger.info(f"Ball grabbed! Total collected: {len(self.collected_balls)}")
        except Exception as e:
            self.logger.error(f"Failed to grab ball: {e}")
    
    def release_balls(self):
        """Release all collected balls"""
        try:
            self.set_servo_angle(self.servo1, config.SERVO_RELEASE)
            self.set_servo_angle(self.servo2, config.SERVO_RELEASE)
            self.set_servo_angle(self.servo3, config.SERVO_RELEASE)
            time.sleep(1.0)  # Allow balls to fall out
            balls_released = len(self.collected_balls)
            self.collected_balls.clear()
            if config.DEBUG_COLLECTION:
                self.logger.info(f"Released {balls_released} balls")
            return balls_released
        except Exception as e:
            self.logger.error(f"Failed to release balls: {e}")
            return 0
    
    # === MOTOR CONTROL ===
    def stop_motors(self):
        """Stop all motors"""
        self.motor_in1.off()
        self.motor_in2.off()
        self.motor_in3.off()
        self.motor_in4.off()
        if config.DEBUG_MOVEMENT:
            self.logger.info("Motors stopped")
    
    def move_forward(self, duration=None, speed=None):
        """Move robot forward"""
        if speed is None:
            speed = self.current_speed
            
        # Motor A forward, Motor B reverse (due to mirrored mounting)
        self.motor_in1.value = speed
        self.motor_in2.off()
        self.motor_in3.off() 
        self.motor_in4.value = speed
        
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"Moving forward at {speed*100:.0f}% speed")
            
        if duration:
            time.sleep(duration)
            self.stop_motors()
    
    def move_backward(self, duration=None, speed=None):
        """Move robot backward"""
        if speed is None:
            speed = self.current_speed
            
        # Reverse of forward movement
        self.motor_in1.off()
        self.motor_in2.value = speed
        self.motor_in3.value = speed
        self.motor_in4.off()
        
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"Moving backward at {speed*100:.0f}% speed")
            
        if duration:
            time.sleep(duration)
            self.stop_motors()
    
    def turn_right(self, duration=None, speed=None):
        """Turn robot right"""
        if speed is None:
            speed = self.current_speed
            
        # Both motors forward (same direction = turn right)
        self.motor_in1.value = speed
        self.motor_in2.off()
        self.motor_in3.value = speed
        self.motor_in4.off()
        
        if config.DEBUG_MOVEMENT:
            self.logger.info("Turning right")
            
        if duration:
            time.sleep(duration)
            self.stop_motors()
    
    def turn_left(self, duration=None, speed=None):
        """Turn robot left"""
        if speed is None:
            speed = self.current_speed
            
        # Both motors reverse (same direction = turn left)
        self.motor_in1.off()
        self.motor_in2.value = speed
        self.motor_in3.off()
        self.motor_in4.value = speed
        
        if config.DEBUG_MOVEMENT:
            self.logger.info("Turning left")
            
        if duration:
            time.sleep(duration)
            self.stop_motors()
    
    def set_speed(self, speed):
        """Set default movement speed (0.0 to 1.0)"""
        self.current_speed = max(0.0, min(1.0, speed))
        if config.DEBUG_MOVEMENT:
            self.logger.info(f"Speed set to {self.current_speed*100:.0f}%")
    
    # === HIGH-LEVEL MOVEMENT FUNCTIONS ===
    def turn_90_right(self):
        """Turn exactly 90 degrees right"""
        self.turn_right(duration=config.TURN_TIME_90_DEGREES)
    
    def turn_90_left(self):
        """Turn exactly 90 degrees left"""
        self.turn_left(duration=config.TURN_TIME_90_DEGREES)
    
    def turn_180(self):
        """Turn around 180 degrees"""
        self.turn_right(duration=config.TURN_TIME_90_DEGREES * 2)
    
    def forward_step(self):
        """Move forward a short distance"""
        self.move_forward(duration=config.FORWARD_TIME_SHORT)
    
    def backward_step(self):
        """Move backward a short distance"""
        self.move_backward(duration=config.FORWARD_TIME_SHORT)
    
    # === BALL COLLECTION SEQUENCE ===
    def attempt_ball_collection(self):
        """Complete ball collection sequence"""
        try:
            # Slow down for precision
            original_speed = self.current_speed
            self.set_speed(config.MOTOR_SPEED_SLOW)
            
            # Open collection mechanism
            self.collection_position()
            
            # Move forward slowly to collect
            self.move_forward(duration=0.5)
            
            # Grab the ball
            self.grab_ball()
            
            # Back up slightly
            self.move_backward(duration=0.3)
            
            # Restore original speed
            self.set_speed(original_speed)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ball collection failed: {e}")
            self.stop_motors()
            return False
    
    def delivery_sequence(self, goal_type="B"):
        """Deliver balls to specified goal"""
        try:
            # Position for delivery
            self.stop_motors()
            time.sleep(0.5)
            
            # Release balls
            balls_delivered = self.release_balls()
            
            # Back away from goal
            self.move_backward(duration=1.0)
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"Delivered {balls_delivered} balls to goal {goal_type}")
                
            return balls_delivered
            
        except Exception as e:
            self.logger.error(f"Ball delivery failed: {e}")
            return 0
    
    # === SERVO ANGLE GETTERS ===
    def get_servo_angles(self):
        """Get current servo angles"""
        try:
            return {
                "servo1": getattr(self.servo1, 'angle', 90),
                "servo2": getattr(self.servo2, 'angle', 90),
                "servo3": getattr(self.servo3, 'angle', 90)
            }
        except Exception as e:
            self.logger.error(f"Failed to get servo angles: {e}")
            return {"servo1": 90, "servo2": 90, "servo3": 90}
    
    # === EMERGENCY AND CLEANUP ===
    def emergency_stop(self):
        """Emergency stop all movement"""
        self.stop_motors()
        self.center_servos()
        self.logger.warning("EMERGENCY STOP activated")
    
    def cleanup(self):
        """Clean shutdown of hardware"""
        try:
            self.stop_motors()
            self.center_servos()
            
            # Close motor GPIO connections
            for component in [self.motor_in1, self.motor_in2, self.motor_in3, self.motor_in4]:
                component.close()
            
            # Deinitialize PCA9685
            self.pca.deinit()
            
            self.logger.info("Hardware cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Hardware cleanup failed: {e}")
    
    # === STATUS METHODS ===
    def get_ball_count(self):
        """Get number of collected balls"""
        return len(self.collected_balls)
    
    def has_balls(self):
        """Check if robot has collected balls"""
        return len(self.collected_balls) > 0
    
    def get_status(self):
        """Get hardware status"""
        return {
            'collected_balls': len(self.collected_balls),
            'current_speed': self.current_speed,
            'speed_percentage': f"{self.current_speed*100:.0f}%",
            'servo_angles': self.get_servo_angles()
        }