import time
import logging
import busio
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from gpiozero import PWMOutputDevice
import threading
import config

class GolfBotHardware:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_hardware()
        self.collected_balls = []
        self.current_speed = config.DEFAULT_SPEED
        
        # Servo position tracking and movement control
        self.servo_positions = {
            'servo1': config.SERVO_CENTER,
            'servo2': config.SERVO_CENTER,
            'servo3': config.SERVO_CENTER
        }
        self.servo_targets = {
            'servo1': config.SERVO_CENTER,
            'servo2': config.SERVO_CENTER,
            'servo3': config.SERVO_CENTER
        }
        
        # Smooth movement settings
        self.servo_moving = {
            'servo1': False,
            'servo2': False,
            'servo3': False
        }
        self.movement_threads = {}
        self.stop_movement = {}
        
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
            self.servos = {
                'servo1': self.servo1,
                'servo2': self.servo2,
                'servo3': self.servo3
            }
            
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
    
    # === SMOOTH SERVO CONTROL ===
    def _move_servo_smooth(self, servo_name: str, target_angle: int, duration: float = 1.0, steps_per_second: int = 30):
        """Smoothly move servo to target angle over specified duration"""
        if servo_name not in self.servos:
            self.logger.error(f"Unknown servo: {servo_name}")
            return
        
        servo_obj = self.servos[servo_name]
        start_angle = self.servo_positions[servo_name]
        target_angle = max(0, min(180, target_angle))  # Clamp to valid range
        
        # Calculate movement parameters
        angle_difference = target_angle - start_angle
        if abs(angle_difference) < 1:  # Already close enough
            return
        
        total_steps = int(duration * steps_per_second)
        angle_per_step = angle_difference / total_steps
        step_delay = 1.0 / steps_per_second
        
        self.servo_moving[servo_name] = True
        self.stop_movement[servo_name] = False
        
        try:
            for step in range(total_steps + 1):
                if self.stop_movement.get(servo_name, False):
                    break
                
                # Calculate current angle
                current_angle = start_angle + (angle_per_step * step)
                current_angle = max(0, min(180, current_angle))
                
                # Move servo
                servo_obj.angle = current_angle
                self.servo_positions[servo_name] = current_angle
                
                # Debug logging for servo1 movements
                if servo_name == 'servo1' and config.DEBUG_MOVEMENT and step % 10 == 0:
                    self.logger.debug(f"Servo1 smooth move: {current_angle:.1f}° (step {step}/{total_steps})")
                
                time.sleep(step_delay)
            
            # Ensure we end at exact target
            servo_obj.angle = target_angle
            self.servo_positions[servo_name] = target_angle
            self.servo_targets[servo_name] = target_angle
            
            if config.DEBUG_MOVEMENT:
                self.logger.info(f"{servo_name} smoothly moved to {target_angle}° in {duration:.1f}s")
                
        except Exception as e:
            self.logger.error(f"Smooth servo movement failed for {servo_name}: {e}")
        finally:
            self.servo_moving[servo_name] = False
    
    def move_servo_smooth(self, servo_name: str, target_angle: int, duration: float = 1.0, wait: bool = False):
        """Public method to initiate smooth servo movement"""
        if servo_name in self.movement_threads and self.movement_threads[servo_name].is_alive():
            # Stop current movement
            self.stop_movement[servo_name] = True
            self.movement_threads[servo_name].join(timeout=0.5)
        
        # Start new movement thread
        thread = threading.Thread(
            target=self._move_servo_smooth,
            args=(servo_name, target_angle, duration),
            daemon=True
        )
        self.movement_threads[servo_name] = thread
        thread.start()
        
        if wait:
            thread.join()
    
    def set_servo_angle(self, servo_obj, angle):
        """Legacy method - now uses smooth movement for servo1"""
        angle = max(0, min(180, angle))  # Clamp to valid range
        
        # Determine which servo this is
        servo_name = None
        for name, obj in self.servos.items():
            if obj == servo_obj:
                servo_name = name
                break
        
        if servo_name == 'servo1':
            # Use smooth movement for servo1
            self.move_servo_smooth('servo1', angle, duration=0.8)
        else:
            # Direct movement for other servos
            try:
                servo_obj.angle = angle
                if servo_name:
                    self.servo_positions[servo_name] = angle
            except Exception as e:
                self.logger.error(f"Failed to set servo angle: {e}")
    
    def set_servo1_angle_smooth(self, angle: int, duration: float = 1.0, wait: bool = False):
        """Specifically for servo1 with smooth movement"""
        self.move_servo_smooth('servo1', angle, duration, wait)
    
    def is_servo_moving(self, servo_name: str = None) -> bool:
        """Check if servo(s) are currently moving"""
        if servo_name:
            return self.servo_moving.get(servo_name, False)
        return any(self.servo_moving.values())
    
    def wait_for_servo_movement(self, servo_name: str = None, timeout: float = 5.0):
        """Wait for servo movement to complete"""
        start_time = time.time()
        
        if servo_name:
            while self.servo_moving.get(servo_name, False) and (time.time() - start_time) < timeout:
                time.sleep(0.1)
        else:
            while any(self.servo_moving.values()) and (time.time() - start_time) < timeout:
                time.sleep(0.1)
    
    def stop_servo_movement(self, servo_name: str = None):
        """Stop ongoing servo movement"""
        if servo_name:
            self.stop_movement[servo_name] = True
        else:
            for name in self.stop_movement:
                self.stop_movement[name] = True
    
    # === ENHANCED COLLECTION METHODS ===
    def servo1_to_collection_position(self, duration: float = 1.0):
        """Smoothly move servo1 to collection position"""
        self.set_servo1_angle_smooth(config.SERVO_COLLECT_OPEN, duration)
        if config.DEBUG_COLLECTION:
            self.logger.info(f"Servo1 moving to collection position ({config.SERVO_COLLECT_OPEN}°) over {duration}s")
    
    def servo1_to_center(self, duration: float = 0.8):
        """Smoothly move servo1 to center position"""
        self.set_servo1_angle_smooth(config.SERVO_CENTER, duration)
        if config.DEBUG_COLLECTION:
            self.logger.info(f"Servo1 moving to center ({config.SERVO_CENTER}°) over {duration}s")
    
    def servo1_to_grab_position(self, duration: float = 1.2):
        """Smoothly move servo1 to grab position"""
        self.set_servo1_angle_smooth(config.SERVO_COLLECT_CLOSE, duration)
        if config.DEBUG_COLLECTION:
            self.logger.info(f"Servo1 moving to grab position ({config.SERVO_COLLECT_CLOSE}°) over {duration}s")
    
    # === ORIGINAL SERVO METHODS (UPDATED) ===
    def center_servos(self):
        """Center all servos to 90 degrees with smooth movement for servo1"""
        try:
            # Smooth movement for servo1
            self.set_servo1_angle_smooth(config.SERVO_CENTER, duration=1.0)
            
            # Direct movement for servo2 and servo3
            self.servos['servo2'].angle = config.SERVO_CENTER
            self.servos['servo3'].angle = config.SERVO_CENTER
            self.servo_positions['servo2'] = config.SERVO_CENTER
            self.servo_positions['servo3'] = config.SERVO_CENTER
            
            # Wait for movements to complete
            self.wait_for_servo_movement('servo1', timeout=2.0)
            
        except Exception as e:
            self.logger.error(f"Failed to center servos: {e}")
        
    def collection_position(self):
        """Move servos to ball collection position"""
        try:
            # Smooth movement for servo1
            self.servo1_to_collection_position(duration=1.0)
            
            # Direct movement for servo2 and servo3
            self.servos['servo2'].angle = config.SERVO_COLLECT_OPEN
            self.servos['servo3'].angle = config.SERVO_COLLECT_OPEN
            self.servo_positions['servo2'] = config.SERVO_COLLECT_OPEN
            self.servo_positions['servo3'] = config.SERVO_COLLECT_OPEN
            
            # Wait for servo1 to reach position
            self.wait_for_servo_movement('servo1', timeout=2.0)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("Servos in collection position")
        except Exception as e:
            self.logger.error(f"Failed to set collection position: {e}")
    
    def grab_ball(self):
        """Close servos to grab a ball with smooth servo1 movement"""
        try:
            # Smooth movement for servo1
            self.servo1_to_grab_position(duration=1.2)
            
            # Direct movement for servo2 and servo3
            self.servos['servo2'].angle = config.SERVO_COLLECT_CLOSE
            self.servos['servo3'].angle = config.SERVO_COLLECT_CLOSE
            self.servo_positions['servo2'] = config.SERVO_COLLECT_CLOSE
            self.servo_positions['servo3'] = config.SERVO_COLLECT_CLOSE
            
            # Wait for servo1 to complete grabbing motion
            self.wait_for_servo_movement('servo1', timeout=2.0)
            time.sleep(0.3)  # Additional time to secure ball
            
            self.collected_balls.append(time.time())  # Track collection time
            if config.DEBUG_COLLECTION:
                self.logger.info(f"Ball grabbed! Total collected: {len(self.collected_balls)}")
        except Exception as e:
            self.logger.error(f"Failed to grab ball: {e}")
    
    def release_balls(self):
        """Release all collected balls with smooth servo1 movement"""
        try:
            # Smooth movement for servo1
            self.set_servo1_angle_smooth(config.SERVO_RELEASE, duration=1.5)
            
            # Direct movement for servo2 and servo3
            self.servos['servo2'].angle = config.SERVO_RELEASE
            self.servos['servo3'].angle = config.SERVO_RELEASE
            self.servo_positions['servo2'] = config.SERVO_RELEASE
            self.servo_positions['servo3'] = config.SERVO_RELEASE
            
            # Wait for servo1 to complete release motion
            self.wait_for_servo_movement('servo1', timeout=2.5)
            time.sleep(1.0)  # Allow balls to fall out
            
            balls_released = len(self.collected_balls)
            self.collected_balls.clear()
            if config.DEBUG_COLLECTION:
                self.logger.info(f"Released {balls_released} balls")
            return balls_released
        except Exception as e:
            self.logger.error(f"Failed to release balls: {e}")
            return 0
    
    # === MOTOR CONTROL (UNCHANGED) ===
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
    
    # === HIGH-LEVEL MOVEMENT FUNCTIONS (UNCHANGED) ===
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
    
    # === ENHANCED BALL COLLECTION SEQUENCE ===
    def attempt_ball_collection(self):
        """Complete ball collection sequence with smooth servo movement"""
        try:
            # Slow down for precision
            original_speed = self.current_speed
            self.set_speed(config.MOTOR_SPEED_SLOW)
            
            # Smoothly open collection mechanism
            self.collection_position()
            
            # Move forward slowly to collect
            self.move_forward(duration=0.5)
            
            # Smoothly grab the ball
            self.grab_ball()
            
            # Back up slightly
            self.move_backward(duration=0.3)
            
            # Restore original speed
            self.set_speed(original_speed)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ball collection failed: {e}")
            self.stop_motors()
            self.stop_servo_movement()  # Stop any ongoing servo movements
            return False
    
    def delivery_sequence(self, goal_type="B"):
        """Deliver balls to specified goal with smooth servo movement"""
        try:
            # Position for delivery
            self.stop_motors()
            time.sleep(0.5)
            
            # Smoothly release balls
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
        """Get current servo angles from tracking"""
        return {
            "servo1": self.servo_positions.get("servo1", 90),
            "servo2": self.servo_positions.get("servo2", 90),
            "servo3": self.servo_positions.get("servo3", 90)
        }
    
    def get_servo_targets(self):
        """Get target servo angles"""
        return {
            "servo1": self.servo_targets.get("servo1", 90),
            "servo2": self.servo_targets.get("servo2", 90),
            "servo3": self.servo_targets.get("servo3", 90)
        }
    
    # === EMERGENCY AND CLEANUP ===
    def emergency_stop(self):
        """Emergency stop all movement"""
        self.stop_motors()
        self.stop_servo_movement()  # Stop all servo movements
        self.center_servos()
        self.logger.warning("EMERGENCY STOP activated")
    
    def cleanup(self):
        """Clean shutdown of hardware"""
        try:
            self.stop_motors()
            self.stop_servo_movement()  # Stop all servo movements
            
            # Wait for movements to stop
            self.wait_for_servo_movement(timeout=3.0)
            
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
        """Get hardware status including servo movement info"""
        return {
            'collected_balls': len(self.collected_balls),
            'current_speed': self.current_speed,
            'speed_percentage': f"{self.current_speed*100:.0f}%",
            'servo_angles': self.get_servo_angles(),
            'servo_targets': self.get_servo_targets(),
            'servo_moving': dict(self.servo_moving),
            'servo1_moving': self.is_servo_moving('servo1')
        }