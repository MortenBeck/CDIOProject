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
            
            # Initialize positions - servos start at center
            self.stop_motors()
            
            self.logger.info("✅ Hardware initialized successfully")
            self.logger.info("✓ PCA9685 ready for servo control")
            self.logger.info(f"✓ Servo 1 on channel {config.SERVO_1_CHANNEL}")
            self.logger.info(f"✓ Servo 2 on channel {config.SERVO_2_CHANNEL}")
            self.logger.info(f"✓ Servo 3 on channel {config.SERVO_3_CHANNEL}")
            self.logger.info(f"✓ Motors configured with PWM control")
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            raise
    
    # === SERVO CONTROL ===
    def set_servo_angle(self, servo_obj, angle):
        """Set servo to specific angle (0-180 degrees) - immediate movement"""
        angle = max(0, min(180, angle))  # Clamp to valid range
        try:
            servo_obj.angle = angle
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"Servo set to {angle}°")
        except Exception as e:
            self.logger.error(f"Failed to set servo angle: {e}")
    
    def set_servo_angle_gradual(self, servo_obj, target_angle, speed_delay=None):
        """Set servo to specific angle gradually to reduce current draw"""
        if speed_delay is None:
            speed_delay = getattr(config, 'SERVO_STEP_DELAY', 0.02)
            
        target_angle = max(0, min(180, target_angle))  # Clamp to valid range
        
        try:
            # Get current angle (default to 90 if unknown)
            current_angle = getattr(servo_obj, 'angle', 90)
            if current_angle is None:
                current_angle = 90
            
            # Calculate step direction and size
            angle_diff = target_angle - current_angle
            if abs(angle_diff) <= 2:  # Already close enough
                servo_obj.angle = target_angle
                return
            
            # Move in 2-degree increments
            step_size = 2 if angle_diff > 0 else -2
            steps = int(abs(angle_diff) / 2)
            
            for i in range(steps):
                current_angle += step_size
                servo_obj.angle = current_angle
                time.sleep(speed_delay)  # Small delay between steps
            
            # Final position
            servo_obj.angle = target_angle
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"Servo moved gradually to {target_angle}°")
            
        except Exception as e:
            self.logger.error(f"Failed to set servo angle gradually: {e}")

    def set_servo_angle_smooth(self, servo_obj, target_angle, duration=None):
        """Set servo angle with smooth movement over specified duration"""
        if duration is None:
            duration = getattr(config, 'SERVO_SMOOTH_DURATION', 0.5)
            
        target_angle = max(0, min(180, target_angle))
        
        try:
            current_angle = getattr(servo_obj, 'angle', 90)
            if current_angle is None:
                current_angle = 90
            
            angle_diff = target_angle - current_angle
            if abs(angle_diff) <= 1:
                servo_obj.angle = target_angle
                return
            
            # Calculate movement parameters
            steps = max(10, int(duration / 0.02))  # At least 10 steps
            angle_step = angle_diff / steps
            time_step = duration / steps
            
            # Smooth movement
            for i in range(steps):
                current_angle += angle_step
                servo_obj.angle = int(current_angle)
                time.sleep(time_step)
            
            # Ensure final position
            servo_obj.angle = target_angle
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"Servo moved smoothly to {target_angle}° in {duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to set servo angle smoothly: {e}")

    # === NEW: SERVO 1 FOUR-STATE SYSTEM ===
    def set_servo1_incremental(self, target_angle):
        """Move servo 1 incrementally by 5-degree steps"""
        try:
            current_angle = getattr(self.servo1, 'angle', 90)
            if current_angle is None:
                current_angle = 90
            
            target_angle = max(0, min(180, target_angle))
            angle_diff = target_angle - current_angle
            
            if abs(angle_diff) <= config.SERVO_1_STEP_SIZE:
                # Close enough, move directly
                self.servo1.angle = target_angle
                if config.DEBUG_MOVEMENT:
                    self.logger.debug(f"Servo 1 moved to {target_angle}°")
                return
            
            # Move in incremental steps
            step_direction = config.SERVO_1_STEP_SIZE if angle_diff > 0 else -config.SERVO_1_STEP_SIZE
            steps = int(abs(angle_diff) / config.SERVO_1_STEP_SIZE)
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"Moving servo 1 from {current_angle}° to {target_angle}° in {steps} steps of {config.SERVO_1_STEP_SIZE}°")
            
            for i in range(steps):
                current_angle += step_direction
                self.servo1.angle = int(current_angle)
                time.sleep(0.05)  # Small delay between steps
            
            # Final position adjustment
            self.servo1.angle = target_angle
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"✅ Servo 1 reached {target_angle}°")
                
        except Exception as e:
            self.logger.error(f"Failed to move servo 1 incrementally: {e}")

    def servo1_to_store(self):
        """Move servo 1 to store position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"📦 Moving servo 1 to STORE position ({config.SERVO_1_STORE}°)")
        self.set_servo1_incremental(config.SERVO_1_STORE)

    def servo1_to_pre_collect(self):
        """Move servo 1 to pre-collect position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🎯 Moving servo 1 to PRE-COLLECT position ({config.SERVO_1_PRE_COLLECT}°)")
        self.set_servo1_incremental(config.SERVO_1_PRE_COLLECT)

    def servo1_to_driving(self):
        """Move servo 1 to driving position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🚗 Moving servo 1 to DRIVING position ({config.SERVO_1_DRIVING}°)")
        self.set_servo1_incremental(config.SERVO_1_DRIVING)

    def servo1_to_collect(self):
        """Move servo 1 to collect position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🤏 Moving servo 1 to COLLECT position ({config.SERVO_1_COLLECT}°)")
        self.set_servo1_incremental(config.SERVO_1_COLLECT)

    def get_servo1_state(self):
        """Get current servo 1 state as string"""
        try:
            current_angle = getattr(self.servo1, 'angle', 90)
            if current_angle is None:
                return "unknown"
            
            # Determine which state we're closest to
            distances = {
                'store': abs(current_angle - config.SERVO_1_STORE),
                'pre-collect': abs(current_angle - config.SERVO_1_PRE_COLLECT),
                'driving': abs(current_angle - config.SERVO_1_DRIVING),
                'collect': abs(current_angle - config.SERVO_1_COLLECT)
            }
            
            closest_state = min(distances.keys(), key=lambda k: distances[k])
            return closest_state
            
        except Exception as e:
            self.logger.error(f"Failed to get servo 1 state: {e}")
            return "unknown"

    def initialize_servo1_for_competition(self):
        """Initialize servo 1 to driving position for competition start"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Initializing servo 1 for competition start...")
            self.servo1_to_driving()
            time.sleep(0.5)  # Allow time to settle
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Servo 1 initialized at driving position")
        except Exception as e:
            self.logger.error(f"Failed to initialize servo 1 for competition: {e}")
    def center_servos(self):
        """Center all servos - servo 1 goes to driving position"""
        try:
            # Check if gradual movement is enabled
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔧 Centering servos...")
            
            if use_gradual:
                # Move servo 1 to driving position using incremental movement
                self.servo1_to_driving()
                time.sleep(sequential_delay)
                # Move other servos gradually
                self.set_servo_angle_gradual(self.servo2, config.SERVO_CENTER)
                time.sleep(sequential_delay)
                self.set_servo_angle_gradual(self.servo3, config.SERVO_CENTER)
                time.sleep(0.3)  # Final settling time
            else:
                # Original immediate movement
                self.servo1_to_driving()
                self.set_servo_angle(self.servo2, config.SERVO_CENTER)
                self.set_servo_angle(self.servo3, config.SERVO_CENTER)
                time.sleep(0.5)  # Allow time to reach position
            
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Servos centered - servo 1 at driving position")
                
        except Exception as e:
            self.logger.error(f"Failed to center servos: {e}")
        
    def collection_position(self):
        """Move servos to ball collection position (legacy method)"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔧 Moving to legacy collection position...")
            
            if use_gradual:
                self.set_servo_angle_gradual(self.servo1, config.SERVO_COLLECT_OPEN)
                time.sleep(sequential_delay)
                self.set_servo_angle_gradual(self.servo2, config.SERVO_COLLECT_OPEN)
                time.sleep(sequential_delay)
                self.set_servo_angle_gradual(self.servo3, config.SERVO_COLLECT_OPEN)
                time.sleep(0.3)
            else:
                self.set_servo_angle(self.servo1, config.SERVO_COLLECT_OPEN)
                self.set_servo_angle(self.servo2, config.SERVO_COLLECT_OPEN)
                self.set_servo_angle(self.servo3, config.SERVO_COLLECT_OPEN)
                time.sleep(0.5)
                
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Legacy collection position set")
                
        except Exception as e:
            self.logger.error(f"Failed to set collection position: {e}")
    
    # === ENHANCED COLLECTION METHODS ===
    def prepare_for_collection(self):
        """Prepare servos for ball collection - servo 1 stays in driving position"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔧 Preparing for collection - keeping servo 1 in driving position...")
            
            if use_gradual:
                # Keep servo 1 in driving position, move other servos to ready position
                self.servo1_to_driving()
                time.sleep(sequential_delay)
                self.set_servo_angle_smooth(self.servo2, config.SERVO_READY_POSITION, duration=0.4)
                time.sleep(sequential_delay)
                self.set_servo_angle_smooth(self.servo3, config.SERVO_READY_POSITION, duration=0.4)
                time.sleep(0.3)
            else:
                self.servo1_to_driving()
                self.set_servo_angle(self.servo2, config.SERVO_READY_POSITION)
                self.set_servo_angle(self.servo3, config.SERVO_READY_POSITION)
                time.sleep(0.5)
                
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Prepared for collection - servo 1 at driving, others ready")
                
        except Exception as e:
            self.logger.error(f"Failed to prepare for collection: {e}")

    # === NEW: ENHANCED COLLECTION SEQUENCE ===
    def enhanced_collection_sequence(self):
        """New collection sequence using servo 1 four-state system"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Starting enhanced collection sequence with four-state system...")
                self.logger.info("   Flow: driving -> pre-collect -> drive forward -> collect -> store -> driving")
            
            # Step 1: From driving to pre-collect position
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 1: Moving servo 1 from DRIVING to PRE-COLLECT")
            self.servo1_to_pre_collect()
            time.sleep(0.2)
            
            # Step 2: Drive forward for collection
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 2: Driving forward for collection")
            self.move_forward(duration=1.2, speed=config.COLLECTION_SPEED)
            time.sleep(0.1)
            
            # Step 3: Move servo 1 to collect position (capture ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 3: Moving servo 1 to COLLECT position (capture)")
            self.servo1_to_collect()
            time.sleep(0.3)
            
            # Step 4: Move servo 1 to store position (secure ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 4: Moving servo 1 to STORE position (secure)")
            self.servo1_to_store()
            time.sleep(0.3)
            
            # Step 5: Move servo 1 back to driving position (ready for next action)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 5: Moving servo 1 to DRIVING position (ready)")
            self.servo1_to_driving()
            time.sleep(0.2)
            
            # Record collection
            self.collected_balls.append(time.time())
            
            if config.DEBUG_COLLECTION:
                current_state = self.get_servo1_state()
                self.logger.info(f"✅ Enhanced collection complete! Servo 1 state: {current_state.upper()}, Total balls: {len(self.collected_balls)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced collection sequence failed: {e}")
            self.stop_motors()
            # Ensure we return to driving position on error
            self.servo1_to_driving()
            return False

    def blind_collection_sequence(self, drive_time):
        """Legacy method - now redirects to enhanced sequence"""
        self.logger.info("Using enhanced three-state collection sequence instead of blind collection")
        return self.enhanced_collection_sequence()
    
    def grab_ball(self):
        """Close servos to grab a ball (legacy method)"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔒 Grabbing ball with legacy method...")
            
            if use_gradual:
                # Move servos sequentially for smoother operation
                self.set_servo_angle_smooth(self.servo1, config.SERVO_COLLECT_CLOSE, duration=0.4)
                time.sleep(sequential_delay)
                self.set_servo_angle_smooth(self.servo2, config.SERVO_COLLECT_CLOSE, duration=0.4)
                time.sleep(sequential_delay)
                self.set_servo_angle_smooth(self.servo3, config.SERVO_COLLECT_CLOSE, duration=0.4)
                time.sleep(0.5)  # Give time to secure ball
            else:
                self.set_servo_angle(self.servo1, config.SERVO_COLLECT_CLOSE)
                self.set_servo_angle(self.servo2, config.SERVO_COLLECT_CLOSE) 
                self.set_servo_angle(self.servo3, config.SERVO_COLLECT_CLOSE)
                time.sleep(0.8)  # Give time to secure ball
                
            self.collected_balls.append(time.time())  # Track collection time
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"✅ Ball grabbed! Total collected: {len(self.collected_balls)}")
                
        except Exception as e:
            self.logger.error(f"Failed to grab ball: {e}")
    
    def release_balls(self):
        """Release all collected balls"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            balls_to_release = len(self.collected_balls)
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"🔓 Releasing {balls_to_release} balls...")
            
            if use_gradual:
                # Move servo 1 to store position for release
                self.servo1_to_store()
                time.sleep(sequential_delay)
                # Move other servos to release position
                self.set_servo_angle_smooth(self.servo2, config.SERVO_RELEASE, duration=0.6)
                time.sleep(sequential_delay)
                self.set_servo_angle_smooth(self.servo3, config.SERVO_RELEASE, duration=0.6)
                time.sleep(1.0)  # Allow balls to fall out
                
                # Return servo 1 to driving position after release
                self.servo1_to_driving()
                time.sleep(0.2)
            else:
                self.servo1_to_store()
                self.set_servo_angle(self.servo2, config.SERVO_RELEASE)
                self.set_servo_angle(self.servo3, config.SERVO_RELEASE)
                time.sleep(1.0)  # Allow balls to fall out
                self.servo1_to_driving()  # Return to driving
                
            self.collected_balls.clear()
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"✅ Released {balls_to_release} balls - servo 1 returned to driving position")
                
            return balls_to_release
            
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
            self.logger.debug("🛑 Motors stopped")
    
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
            self.logger.debug(f"⬆️ Moving forward at {speed*100:.0f}% speed")
            
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
            self.logger.debug(f"⬇️ Moving backward at {speed*100:.0f}% speed")
            
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
            self.logger.debug(f"↗️ Turning right at {speed*100:.0f}% speed")
            
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
            self.logger.debug(f"↖️ Turning left at {speed*100:.0f}% speed")
            
        if duration:
            time.sleep(duration)
            self.stop_motors()
    
    def set_speed(self, speed):
        """Set default movement speed (0.0 to 1.0)"""
        self.current_speed = max(0.0, min(1.0, speed))
        if config.DEBUG_MOVEMENT:
            self.logger.debug(f"⚡ Speed set to {self.current_speed*100:.0f}%")
    
    # === HIGH-LEVEL MOVEMENT FUNCTIONS ===
    def turn_90_right(self):
        """Turn exactly 90 degrees right"""
        if config.DEBUG_MOVEMENT:
            self.logger.debug("🔄 Turning 90° right")
        self.turn_right(duration=config.TURN_TIME_90_DEGREES)
    
    def turn_90_left(self):
        """Turn exactly 90 degrees left"""
        if config.DEBUG_MOVEMENT:
            self.logger.debug("🔄 Turning 90° left")
        self.turn_left(duration=config.TURN_TIME_90_DEGREES)
    
    def turn_180(self):
        """Turn around 180 degrees"""
        if config.DEBUG_MOVEMENT:
            self.logger.debug("🔄 Turning 180°")
        self.turn_right(duration=config.TURN_TIME_90_DEGREES * 2)
    
    def forward_step(self):
        """Move forward a short distance"""
        if config.DEBUG_MOVEMENT:
            self.logger.debug("👣 Forward step")
        self.move_forward(duration=config.FORWARD_TIME_SHORT)
    
    def backward_step(self):
        """Move backward a short distance"""
        if config.DEBUG_MOVEMENT:
            self.logger.debug("👣 Backward step")
        self.move_backward(duration=config.FORWARD_TIME_SHORT)
    
    # === BALL COLLECTION SEQUENCE (LEGACY) ===
    def attempt_ball_collection(self):
        """Complete ball collection sequence (legacy method)"""
        try:
            # Slow down for precision
            original_speed = self.current_speed
            self.set_speed(config.MOTOR_SPEED_SLOW)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🤖 Starting legacy ball collection sequence...")
            
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
            
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Legacy collection sequence completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Legacy ball collection failed: {e}")
            self.stop_motors()
            self.set_speed(original_speed)
            return False
    
    def delivery_sequence(self, goal_type="B"):
        """Deliver balls to specified goal"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info(f"🚚 Starting ball delivery to goal {goal_type}")
            
            # Position for delivery
            self.stop_motors()
            time.sleep(0.5)
            
            # Release balls
            balls_delivered = self.release_balls()
            
            # Back away from goal
            self.move_backward(duration=1.0)
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"✅ Delivered {balls_delivered} balls to goal {goal_type}")
                
            return balls_delivered
            
        except Exception as e:
            self.logger.error(f"❌ Ball delivery failed: {e}")
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
        self.center_servos()  # This will set servo 1 to driving position
        self.logger.warning("🛑 EMERGENCY STOP activated")
    
    def cleanup(self):
        """Clean shutdown of hardware"""
        try:
            if config.DEBUG_MOVEMENT:
                self.logger.info("🧹 Starting hardware cleanup...")
            
            self.stop_motors()
            self.center_servos()  # This will set servo 1 to driving position
            
            # Close motor GPIO connections
            for component in [self.motor_in1, self.motor_in2, self.motor_in3, self.motor_in4]:
                if hasattr(component, 'close'):
                    component.close()
            
            # Deinitialize PCA9685
            if hasattr(self.pca, 'deinit'):
                self.pca.deinit()
            
            self.logger.info("✅ Hardware cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Hardware cleanup failed: {e}")
    
    # === STATUS METHODS ===
    def get_ball_count(self):
        """Get number of collected balls"""
        return len(self.collected_balls)
    
    def has_balls(self):
        """Check if robot has collected balls"""
        return len(self.collected_balls) > 0
    
    def get_status(self):
        """Get comprehensive hardware status"""
        use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
        servo_angles = self.get_servo_angles()
        servo1_state = self.get_servo1_state()
        
        # Get motor status
        motor_status = {
            'in1_active': self.motor_in1.is_active if hasattr(self.motor_in1, 'is_active') else False,
            'in2_active': self.motor_in2.is_active if hasattr(self.motor_in2, 'is_active') else False,
            'in3_active': self.motor_in3.is_active if hasattr(self.motor_in3, 'is_active') else False,
            'in4_active': self.motor_in4.is_active if hasattr(self.motor_in4, 'is_active') else False,
        }
        
        return {
            'collected_balls': len(self.collected_balls),
            'current_speed': self.current_speed,
            'speed_percentage': f"{self.current_speed*100:.0f}%",
            'servo_angles': servo_angles,
            'servo1_state': servo1_state,
            'motor_status': motor_status,
            'gradual_movement': use_gradual,
            'collection_method': 'enhanced_four_state_collection',
            'hardware_ready': True
        }
    
    # === ENHANCED STATUS METHODS ===
    def log_status_summary(self):
        """Log a comprehensive status summary"""
        status = self.get_status()
        servo_angles = status['servo_angles']
        
        self.logger.info("🔧 HARDWARE STATUS SUMMARY:")
        self.logger.info(f"   Balls collected: {status['collected_balls']}")
        self.logger.info(f"   Current speed: {status['speed_percentage']}")
        self.logger.info(f"   Servo angles: S1={servo_angles['servo1']}° ({status['servo1_state']}) S2={servo_angles['servo2']}° S3={servo_angles['servo3']}°")
        self.logger.info(f"   Gradual movement: {status['gradual_movement']}")
        self.logger.info(f"   Collection method: {status['collection_method']}")
    
    def test_all_systems(self):
        """Test all hardware systems"""
        try:
            self.logger.info("🧪 Testing all hardware systems...")
            
            # Test motors
            self.logger.info("Testing motors...")
            self.move_forward(duration=0.2)
            time.sleep(0.2)
            self.move_backward(duration=0.2)
            time.sleep(0.2)
            self.turn_right(duration=0.2)
            time.sleep(0.2)
            self.turn_left(duration=0.2)
            time.sleep(0.2)
            
            # Test servo 1 four-state system
            self.logger.info("Testing servo 1 four-state system...")
            self.servo1_to_driving()
            time.sleep(1)
            self.servo1_to_pre_collect()
            time.sleep(1)
            self.servo1_to_collect()
            time.sleep(1)
            self.servo1_to_store()
            time.sleep(1)
            self.servo1_to_driving()  # Return to driving
            time.sleep(1)
            
            # Test other servos
            self.logger.info("Testing servos 2 and 3...")
            self.set_servo_angle(self.servo2, config.SERVO_COLLECT_CLOSE)
            self.set_servo_angle(self.servo3, config.SERVO_COLLECT_CLOSE)
            time.sleep(1)
            
            # Return to center
            self.center_servos()
            
            self.logger.info("✅ All systems test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System test failed: {e}")
            return False