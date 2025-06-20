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
            
            # Setup servos - SS and SF only
            self.servo_ss = servo.Servo(self.pca.channels[config.SERVO_SS_CHANNEL])
            self.servo_sf = servo.Servo(self.pca.channels[config.SERVO_SF_CHANNEL])
            
            # Setup motors with PWM for speed control
            self.motor_in1 = PWMOutputDevice(config.MOTOR_IN1)
            self.motor_in2 = PWMOutputDevice(config.MOTOR_IN2)
            self.motor_in3 = PWMOutputDevice(config.MOTOR_IN3)
            self.motor_in4 = PWMOutputDevice(config.MOTOR_IN4)
            
            # Initialize positions
            self.stop_motors()
            
            self.logger.info("✅ Hardware initialized successfully")
            self.logger.info("✓ PCA9685 ready for servo control")
            self.logger.info(f"✓ Servo SS on channel {config.SERVO_SS_CHANNEL}")
            self.logger.info(f"✓ Servo SF on channel {config.SERVO_SF_CHANNEL}")
            self.logger.info(f"✓ Motors configured with PWM control")
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            raise
    
    # === SERVO CONTROL ===
    def set_servo_angle(self, servo_obj, angle):
        """Set servo to specific angle (0-180 degrees) - immediate movement"""
        angle = max(0, min(180, angle))
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
            
        target_angle = max(0, min(180, target_angle))
        
        try:
            current_angle = getattr(servo_obj, 'angle', 90)
            if current_angle is None:
                current_angle = 90
            
            angle_diff = target_angle - current_angle
            if abs(angle_diff) <= 2:
                servo_obj.angle = target_angle
                return
            
            step_size = 2 if angle_diff > 0 else -2
            steps = int(abs(angle_diff) / 2)
            
            for i in range(steps):
                current_angle += step_size
                servo_obj.angle = current_angle
                time.sleep(speed_delay)
            
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
            
            steps = max(10, int(duration / 0.02))
            angle_step = angle_diff / steps
            time_step = duration / steps
            
            for i in range(steps):
                current_angle += angle_step
                servo_obj.angle = int(current_angle)
                time.sleep(time_step)
            
            servo_obj.angle = target_angle
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"Servo moved smoothly to {target_angle}° in {duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to set servo angle smoothly: {e}")

    # === SERVO SS (MAIN COLLECTION) FOUR-STATE SYSTEM ===
    def set_servo_ss_incremental(self, target_angle):
        """Move servo SS incrementally by 5-degree steps"""
        try:
            current_angle = getattr(self.servo_ss, 'angle', 90)
            if current_angle is None:
                current_angle = 90
            
            target_angle = max(0, min(180, target_angle))
            angle_diff = target_angle - current_angle
            
            if abs(angle_diff) <= config.SERVO_SS_STEP_SIZE:
                self.servo_ss.angle = target_angle
                if config.DEBUG_MOVEMENT:
                    self.logger.debug(f"Servo SS moved to {target_angle}°")
                return
            
            step_direction = config.SERVO_SS_STEP_SIZE if angle_diff > 0 else -config.SERVO_SS_STEP_SIZE
            steps = int(abs(angle_diff) / config.SERVO_SS_STEP_SIZE)
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"Moving servo SS from {current_angle}° to {target_angle}° in {steps} steps")
            
            for i in range(steps):
                current_angle += step_direction
                self.servo_ss.angle = int(current_angle)
                time.sleep(0.05)
            
            self.servo_ss.angle = target_angle
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"✅ Servo SS reached {target_angle}°")
                
        except Exception as e:
            self.logger.error(f"Failed to move servo SS incrementally: {e}")

    def servo_ss_to_store(self):
        """Move servo SS to store position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"📦 Moving servo SS to STORE position ({config.SERVO_SS_STORE}°)")
        self.set_servo_ss_incremental(config.SERVO_SS_STORE)

    def servo_ss_to_pre_collect(self):
        """Move servo SS to pre-collect position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🎯 Moving servo SS to PRE-COLLECT position ({config.SERVO_SS_PRE_COLLECT}°)")
        self.set_servo_ss_incremental(config.SERVO_SS_PRE_COLLECT)

    def servo_ss_to_driving(self):
        """Move servo SS to driving position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🚗 Moving servo SS to DRIVING position ({config.SERVO_SS_DRIVING}°)")
        self.set_servo_ss_incremental(config.SERVO_SS_DRIVING)

    def servo_ss_to_collect(self):
        """Move servo SS to collect position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🤏 Moving servo SS to COLLECT position ({config.SERVO_SS_COLLECT}°)")
        self.set_servo_ss_incremental(config.SERVO_SS_COLLECT)

    def get_servo_ss_state(self):
        """Get current servo SS state as string"""
        try:
            current_angle = getattr(self.servo_ss, 'angle', 90)
            if current_angle is None:
                return "unknown"
            
            distances = {
                'store': abs(current_angle - config.SERVO_SS_STORE),
                'pre-collect': abs(current_angle - config.SERVO_SS_PRE_COLLECT),
                'driving': abs(current_angle - config.SERVO_SS_DRIVING),
                'collect': abs(current_angle - config.SERVO_SS_COLLECT)
            }
            
            closest_state = min(distances.keys(), key=lambda k: distances[k])
            return closest_state
            
        except Exception as e:
            self.logger.error(f"Failed to get servo SS state: {e}")
            return "unknown"

    def initialize_servos_for_competition(self):
        """Initialize servos for competition start"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Initializing servos for competition start...")
            self.servo_ss_to_driving()
            self.set_servo_angle(self.servo_sf, config.SERVO_SF_READY_POSITION)
            time.sleep(0.5)
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Servos initialized - SS at driving, SF at ready")
        except Exception as e:
            self.logger.error(f"Failed to initialize servos for competition: {e}")
    
    def center_servos(self):
        """Center servos - SS goes to driving position, SF to ready"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔧 Centering servos...")
            
            if use_gradual:
                self.servo_ss_to_driving()
                time.sleep(sequential_delay)
                self.set_servo_angle_gradual(self.servo_sf, config.SERVO_SF_READY_POSITION)
                time.sleep(0.3)
            else:
                self.servo_ss_to_driving()
                self.set_servo_angle(self.servo_sf, config.SERVO_SF_READY_POSITION)
                time.sleep(0.5)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Servos centered - SS at driving, SF at ready")
                
        except Exception as e:
            self.logger.error(f"Failed to center servos: {e}")
    
    # === ENHANCED COLLECTION SEQUENCE (LEGACY) ===
    def enhanced_collection_sequence(self):
        """Enhanced collection sequence using servo SS four-state system (LEGACY)"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Starting enhanced collection sequence with SS/SF system...")
                self.logger.info("   Flow: driving -> pre-collect -> drive forward -> collect -> store -> driving")
            
            # Step 1: From driving to pre-collect position
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 1: Moving servo SS from DRIVING to PRE-COLLECT")
            self.servo_ss_to_pre_collect()
            time.sleep(0.2)
            
            # Step 2: Drive forward for collection
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 2: Driving forward for collection")
            self.move_forward(duration=config.ENHANCED_COLLECTION_DRIVE_TIME, speed=config.COLLECTION_SPEED)
            time.sleep(0.1)
            
            # Step 3: Move servo SS to collect position (capture ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 3: Moving servo SS to COLLECT position (capture)")
            self.servo_ss_to_collect()
            time.sleep(0.3)
            
            # Step 4: Move servo SS to store position (secure ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 4: Moving servo SS to STORE position (secure)")
            self.servo_ss_to_store()
            time.sleep(0.3)
            
            # Step 5: Move servo SS back to driving position (ready for next action)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 5: Moving servo SS to DRIVING position (ready)")
            self.servo_ss_to_driving()
            time.sleep(0.2)
            
            # Record collection
            self.collected_balls.append(time.time())
            
            if config.DEBUG_COLLECTION:
                current_state = self.get_servo_ss_state()
                self.logger.info(f"✅ Enhanced collection complete! Servo SS state: {current_state.upper()}, Total balls: {len(self.collected_balls)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced collection sequence failed: {e}")
            self.stop_motors()
            self.servo_ss_to_driving()
            return False
    
    # === NEW: OPTIMIZED COLLECTION SEQUENCE FOR COLLECTION ZONE ===
    def optimized_collection_sequence(self):
        """Optimized collection sequence for when ball is already in collection zone"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Starting OPTIMIZED collection sequence for collection zone...")
                self.logger.info(f"   Using: Speed={config.CENTERING_2_COLLECTION_SPEED}, Time={config.CENTERING_2_COLLECTION_TIME}s")
                self.logger.info("   Flow: (pre-collect already set) -> slow drive -> collect -> store -> driving")
            
            # Servo SS should already be in pre-collect position from CENTERING_2
            current_state = self.get_servo_ss_state()
            if current_state != "pre-collect":
                if config.DEBUG_COLLECTION:
                    self.logger.info("Setting servo SS to PRE-COLLECT position")
                self.servo_ss_to_pre_collect()
                time.sleep(0.15)
            
            # Step 1: Slow, precise drive into collection zone
            if config.DEBUG_COLLECTION:
                self.logger.info(f"Step 1: Slow drive for precise collection (speed: {config.CENTERING_2_COLLECTION_SPEED})")
            self.move_forward(duration=config.CENTERING_2_COLLECTION_TIME, speed=config.CENTERING_2_COLLECTION_SPEED)
            time.sleep(0.05)
            
            # Step 2: Quick collect movement
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 2: Moving servo SS to COLLECT position (capture)")
            self.servo_ss_to_collect()
            time.sleep(0.2)
            
            # Step 3: Secure ball in store position
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 3: Moving servo SS to STORE position (secure)")
            self.servo_ss_to_store()
            time.sleep(0.25)
            
            # Step 4: Return to driving position
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 4: Moving servo SS to DRIVING position (ready)")
            self.servo_ss_to_driving()
            time.sleep(0.15)
            
            # Record collection
            self.collected_balls.append(time.time())
            
            if config.DEBUG_COLLECTION:
                current_state = self.get_servo_ss_state()
                self.logger.info(f"✅ Optimized collection complete! Servo SS state: {current_state.upper()}, Total balls: {len(self.collected_balls)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Optimized collection sequence failed: {e}")
            self.stop_motors()
            self.servo_ss_to_driving()
            return False
    
    def release_balls(self):
        """Release all collected balls"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            balls_to_release = len(self.collected_balls)
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"🔓 Releasing {balls_to_release} balls...")
            
            if use_gradual:
                self.servo_ss_to_store()
                time.sleep(sequential_delay)
                self.set_servo_angle_smooth(self.servo_sf, config.SERVO_RELEASE, duration=0.6)
                time.sleep(1.0)
                self.servo_ss_to_driving()
                time.sleep(0.2)
            else:
                self.servo_ss_to_store()
                self.set_servo_angle(self.servo_sf, config.SERVO_RELEASE)
                time.sleep(1.0)
                self.servo_ss_to_driving()
                
            self.collected_balls.clear()
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"✅ Released {balls_to_release} balls - servo SS returned to driving position")
                
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
        """Move robot forward with motor compensation for slow speeds"""
        if speed is None:
            speed = self.current_speed
        
        # Apply motor compensation for slow speeds to prevent drift
        if speed <= getattr(config, 'SLOW_SPEED_THRESHOLD', 0.3):
            left_speed = speed * getattr(config, 'MOTOR_LEFT_COMPENSATION', 1.0)
            right_speed = speed * getattr(config, 'MOTOR_RIGHT_COMPENSATION', 1.0)
            
            # Ensure speeds don't exceed 1.0
            left_speed = min(1.0, left_speed)
            right_speed = min(1.0, right_speed)
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"⬆️ Moving forward with compensation: L={left_speed:.3f}, R={right_speed:.3f}")
        else:
            left_speed = right_speed = speed
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"⬆️ Moving forward at {speed*100:.0f}% speed")
        
        # Left motors (IN1, IN4), Right motors (IN3)  
        self.motor_in1.value = left_speed   # Left motor A
        self.motor_in2.off()
        self.motor_in3.off() 
        self.motor_in4.value = right_speed  # Right motor B
        
        if duration:
            time.sleep(duration)
            self.stop_motors()
    
    def move_backward(self, duration=None, speed=None):
        """Move robot backward"""
        if speed is None:
            speed = self.current_speed
            
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
    
    def delivery_sequence(self, goal_type="B"):
        """Deliver balls to specified goal"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info(f"🚚 Starting ball delivery to goal {goal_type}")
            
            self.stop_motors()
            time.sleep(0.5)
            
            balls_delivered = self.release_balls()
            
            self.move_backward(duration=1.0)
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"✅ Delivered {balls_delivered} balls to goal {goal_type}")
                
            return balls_delivered
            
        except Exception as e:
            self.logger.error(f"❌ Ball delivery failed: {e}")
            return 0
    
    # === SERVO ANGLE GETTERS ===
    def get_servo_angles(self):
        """Get current servo angles - SS and SF only"""
        try:
            return {
                "servo_ss": getattr(self.servo_ss, 'angle', 90),
                "servo_sf": getattr(self.servo_sf, 'angle', 90)
            }
        except Exception as e:
            self.logger.error(f"Failed to get servo angles: {e}")
            return {"servo_ss": 90, "servo_sf": 90}
    
    # === EMERGENCY AND CLEANUP ===
    def emergency_stop(self):
        """Emergency stop all movement"""
        self.stop_motors()
        self.center_servos()
        self.logger.warning("🛑 EMERGENCY STOP activated")
    
    def cleanup(self):
        """Clean shutdown of hardware"""
        try:
            if config.DEBUG_MOVEMENT:
                self.logger.info("🧹 Starting hardware cleanup...")
            
            self.stop_motors()
            self.center_servos()
            
            for component in [self.motor_in1, self.motor_in2, self.motor_in3, self.motor_in4]:
                if hasattr(component, 'close'):
                    component.close()
            
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
        servo_ss_state = self.get_servo_ss_state()
        
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
            'servo_ss_state': servo_ss_state,
            'motor_status': motor_status,
            'gradual_movement': use_gradual,
            'collection_method': 'two_phase_optimized_collection',
            'hardware_ready': True
        }
    
    def log_status_summary(self):
        """Log a comprehensive status summary"""
        status = self.get_status()
        servo_angles = status['servo_angles']
        
        self.logger.info("🔧 HARDWARE STATUS SUMMARY:")
        self.logger.info(f"   Balls collected: {status['collected_balls']}")
        self.logger.info(f"   Current speed: {status['speed_percentage']}")
        self.logger.info(f"   Servo angles: SS={servo_angles['servo_ss']}° ({status['servo_ss_state']}) SF={servo_angles['servo_sf']}°")
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
            
            # Test servo SS four-state system
            self.logger.info("Testing servo SS four-state system...")
            self.servo_ss_to_driving()
            time.sleep(1)
            self.servo_ss_to_pre_collect()
            time.sleep(1)
            self.servo_ss_to_collect()
            time.sleep(1)
            self.servo_ss_to_store()
            time.sleep(1)
            self.servo_ss_to_driving()
            time.sleep(1)
            
            # Test servo SF
            self.logger.info("Testing servo SF...")
            self.set_servo_angle(self.servo_sf, config.SERVO_SF_CATCH_POSITION)
            time.sleep(1)
            
            self.center_servos()
            
            self.logger.info("✅ All systems test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System test failed: {e}")
            return False
