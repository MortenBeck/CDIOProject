import time
import logging
import busio
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import config

class ServoController:
    """Handles all servo control and positioning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_servos()
        
    def setup_servos(self):
        """Initialize PCA9685 and servo connections"""
        try:
            self.logger.info("Initializing PCA9685 for servo control...")
            
            # Setup PCA9685 for servo control
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c, address=config.PCA9685_ADDRESS)
            self.pca.frequency = config.PCA9685_FREQUENCY
            
            # Setup servos - only two now: SS and SF
            self.servo_ss = servo.Servo(self.pca.channels[config.SERVO_SS_CHANNEL])
            self.servo_sf = servo.Servo(self.pca.channels[config.SERVO_SF_CHANNEL]) 
            
            self.logger.info("✅ Servos initialized successfully")
            self.logger.info(f"✓ Servo SS on channel {config.SERVO_SS_CHANNEL}")
            self.logger.info(f"✓ Servo SF on channel {config.SERVO_SF_CHANNEL}")
            
        except Exception as e:
            self.logger.error(f"Servo initialization failed: {e}")
            raise
    
    # === BASIC SERVO CONTROL ===
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

    # === SERVO SS (SERVO 1) FOUR-STATE SYSTEM ===
    def set_servo_ss_incremental(self, target_angle):
        """Move servo SS incrementally by 5-degree steps"""
        try:
            current_angle = getattr(self.servo_ss, 'angle', 90)
            if current_angle is None:
                current_angle = 90
            
            target_angle = max(0, min(180, target_angle))
            angle_diff = target_angle - current_angle
            
            if abs(angle_diff) <= config.SERVO_SS_STEP_SIZE:
                # Close enough, move directly
                self.servo_ss.angle = target_angle
                if config.DEBUG_MOVEMENT:
                    self.logger.debug(f"Servo SS moved to {target_angle}°")
                return
            
            # Move in incremental steps
            step_direction = config.SERVO_SS_STEP_SIZE if angle_diff > 0 else -config.SERVO_SS_STEP_SIZE
            steps = int(abs(angle_diff) / config.SERVO_SS_STEP_SIZE)
            
            if config.DEBUG_MOVEMENT:
                self.logger.debug(f"Moving servo SS from {current_angle}° to {target_angle}° in {steps} steps of {config.SERVO_SS_STEP_SIZE}°")
            
            for i in range(steps):
                current_angle += step_direction
                self.servo_ss.angle = int(current_angle)
                time.sleep(0.05)  # Small delay between steps
            
            # Final position adjustment
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
            
            # Determine which state we're closest to
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

    # === SERVO SF (SERVO 2) CONTROL ===
    def servo_sf_to_ready(self):
        """Move servo SF to ready position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🔧 Moving servo SF to READY position ({config.SERVO_SF_READY}°)")
        self.set_servo_angle_gradual(self.servo_sf, config.SERVO_SF_READY)

    def servo_sf_to_catch(self):
        """Move servo SF to catch position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🤏 Moving servo SF to CATCH position ({config.SERVO_SF_CATCH}°)")
        self.set_servo_angle_gradual(self.servo_sf, config.SERVO_SF_CATCH)

    def servo_sf_to_release(self):
        """Move servo SF to release position"""
        if config.DEBUG_COLLECTION:
            self.logger.info(f"🔓 Moving servo SF to RELEASE position ({config.SERVO_SF_RELEASE}°)")
        self.set_servo_angle_gradual(self.servo_sf, config.SERVO_SF_RELEASE)

    # === COMBINED SERVO OPERATIONS ===
    def initialize_servos_for_competition(self):
        """Initialize both servos for competition start"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Initializing servos for competition start...")
            self.servo_ss_to_driving()
            time.sleep(0.3)
            self.servo_sf_to_ready()
            time.sleep(0.5)  # Allow time to settle
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Servos initialized - SS at driving, SF at ready")
        except Exception as e:
            self.logger.error(f"Failed to initialize servos for competition: {e}")
    
    def center_servos(self):
        """Center both servos - SS goes to driving position, SF to ready"""
        try:
            # Check if gradual movement is enabled
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔧 Centering servos...")
            
            if use_gradual:
                # Move SS to driving position using incremental movement
                self.servo_ss_to_driving()
                time.sleep(sequential_delay)
                # Move SF to ready position
                self.servo_sf_to_ready()
                time.sleep(0.3)  # Final settling time
            else:
                # Original immediate movement
                self.servo_ss_to_driving()
                self.servo_sf_to_ready()
                time.sleep(0.5)  # Allow time to reach position
            
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Servos centered - SS at driving, SF at ready")
                
        except Exception as e:
            self.logger.error(f"Failed to center servos: {e}")
    
    def get_servo_angles(self):
        """Get current servo angles - ensures no None values"""
        try:
            ss_angle = getattr(self.servo_ss, 'angle', None)
            sf_angle = getattr(self.servo_sf, 'angle', None)
            
            # Ensure we never return None - use default values
            ss_angle = 90 if ss_angle is None else ss_angle
            sf_angle = 90 if sf_angle is None else sf_angle
            
            return {
                "servo_ss": ss_angle,
                "servo_sf": sf_angle
            }
        except Exception as e:
            self.logger.error(f"Failed to get servo angles: {e}")
            # Return safe default values instead of None
            return {"servo_ss": 90, "servo_sf": 90}
    
    def test_servos(self):
        """Test all servo systems"""
        try:
            self.logger.info("Testing servo SS four-state system...")
            self.servo_ss_to_driving()
            time.sleep(1)
            self.servo_ss_to_pre_collect()
            time.sleep(1)
            self.servo_ss_to_collect()
            time.sleep(1)
            self.servo_ss_to_store()
            time.sleep(1)
            self.servo_ss_to_driving()  # Return to driving
            time.sleep(1)
            
            # Test servo SF
            self.logger.info("Testing servo SF...")
            self.servo_sf_to_ready()
            time.sleep(1)
            self.servo_sf_to_catch()
            time.sleep(1)
            self.servo_sf_to_release()
            time.sleep(1)
            self.servo_sf_to_ready()  # Return to ready
            time.sleep(1)
            
            # Return to center
            self.center_servos()
            
            self.logger.info("✅ Servo test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Servo test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean shutdown of servos"""
        try:
            if config.DEBUG_MOVEMENT:
                self.logger.info("🧹 Cleaning up servos...")
            
            self.center_servos()  # This will set SS to driving, SF to ready
            
            # Deinitialize PCA9685
            if hasattr(self.pca, 'deinit'):
                self.pca.deinit()
            
            self.logger.info("✅ Servo cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Servo cleanup failed: {e}")