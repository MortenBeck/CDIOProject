import time
import logging
from gpiozero import PWMOutputDevice
import config

class MotorController:
    """Handles all motor movement and control with power management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_speed = config.DEFAULT_SPEED
        self.motors_powered = False
        self.setup_motors()
        
    def setup_motors(self):
        """Initialize motor GPIO connections and turn motors ON"""
        try:
            self.logger.info("Initializing motor controllers...")
            
            # Setup motors with PWM for speed control
            self.motor_in1 = PWMOutputDevice(config.MOTOR_IN1)
            self.motor_in2 = PWMOutputDevice(config.MOTOR_IN2)
            self.motor_in3 = PWMOutputDevice(config.MOTOR_IN3)
            self.motor_in4 = PWMOutputDevice(config.MOTOR_IN4)
            
            # Start with motors stopped but powered
            self.stop_motors()
            self.power_on_motors()
            
            self.logger.info("✅ Motors initialized successfully")
            self.logger.info(f"✓ Motor A: GPIO {config.MOTOR_IN1}, {config.MOTOR_IN2}")
            self.logger.info(f"✓ Motor B: GPIO {config.MOTOR_IN3}, {config.MOTOR_IN4}")
            
        except Exception as e:
            self.logger.error(f"Motor initialization failed: {e}")
            raise
    
    def power_on_motors(self):
        """Explicitly turn motors ON for operation"""
        self.motors_powered = True
        if config.DEBUG_MOVEMENT:
            self.logger.info("🔋 Motors POWERED ON - ready for operation")
    
    def power_off_motors(self):
        """Explicitly turn motors OFF for power saving"""
        self.stop_motors()
        self.motors_powered = False
        if config.DEBUG_MOVEMENT:
            self.logger.info("🔋 Motors POWERED OFF - saving battery")
    
    def stop_motors(self):
        """Stop all motors"""
        if hasattr(self, 'motor_in1'):  # Check if motors are initialized
            self.motor_in1.off()
            self.motor_in2.off()
            self.motor_in3.off()
            self.motor_in4.off()
        if config.DEBUG_MOVEMENT:
            self.logger.debug("🛑 Motors stopped")
    
    def _check_motors_powered(self):
        """Check if motors are powered before movement"""
        if not self.motors_powered:
            self.logger.warning("⚠️ Attempted motor movement while powered OFF")
            return False
        return True
    
    def move_forward(self, duration=None, speed=None):
        """Move robot forward"""
        if not self._check_motors_powered():
            return
            
        if speed is None:
            speed = self.current_speed
            
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
        if not self._check_motors_powered():
            return
            
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
        if not self._check_motors_powered():
            return
            
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
        if not self._check_motors_powered():
            return
            
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
    
    def get_motor_status(self):
        """Get current motor status"""
        return {
            'in1_active': self.motor_in1.is_active if hasattr(self.motor_in1, 'is_active') else False,
            'in2_active': self.motor_in2.is_active if hasattr(self.motor_in2, 'is_active') else False,
            'in3_active': self.motor_in3.is_active if hasattr(self.motor_in3, 'is_active') else False,
            'in4_active': self.motor_in4.is_active if hasattr(self.motor_in4, 'is_active') else False,
            'current_speed': self.current_speed,
            'speed_percentage': f"{self.current_speed*100:.0f}%",
            'motors_powered': self.motors_powered
        }
    
    def test_motors(self):
        """Test all motor movements"""
        try:
            if not self.motors_powered:
                self.logger.warning("Motors not powered - turning on for test")
                self.power_on_motors()
                
            self.logger.info("Testing motors...")
            self.move_forward(duration=0.2)
            time.sleep(0.2)
            self.move_backward(duration=0.2)
            time.sleep(0.2)
            self.turn_right(duration=0.2)
            time.sleep(0.2)
            self.turn_left(duration=0.2)
            time.sleep(0.2)
            self.logger.info("✅ Motor test completed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Motor test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean shutdown of motors with power OFF"""
        try:
            if config.DEBUG_MOVEMENT:
                self.logger.info("🧹 Cleaning up motors...")
            
            self.power_off_motors()
            
            for component in [self.motor_in1, self.motor_in2, self.motor_in3, self.motor_in4]:
                if hasattr(component, 'close'):
                    component.close()
            
            self.logger.info("✅ Motor cleanup completed - motors POWERED OFF")
            
        except Exception as e:
            self.logger.error(f"❌ Motor cleanup failed: {e}")