import time
import logging
from gpiozero import PWMOutputDevice
import config

class MotorController:
    """Handles all motor movement and control"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_speed = config.DEFAULT_SPEED
        self.setup_motors()
        
    def setup_motors(self):
        """Initialize motor GPIO connections"""
        try:
            self.logger.info("Initializing motor controllers...")
            
            # Setup motors with PWM for speed control
            self.motor_in1 = PWMOutputDevice(config.MOTOR_IN1)
            self.motor_in2 = PWMOutputDevice(config.MOTOR_IN2)
            self.motor_in3 = PWMOutputDevice(config.MOTOR_IN3)
            self.motor_in4 = PWMOutputDevice(config.MOTOR_IN4)
            
            # Start with motors stopped
            self.stop_motors()
            
            self.logger.info("✅ Motors initialized successfully")
            self.logger.info(f"✓ Motor A: GPIO {config.MOTOR_IN1}, {config.MOTOR_IN2}")
            self.logger.info(f"✓ Motor B: GPIO {config.MOTOR_IN3}, {config.MOTOR_IN4}")
            
        except Exception as e:
            self.logger.error(f"Motor initialization failed: {e}")
            raise
    
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
    
    def get_motor_status(self):
        """Get current motor status"""
        return {
            'in1_active': self.motor_in1.is_active if hasattr(self.motor_in1, 'is_active') else False,
            'in2_active': self.motor_in2.is_active if hasattr(self.motor_in2, 'is_active') else False,
            'in3_active': self.motor_in3.is_active if hasattr(self.motor_in3, 'is_active') else False,
            'in4_active': self.motor_in4.is_active if hasattr(self.motor_in4, 'is_active') else False,
            'current_speed': self.current_speed,
            'speed_percentage': f"{self.current_speed*100:.0f}%"
        }
    
    def test_motors(self):
        """Test all motor movements"""
        try:
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
        """Clean shutdown of motors"""
        try:
            if config.DEBUG_MOVEMENT:
                self.logger.info("🧹 Cleaning up motors...")
            
            self.stop_motors()
            
            # Close motor GPIO connections
            for component in [self.motor_in1, self.motor_in2, self.motor_in3, self.motor_in4]:
                if hasattr(component, 'close'):
                    component.close()
            
            self.logger.info("✅ Motor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Motor cleanup failed: {e}")