import logging
from motor_controller import MotorController
from servo_controller import ServoController
from ball_collection import BallCollectionSystem
import config

class GolfBotHardware:
    """Main hardware interface combining all subsystems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_hardware()
        
    def setup_hardware(self):
        """Initialize all hardware subsystems"""
        try:
            self.logger.info("Initializing GolfBot hardware subsystems...")
            
            # Initialize individual controllers
            self.motor_controller = MotorController()
            self.servo_controller = ServoController()
            self.ball_collection = BallCollectionSystem(
                self.motor_controller, 
                self.servo_controller
            )
            
            # Explicitly turn on motors after initialization
            self.logger.info("🔋 Turning ON motors for operation")
            
            self.logger.info("✅ All hardware subsystems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            raise
    
    # === MOTOR CONTROL DELEGATION ===
    def stop_motors(self):
        """Stop all motors"""
        return self.motor_controller.stop_motors()
    
    def move_forward(self, duration=None, speed=None):
        """Move robot forward"""
        return self.motor_controller.move_forward(duration, speed)
    
    def move_backward(self, duration=None, speed=None):
        """Move robot backward"""
        return self.motor_controller.move_backward(duration, speed)
    
    def turn_right(self, duration=None, speed=None):
        """Turn robot right"""
        return self.motor_controller.turn_right(duration, speed)
    
    def turn_left(self, duration=None, speed=None):
        """Turn robot left"""
        return self.motor_controller.turn_left(duration, speed)
    
    def set_speed(self, speed):
        """Set default movement speed (0.0 to 1.0)"""
        return self.motor_controller.set_speed(speed)
    
    @property
    def current_speed(self):
        """Get current speed"""
        return self.motor_controller.current_speed
    
    # === SERVO SS CONTROL DELEGATION ===
    def servo_ss_to_store(self):
        """Move servo SS to store position"""
        return self.servo_controller.servo_ss_to_store()
    
    def servo_ss_to_pre_collect(self):
        """Move servo SS to pre-collect position"""
        return self.servo_controller.servo_ss_to_pre_collect()
    
    def servo_ss_to_driving(self):
        """Move servo SS to driving position"""
        return self.servo_controller.servo_ss_to_driving()
    
    def servo_ss_to_collect(self):
        """Move servo SS to collect position"""
        return self.servo_controller.servo_ss_to_collect()
    
    def get_servo_ss_state(self):
        """Get current servo SS state"""
        return self.servo_controller.get_servo_ss_state()
    
    # === SERVO SF CONTROL DELEGATION ===
    def servo_sf_to_open(self):
        """Move servo SF to open position (for delivery only)"""
        return self.servo_controller.servo_sf_to_open()
    
    def servo_sf_to_closed(self):
        """Move servo SF to closed position (default state)"""
        return self.servo_controller.servo_sf_to_closed()
    
    def get_servo_sf_state(self):
        """Get current servo SF state"""
        return self.servo_controller.get_servo_sf_state()
    
    # === LEGACY SF METHODS FOR BACKWARD COMPATIBILITY ===
    def servo_sf_to_ready(self):
        """Legacy method - redirects to closed"""
        return self.servo_controller.servo_sf_to_closed()
    
    def servo_sf_to_catch(self):
        """Legacy method - redirects to closed"""
        return self.servo_controller.servo_sf_to_closed()
    
    def servo_sf_to_release(self):
        """Legacy method - redirects to open"""
        return self.servo_controller.servo_sf_to_open()
    
    # === SERVO SYSTEM OPERATIONS ===
    def initialize_servos_for_competition(self):
        """Initialize both servos for competition start - SS driving, SF closed"""
        return self.servo_controller.initialize_servos_for_competition()
    
    def center_servos(self):
        """Center both servos - SS driving, SF closed"""
        return self.servo_controller.center_servos()
    
    def get_servo_angles(self):
        """Get current servo angles"""
        return self.servo_controller.get_servo_angles()
    
    # === BALL COLLECTION DELEGATION ===
    def enhanced_collection_sequence(self):
        """Execute enhanced ball collection sequence"""
        return self.ball_collection.enhanced_collection_sequence()
    
    def collection_position(self):
        """Move to collection position (legacy)"""
        return self.ball_collection.collection_position()
    
    def grab_ball(self):
        """Grab ball (legacy method)"""
        return self.ball_collection.grab_ball()
    
    def release_balls(self):
        """Release all collected balls"""
        return self.ball_collection.release_balls()
    
    def get_ball_count(self):
        """Get number of collected balls"""
        return self.ball_collection.get_ball_count()
    
    def has_balls(self):
        """Check if robot has collected balls"""
        return self.ball_collection.has_balls()
    
    @property
    def collected_balls(self):
        """Get collected balls list for backward compatibility"""
        return self.ball_collection.collected_balls
    
    # === SYSTEM-WIDE OPERATIONS ===
    def emergency_stop(self):
        """Emergency stop all movement"""
        self.motor_controller.stop_motors()
        self.servo_controller.center_servos()
        self.logger.warning("🛑 EMERGENCY STOP activated")
    
    def get_status(self):
        """Get comprehensive hardware status"""
        motor_status = self.motor_controller.get_motor_status()
        servo_angles = self.servo_controller.get_servo_angles()
        servo_ss_state = self.servo_controller.get_servo_ss_state()
        servo_sf_state = self.servo_controller.get_servo_sf_state()
        collection_status = self.ball_collection.get_collection_status()
        
        use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
        
        return {
            'collected_balls': collection_status['balls_collected'],
            'current_speed': motor_status['current_speed'],
            'speed_percentage': motor_status['speed_percentage'],
            'servo_angles': servo_angles,
            'servo_ss_state': servo_ss_state,
            'servo_sf_state': servo_sf_state,
            'motor_status': {
                'in1_active': motor_status['in1_active'],
                'in2_active': motor_status['in2_active'],
                'in3_active': motor_status['in3_active'],
                'in4_active': motor_status['in4_active'],
            },
            'gradual_movement': use_gradual,
            'collection_method': collection_status['collection_method'],
            'hardware_ready': True
        }
    
    def log_status_summary(self):
        """Log a comprehensive status summary"""
        status = self.get_status()
        servo_angles = status['servo_angles']
        
        self.logger.info("🔧 HARDWARE STATUS SUMMARY:")
        self.logger.info(f"   Balls collected: {status['collected_balls']}")
        self.logger.info(f"   Current speed: {status['speed_percentage']}")
        self.logger.info(f"   Servo angles: SS={servo_angles['servo_ss']}° ({status['servo_ss_state']}) SF={servo_angles['servo_sf']}° ({status['servo_sf_state']})")
        self.logger.info(f"   Gradual movement: {status['gradual_movement']}")
        self.logger.info(f"   Collection method: {status['collection_method']}")
    
    def test_all_systems(self):
        """Test all hardware systems"""
        try:
            self.logger.info("🧪 Testing all hardware systems...")
            
            # Test motors
            motor_test_result = self.motor_controller.test_motors()
            if not motor_test_result:
                return False
            
            # Test servos
            servo_test_result = self.servo_controller.test_servos()
            if not servo_test_result:
                return False
            
            self.logger.info("✅ All systems test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean shutdown of all hardware"""
        try:
            if config.DEBUG_MOVEMENT:
                self.logger.info("🧹 Starting hardware cleanup...")
            
            # Explicitly turn OFF motors before cleanup
            self.logger.info("🔋 Turning OFF motors for power saving")
            self.motor_controller.stop_motors()
            
            # Clean up subsystems
            self.motor_controller.cleanup()
            self.servo_controller.cleanup()
            
            self.logger.info("✅ Hardware cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Hardware cleanup failed: {e}")