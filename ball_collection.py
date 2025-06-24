import time
import logging
import config

class BallCollectionSystem:
    """Handles ball collection sequences and tracking"""
    
    def __init__(self, motor_controller, servo_controller):
        self.logger = logging.getLogger(__name__)
        self.motors = motor_controller
        self.servos = servo_controller
        self.collected_balls = []
        
    def enhanced_collection_sequence(self):
        """Enhanced collection sequence using SS only - SF stays closed"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Starting enhanced collection sequence with SS only...")
                self.logger.info("   Flow: SS driving -> pre-collect -> drive forward -> collect -> store -> collect -> store -> driving")
                self.logger.info("   SF: stays closed during collection")
            
            # Step 1: Move SS from driving to pre-collect position
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 1: Moving SS from DRIVING to PRE-COLLECT")
            self.servos.servo_ss_to_pre_collect()
            time.sleep(0.2)
            
            # Step 2: Drive forward for collection
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 2: Driving forward for collection")
            self.motors.move_forward(duration=config.FIXED_COLLECTION_DRIVE_TIME, speed=config.COLLECTION_SPEED)
            time.sleep(0.1)
            
            # Step 3: SS captures ball (first collect)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 3: SS collect position (first capture)")
            self.servos.servo_ss_to_collect()
            time.sleep(0.3)
            
            # Step 4: Move SS to store position (secure ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 4: Moving SS to STORE position (secure first ball)")
            self.servos.servo_ss_to_store()
            time.sleep(0.3)
            
            # Step 5: SS captures ball again (second collect)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 5: SS collect position (second capture)")
            self.servos.servo_ss_to_collect()
            time.sleep(0.3)
            
            # Step 6: Move SS to store position again (secure ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 6: Moving SS to STORE position (secure second ball)")
            self.servos.servo_ss_to_store()
            time.sleep(0.3)
            
            # Step 7: Return SS to driving position
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 7: Returning SS to driving position")
            self.servos.servo_ss_to_driving()
            time.sleep(0.2)
            
            # Record collection
            self.collected_balls.append(time.time())
            
            if config.DEBUG_COLLECTION:
                ss_state = self.servos.get_servo_ss_state()
                self.logger.info(f"✅ Enhanced collection complete! SS state: {ss_state.upper()}, Total balls: {len(self.collected_balls)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced collection sequence failed: {e}")
            self.motors.stop_motors()
            # Ensure we return to ready positions on error
            self.servos.servo_ss_to_driving()
            return False
    
    def collection_position(self):
        """Move servos to ball collection position (legacy method)"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔧 Moving to legacy collection position...")
            
            if use_gradual:
                self.servos.set_servo_angle_gradual(self.servos.servo_ss, config.SERVO_COLLECT_OPEN)
                time.sleep(0.3)
            else:
                self.servos.set_servo_angle(self.servos.servo_ss, config.SERVO_COLLECT_OPEN)
                time.sleep(0.5)
                
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Legacy collection position set")
                
        except Exception as e:
            self.logger.error(f"Failed to set collection position: {e}")
    
    def grab_ball(self):
        """Close servos to grab a ball (legacy method)"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔒 Grabbing ball with legacy method...")
            
            if use_gradual:
                self.servos.set_servo_angle_smooth(self.servos.servo_ss, config.SERVO_COLLECT_CLOSE, duration=0.4)
                time.sleep(0.5)  # Give time to secure ball
            else:
                self.servos.set_servo_angle(self.servos.servo_ss, config.SERVO_COLLECT_CLOSE)
                time.sleep(0.8)  # Give time to secure ball
                
            self.collected_balls.append(time.time())  # Track collection time
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"✅ Ball grabbed! Total collected: {len(self.collected_balls)}")
                
        except Exception as e:
            self.logger.error(f"Failed to grab ball: {e}")
    
    def release_balls(self):
        """Release all collected balls - opens SF for delivery"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            balls_to_release = len(self.collected_balls)
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"🔓 Releasing {balls_to_release} balls for delivery...")
            
            if use_gradual:
                # Move SS to store position for release
                self.servos.servo_ss_to_store()
                time.sleep(sequential_delay)
                # Open SF for ball release
                self.servos.servo_sf_to_open()
                time.sleep(1.5)  # Allow balls to fall out
                
                # Return servos to ready positions after release
                self.servos.servo_ss_to_driving()
                time.sleep(0.1)
                self.servos.servo_sf_to_closed()  # Close SF after delivery
                time.sleep(0.2)
            else:
                self.servos.servo_ss_to_store()
                self.servos.servo_sf_to_open()
                time.sleep(1.5)  # Allow balls to fall out
                self.servos.servo_ss_to_driving()  # Return to driving
                self.servos.servo_sf_to_closed()   # Close SF after delivery
                
            self.collected_balls.clear()
            
            if config.DEBUG_COLLECTION:
                self.logger.info(f"✅ Released {balls_to_release} balls - servos returned to ready positions")
                
            return balls_to_release
            
        except Exception as e:
            self.logger.error(f"Failed to release balls: {e}")
            return 0
    
    def get_ball_count(self):
        """Get number of collected balls"""
        return len(self.collected_balls)
    
    def has_balls(self):
        """Check if robot has collected balls"""
        return len(self.collected_balls) > 0
    
    def get_collection_status(self):
        """Get detailed collection status"""
        return {
            'balls_collected': len(self.collected_balls),
            'collection_times': self.collected_balls.copy(),
            'has_balls': self.has_balls(),
            'collection_method': 'enhanced_ss_only_collection'
        }