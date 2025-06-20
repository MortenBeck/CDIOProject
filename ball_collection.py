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
        """New collection sequence using SS four-state system and SF assist"""
        try:
            if config.DEBUG_COLLECTION:
                self.logger.info("🚀 Starting enhanced collection sequence with two-servo system...")
                self.logger.info("   Flow: SS driving -> pre-collect -> drive forward -> collect -> store -> driving")
                self.logger.info("   SF: ready -> catch -> ready")
            
            # Step 1: Prepare SF for catching
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 1: Preparing SF for catching")
            self.servos.servo_sf_to_ready()
            time.sleep(0.2)
            
            # Step 2: Move SS from driving to pre-collect position
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 2: Moving SS from DRIVING to PRE-COLLECT")
            self.servos.servo_ss_to_pre_collect()
            time.sleep(0.2)
            
            # Step 3: Drive forward for collection
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 3: Driving forward for collection")
            self.motors.move_forward(duration=1.05, speed=config.COLLECTION_SPEED)
            time.sleep(0.1)
            
            # Step 4: Coordinate collection - SS captures, SF assists
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 4: Coordinated collection - SS collect, SF catch")
            self.servos.servo_ss_to_collect()
            time.sleep(0.15)
            self.servos.servo_sf_to_catch()
            time.sleep(0.3)
            
            # Step 5: Move SS to store position (secure ball)
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 5: Moving SS to STORE position (secure)")
            self.servos.servo_ss_to_store()
            time.sleep(0.3)
            
            # Step 6: Return both servos to ready positions
            if config.DEBUG_COLLECTION:
                self.logger.info("Step 6: Returning servos to ready positions")
            self.servos.servo_ss_to_driving()
            time.sleep(0.1)
            self.servos.servo_sf_to_ready()
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
            self.servos.servo_sf_to_ready()
            return False
    
    def collection_position(self):
        """Move servos to ball collection position (legacy method)"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔧 Moving to legacy collection position...")
            
            if use_gradual:
                self.servos.set_servo_angle_gradual(self.servos.servo_ss, config.SERVO_COLLECT_OPEN)
                time.sleep(sequential_delay)
                self.servos.set_servo_angle_gradual(self.servos.servo_sf, config.SERVO_COLLECT_OPEN)
                time.sleep(0.3)
            else:
                self.servos.set_servo_angle(self.servos.servo_ss, config.SERVO_COLLECT_OPEN)
                self.servos.set_servo_angle(self.servos.servo_sf, config.SERVO_COLLECT_OPEN)
                time.sleep(0.5)
                
            if config.DEBUG_COLLECTION:
                self.logger.info("✅ Legacy collection position set")
                
        except Exception as e:
            self.logger.error(f"Failed to set collection position: {e}")
    
    def grab_ball(self):
        """Close servos to grab a ball (legacy method)"""
        try:
            use_gradual = getattr(config, 'SERVO_GRADUAL_MOVEMENT', True)
            sequential_delay = getattr(config, 'SERVO_SEQUENTIAL_DELAY', 0.1)
            
            if config.DEBUG_COLLECTION:
                self.logger.info("🔒 Grabbing ball with legacy method...")
            
            if use_gradual:
                # Move servos sequentially for smoother operation
                self.servos.set_servo_angle_smooth(self.servos.servo_ss, config.SERVO_COLLECT_CLOSE, duration=0.4)
                time.sleep(sequential_delay)
                self.servos.set_servo_angle_smooth(self.servos.servo_sf, config.SERVO_COLLECT_CLOSE, duration=0.4)
                time.sleep(0.5)  # Give time to secure ball
            else:
                self.servos.set_servo_angle(self.servos.servo_ss, config.SERVO_COLLECT_CLOSE)
                self.servos.set_servo_angle(self.servos.servo_sf, config.SERVO_COLLECT_CLOSE)
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
                # Move SS to store position for release
                self.servos.servo_ss_to_store()
                time.sleep(sequential_delay)
                # Move SF to release position
                self.servos.servo_sf_to_release()
                time.sleep(1.0)  # Allow balls to fall out
                
                # Return servos to ready positions after release
                self.servos.servo_ss_to_driving()
                time.sleep(0.1)
                self.servos.servo_sf_to_ready()
                time.sleep(0.2)
            else:
                self.servos.servo_ss_to_store()
                self.servos.servo_sf_to_release()
                time.sleep(1.0)  # Allow balls to fall out
                self.servos.servo_ss_to_driving()  # Return to driving
                self.servos.servo_sf_to_ready()    # Return to ready
                
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
            'collection_method': 'enhanced_two_servo_collection'
        }