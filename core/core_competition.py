import time
import logging
from typing import Optional
import config

class CompetitionManager:
    """Manages competition timing, scoring, and overall flow"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.is_active = False
        self.time_limit = config.COMPETITION_TIME
        
    def start_competition(self):
        """Start the competition timer"""
        self.start_time = time.time()
        self.end_time = None
        self.is_active = True
        
        self.logger.info("COMPETITION STARTED!")
        self.logger.info(f"Time limit: {self.time_limit} seconds ({self.time_limit//60}:{self.time_limit%60:02d})")
        self.logger.info("Using two-phase collection: Centering_1 + Centering_2 + Collection")
    
    def end_competition(self, hardware=None):
        """End the competition and log results"""
        self.end_time = time.time()
        self.is_active = False
        
        elapsed_time = self.get_elapsed_time()
        
        self.logger.info("=" * 60)
        self.logger.info("COMPETITION ENDED!")
        self.logger.info("=" * 60)
        self.logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time//60:.0f}:{elapsed_time%60:02.0f})")
        
        if hardware:
            balls_collected = hardware.get_ball_count()
            self.logger.info(f"Balls collected: {balls_collected}")
            
            # Calculate basic score
            score = self._calculate_basic_score(balls_collected, elapsed_time)
            self.logger.info(f"Basic score: {score} points")
        
        self.logger.info(f"Collection system: Two-Phase (Centering_1 + Centering_2 + Optimized Collection)")
        self.logger.info("=" * 60)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed competition time"""
        if not self.start_time:
            return 0.0
        
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time
    
    def get_time_remaining(self) -> float:
        """Get remaining competition time"""
        if not self.is_active or not self.start_time:
            return self.time_limit
        
        elapsed = self.get_elapsed_time()
        remaining = max(0, self.time_limit - elapsed)
        return remaining
    
    def is_time_up(self) -> bool:
        """Check if competition time is up"""
        return self.get_time_remaining() <= 0
    
    def get_time_remaining_formatted(self) -> str:
        """Get formatted time remaining string"""
        remaining = self.get_time_remaining()
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes}:{seconds:02d}"
    
    def should_enter_endgame_mode(self) -> bool:
        """Check if should enter endgame mode (e.g., focus on delivery)"""
        remaining = self.get_time_remaining()
        return remaining <= 120  # Last 2 minutes
    
    def _calculate_basic_score(self, balls_collected: int, elapsed_time: float) -> int:
        """Calculate basic score based on balls collected and time"""
        # Basic scoring - can be enhanced later
        ball_score = balls_collected * config.GOAL_B_POINTS  # Assume Goal B for basic calculation
        
        # Time bonus for remaining time
        remaining_time = max(0, self.time_limit - elapsed_time)
        time_bonus = int(remaining_time * config.TIME_BONUS_PER_SECOND)
        
        total_score = ball_score + time_bonus
        return total_score
    
    def get_competition_status(self) -> dict:
        """Get comprehensive competition status"""
        return {
            'is_active': self.is_active,
            'elapsed_time': self.get_elapsed_time(),
            'remaining_time': self.get_time_remaining(),
            'remaining_formatted': self.get_time_remaining_formatted(),
            'time_limit': self.time_limit,
            'is_time_up': self.is_time_up(),
            'endgame_mode': self.should_enter_endgame_mode(),
            'start_time': self.start_time,
            'end_time': self.end_time
        }
    
    def log_status_update(self, hardware=None):
        """Log periodic status update"""
        if not self.is_active:
            return
        
        remaining = self.get_time_remaining()
        elapsed = self.get_elapsed_time()
        
        status_msg = f"⏱️  Time: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining"
        
        if hardware:
            balls = hardware.get_ball_count()
            status_msg += f" | Balls: {balls}"
        
        if self.should_enter_endgame_mode():
            status_msg += " | ENDGAME MODE"
        
        self.logger.info(status_msg)
    
    def check_time_warnings(self):
        """Check and log time warnings"""
        remaining = self.get_time_remaining()
        
        # Warning thresholds (in seconds)
        warnings = [300, 180, 120, 60, 30, 10]  # 5min, 3min, 2min, 1min, 30s, 10s
        
        for warning_time in warnings:
            # Check if we just crossed this threshold
            if warning_time - 1 < remaining <= warning_time:
                if warning_time >= 60:
                    time_str = f"{warning_time//60} minute{'s' if warning_time > 60 else ''}"
                else:
                    time_str = f"{warning_time} seconds"
                
                self.logger.warning(f"⚠️  {time_str} remaining!")
                break
