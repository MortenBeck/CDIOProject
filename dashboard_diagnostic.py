#!/usr/bin/env python3
"""
Dashboard Diagnostic Tool - Check why dashboard isn't appearing
"""

import os
import sys
import cv2

def check_dashboard_import():
    """Check if dashboard can be imported"""
    print("=== DASHBOARD IMPORT CHECK ===")
    try:
        from dashboard import GolfBotDashboard
        print("✅ Dashboard import: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ Dashboard import: FAILED - {e}")
        return False
    except Exception as e:
        print(f"❌ Dashboard import: ERROR - {e}")
        return False

def check_display_available():
    """Check if display/X11 is available"""
    print("\n=== DISPLAY AVAILABILITY CHECK ===")
    
    # Check DISPLAY environment variable
    display_env = os.environ.get('DISPLAY')
    print(f"DISPLAY environment: {display_env}")
    
    if display_env is None:
        print("❌ No DISPLAY environment variable - running headless")
        return False
    
    # Try to create a test window
    try:
        cv2.namedWindow('test_window', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('test_window')
        print("✅ OpenCV window creation: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ OpenCV window creation: FAILED - {e}")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n=== DEPENDENCIES CHECK ===")
    
    required_modules = [
        'cv2', 'numpy', 'time', 'logging', 'config'
    ]
    
    all_good = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}: Available")
        except ImportError:
            print(f"❌ {module}: Missing")
            all_good = False
    
    return all_good

def test_minimal_dashboard():
    """Test minimal dashboard functionality"""
    print("\n=== MINIMAL DASHBOARD TEST ===")
    
    try:
        from dashboard import GolfBotDashboard
        dashboard = GolfBotDashboard()
        print("✅ Dashboard creation: SUCCESS")
        
        # Create a test frame
        import numpy as np
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:, :] = (50, 50, 50)  # Dark gray
        
        # Add some text
        cv2.putText(test_frame, "Dashboard Test Frame", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print("✅ Test frame creation: SUCCESS")
        
        # Try to create dashboard
        from enum import Enum
        class TestState(Enum):
            TESTING = "testing"
        
        class MockVision:
            def __init__(self):
                self.arena_detected = True
                self.current_target = None
                self._last_detected_balls = []
                self.detected_walls = []
        
        class MockHardware:
            def get_ball_count(self):
                return 2
            def get_servo_angles(self):
                return {"servo1": 90, "servo2": 90, "servo3": 90}
        
        vision = MockVision()
        hardware = MockHardware()
        state = TestState.TESTING
        
        dashboard_frame = dashboard.create_dashboard(
            test_frame, state, vision, hardware, None
        )
        
        print("✅ Dashboard frame creation: SUCCESS")
        
        # Try to show dashboard
        key = dashboard.show("Dashboard Test")
        print("✅ Dashboard display: SUCCESS")
        print("Press any key in the dashboard window to continue...")
        
        # Wait for keypress
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def check_main_configuration():
    """Check main.py configuration"""
    print("\n=== MAIN.PY CONFIGURATION CHECK ===")
    
    try:
        # Check if config.SHOW_CAMERA_FEED is True
        import config
        show_feed = getattr(config, 'SHOW_CAMERA_FEED', False)
        print(f"SHOW_CAMERA_FEED: {show_feed}")
        if not show_feed:
            print("⚠️  Camera feed is disabled in config")
        
        debug_vision = getattr(config, 'DEBUG_VISION', False)
        print(f"DEBUG_VISION: {debug_vision}")
        
        return True
    except Exception as e:
        print(f"❌ Config check: FAILED - {e}")
        return False

def run_full_diagnostic():
    """Run complete diagnostic"""
    print("🔍 GOLFBOT DASHBOARD DIAGNOSTIC")
    print("=" * 50)
    
    results = {
        'dashboard_import': check_dashboard_import(),
        'display_available': check_display_available(),
        'dependencies': check_dependencies(),
        'config': check_main_configuration()
    }
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY:")
    print("=" * 50)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All checks passed! Testing dashboard...")
        if test_minimal_dashboard():
            print("\n✅ Dashboard should work correctly!")
            print("\nIf dashboard still doesn't appear in main.py:")
            print("1. Make sure you select option '1' (Dashboard Mode)")
            print("2. Check that you're running on a system with display")
            print("3. Try option '2' (Legacy Overlay Mode) as fallback")
        else:
            print("\n❌ Dashboard test failed despite passing checks")
    else:
        print("\n❌ Some checks failed. Dashboard may not work.")
        print("\nTroubleshooting suggestions:")
        
        if not results['display_available']:
            print("- Run on a system with display (not headless)")
            print("- Use SSH with X11 forwarding: ssh -X user@host")
            print("- Use VNC or similar remote desktop")
        
        if not results['dashboard_import']:
            print("- Check that dashboard.py exists in the same directory")
            print("- Verify all dashboard dependencies are installed")
        
        if not results['dependencies']:
            print("- Install missing Python packages")
        
        print("\nAs a workaround, use Legacy Overlay Mode (option 2)")

if __name__ == "__main__":
    try:
        run_full_diagnostic()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\n\nDiagnostic error: {e}")
        import traceback
        traceback.print_exc()