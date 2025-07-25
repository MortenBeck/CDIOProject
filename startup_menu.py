import config

def show_startup_menu():
    """Show startup menu with options"""
    print("\n" + "="*60)
    print("🤖 GOLFBOT CONTROL SYSTEM - WHITE BALL COLLECTION ONLY")
    print("="*60)
    print("1. Start Competition (Dashboard Mode)")
    print("2. Start Competition (Legacy Overlay Mode)")
    print("3. Hardware Testing") 
    print("4. Delivery System Test (Green Target Detection)")
    print("5. Exit")
    print("="*60)
    print("FEATURES:")
    print("• White ball detection and collection only")
    print("• Ball centering before collection (X+Y axis)")
    print("• Enhanced servo collection sequence")
    print("• Faster centering adjustments (2x speed)")
    print("• Modular boundary avoidance system")
    print("• Clean dashboard interface (option 1) - Camera + Side panels")
    print("• Legacy overlay mode (option 2) - All info on camera")
    print("• NEW: Green delivery zone detection and navigation (option 4)")
    print("="*60)
    
    while True:
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == '1':
                return 'competition_dashboard'
            elif choice == '2':
                return 'competition_legacy'
            elif choice == '3':
                return 'testing'
            elif choice == '4':
                return 'delivery'
            elif choice == '5':
                return 'exit'
            else:
                print("Invalid choice. Enter 1, 2, 3, 4, or 5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return 'exit'
        except EOFError:
            print("\nExiting...")
            return 'exit'
        except Exception as e:
            print(f"Input error: {e}")
            return 'exit'

def show_competition_info(use_dashboard):
    """Show competition information before starting"""
    interface_mode = "Dashboard" if use_dashboard else "Legacy Overlay"
    print(f"\n🏁 Entering Competition Mode with {interface_mode} Interface...")
    
    print("\n🚀 Robot ready with enhanced white ball collection system!")
    print("   - White ball detection and collection only")
    print("   - Ball centering for precision targeting (X+Y axis)")
    print("   - Enhanced servo collection sequence") 
    print("   - HoughCircles + Arena boundary detection")
    print("   - Enhanced servo control with gradual movement")
    print("   - Modular boundary avoidance system")
    print(f"   - {interface_mode} interface for monitoring")
    print(f"\n⚙️  Configuration:")
    print(f"   - X-centering tolerance: ±{config.CENTERING_TOLERANCE} pixels")
    print(f"   - Y-centering tolerance: ±{config.CENTERING_DISTANCE_TOLERANCE} pixels")
    print(f"   - Collection speed: {config.COLLECTION_SPEED}")
    print(f"   - Centering turn speed: {config.CENTERING_TURN_DURATION}s (FASTER)")
    print(f"   - Centering drive speed: {config.CENTERING_DRIVE_DURATION}s")
    print(f"   - Interface mode: {interface_mode}")
    print(f"   - Target: WHITE BALLS ONLY")
    print(f"   - Boundary system: Modular avoidance")
    print("\nPress Enter to start competition...")
    input()

def show_delivery_info():
    """Show delivery system information before starting"""
    print(f"\n🚚 Entering Delivery System Test Mode...")
    
    print("\n🎯 Delivery system features:")
    print("   - Green target/zone detection using HSV color filtering")
    print("   - Smart search pattern (left/right scanning)")
    print("   - Target centering before approach")
    print("   - Automatic ball release at delivery zones")
    print("   - Real-time visual feedback with overlay")
    print("   - Confidence-based target selection")
    
    print(f"\n⚙️  Delivery Configuration:")
    print(f"   - Green HSV range: [40-80, 50-255, 50-255]")
    print(f"   - Centering tolerance: ±30px X, ±25px Y")
    print(f"   - Search turn duration: 0.8s")
    print(f"   - Approach speed: 40%")
    print(f"   - Min target area: 500 pixels")
    print(f"   - Max target area: 50,000 pixels")
    
    print("\n🎮 Controls:")
    print("   - Press 'q' in camera window to quit")
    print("   - System will automatically search, center, and approach green targets")
    print("   - Balls will be released automatically when delivery zone is reached")
    
    print("\nPress Enter to start delivery test...")
    input()