"""
Startup menu interface for GolfBot
"""

def show_startup_menu():
    """Show startup menu with options"""
    print("\n" + "="*60)
    print("🤖 GOLFBOT CONTROL SYSTEM - TWO-PHASE COLLECTION")
    print("="*60)
    print("1. Start Competition (Full Dashboard)")
    print("2. Start Competition (Compact Dashboard)")
    print("3. Hardware Testing") 
    print("4. Exit")
    print("="*60)
    print("FEATURES:")
    print("• Phase 1: X+Y centering for ball alignment")
    print("• Phase 2: Collection zone positioning with servo pre-collect")
    print("• Optimized collection sequence in green zone")
    print("• Full dashboard interface (option 1)")
    print("• Compact dashboard with 25% camera scale (option 2)")
    print("="*60)
    
    while True:
        try:
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                return 'competition_dashboard'
            elif choice == '2':
                return 'competition_legacy'
            elif choice == '3':
                return 'testing'
            elif choice == '4':
                return 'exit'
            else:
                print("Invalid choice. Enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return 'exit'
        except EOFError:
            print("\nExiting...")
            return 'exit'
        except Exception as e:
            print(f"Input error: {e}")
            return 'exit'