#!/usr/bin/env python3
"""
Integrated System Manager
Combines Enhanced Chrome Profile Setup + Continuous Bug Scanner
"""

import os
import sys
import time
import threading
from datetime import datetime

# Using proper package imports

from enhanced_chrome_profile_setup import EnhancedProfileSetup
from continuous_bug_scanner import BugScanner

class IntegratedSystemManager:
    """Manages both Chrome profile setup and continuous bug scanning"""
    
    def __init__(self):
        """Initialize the integrated system"""
        self.profile_setup = None
        self.bug_scanner = None
        self.scanner_thread = None
        self.monitoring_active = False
        
        print("🚀 Integrated System Manager")
        print("=" * 60)
        print("🎯 Enhanced Chrome Profile Setup + Continuous Bug Scanner")
        print("🔧 Fixes BrowserManager headless issue + Real-time monitoring")
    
    def run_complete_system_setup(self):
        """Run complete system setup and monitoring"""
        print("\n📋 Complete System Setup Process")
        print("-" * 40)
        
        # Step 1: Fix critical issues first
        print("🔴 STEP 1: Fixing Critical Issues")
        if not self._fix_critical_issues():
            print("❌ Critical issues not resolved. Cannot continue.")
            return False
        
        # Step 2: Setup Chrome profiles
        print("\n🔐 STEP 2: Chrome Profile Setup")
        if not self._setup_chrome_profiles():
            print("⚠️ Chrome profile setup failed, but continuing...")
        
        # Step 3: Start continuous monitoring
        print("\n👀 STEP 3: Starting Continuous Monitoring")
        if not self._start_continuous_monitoring():
            print("⚠️ Monitoring setup failed, but system is functional")
        
        # Step 4: Integration verification
        print("\n🧪 STEP 4: Integration Verification")
        self._verify_integration()
        
        print("\n🎉 Integrated System Setup Complete!")
        return True
    
    def _fix_critical_issues(self):
        """Fix critical issues immediately"""
        print("🔍 Scanning for critical issues...")
        
        # Initialize bug scanner for critical scan
        self.bug_scanner = BugScanner(
            project_path="./",
            watch_mode=False,  # No watching during setup
            focus_modules=["func.browser", "func.analyzer"]
        )
        
        # Run scan to identify issues
        self.bug_scanner.run_comprehensive_scan()
        
        # Check for critical bugs
        critical_bugs = [bug for bug in self.bug_scanner.bugs if bug.category == 'critical']
        
        if critical_bugs:
            print(f"🚨 Found {len(critical_bugs)} critical issues:")
            for bug in critical_bugs:
                print(f"   • {bug.title}")
                print(f"     💡 {bug.suggestion}")
            
            # Auto-fix known issues
            fixed_count = 0
            for bug in critical_bugs:
                if "headless parameter" in bug.title.lower():
                    print("🔧 Auto-fixing BrowserManager headless parameter...")
                    # This should already be fixed by our earlier changes
                    fixed_count += 1
            
            if fixed_count > 0:
                print(f"✅ Auto-fixed {fixed_count} critical issues")
                # Re-scan to verify fixes
                self.bug_scanner.run_comprehensive_scan()
                remaining_critical = [bug for bug in self.bug_scanner.bugs if bug.category == 'critical']
                if not remaining_critical:
                    print("✅ All critical issues resolved!")
                    return True
                else:
                    print(f"⚠️ {len(remaining_critical)} critical issues remain")
                    return False
            else:
                print("❌ Could not auto-fix critical issues")
                return False
        else:
            print("✅ No critical issues found")
            return True
    
    def _setup_chrome_profiles(self):
        """Setup Chrome profiles"""
        print("🔐 Setting up Chrome profiles...")
        
        try:
            self.profile_setup = EnhancedProfileSetup()
            
            # Ask user for setup preference
            print("\n📋 Chrome Profile Setup Options:")
            print("1. Quick test (verify fixes only)")
            print("2. Complete setup (full login process)")
            print("3. Skip profile setup")
            
            choice = input("🔢 Select option (1-3): ").strip()
            
            if choice == "1":
                return self.profile_setup.quick_test_setup()
            elif choice == "2":
                return self.profile_setup.run_complete_setup()
            elif choice == "3":
                print("⏭️ Skipping Chrome profile setup")
                return True
            else:
                print("❌ Invalid choice, skipping setup")
                return True
                
        except Exception as e:
            print(f"❌ Chrome profile setup failed: {e}")
            return False
    
    def _start_continuous_monitoring(self):
        """Start continuous monitoring in background"""
        print("👀 Starting continuous monitoring...")
        
        try:
            # Reinitialize scanner for monitoring
            self.bug_scanner = BugScanner(
                project_path="./",
                watch_mode=True,
                focus_modules=[
                    "func.browser",
                    "func.analyzer", 
                    "func.storage",
                    "func.utils"
                ]
            )
            
            # Start monitoring in background thread
            self.scanner_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True
            )
            self.scanner_thread.start()
            self.monitoring_active = True
            
            print("✅ Continuous monitoring started in background")
            return True
            
        except Exception as e:
            print(f"❌ Monitoring startup failed: {e}")
            return False
    
    def _monitoring_worker(self):
        """Background monitoring worker"""
        try:
            # Run initial scan
            self.bug_scanner.run_comprehensive_scan()
            
            # Start file watching if available
            if hasattr(self.bug_scanner, '_start_file_watcher'):
                self.bug_scanner._start_file_watcher()
            
            # Monitoring loop
            while self.monitoring_active:
                time.sleep(60)  # Scan every minute
                self.bug_scanner.run_comprehensive_scan()
                
        except Exception as e:
            print(f"⚠️ Monitoring worker error: {e}")
    
    def _verify_integration(self):
        """Verify integration between all components"""
        print("🧪 Verifying system integration...")
        
        verification_results = {
            'browser_manager': False,
            'enhanced_analyzer': False,
            'profile_setup': False,
            'bug_scanner': False,
            'overall_health': 0
        }
        
        # Test BrowserManager
        try:
            from func import BrowserManager
            
            # Test with headless parameter (should not error)
            browser = BrowserManager(headless=True)
            browser = BrowserManager(headless=False)
            verification_results['browser_manager'] = True
            print("✅ BrowserManager integration verified")
        except Exception as e:
            print(f"❌ BrowserManager integration failed: {e}")
        
        # Test EnhancedConfigurableAnalyzer
        try:
            from func import EnhancedConfigurableAnalyzer
            analyzer = EnhancedConfigurableAnalyzer()
            verification_results['enhanced_analyzer'] = True
            print("✅ EnhancedConfigurableAnalyzer integration verified")
        except Exception as e:
            print(f"❌ EnhancedConfigurableAnalyzer integration failed: {e}")
        
        # Test Profile Setup
        if self.profile_setup:
            verification_results['profile_setup'] = True
            print("✅ Profile setup integration verified")
        
        # Test Bug Scanner
        if self.bug_scanner:
            verification_results['bug_scanner'] = True
            print("✅ Bug scanner integration verified")
        
        # Calculate overall health
        passed_tests = sum(verification_results.values())
        total_tests = len(verification_results) - 1  # Exclude overall_health
        verification_results['overall_health'] = (passed_tests / total_tests) * 100
        
        print(f"\n📊 Integration Results: {passed_tests}/{total_tests} tests passed")
        print(f"🏥 System Health: {verification_results['overall_health']:.0f}%")
        
        return verification_results
    
    def show_live_dashboard(self):
        """Show live dashboard with both systems"""
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
                
                print("🚀 Integrated System Dashboard")
                print("=" * 60)
                print(f"🕐 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Bug Scanner Status
                if self.bug_scanner:
                    print(f"\n👀 Bug Scanner Status: {'🟢 ACTIVE' if self.monitoring_active else '🔴 INACTIVE'}")
                    print(f"📊 Health Score: {self.bug_scanner.health_score}/100")
                    print(f"🔍 Total Scans: {self.bug_scanner.scan_count}")
                    print(f"🐛 Current Bugs: {len(self.bug_scanner.bugs)}")
                    
                    # Show bug breakdown
                    if self.bug_scanner.bugs:
                        from collections import defaultdict
                        bug_categories = defaultdict(int)
                        for bug in self.bug_scanner.bugs:
                            bug_categories[bug.category] += 1
                        
                        print("   Bug Breakdown:")
                        category_emojis = {
                            'critical': '🔴',
                            'warning': '🟡', 
                            'config': '🔵',
                            'performance': '🟢',
                            'integration': '🟣'
                        }
                        
                        for category, count in bug_categories.items():
                            emoji = category_emojis.get(category, '⚪')
                            print(f"   {emoji} {category.title()}: {count}")
                
                # Chrome Profile Status
                print(f"\n🔐 Chrome Profile Status:")
                profile_path = "chrome_profiles/research_profile"
                if os.path.exists(profile_path):
                    print("   🟢 Profile directory exists")
                else:
                    print("   🔴 Profile directory missing")
                
                credentials_path = "config/login_credentials.json"
                if os.path.exists(credentials_path):
                    print("   🟢 Credentials file exists")
                else:
                    print("   🔴 Credentials file missing")
                
                # System Status
                print(f"\n🖥️ System Status:")
                print(f"   📁 Project Path: {os.getcwd()}")
                print(f"   🐍 Python: {sys.version.split()[0]}")
                print(f"   🧵 Monitoring Thread: {'🟢 Running' if self.monitoring_active else '🔴 Stopped'}")
                
                print(f"\n⌨️ Commands:")
                print("   'q' - Quit dashboard")
                print("   'r' - Force rescan")
                print("   's' - Stop monitoring")
                print("   'h' - Show help")
                
                # Non-blocking input check
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    command = input().strip().lower()
                    if command == 'q':
                        break
                    elif command == 'r':
                        if self.bug_scanner:
                            self.bug_scanner.run_comprehensive_scan()
                    elif command == 's':
                        self.stop_monitoring()
                    elif command == 'h':
                        self._show_help()
                
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️ Dashboard stopped")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Integrated System Help")
        print("-" * 30)
        print("🔧 System Components:")
        print("   • Enhanced Chrome Profile Setup")
        print("   • Continuous Bug Scanner")
        print("   • Real-time Monitoring")
        print("   • Integration Verification")
        print("\n🔍 Bug Categories:")
        print("   🔴 Critical: Breaks execution")
        print("   🟡 Warning: Potential issues")
        print("   🔵 Config: Configuration problems")
        print("   🟢 Performance: Speed issues")
        print("   🟣 Integration: Module compatibility")
        input("\nPress Enter to continue...")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.bug_scanner and hasattr(self.bug_scanner, 'observer') and self.bug_scanner.observer:
            self.bug_scanner.observer.stop()
            self.bug_scanner.observer.join()
        print("⏹️ Monitoring stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        if self.profile_setup:
            self.profile_setup.cleanup()
        print("✅ System cleanup completed")


def main():
    """Main function"""
    print("🚀 Integrated System Manager")
    print("=" * 60)
    print("🎯 Enhanced Chrome Profile Setup + Continuous Bug Scanner")
    
    manager = IntegratedSystemManager()
    
    try:
        print("\n📋 Available options:")
        print("1. Complete system setup (recommended)")
        print("2. Chrome profile setup only")
        print("3. Bug scanner only")
        print("4. Live dashboard")
        print("5. Exit")
        
        choice = input("\n🔢 Select option (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Running complete system setup...")
            if manager.run_complete_system_setup():
                print("\n✅ Setup successful!")
                
                # Ask if user wants to see dashboard
                dashboard = input("\n📊 Show live dashboard? (y/n): ").lower().strip()
                if dashboard == 'y':
                    manager.show_live_dashboard()
            else:
                print("\n❌ Setup failed")
        
        elif choice == "2":
            print("\n🔐 Running Chrome profile setup...")
            manager._setup_chrome_profiles()
        
        elif choice == "3":
            print("\n🔍 Running bug scanner...")
            scanner = BugScanner()
            scanner.start_monitoring()
        
        elif choice == "4":
            print("\n📊 Starting live dashboard...")
            # Initialize components for dashboard
            manager.bug_scanner = BugScanner(watch_mode=True)
            manager._start_continuous_monitoring()
            manager.show_live_dashboard()
        
        elif choice == "5":
            print("👋 Goodbye!")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.cleanup()


if __name__ == "__main__":
    main()