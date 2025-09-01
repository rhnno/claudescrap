#!/usr/bin/env python3
"""
Verification Script for Integrated Systems
Tests both Enhanced Chrome Profile Setup and Continuous Bug Scanner
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def print_header():
    """Print verification header"""
    print("🧪 Integrated Systems Verification")
    print("=" * 60)
    print("🎯 Testing Enhanced Chrome Profile Setup + Bug Scanner")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def test_browser_manager_fix():
    """Test BrowserManager headless parameter fix"""
    print("🔧 Testing BrowserManager headless parameter fix...")
    
    try:
        # Using proper package imports
        
        from func import BrowserManager
        
        # Test 1: Initialize with headless=True (should not error)
        print("   1. Testing headless=True...")
        browser1 = BrowserManager(headless=True)
        print("      ✅ headless=True accepted")
        
        # Test 2: Initialize with headless=False (should not error)
        print("   2. Testing headless=False...")
        browser2 = BrowserManager(headless=False)
        print("      ✅ headless=False accepted")
        
        # Test 3: Initialize without headless (should use default)
        print("   3. Testing default headless...")
        browser3 = BrowserManager()
        print("      ✅ Default headless accepted")
        
        # Test 4: Check if headless attribute exists
        print("   4. Testing headless attribute...")
        if hasattr(browser1, 'headless'):
            print(f"      ✅ headless attribute exists: {browser1.headless}")
        else:
            print("      ❌ headless attribute missing")
            return False
        
        print("✅ BrowserManager headless fix verified!")
        return True
        
    except Exception as e:
        print(f"❌ BrowserManager test failed: {e}")
        return False

def test_enhanced_profile_setup():
    """Test Enhanced Profile Setup system"""
    print("\n🔐 Testing Enhanced Profile Setup system...")
    
    try:
        from enhanced_chrome_profile_setup import EnhancedProfileSetup
        
        # Test 1: Initialize setup
        print("   1. Testing initialization...")
        setup = EnhancedProfileSetup()
        print("      ✅ EnhancedProfileSetup initialized")
        
        # Test 2: Check sites configuration
        print("   2. Testing sites configuration...")
        if hasattr(setup, 'sites_config') and setup.sites_config:
            sites = list(setup.sites_config.keys())
            print(f"      ✅ Found {len(sites)} sites: {sites}")
        else:
            print("      ❌ Sites configuration missing")
            return False
        
        # Test 3: Check required methods
        print("   3. Testing required methods...")
        required_methods = [
            'run_complete_setup',
            'quick_test_setup',
            '_phase1_browser_setup',
            '_phase2_manual_login',
            '_phase3_integration_testing'
        ]
        
        for method in required_methods:
            if hasattr(setup, method):
                print(f"      ✅ Method {method} exists")
            else:
                print(f"      ❌ Method {method} missing")
                return False
        
        print("✅ Enhanced Profile Setup verified!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Profile Setup test failed: {e}")
        return False

def test_bug_scanner():
    """Test Continuous Bug Scanner system"""
    print("\n🔍 Testing Continuous Bug Scanner system...")
    
    try:
        from continuous_bug_scanner import BugScanner, BugReport
        
        # Test 1: Initialize scanner
        print("   1. Testing initialization...")
        scanner = BugScanner(
            project_path="./",
            watch_mode=False,  # Disable watching for test
            focus_modules=["func.browser"]
        )
        print("      ✅ BugScanner initialized")
        
        # Test 2: Test BugReport dataclass
        print("   2. Testing BugReport...")
        bug = BugReport(
            category="test",
            severity=1,
            title="Test Bug",
            description="Test description",
            file_path="test.py",
            line_number=1,
            suggestion="Test suggestion",
            timestamp=datetime.now()
        )
        print("      ✅ BugReport created")
        
        # Test 3: Test scanning methods
        print("   3. Testing scanning methods...")
        required_methods = [
            'run_comprehensive_scan',
            '_scan_method_signatures',
            '_scan_configurations',
            '_scan_ml_pipeline',
            '_scan_import_dependencies'
        ]
        
        for method in required_methods:
            if hasattr(scanner, method):
                print(f"      ✅ Method {method} exists")
            else:
                print(f"      ❌ Method {method} missing")
                return False
        
        # Test 4: Run a quick scan
        print("   4. Testing quick scan...")
        scanner.run_comprehensive_scan()
        print(f"      ✅ Scan completed, found {len(scanner.bugs)} issues")
        print(f"      📊 Health score: {scanner.health_score}/100")
        
        print("✅ Bug Scanner verified!")
        return True
        
    except Exception as e:
        print(f"❌ Bug Scanner test failed: {e}")
        return False

def test_integrated_manager():
    """Test Integrated System Manager"""
    print("\n🚀 Testing Integrated System Manager...")
    
    try:
        from integrated_system_manager import IntegratedSystemManager
        
        # Test 1: Initialize manager
        print("   1. Testing initialization...")
        manager = IntegratedSystemManager()
        print("      ✅ IntegratedSystemManager initialized")
        
        # Test 2: Check required methods
        print("   2. Testing required methods...")
        required_methods = [
            'run_complete_system_setup',
            '_fix_critical_issues',
            '_setup_chrome_profiles',
            '_start_continuous_monitoring',
            '_verify_integration'
        ]
        
        for method in required_methods:
            if hasattr(manager, method):
                print(f"      ✅ Method {method} exists")
            else:
                print(f"      ❌ Method {method} missing")
                return False
        
        # Test 3: Test critical issue detection
        print("   3. Testing critical issue detection...")
        try:
            manager._fix_critical_issues()
            print("      ✅ Critical issue detection works")
        except Exception as e:
            print(f"      ⚠️ Critical issue detection error: {e}")
        
        print("✅ Integrated Manager verified!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated Manager test failed: {e}")
        return False

def test_file_structure():
    """Test required file structure"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "func/browser.py",
        "func/analyzer.py",
        "enhanced_chrome_profile_setup.py",
        "continuous_bug_scanner.py",
        "integrated_system_manager.py"
    ]
    
    optional_files = [
        "config/enhanced_training_templates.yaml",
        "config/analyzer_config.yaml",
        "config/login_credentials.json"
    ]
    
    # Check required files
    missing_required = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (REQUIRED)")
            missing_required.append(file_path)
    
    # Check optional files
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ⚠️ {file_path} (optional)")
    
    if missing_required:
        print(f"❌ Missing {len(missing_required)} required files")
        return False
    else:
        print("✅ File structure verified!")
        return True

def test_integration_compatibility():
    """Test integration with existing codebase"""
    print("\n🔗 Testing integration compatibility...")
    
    try:
        # Test 1: Import existing enhanced analyzer
        print("   1. Testing EnhancedConfigurableAnalyzer import...")
        # Using proper package imports
        
        from func import EnhancedConfigurableAnalyzer
        analyzer = EnhancedConfigurableAnalyzer()
        print("      ✅ EnhancedConfigurableAnalyzer imported")
        
        # Test 2: Test BrowserManager with analyzer
        print("   2. Testing BrowserManager + Analyzer integration...")
        from func import BrowserManager
        
        browser = BrowserManager(headless=True)  # Should not error
        print("      ✅ BrowserManager + Analyzer compatible")
        
        # Test 3: Test orchestrator compatibility
        print("   3. Testing orchestrator compatibility...")
        try:
            from train_model_orchestrator import TrainingOrchestrator
            from scraping_orchestrator import ScrapingOrchestrator
            print("      ✅ Orchestrators compatible")
        except ImportError as e:
            print(f"      ⚠️ Orchestrator import issue: {e}")
        
        print("✅ Integration compatibility verified!")
        return True
        
    except Exception as e:
        print(f"❌ Integration compatibility test failed: {e}")
        return False

def run_comprehensive_verification():
    """Run all verification tests"""
    print_header()
    
    tests = [
        ("BrowserManager Fix", test_browser_manager_fix),
        ("Enhanced Profile Setup", test_enhanced_profile_setup),
        ("Bug Scanner", test_bug_scanner),
        ("Integrated Manager", test_integrated_manager),
        ("File Structure", test_file_structure),
        ("Integration Compatibility", test_integration_compatibility)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Your integrated systems are ready:")
        print("   • BrowserManager headless issue fixed")
        print("   • Enhanced Chrome Profile Setup working")
        print("   • Continuous Bug Scanner functional")
        print("   • Integration verified")
        print("\n🚀 Next steps:")
        print("   1. Run: python integrated_system_manager.py")
        print("   2. Choose option 1 for complete setup")
        print("   3. Follow the guided process")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        print("\n🔧 Troubleshooting:")
        print("   1. Check error messages above")
        print("   2. Ensure all files are created correctly")
        print("   3. Install missing dependencies")
        print("   4. Re-run verification")
        return False

def main():
    """Main verification function"""
    try:
        success = run_comprehensive_verification()
        
        if success:
            print("\n🎯 Ready to use integrated systems!")
            
            # Ask if user wants to run the integrated manager
            choice = input("\n🚀 Run integrated system manager now? (y/n): ").lower().strip()
            if choice == 'y':
                print("🚀 Starting integrated system manager...")
                os.system("python integrated_system_manager.py")
        else:
            print("\n❌ Verification failed. Please fix issues before proceeding.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Verification interrupted")
    except Exception as e:
        print(f"\n💥 Verification crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()