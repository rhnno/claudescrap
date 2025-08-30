#!/usr/bin/env python3
"""
Quick test to verify all fixes are working
"""

import sys
import os
from pathlib import Path

def test_directory_creation():
    """Test that directories are created properly"""
    print("📁 Testing directory creation...")
    
    expected_dirs = ['logs', 'data', 'data/raw_html', 'data/processed', 'models', 'config']
    
    for directory in expected_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        if os.path.exists(directory):
            print(f"✅ {directory} created successfully")
        else:
            print(f"❌ Failed to create {directory}")
            return False
    
    return True

def test_main_import():
    """Test that main.py can be imported without errors"""
    print("\n📦 Testing main.py import...")
    
    try:
        import main
        print("✅ main.py imported successfully")
        return True
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False

def test_analyzer_import():
    """Test that analyzer can be imported and initialized"""
    print("\n🧠 Testing analyzer import and initialization...")
    
    try:
        sys.path.insert(0, 'func')
        from analyzer import ConfigurableAnalyzer
        
        # Test initialization
        analyzer = ConfigurableAnalyzer()
        print("✅ ConfigurableAnalyzer initialized successfully")
        
        # Test config structure
        if 'paths' in analyzer.config:
            print("✅ Configuration structure is correct")
        else:
            print("❌ Configuration structure is incorrect")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_scraper_app():
    """Test that EnhancedScraperApp can be created"""
    print("\n🚀 Testing EnhancedScraperApp...")
    
    try:
        from main import EnhancedScraperApp
        
        # Test app creation
        app = EnhancedScraperApp()
        print("✅ EnhancedScraperApp created successfully")
        
        # Test config loading
        if app.config:
            print("✅ App configuration loaded")
        else:
            print("❌ App configuration failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ EnhancedScraperApp test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_setup():
    """Test that logging works properly"""
    print("\n📝 Testing logging setup...")
    
    try:
        import logging
        from pathlib import Path
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Test log file creation
        log_file = log_dir / 'test.log'
        
        # Configure test logger
        test_logger = logging.getLogger('test_logger')
        handler = logging.FileHandler(str(log_file))
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)
        
        # Test logging
        test_logger.info("Test log message")
        
        if log_file.exists():
            print("✅ Logging setup working correctly")
            # Clean up test log
            log_file.unlink()
            return True
        else:
            print("❌ Log file not created")
            return False
            
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Testing All Fixes")
    print("=" * 40)
    
    tests = [
        ("Directory Creation", test_directory_creation),
        ("Main Import", test_main_import),
        ("Analyzer Import", test_analyzer_import),
        ("Enhanced Scraper App", test_enhanced_scraper_app),
        ("Logging Setup", test_logging_setup),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*10} {test_name} {'='*10}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL FIXES WORKING! Ready to run main.py")
        print("\n📝 You can now run:")
        print("   python main.py")
        print("   python test_main.py")
    else:
        print("❌ Some fixes still need work")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)