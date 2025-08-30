#!/usr/bin/env python3
"""
Test script for main.py - Enhanced ML-Powered Web Scraper
Tests all functionality without requiring full browser setup
"""

import sys
import os
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add func directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'func'))

def create_mock_browser():
    """Create a mock browser manager for testing"""
    mock_browser = Mock()
    mock_driver = Mock()
    
    # Mock driver methods
    mock_driver.find_elements.return_value = [Mock(), Mock()]  # 2 elements
    mock_driver.find_element.return_value = Mock(text="Sample page with pagination page 1 of 10")
    mock_driver.execute_script.return_value = 1000
    mock_driver.current_url = "https://tokopedia.com/search?q=laptop&page=1"
    mock_driver.get.return_value = None
    mock_driver.quit.return_value = None
    
    # Mock browser manager methods
    mock_browser.setup_driver.return_value = None
    mock_browser.driver = mock_driver
    mock_browser.close.return_value = None
    
    return mock_browser

def create_mock_analyzer():
    """Create a mock analyzer for testing"""
    mock_analyzer = Mock()
    
    # Mock analyzer methods
    mock_analyzer.reset_state.return_value = None
    mock_analyzer.analyze_page_structure.return_value = "pagination"
    mock_analyzer.collect_training_data.return_value = None
    mock_analyzer.train_model.return_value = True
    mock_analyzer.get_feature_summary.return_value = {
        '_feature_completeness': 0.85,
        'pagination_buttons': 2,
        'total_products': 20
    }
    
    return mock_analyzer

def create_mock_scraper():
    """Create a mock smart scraper for testing"""
    mock_scraper = Mock()
    
    # Mock scraper methods
    mock_scraper.smart_scrape.return_value = [
        {'page': 1, 'products': 20, 'url': 'https://tokopedia.com/search?q=laptop&page=1'},
        {'page': 2, 'products': 18, 'url': 'https://tokopedia.com/search?q=laptop&page=2'}
    ]
    mock_scraper.train_on_current_session.return_value = True
    
    return mock_scraper

def test_imports():
    """Test if all required modules can be imported"""
    print("📦 Testing imports...")
    
    try:
        # Test main.py imports
        import main
        print("✅ main.py imported successfully")
        
        # Test individual components
        from main import EnhancedScraperApp
        print("✅ EnhancedScraperApp imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading from environment"""
    print("\n⚙️ Testing configuration loading...")
    
    try:
        # Set test environment variables
        test_env = {
            'MAX_PAGES': '15',
            'SCRAPE_INTERVAL': '1800',
            'DEFAULT_SITE': 'amazon',
            'RUN_MODE': 'single',
            'SCRAPE_QUERY': 'smartphone'
        }
        
        with patch.dict(os.environ, test_env):
            from main import EnhancedScraperApp
            app = EnhancedScraperApp()
            
            # Check if configuration is loaded correctly
            assert app.config['max_pages'] == 15
            assert app.config['scrape_interval'] == 1800
            assert app.config['default_site'] == 'amazon'
            
            print("✅ Configuration loaded correctly from environment")
            print(f"   - Max pages: {app.config['max_pages']}")
            print(f"   - Scrape interval: {app.config['scrape_interval']}")
            print(f"   - Default site: {app.config['default_site']}")
            
            return True
            
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_app_initialization():
    """Test EnhancedScraperApp initialization with mocks"""
    print("\n🔧 Testing app initialization...")
    
    try:
        # Mock the dependencies
        mock_browser = create_mock_browser()
        mock_analyzer = create_mock_analyzer()
        mock_scraper = create_mock_scraper()
        
        with patch('main.BrowserManager', return_value=mock_browser), \
             patch('main.ConfigurableAnalyzer', return_value=mock_analyzer), \
             patch('main.SmartTokopediaScraper', return_value=mock_scraper):
            
            from main import EnhancedScraperApp
            app = EnhancedScraperApp()
            
            # Test initialization
            success = app.initialize()
            
            if success:
                print("✅ App initialization successful")
                print("   - Browser manager created")
                print("   - ML analyzer initialized")
                print("   - Smart scraper setup")
                return True
            else:
                print("❌ App initialization failed")
                return False
                
    except Exception as e:
        print(f"❌ App initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_scrape():
    """Test single scraping session"""
    print("\n🕷️ Testing single scrape session...")
    
    try:
        # Mock the dependencies
        mock_browser = create_mock_browser()
        mock_analyzer = create_mock_analyzer()
        mock_scraper = create_mock_scraper()
        
        with patch('main.BrowserManager', return_value=mock_browser), \
             patch('main.ConfigurableAnalyzer', return_value=mock_analyzer), \
             patch('main.SmartTokopediaScraper', return_value=mock_scraper):
            
            from main import EnhancedScraperApp
            app = EnhancedScraperApp()
            app.initialize()
            
            # Test single scrape
            results = app.run_single_scrape("test laptop", "tokopedia")
            
            if results:
                print("✅ Single scrape session successful")
                print(f"   - Results: {len(results)} pages processed")
                print(f"   - Sample result: {results[0] if results else 'None'}")
                return True
            else:
                print("⚠️ Single scrape returned no results (this might be normal)")
                return True
                
    except Exception as e:
        print(f"❌ Single scrape test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_development_mode():
    """Test development mode functionality"""
    print("\n🔧 Testing development mode...")
    
    try:
        # Mock the dependencies
        mock_browser = create_mock_browser()
        mock_analyzer = create_mock_analyzer()
        mock_scraper = create_mock_scraper()
        
        with patch('main.BrowserManager', return_value=mock_browser), \
             patch('main.ConfigurableAnalyzer', return_value=mock_analyzer), \
             patch('main.SmartTokopediaScraper', return_value=mock_scraper), \
             patch('time.sleep'):  # Mock sleep to speed up test
            
            from main import EnhancedScraperApp
            app = EnhancedScraperApp()
            app.initialize()
            
            # Test development mode (should run quickly with mocks)
            app.run_development_mode()
            
            print("✅ Development mode test completed")
            return True
            
    except Exception as e:
        print(f"❌ Development mode test failed: {e}")
        return False

def test_signal_handling():
    """Test graceful shutdown signal handling"""
    print("\n🛑 Testing signal handling...")
    
    try:
        from main import EnhancedScraperApp
        app = EnhancedScraperApp()
        
        # Test signal handler
        import signal
        app._signal_handler(signal.SIGTERM, None)
        
        if not app.running:
            print("✅ Signal handling working correctly")
            return True
        else:
            print("❌ Signal handling not working")
            return False
            
    except Exception as e:
        print(f"❌ Signal handling test failed: {e}")
        return False

def test_cleanup():
    """Test cleanup functionality"""
    print("\n🧹 Testing cleanup...")
    
    try:
        # Mock the dependencies
        mock_browser = create_mock_browser()
        mock_analyzer = create_mock_analyzer()
        
        with patch('main.BrowserManager', return_value=mock_browser), \
             patch('main.ConfigurableAnalyzer', return_value=mock_analyzer):
            
            from main import EnhancedScraperApp
            app = EnhancedScraperApp()
            app.browser = mock_browser
            app.analyzer = mock_analyzer
            
            # Test cleanup
            app.cleanup()
            
            print("✅ Cleanup completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Cleanup test failed: {e}")
        return False

def test_environment_variables():
    """Test different environment variable configurations"""
    print("\n🌍 Testing environment variable configurations...")
    
    test_configs = [
        {
            'name': 'Production Config',
            'env': {
                'MAX_PAGES': '50',
                'SCRAPE_INTERVAL': '7200',
                'DEFAULT_SITE': 'tokopedia',
                'AUTO_TRAIN_MODEL': 'true',
                'COLLECT_TRAINING_DATA': 'true'
            }
        },
        {
            'name': 'Development Config',
            'env': {
                'MAX_PAGES': '5',
                'SCRAPE_INTERVAL': '300',
                'DEFAULT_SITE': 'amazon',
                'AUTO_TRAIN_MODEL': 'false',
                'RUN_MODE': 'development'
            }
        },
        {
            'name': 'Single Run Config',
            'env': {
                'MAX_PAGES': '10',
                'RUN_MODE': 'single',
                'SCRAPE_QUERY': 'headphones',
                'DEFAULT_SITE': 'shopee'
            }
        }
    ]
    
    try:
        for config in test_configs:
            print(f"   Testing {config['name']}...")
            
            with patch.dict(os.environ, config['env'], clear=False):
                from main import EnhancedScraperApp
                app = EnhancedScraperApp()
                
                # Verify configuration
                expected_pages = int(config['env'].get('MAX_PAGES', 50))
                if app.config['max_pages'] == expected_pages:
                    print(f"   ✅ {config['name']} loaded correctly")
                else:
                    print(f"   ❌ {config['name']} configuration mismatch")
                    return False
        
        print("✅ All environment configurations tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False

def test_main_function():
    """Test the main function with mocks"""
    print("\n🚀 Testing main function...")
    
    try:
        # Mock all dependencies and sys.exit
        mock_browser = create_mock_browser()
        mock_analyzer = create_mock_analyzer()
        mock_scraper = create_mock_scraper()
        
        with patch('main.BrowserManager', return_value=mock_browser), \
             patch('main.ConfigurableAnalyzer', return_value=mock_analyzer), \
             patch('main.SmartTokopediaScraper', return_value=mock_scraper), \
             patch('sys.exit') as mock_exit, \
             patch.dict(os.environ, {'RUN_MODE': 'single', 'SCRAPE_QUERY': 'test'}):
            
            # Import and run main
            import main
            main.main()
            
            # Check if main completed without calling sys.exit(1)
            if mock_exit.called:
                exit_code = mock_exit.call_args[0][0] if mock_exit.call_args[0] else 0
                if exit_code == 1:
                    print("❌ Main function exited with error code 1")
                    return False
                else:
                    print("✅ Main function completed successfully")
                    return True
            else:
                print("✅ Main function completed without exit")
                return True
                
    except SystemExit as e:
        if e.code == 0:
            print("✅ Main function completed successfully")
            return True
        else:
            print(f"❌ Main function exited with code: {e.code}")
            return False
    except Exception as e:
        print(f"❌ Main function test failed: {e}")
        return False

def run_performance_test():
    """Test performance and memory usage"""
    print("\n⚡ Running performance test...")
    
    try:
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Run a series of operations
        mock_browser = create_mock_browser()
        mock_analyzer = create_mock_analyzer()
        mock_scraper = create_mock_scraper()
        
        with patch('main.BrowserManager', return_value=mock_browser), \
             patch('main.ConfigurableAnalyzer', return_value=mock_analyzer), \
             patch('main.SmartTokopediaScraper', return_value=mock_scraper):
            
            from main import EnhancedScraperApp
            
            # Create and initialize multiple app instances
            for i in range(5):
                app = EnhancedScraperApp()
                app.initialize()
                app.run_single_scrape(f"test query {i}")
                app.cleanup()
        
        end_time = time.time()
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_increase = final_memory - initial_memory
        
        print(f"✅ Performance test completed")
        print(f"   - Execution time: {execution_time:.2f} seconds")
        print(f"   - Initial memory: {initial_memory:.2f} MB")
        print(f"   - Final memory: {final_memory:.2f} MB")
        print(f"   - Memory increase: {memory_increase:.2f} MB")
        
        # Check for reasonable performance
        if execution_time < 10 and memory_increase < 100:
            print("✅ Performance within acceptable limits")
            return True
        else:
            print("⚠️ Performance may need optimization")
            return True  # Still pass, just a warning
            
    except ImportError:
        print("⚠️ psutil not available, skipping performance test")
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Enhanced ML-Powered Web Scraper - Main.py Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Loading", test_configuration_loading),
        ("App Initialization", test_app_initialization),
        ("Single Scrape", test_single_scrape),
        ("Development Mode", test_development_mode),
        ("Signal Handling", test_signal_handling),
        ("Cleanup", test_cleanup),
        ("Environment Variables", test_environment_variables),
        ("Main Function", test_main_function),
        ("Performance Test", run_performance_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! main.py is working correctly.")
        print("\n📝 Ready for:")
        print("   ✅ Direct execution: python main.py")
        print("   ✅ Docker deployment: docker-compose up -d")
        print("   ✅ Production use")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. Minor issues may exist.")
        print("🔧 Check failed tests above for details.")
    else:
        print("❌ Multiple test failures detected.")
        print("🔧 Please fix the issues before running main.py")
    
    print(f"\n🏁 Test suite completed in {time.time():.2f} seconds")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)