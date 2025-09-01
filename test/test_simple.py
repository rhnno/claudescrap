#!/usr/bin/env python3
"""
Simple test script that avoids browser initialization issues
"""

import sys
import os
import time
from datetime import datetime

# Add func directory to path
# Using proper package imports

class TestLogger:
    """Logger that outputs to both console and file"""
    def __init__(self, filename="result_test.txt"):
        self.filename = filename
        self.start_time = datetime.now()
        
        # Initialize the test result file
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SIMPLE WEB SCRAPER - TEST RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Test Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
    
    def log(self, message):
        """Log message to both console and file"""
        print(message)
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
    
    def log_test_result(self, test_name, passed, details="", duration=0):
        """Log detailed test result"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        message = f"{status} - {test_name}"
        if duration > 0:
            message += f" ({duration:.2f}s)"
        
        self.log(message)
        
        if details:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(f"  Details: {details}\n")
    
    def finalize(self, passed, total, execution_time):
        """Write final summary to file"""
        end_time = datetime.now()
        success_rate = (passed / total) * 100
        
        summary = f"""
{"-" * 70}
FINAL TEST SUMMARY
{"-" * 70}
Tests Passed: {passed}/{total}
Success Rate: {success_rate:.1f}%
Total Duration: {execution_time:.2f}s
Test Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}

Overall Status: {"🎉 ALL TESTS PASSED" if passed == total else "❌ SOME TESTS FAILED"}
{"-" * 70}
"""
        
        self.log(summary)

# Initialize logger
logger = TestLogger()

def test_config_files():
    """Test if configuration files exist"""
    logger.log("📋 Testing configuration files...")
    
    config_files = [
        'config/analyzer_config.yaml',
        'config/enhanced_analyzer_config.yaml',
        'config/parsers.yaml'
    ]
    
    missing_files = []
    for config_file in config_files:
        if not os.path.exists(config_file):
            missing_files.append(config_file)
        else:
            logger.log(f"   ✅ {config_file} exists")
    
    if missing_files:
        error_msg = f"Missing config files: {', '.join(missing_files)}"
        logger.log(f"❌ {error_msg}")
        return False, error_msg
    else:
        return True, f"All {len(config_files)} config files found"

def test_func_imports():
    """Test importing func modules individually"""
    logger.log("📦 Testing func module imports...")
    
    modules_to_test = [
        ('func.utils', 'RandomUtils'),
        ('func.storage', 'DataStorage'),
        ('func.parser', None),
        ('func.scraper', None)
    ]
    
    imported_modules = []
    failed_imports = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            if class_name:
                getattr(module, class_name)
                logger.log(f"   ✅ {module_name}.{class_name} imported")
                imported_modules.append(f"{module_name}.{class_name}")
            else:
                logger.log(f"   ✅ {module_name} imported")
                imported_modules.append(module_name)
        except Exception as e:
            error_msg = f"{module_name}: {e}"
            logger.log(f"   ❌ {error_msg}")
            failed_imports.append(error_msg)
    
    if failed_imports:
        return False, f"Failed imports: {'; '.join(failed_imports)}"
    else:
        return True, f"Successfully imported {len(imported_modules)} modules"

def test_directory_structure():
    """Test if required directories exist"""
    logger.log("📁 Testing directory structure...")
    
    required_dirs = [
        'func',
        'config',
        'docs',
        'data',
        'logs'
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            existing_dirs.append(directory)
            logger.log(f"   ✅ {directory}/ exists")
        else:
            missing_dirs.append(directory)
            logger.log(f"   ❌ {directory}/ missing")
    
    if missing_dirs:
        return False, f"Missing directories: {', '.join(missing_dirs)}"
    else:
        return True, f"All {len(required_dirs)} directories found"

def test_main_import_safe():
    """Test main.py import without browser initialization"""
    logger.log("🚀 Testing main.py import (safe mode)...")
    
    try:
        # Set environment to prevent browser initialization
        os.environ['TESTING_MODE'] = 'true'
        
        # Try to import main module
        import main
        logger.log("   ✅ main.py imported successfully")
        
        # Check if EnhancedScraperApp class exists
        if hasattr(main, 'EnhancedScraperApp'):
            logger.log("   ✅ EnhancedScraperApp class found")
            return True, "main.py imported successfully with EnhancedScraperApp class"
        else:
            return False, "EnhancedScraperApp class not found in main.py"
            
    except Exception as e:
        error_msg = f"Failed to import main.py: {e}"
        logger.log(f"❌ {error_msg}")
        return False, error_msg
    finally:
        # Clean up environment
        if 'TESTING_MODE' in os.environ:
            del os.environ['TESTING_MODE']

def test_requirements():
    """Test if required Python packages are available"""
    logger.log("🐍 Testing Python requirements...")
    
    required_packages = [
        'selenium',
        'requests',
        'yaml',
        'sklearn',
        'pandas',
        'numpy'
    ]
    
    available_packages = []
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            available_packages.append(package)
            logger.log(f"   ✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            logger.log(f"   ❌ {package} missing")
    
    if missing_packages:
        return False, f"Missing packages: {', '.join(missing_packages)}"
    else:
        return True, f"All {len(required_packages)} required packages available"

def test_docker_files():
    """Test if Docker-related files exist"""
    logger.log("🐳 Testing Docker configuration...")
    
    docker_files = [
        'docker-compose.yml',
        'dockerfile',
        'requirements.txt'
    ]
    
    existing_files = []
    missing_files = []
    
    for docker_file in docker_files:
        if os.path.exists(docker_file):
            existing_files.append(docker_file)
            logger.log(f"   ✅ {docker_file} exists")
        else:
            missing_files.append(docker_file)
            logger.log(f"   ❌ {docker_file} missing")
    
    if missing_files:
        return False, f"Missing Docker files: {', '.join(missing_files)}"
    else:
        return True, f"All {len(docker_files)} Docker files found"

def main():
    """Main test function"""
    start_time = time.time()
    
    logger.log("🧪 Simple Web Scraper - Test Suite")
    logger.log("=" * 60)
    
    tests = [
        ("Configuration Files", test_config_files),
        ("Directory Structure", test_directory_structure),
        ("Func Module Imports", test_func_imports),
        ("Python Requirements", test_requirements),
        ("Docker Files", test_docker_files),
        ("Main Import (Safe)", test_main_import_safe),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.log(f"\n{'='*20} {test_name} {'='*20}")
        test_start_time = time.time()
        
        try:
            success, details = test_func()
            test_duration = time.time() - test_start_time
            
            if success:
                passed += 1
                logger.log_test_result(test_name, True, details, test_duration)
            else:
                logger.log_test_result(test_name, False, details, test_duration)
                
        except Exception as e:
            test_duration = time.time() - test_start_time
            error_details = f"Test crashed with exception: {e}"
            logger.log_test_result(test_name, False, error_details, test_duration)
    
    execution_time = time.time() - start_time
    
    logger.log("\n" + "=" * 60)
    logger.log(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.log("🎉 ALL TESTS PASSED! System is ready.")
        logger.log("\n📝 Ready for:")
        logger.log("   ✅ Docker deployment: docker-compose up -d")
        logger.log("   ✅ Local testing")
    elif passed >= total * 0.8:
        logger.log("⚠️ Most tests passed. Minor issues may exist.")
        logger.log("🔧 Check failed tests above for details.")
    else:
        logger.log("❌ Multiple test failures detected.")
        logger.log("🔧 Please fix the issues before deployment.")
    
    logger.log(f"\n🏁 Test suite completed in {execution_time:.2f} seconds")
    
    # Finalize the log file
    logger.finalize(passed, total, execution_time)
    logger.log(f"\n📄 Detailed test results saved to: result_test.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)