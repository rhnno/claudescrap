#!/usr/bin/env python3
"""
Validation script to check if test infrastructure is properly set up.

This script performs basic validation of the test environment before
running the full test suite.
"""
import sys
import os
import importlib.util
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 10):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires 3.10+")
        return False


def check_required_packages():
    """Check if required packages are installed."""
    print("\n🔍 Checking required packages...")
    
    required_packages = [
        'pytest',
        'pytest_asyncio',
        'asyncio',
        'unittest.mock',
        'datetime',
        'uuid'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'pytest_asyncio':
                import pytest_asyncio
                print(f"✅ {package} - OK")
            elif package == 'unittest.mock':
                from unittest import mock
                print(f"✅ {package} - OK")
            else:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    print(f"✅ {package} - OK")
                else:
                    print(f"❌ {package} - Missing")
                    missing_packages.append(package)
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0


def check_project_structure():
    """Check if project structure is correct."""
    print("\n🔍 Checking project structure...")
    
    required_files = [
        'src/services/scraper_service.py',
        'tests/__init__.py',
        'tests/conftest.py',
        'tests/test_scraper_service.py',
        'tests/test_integration.py',
        'tests/test_api.py',
        'tests/test_performance.py',
        'pytest.ini'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - Found")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0


def check_test_imports():
    """Check if test modules can be imported."""
    print("\n🔍 Checking test imports...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    test_modules = [
        'tests.conftest',
        'tests.test_scraper_service', 
        'tests.test_integration',
        'tests.test_api',
        'tests.test_performance'
    ]
    
    import_errors = []
    
    for module in test_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} - Import OK")
        except ImportError as e:
            print(f"❌ {module} - Import Failed: {e}")
            import_errors.append(module)
    
    return len(import_errors) == 0


def check_source_imports():
    """Check if source modules can be imported."""
    print("\n🔍 Checking source imports...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    try:
        from src.services.scraper_service import ScraperService
        print("✅ ScraperService - Import OK")
        return True
    except ImportError as e:
        print(f"❌ ScraperService - Import Failed: {e}")
        print("Note: This is expected if dependencies like Selenium are not installed")
        return False


def run_basic_test():
    """Run a basic test to verify pytest works."""
    print("\n🔍 Running basic test...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_scraper_service.py::TestScraperServiceInitialization::test_init_creates_required_attributes',
            '-v', '--tb=short', '--no-cov'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Basic test - PASSED")
            return True
        else:
            print("❌ Basic test - FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Basic test - ERROR: {e}")
        return False


def main():
    """Main validation function."""
    print("🚀 ScraperService Test Infrastructure Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages), 
        ("Project Structure", check_project_structure),
        ("Test Imports", check_test_imports),
        ("Source Imports", check_source_imports),
        ("Basic Test", run_basic_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"💥 {check_name} - ERROR: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} checks passed")
    success_rate = (passed / total) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\n🎉 All validations passed! Test infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Run: python tests/test_runner.py --unit")
        print("  2. Run: python tests/test_runner.py --all") 
        print("  3. Run: make test-ci (if make is available)")
        return True
    else:
        print(f"\n⚠️ {total - passed} validation(s) failed.")
        print("\nTroubleshooting:")
        print("  1. Install missing dependencies: pip install -r requirements_test.txt")
        print("  2. Check file paths and project structure")
        print("  3. Verify Python version is 3.10+")
        return False


if __name__ == "__main__":
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    success = main()
    sys.exit(0 if success else 1)