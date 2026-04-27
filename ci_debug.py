#!/usr/bin/env python3
"""
CI/CD Debug Script

This script helps debug CI/CD issues by testing basic functionality
without complex dependencies.
"""
import sys
import os

def test_basic_imports():
    """Test basic Python imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import asyncio
        print("✅ asyncio - OK")
        
        import unittest.mock
        print("✅ unittest.mock - OK")
        
        import datetime
        print("✅ datetime - OK")
        
        import uuid
        print("✅ uuid - OK")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pytest_availability():
    """Test if pytest is available."""
    print("\n🔍 Testing pytest availability...")
    
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} - OK")
        return True
    except ImportError as e:
        print(f"❌ pytest not available: {e}")
        return False

def test_src_path():
    """Test src path setup."""
    print("\n🔍 Testing src path setup...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
        print(f"✅ Added {src_path} to Python path")
        return True
    else:
        print(f"❌ src directory not found at {src_path}")
        return False

def test_service_import():
    """Test ScraperService import with mocking."""
    print("\n🔍 Testing ScraperService import...")
    
    try:
        # Mock dependencies before import
        from unittest.mock import Mock, patch
        
        with patch('src.services.scraper_service.DatabaseManager') as mock_db:
            with patch('src.services.scraper_service.Utils') as mock_utils:
                from src.services.scraper_service import ScraperService
                print("✅ ScraperService import - OK")
                
                # Test basic initialization
                service = ScraperService()
                print("✅ ScraperService initialization - OK")
                
                return True
                
    except ImportError as e:
        print(f"❌ ScraperService import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ ScraperService error: {e}")
        return False

def main():
    """Main CI debug function."""
    print("🚀 CI/CD Debug Script")
    print("=" * 50)
    
    results = [
        test_basic_imports(),
        test_pytest_availability(),
        test_src_path(),
        test_service_import()
    ]
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All CI debug tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed - check CI environment setup")
        return 1

if __name__ == "__main__":
    sys.exit(main())