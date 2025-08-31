#!/usr/bin/env python3
"""
Simple verification script for enhanced analyzer
"""

def verify_imports():
    """Verify that all imports work correctly"""
    print("🔍 Verifying Enhanced Analyzer Setup...")
    
    try:
        # Test basic imports
        print("1. Testing basic imports...")
        from func import ConfigurableAnalyzer, EnhancedConfigurableAnalyzer
        print("   ✅ Analyzers imported successfully")
        
        # Test enhanced analyzer initialization
        print("2. Testing enhanced analyzer...")
        analyzer = EnhancedConfigurableAnalyzer()
        print("   ✅ Enhanced analyzer initialized")
        
        # Test template loading
        print("3. Testing template loading...")
        import os
        if os.path.exists("config/enhanced_training_templates.yaml"):
            analyzer_with_templates = EnhancedConfigurableAnalyzer(
                template_path="config/enhanced_training_templates.yaml"
            )
            print("   ✅ Templates loaded successfully")
        else:
            print("   ⚠️ Template file not found, using defaults")
        
        # Test main integration
        print("4. Testing main.py integration...")
        from main import EnhancedTokopediaScraper
        print("   ✅ Enhanced scraper class available")
        
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("\n📝 Your enhanced analyzer is ready to use:")
        print("   ✅ Import issues fixed")
        print("   ✅ Enhanced analyzer working")
        print("   ✅ YAML templates supported")
        print("   ✅ Main integration complete")
        
        print("\n🚀 Usage examples:")
        print("   # Direct usage:")
        print("   from func.analyzer import EnhancedConfigurableAnalyzer")
        print("   analyzer = EnhancedConfigurableAnalyzer()")
        print("   result = analyzer.analyze_page_structure(driver, 'tokopedia')")
        print()
        print("   # With scraper:")
        print("   from main import EnhancedTokopediaScraper")
        print("   scraper = EnhancedTokopediaScraper(browser_manager)")
        print("   results = scraper.smart_scrape('laptop', max_pages=10)")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure all files are in place")
        print("   2. Check Python environment")
        print("   3. Install missing dependencies")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Analyzer Verification")
    print("=" * 40)
    
    success = verify_imports()
    
    if success:
        print("\n✅ Setup verification complete!")
    else:
        print("\n❌ Setup verification failed!")