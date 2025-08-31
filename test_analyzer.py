#!/usr/bin/env python3
"""
Test script for the Enhanced Analyzer
"""

import sys
import os

# Add func directory to path
# Using proper package imports

def test_analyzer():
    """Test the ConfigurableAnalyzer"""
    print("🧪 Testing Enhanced ML-Powered Analyzer...")
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from func import ConfigurableAnalyzer
        print("✅ ConfigurableAnalyzer imported successfully")
        
        # Test initialization
        print("🔧 Testing initialization...")
        
        # Try to find existing config files in order of preference
        config_paths = [
            'config/enhanced_analyzer_config.yaml',
            'config/analyzer_config.yaml'
        ]
        
        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                print(f"📄 Using config: {config_path}")
                break
        
        if config_path is None:
            print("📄 Using default configuration (no config file found)")
        
        # Initialize with found config or None for defaults
        analyzer = ConfigurableAnalyzer(config_path)
        print("✅ Analyzer initialized successfully")
        
        # Test configuration
        print("⚙️ Testing configuration...")
        config_sections = list(analyzer.config.keys())
        print(f"✅ Config sections loaded: {config_sections}")
        
        # Test feature extraction (mock)
        print("🔍 Testing feature extraction...")
        
        # Create a mock driver for testing
        class MockDriver:
            def find_elements(self, by, selector):
                return []  # Return empty list for testing
            
            def find_element(self, by, selector):
                class MockElement:
                    text = "Sample page text with pagination page 1 of 10"
                return MockElement()
            
            def execute_script(self, script):
                if "scrollHeight" in script:
                    return 2000
                elif "innerHeight" in script:
                    return 800
                elif "pageYOffset" in script:
                    return 0
                elif "querySelectorAll" in script:
                    return 50
                elif "performance" in script:
                    return 2
                else:
                    return 1
            
            @property
            def current_url(self):
                return "https://example.com/search?page=1"
        
        mock_driver = MockDriver()
        
        # Test feature extraction
        features = analyzer._extract_features(mock_driver)
        print(f"✅ Feature extraction successful: {len(features)} features extracted")
        
        # Show some key features
        key_features = ['pagination_buttons', 'next_button', 'total_products', 'page_text_length']
        for feature in key_features:
            if feature in features:
                print(f"   - {feature}: {features[feature]}")
        
        # Test feature vector conversion
        print("🔢 Testing feature vector conversion...")
        feature_vector = analyzer._features_to_vector(features)
        print(f"✅ Feature vector created: shape {feature_vector.shape}")
        
        # Test rule-based analysis
        print("📋 Testing rule-based analysis...")
        result = analyzer._rule_based_analysis(mock_driver)
        print(f"✅ Rule-based analysis result: {result}")
        
        # Test training data collection
        print("📊 Testing training data collection...")
        analyzer.collect_training_data(mock_driver, "pagination", "test_site")
        print("✅ Training data collection successful")
        
        print("\n🎉 All tests passed! Enhanced Analyzer is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install numpy pandas scikit-learn PyYAML selenium beautifulsoup4")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n📋 Testing configuration system...")
    
    try:
        # Test if config files exist
        config_files = [
            'config/enhanced_analyzer_config.yaml',
            'config/analyzer_config.yaml',
            'config/feature_templates/ecommerce_template.yaml'
        ]
        
        # Create missing config directory if needed
        os.makedirs('config', exist_ok=True)
        os.makedirs('config/feature_templates', exist_ok=True)
        
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✅ Found: {config_file}")
            else:
                print(f"⚠️ Missing: {config_file}")
        
        # Test YAML loading
        import yaml
        test_config = {
            'test': 'value',
            'nested': {'key': 'value'}
        }
        yaml_str = yaml.dump(test_config)
        loaded = yaml.safe_load(yaml_str)
        print("✅ YAML processing working")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Enhanced ML-Powered Web Scraper - Test Suite")
    print("=" * 50)
    
    # Test configuration
    config_ok = test_configuration()
    
    # Test analyzer
    analyzer_ok = test_analyzer()
    
    print("\n" + "=" * 50)
    if config_ok and analyzer_ok:
        print("🎉 ALL TESTS PASSED! The Enhanced Analyzer is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Run with Docker: docker-compose up -d")
        print("   2. Or run directly: python main.py")
        print("   3. Check logs for detailed output")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Install missing dependencies")
        print("   2. Check file paths and permissions")
        print("   3. Verify configuration files")

if __name__ == "__main__":
    main()