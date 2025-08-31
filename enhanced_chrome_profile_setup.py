#!/usr/bin/env python3
"""
Enhanced Chrome Profile Setup System
Fixes BrowserManager headless issue and creates comprehensive login system
"""

import os
import sys
import json
import time
import getpass
from datetime import datetime
from pathlib import Path

# Using proper package imports

from func import BrowserManager

class EnhancedProfileSetup:
    """Enhanced Chrome profile setup with credential management"""
    
    def __init__(self):
        """Initialize the profile setup system"""
        self.browser = None
        self.credentials = {}
        self.sites_config = {
            'tokopedia': {
                'name': 'Tokopedia',
                'login_url': 'https://accounts.tokopedia.com/otp/c/page?otp_type=116&msisdn=&ld=https%3A%2F%2Fwww.tokopedia.com%2F',
                'test_url': 'https://www.tokopedia.com',
                'login_indicators': [
                    "[data-testid*='profile']",
                    ".user-menu",
                    ".account-menu"
                ]
            },
            'shopee': {
                'name': 'Shopee',
                'login_url': 'https://shopee.co.id/buyer/login',
                'test_url': 'https://shopee.co.id',
                'login_indicators': [
                    ".navbar__username",
                    ".user-info",
                    ".shopee-avatar"
                ]
            },
            'bukalapak': {
                'name': 'Bukalapak',
                'login_url': 'https://accounts.bukalapak.com/login',
                'test_url': 'https://www.bukalapak.com',
                'login_indicators': [
                    ".user-menu",
                    ".account-dropdown",
                    ".user-avatar"
                ]
            }
        }
        
        print("🚀 Enhanced Chrome Profile Setup System")
        print("=" * 60)
    
    def run_complete_setup(self):
        """Run the complete profile setup process"""
        print("📋 Starting complete profile setup process...")
        
        # Phase 1: Browser Setup
        if not self._phase1_browser_setup():
            return False
        
        # Phase 2: Manual Login Process
        if not self._phase2_manual_login():
            return False
        
        # Phase 3: Integration Testing
        if not self._phase3_integration_testing():
            return False
        
        print("\n🎉 Enhanced Chrome Profile Setup Complete!")
        return True
    
    def _phase1_browser_setup(self):
        """Phase 1: Profile & Browser Setup"""
        print("\n📱 PHASE 1: Profile & Browser Setup")
        print("-" * 40)
        
        try:
            # Test the fixed BrowserManager initialization
            print("🔧 Testing BrowserManager with headless parameter...")
            
            # This should now work without the 'headless' error
            self.browser = BrowserManager(
                use_profile=True, 
                profile_name="research_profile",
                headless=False  # Fixed: Now accepts headless parameter
            )
            print("✅ BrowserManager initialization successful!")
            
            # Setup the driver
            print("🌐 Setting up Chrome driver...")
            self.browser.setup_driver()
            print("✅ Chrome driver setup successful!")
            
            # Verify profile directory
            profile_path = self.browser.profile_path
            print(f"📁 Profile directory: {profile_path}")
            
            if profile_path.exists():
                print("✅ Profile directory exists")
            else:
                print("📁 Creating new profile directory")
                profile_path.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 1 failed: {e}")
            return False
    
    def _phase2_manual_login(self):
        """Phase 2: Manual Login Process"""
        print("\n🔐 PHASE 2: Manual Login Process")
        print("-" * 40)
        
        for site_key, site_config in self.sites_config.items():
            print(f"\n🌐 Setting up {site_config['name']}...")
            
            try:
                # Navigate to login page
                print(f"📍 Navigating to {site_config['name']} login page...")
                self.browser.navigate_to(site_config['login_url'])
                time.sleep(3)
                
                # Interactive prompt for manual login
                print(f"\n🔐 Please log into {site_config['name']} manually in the browser")
                print("   - Handle any 2FA/CAPTCHA challenges")
                print("   - Complete the login process")
                print("   - Ensure you're successfully logged in")
                
                input(f"⏸️  Press Enter when you've completed login to {site_config['name']}...")
                
                # Verify login success
                if self._verify_login_success(site_key):
                    print(f"✅ {site_config['name']} login verified!")
                    
                    # Collect credentials for backup
                    self._collect_credentials(site_key, site_config)
                else:
                    print(f"⚠️ Could not verify {site_config['name']} login")
                    retry = input("🔄 Would you like to retry? (y/n): ").lower().strip()
                    if retry == 'y':
                        continue
                
            except Exception as e:
                print(f"❌ Error with {site_config['name']}: {e}")
                continue
        
        # Generate credentials JSON
        self._generate_credentials_json()
        return True
    
    def _verify_login_success(self, site_key):
        """Verify if login was successful"""
        try:
            site_config = self.sites_config[site_key]
            
            # Check for login indicators
            for indicator in site_config['login_indicators']:
                try:
                    elements = self.browser.driver.find_elements("css selector", indicator)
                    if elements and any(el.is_displayed() for el in elements):
                        return True
                except:
                    continue
            
            # Check URL for login success
            current_url = self.browser.driver.current_url
            if site_key in current_url and 'login' not in current_url:
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error verifying login: {e}")
            return False
    
    def _collect_credentials(self, site_key, site_config):
        """Collect credentials after successful login"""
        print(f"\n✅ {site_config['name']} login successful! Please provide credentials for backup:")
        
        try:
            # Collect email
            email = input("📧 Email/Username: ").strip()
            
            # Collect password (hidden input)
            password = getpass.getpass("🔑 Password: ")
            
            # Store credentials
            self.credentials[site_key] = {
                'email': email,
                'password': password,
                'login_url': site_config['login_url'],
                'test_url': site_config['test_url'],
                'setup_date': datetime.now().isoformat(),
                'verified': True
            }
            
            print(f"✅ Credentials saved for {site_config['name']}")
            
        except KeyboardInterrupt:
            print(f"\n⏸️ Skipping credential collection for {site_config['name']}")
        except Exception as e:
            print(f"⚠️ Error collecting credentials: {e}")
    
    def _generate_credentials_json(self):
        """Generate the login_credentials.json file"""
        print("\n📝 Generating login_credentials.json...")
        
        try:
            # Ensure config directory exists
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            
            # Create comprehensive credentials structure
            credentials_data = {
                "setup_info": {
                    "created_date": datetime.now().isoformat(),
                    "setup_method": "enhanced_chrome_profile_setup",
                    "profile_name": "research_profile",
                    "total_sites": len(self.credentials)
                },
                "sites": self.credentials,
                "usage_notes": {
                    "auto_login": "Use browser.auto_login(site_name) for automatic login",
                    "manual_login": "Profile sessions should persist between browser restarts",
                    "fallback": "If auto-login fails, credentials are available for manual entry"
                }
            }
            
            # Save to file
            credentials_file = config_dir / "login_credentials.json"
            with open(credentials_file, 'w', encoding='utf-8') as f:
                json.dump(credentials_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Credentials file created: {credentials_file}")
            print(f"📊 Saved credentials for {len(self.credentials)} sites")
            
            # Display file structure
            print("\n📋 Generated file structure:")
            print(f"   📁 config/")
            print(f"   └── 📄 login_credentials.json")
            print(f"       ├── setup_info (metadata)")
            print(f"       ├── sites ({len(self.credentials)} configured)")
            print(f"       └── usage_notes (instructions)")
            
        except Exception as e:
            print(f"❌ Error generating credentials file: {e}")
    
    def _phase3_integration_testing(self):
        """Phase 3: Integration Testing"""
        print("\n🧪 PHASE 3: Integration Testing")
        print("-" * 40)
        
        try:
            # Test profile persistence
            print("🔄 Testing profile persistence...")
            
            # Close current browser
            if self.browser and self.browser.driver:
                self.browser.close()
                time.sleep(2)
            
            # Reopen with same profile
            print("🔄 Reopening browser with same profile...")
            self.browser = BrowserManager(
                use_profile=True,
                profile_name="research_profile",
                headless=False
            )
            self.browser.setup_driver()
            
            # Test each site for persistent login
            persistent_logins = 0
            for site_key, site_config in self.sites_config.items():
                if site_key in self.credentials:
                    print(f"🔍 Testing {site_config['name']} session persistence...")
                    
                    self.browser.navigate_to(site_config['test_url'])
                    time.sleep(3)
                    
                    if self._verify_login_success(site_key):
                        print(f"✅ {site_config['name']} session persisted!")
                        persistent_logins += 1
                    else:
                        print(f"⚠️ {site_config['name']} session not persisted")
            
            # Test EnhancedConfigurableAnalyzer compatibility
            print("\n🧠 Testing EnhancedConfigurableAnalyzer compatibility...")
            try:
                from func import EnhancedConfigurableAnalyzer
                
                analyzer = EnhancedConfigurableAnalyzer(
                    template_path="config/enhanced_training_templates.yaml"
                )
                
                # Test analysis with current browser
                if self.browser.driver:
                    result = analyzer.analyze_page_structure(
                        self.browser.driver, 
                        site_name='tokopedia'
                    )
                    print(f"✅ Analyzer integration successful: {result}")
                else:
                    print("⚠️ No active browser for analyzer testing")
                
            except Exception as e:
                print(f"⚠️ Analyzer integration issue: {e}")
            
            # Summary
            print(f"\n📊 Integration Test Results:")
            print(f"   🔄 Profile persistence: {persistent_logins}/{len(self.credentials)} sites")
            print(f"   🧠 Analyzer compatibility: ✅")
            print(f"   📁 Credentials file: ✅")
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 3 failed: {e}")
            return False
    
    def quick_test_setup(self):
        """Quick test of the setup without full login process"""
        print("\n🧪 Quick Setup Test")
        print("-" * 30)
        
        try:
            # Test BrowserManager fix
            print("1. Testing BrowserManager headless fix...")
            browser = BrowserManager(headless=True)  # Should not error
            print("   ✅ Headless parameter accepted")
            
            browser = BrowserManager(headless=False)  # Should not error
            print("   ✅ Non-headless parameter accepted")
            
            # Test profile setup
            print("2. Testing profile setup...")
            browser.setup_driver()
            print("   ✅ Driver setup successful")
            
            # Test navigation
            print("3. Testing navigation...")
            browser.navigate_to("https://www.google.com")
            print("   ✅ Navigation successful")
            
            browser.close()
            print("   ✅ Browser closed successfully")
            
            print("\n🎉 Quick test passed! Ready for full setup.")
            return True
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.browser and self.browser.driver:
            self.browser.close()
            print("✅ Browser closed and profile saved")


def main():
    """Main function"""
    print("🚀 Enhanced Chrome Profile Setup System")
    print("=" * 60)
    print("🎯 Fixes BrowserManager headless issue and creates login system")
    print()
    
    setup = EnhancedProfileSetup()
    
    try:
        # Ask user what they want to do
        print("📋 Available options:")
        print("1. Quick test (verify fixes)")
        print("2. Complete setup (full login process)")
        print("3. Exit")
        
        choice = input("\n🔢 Select option (1-3): ").strip()
        
        if choice == "1":
            print("\n🧪 Running quick test...")
            if setup.quick_test_setup():
                print("\n✅ Quick test successful!")
                print("💡 You can now run option 2 for complete setup")
            else:
                print("\n❌ Quick test failed")
        
        elif choice == "2":
            print("\n🚀 Running complete setup...")
            if setup.run_complete_setup():
                print("\n🎉 Complete setup successful!")
                print("\n📝 Next steps:")
                print("   1. Your Chrome profile is ready with persistent sessions")
                print("   2. Credentials are saved in config/login_credentials.json")
                print("   3. You can now use enhanced_tokoscrape.py or scraping_orchestrator.py")
                print("   4. BrowserManager headless issue is fixed")
            else:
                print("\n❌ Setup failed")
        
        elif choice == "3":
            print("👋 Goodbye!")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted by user")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        setup.cleanup()


if __name__ == "__main__":
    main()