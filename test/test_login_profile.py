#!/usr/bin/env python3
"""
Test script for Chrome profile with auto-login functionality
Run this to set up your research profile with login credentials
"""

import sys
import os
# Using proper package imports

from func import BrowserManager
import time

def test_profile_setup():
    """Test Chrome profile setup and login functionality"""
    print("🧪 Testing Chrome Profile with Auto-Login")
    print("=" * 50)
    
    # Initialize browser with profile
    browser = BrowserManager(use_profile=True, profile_name="research_profile")
    
    try:
        # Setup driver
        print("🔧 Setting up Chrome driver with research profile...")
        browser.setup_driver()
        
        # Test sites
        sites_to_test = ['tokopedia', 'shopee', 'bukalapak']
        
        for site in sites_to_test:
            print(f"\n🌐 Testing {site.upper()}...")
            
            # Check if already logged in
            if site == 'tokopedia':
                browser.navigate_to("https://www.tokopedia.com")
            elif site == 'shopee':
                browser.navigate_to("https://shopee.co.id")
            elif site == 'bukalapak':
                browser.navigate_to("https://www.bukalapak.com")
            
            time.sleep(2)
            
            # Check login status
            is_logged_in = browser.check_login_status(site)
            
            if not is_logged_in:
                print(f"🔐 Not logged in to {site}")
                print(f"💡 You can manually login now, or set up auto-login credentials")
                
                # Ask user if they want to try auto-login
                response = input(f"Try auto-login to {site}? (y/n): ").lower().strip()
                if response == 'y':
                    success = browser.auto_login(site)
                    if success:
                        print(f"✅ Auto-login to {site} successful!")
                    else:
                        print(f"❌ Auto-login to {site} failed")
                        print("💡 You may need to:")
                        print("   1. Update credentials in config/login_credentials.json")
                        print("   2. Handle 2FA manually")
                        print("   3. Complete CAPTCHA if required")
                else:
                    print(f"⏭️ Skipping auto-login for {site}")
            else:
                print(f"✅ Already logged in to {site}")
            
            # Wait before next site
            time.sleep(3)
        
        print("\n🎉 Profile testing completed!")
        print("\n📋 Summary:")
        print("✅ Chrome profile created and configured")
        print("✅ Anti-detection measures applied")
        print("✅ Login functionality tested")
        print("\n💡 Next steps:")
        print("1. Fill in your credentials in config/login_credentials.json")
        print("2. Run your main scraper with the enhanced browser")
        print("3. Your login sessions will be saved in the profile")
        
        # Keep browser open for manual testing
        input("\n⏸️ Press Enter to close browser...")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        browser.close()

def show_credentials_template():
    """Show the credentials template structure"""
    print("\n📝 Credentials File Structure:")
    print("=" * 40)
    print("""
{
  "tokopedia": {
    "email": "your-email@example.com",
    "password": "your-password",
    "login_url": "https://accounts.tokopedia.com/otp/c/page?otp_type=116&msisdn=&ld=https%3A%2F%2Fwww.tokopedia.com%2F"
  },
  "shopee": {
    "email": "your-email@example.com", 
    "password": "your-password",
    "login_url": "https://shopee.co.id/buyer/login"
  },
  "bukalapak": {
    "email": "your-email@example.com",
    "password": "your-password", 
    "login_url": "https://accounts.bukalapak.com/login"
  }
}
    """)
    print("📍 File location: config/login_credentials.json")
    print("🔒 This file will be created automatically when you run the test")

def quick_login_test(site_name):
    """Quick test for a specific site"""
    print(f"🚀 Quick login test for {site_name}")
    
    browser = BrowserManager(use_profile=True)
    
    try:
        browser.setup_driver()
        success = browser.ensure_login(site_name)
        
        if success:
            print(f"✅ Successfully logged in to {site_name}")
            
            # Test a search to verify functionality
            if site_name == 'tokopedia':
                browser.navigate_to("https://www.tokopedia.com/search?st=product&q=laptop")
            elif site_name == 'shopee':
                browser.navigate_to("https://shopee.co.id/search?keyword=laptop")
            elif site_name == 'bukalapak':
                browser.navigate_to("https://www.bukalapak.com/products?search%5Bkeywords%5D=laptop")
            
            time.sleep(3)
            print(f"🔍 Navigated to {site_name} search page")
            input("⏸️ Press Enter to close...")
        else:
            print(f"❌ Failed to login to {site_name}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        browser.close()

if __name__ == "__main__":
    print("🔧 Chrome Profile & Auto-Login Setup")
    print("=" * 50)
    
    print("\nChoose an option:")
    print("1. Full profile setup and testing")
    print("2. Show credentials template")
    print("3. Quick login test for Tokopedia")
    print("4. Quick login test for Shopee") 
    print("5. Quick login test for Bukalapak")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        test_profile_setup()
    elif choice == "2":
        show_credentials_template()
    elif choice == "3":
        quick_login_test("tokopedia")
    elif choice == "4":
        quick_login_test("shopee")
    elif choice == "5":
        quick_login_test("bukalapak")
    else:
        print("❌ Invalid choice")
        
    print("\n👋 Done!")