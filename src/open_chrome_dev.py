#!/usr/bin/env python3
"""
Chrome Development Profile Setup using BrowserManager
Integrates with the project's modular architecture
"""

import os
import sys
import time
from pathlib import Path

# Using proper package imports
from utils.browser import BrowserManager
   

def open_chrome_with_browser_manager():
    """Open Chrome using BrowserManager with research profile"""
    
    try:
        # Initialize BrowserManager with research profile
        browser = BrowserManager(
            headless=False,
            use_profile=True,
            profile_name="research_profile"
        )
        
        # Setup the driver
        browser.setup_driver()
        print("✅ BrowserManager initialized successfully")
        
        return browser
        
    except Exception as e:
        print(f"❌ BrowserManager setup failed: {e}")
        return None


def setup_development_sites(browser):
    """Navigate to development sites for manual login"""
    
    # URLs to visit (Indonesian e-commerce sites)
    sites = [
        {"name": "Tokopedia", "url": "https://www.tokopedia.com"},
        {"name": "Shopee", "url": "https://shopee.co.id"},
        {"name": "Bukalapak", "url": "https://www.bukalapak.com"}
    ]
    
    try:
        print("🚀 Starting development site setup...")
        print("🔧 Features enabled:")
        print("   - Research profile persistence")
        print("   - Anti-automation detection")
        print("   - Session management")
        print("   - BrowserManager integration")
        
        print("✅ Chrome opened with BrowserManager")
        print(f"📁 Profile: research_profile")
        
        # Visit each e-commerce site
        for i, site in enumerate(sites, 1):
            print(f"\n🌐 [{i}/{len(sites)}] Opening {site['name']}...")
            browser.navigate_to(site['url'])
            
            print(f"✅ Loaded: {site['url']}")
            print(f"📝 Please log into {site['name']} manually")
            print("   - Complete any 2FA/CAPTCHA verification")
            print("   - Save passwords when prompted")
            print("   - Your session will be saved in the profile")
            
            # Wait for user to complete login
            input(f"⏸️  Press Enter when you're done logging into {site['name']}...")
        
        print("\n🎉 All sites visited!")
        print("💾 Login sessions are saved in your research profile")
        print("\n📊 Profile Summary:")
        print(f"   📁 Location: {browser.profile_path}")
        print(f"   🌐 Sites configured: {len(sites)}")
        print("   🔐 Sessions: Saved and persistent")
        
        # Keep browser open for additional manual work
        keep_open = input("\n🔄 Keep browser open for additional setup? (y/n): ").lower().strip()
        
        if keep_open in ['y', 'yes']:
            print("🌐 Browser staying open...")
            print("💡 You can:")
            print("   - Visit additional sites")
            print("   - Test your login sessions")
            print("   - Bookmark important pages")
            print("   - Configure browser settings")
            input("\n⏸️  Press Enter when completely done...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with BrowserManager: {e}")
        print("🔧 Troubleshooting:")
        print("   1. Make sure Chrome is installed")
        print("   2. Check BrowserManager configuration")
        print("   3. Close any existing Chrome instances")
        return False
        
    finally:
        # Browser cleanup is handled by BrowserManager
        pass

def check_profile():
    """Check if profile directory exists and show info"""
    profile_dir = os.path.abspath("chrome_profiles/research_profile")
    
    if os.path.exists(profile_dir):
        print(f"✅ Profile directory exists: {profile_dir}")
        
        # Check for some common Chrome profile files
        profile_files = [
            "Preferences", "Local State", "History", 
            "Cookies", "Login Data", "Web Data"
        ]
        
        existing_files = []
        for file in profile_files:
            if os.path.exists(os.path.join(profile_dir, file)):
                existing_files.append(file)
        
        if existing_files:
            print(f"📄 Found {len(existing_files)} profile files:")
            for file in existing_files:
                print(f"   - {file}")
            print("🎉 Profile appears to be initialized!")
        else:
            print("📄 Profile directory is empty (new profile)")
    else:
        print(f"📁 Profile directory will be created: {profile_dir}")

if __name__ == "__main__":
    print("🌐 Chrome with Selenium - Research Profile")
    print("=" * 50)
    
    # Check current profile status
    check_profile()
    print()

    
    # Ask user if they want to proceed
    response = input("🚀 Open Chrome with Selenium using research profile? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        success = open_chrome_selenium()
        
        if success:
            print("\n🎉 Success! Your research profile is ready.")
            print("\n📝 Next steps:")
            print("   1. Use this profile for consistent scraping sessions")
            print("   2. Login sessions will persist between runs")
            print("   3. Integrate with your scraper using the same profile")
            print("\n🔧 Integration example:")
            print("   from src.utils.browser import BrowserManager")
            print("   browser = BrowserManager(use_profile=True, profile_name='research_profile')")
        else:
            print("\n❌ Failed to open Chrome with Selenium")
            print("🔧 Troubleshooting:")
            print("   1. Make sure Google Chrome is installed")
            print("   2. Install ChromeDriver or webdriver-manager")
            print("   3. Close any existing Chrome instances")
            print("   4. Try: pip install selenium webdriver-manager")
    else:
        print("👋 Cancelled. Run again when ready!")