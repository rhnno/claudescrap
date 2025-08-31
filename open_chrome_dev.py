#!/usr/bin/env python3
"""
Simple script to open Chrome using Selenium with research profile
"""

import os
import sys
import time
from pathlib import Path

def install_selenium():
    """Install selenium if not available"""
    try:
        import selenium
        return True
    except ImportError:
        print("📦 Installing selenium...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium", "webdriver-manager"])
            print("✅ Selenium installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install selenium")
            return False

def open_chrome_selenium():
    """Open Chrome using Selenium with research profile"""
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        print("✅ Selenium imported successfully")
    except ImportError as e:
        print(f"❌ Selenium import failed: {e}")
        return False
    
    # Profile directory
    profile_dir = os.path.abspath("chrome_profiles/research_profile")
    
    # Create profile directory if it doesn't exist
    os.makedirs(profile_dir, exist_ok=True)
    print(f"📁 Profile directory: {profile_dir}")
    
    # Setup Chrome options
    chrome_options = Options()
    
    # Profile and user data
    chrome_options.add_argument(f"--user-data-dir={profile_dir}")
    
    # Development and anti-detection options
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    chrome_options.add_argument("--no-first-run")
    chrome_options.add_argument("--no-default-browser-check")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-translate")
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    chrome_options.add_argument("--disable-device-discovery-notifications")
    chrome_options.add_argument("--start-maximized")
    
    # Remove automation indicators
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # URLs to visit (Indonesian e-commerce sites)
    sites = [
        {"name": "Tokopedia", "url": "https://www.tokopedia.com"},
        {"name": "Shopee", "url": "https://shopee.co.id"},
        {"name": "Bukalapak", "url": "https://www.bukalapak.com"}
    ]
    
    driver = None
    
    try:
        print("🚀 Starting Chrome with Selenium...")
        print("🔧 Features enabled:")
        print("   - Research profile persistence")
        print("   - Anti-automation detection")
        print("   - Web security disabled")
        print("   - Popup blocking disabled")
        
        # Try to use webdriver-manager for automatic ChromeDriver management
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            print("✅ Using webdriver-manager for ChromeDriver")
        except ImportError:
            print("⚠️ webdriver-manager not available, using system ChromeDriver")
            service = Service()  # Will use ChromeDriver from PATH
        
        # Create WebDriver instance
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Remove automation indicators
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print("✅ Chrome opened with Selenium")
        print(f"📁 Profile: research_profile")
        
        # Visit each e-commerce site
        for i, site in enumerate(sites, 1):
            print(f"\n🌐 [{i}/{len(sites)}] Opening {site['name']}...")
            driver.get(site['url'])
            
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
        print(f"   📁 Location: {profile_dir}")
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
        print(f"❌ Error with Selenium: {e}")
        print("🔧 Troubleshooting:")
        print("   1. Make sure Chrome is installed")
        print("   2. Install ChromeDriver or webdriver-manager")
        print("   3. Close any existing Chrome instances")
        return False
        
    finally:
        if driver:
            try:
                driver.quit()
                print("✅ Browser closed successfully")
            except:
                print("⚠️ Browser may still be running")

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
    
    # Check if selenium is available
    if not install_selenium():
        print("❌ Cannot proceed without selenium")
        sys.exit(1)
    
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
            print("   from func.browser import BrowserManager")
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