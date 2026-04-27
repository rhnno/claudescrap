from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

class BrowserManager:
    """Handles all browser-related operations with persistent profile support.
    
    This class manages Chrome browser instances with advanced anti-detection features,
    persistent profile management, and automated login capabilities for e-commerce sites.
    
    Attributes:
        driver (Optional[webdriver.Chrome]): Chrome WebDriver instance
        wait (Optional[WebDriverWait]): WebDriver wait instance for element waiting
        use_profile (bool): Whether to use persistent Chrome profile
        profile_name (str): Name of the Chrome profile to use
        headless (bool): Whether to run browser in headless mode
        profile_path (Path): Path to Chrome profile directory
        credentials_file (str): Path to login credentials JSON file
    
    Example:
        >>> browser = BrowserManager(use_profile=True, headless=False)
        >>> browser.setup_driver()
        >>> browser.navigate_to("https://tokopedia.com")
        >>> browser.auto_login("tokopedia")
        >>> browser.close()
    
    Note:
        Implements multiple anti-detection techniques and supports persistent
        sessions across browser restarts through profile management.
    """
    
    def __init__(self, use_profile: bool = True, profile_name: str = "research_profile", headless: bool = False) -> None:
        """Initialize BrowserManager with configuration options.
        
        Sets up the browser manager with specified profile and display settings.
        
        Args:
            use_profile (bool, optional): Enable persistent Chrome profile. Defaults to True.
            profile_name (str, optional): Name of Chrome profile. Defaults to "research_profile".
            headless (bool, optional): Run browser in headless mode. Defaults to False.
        
        Note:
            Creates profile directory if it doesn't exist and sets up credentials file path.
        """
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        self.use_profile = use_profile
        self.profile_name = profile_name
        self.headless = headless
        self.profile_path = self._get_profile_path()
        self.credentials_file = "config/login_credentials.json"
    
    def _get_profile_path(self) -> Path:
        """Get the path for Chrome profile directory.
        
        Creates and returns the path to the Chrome profile directory,
        ensuring the directory exists.
        
        Returns:
            Path: Path object pointing to the profile directory
        
        Note:
            Creates chrome_profiles directory if it doesn't exist.
        """
        profile_dir = Path("chrome_profiles")
        profile_dir.mkdir(exist_ok=True)
        return profile_dir / self.profile_name
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load login credentials from config file"""
        try:
            if os.path.exists(self.credentials_file):
                with open(self.credentials_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default credentials file
                default_creds = {
                    "tokopedia": {
                        "email": "",
                        "password": "",
                        "login_url": "https://accounts.tokopedia.com/otp/c/page?otp_type=116&msisdn=&ld=https%3A%2F%2Fwww.tokopedia.com%2F"
                    },
                    "shopee": {
                        "email": "",
                        "password": "",
                        "login_url": "https://shopee.co.id/buyer/login"
                    },
                    "bukalapak": {
                        "email": "",
                        "password": "",
                        "login_url": "https://accounts.bukalapak.com/login"
                    }
                }
                os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
                with open(self.credentials_file, 'w', encoding='utf-8') as f:
                    json.dump(default_creds, f, indent=2)
                print(f"📝 Created credentials template at {self.credentials_file}")
                print("🔑 Please fill in your login credentials before using auto-login")
                return default_creds
        except Exception as e:
            print(f"⚠️ Error loading credentials: {e}")
            return {}

    def setup_driver(self) -> Optional[webdriver.Chrome]:
        """Setup Chrome driver with persistent profile and anti-detection options.
        
        Initializes Chrome WebDriver with comprehensive anti-detection measures,
        performance optimizations, and persistent profile support.
        
        Returns:
            Optional[webdriver.Chrome]: Configured Chrome WebDriver instance or None if failed
        
        Raises:
            Exception: Various WebDriver initialization errors
        
        Example:
            >>> browser = BrowserManager()
            >>> driver = browser.setup_driver()
            >>> if driver:
            ...     print("Browser ready for scraping")
        
        Note:
            Implements multiple anti-detection techniques including:
            - User agent rotation
            - WebDriver property masking
            - Automation extension disabling
            - Memory and performance optimizations
        """
        options = webdriver.ChromeOptions()
        
        # Headless mode if requested
        if self.headless:
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            print("🔧 Running in headless mode")
        
        # Anti-detection options
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Profile setup
        if self.use_profile:
            options.add_argument(f'--user-data-dir={self.profile_path}')
            options.add_argument('--profile-directory=Default')
            print(f"🔧 Using Chrome profile: {self.profile_path}")
        
        # Browser optimization flags for faster startup (from memory requirements)
        options.add_argument('--no-first-run')
        options.add_argument('--no-default-browser-check')
        options.add_argument('--disable-default-apps')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        
        # Enhanced stealth options
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-gpu')
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--remote-debugging-address=0.0.0.0")
        options.add_argument('--disable-images')  # Faster loading
        options.add_argument('--disable-javascript')  # Can be removed if JS is needed
        
        # Rotate user agents for better stealth
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        ]
        import random
        selected_ua = random.choice(user_agents)
        options.add_argument(f'--user-agent={selected_ua}')
        
        # Additional preferences
        prefs = {
            "profile.default_content_setting_values": {
                "notifications": 2,  # Block notifications
                "geolocation": 2,    # Block location requests
                "media_stream": 2,   # Block camera/mic
            },
            "profile.managed_default_content_settings": {
                "images": 2  # Block images for faster loading
            }
        }
        options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = webdriver.Chrome(options=options)
            
            # Execute stealth scripts
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
            self.driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})")
            
            self.wait = WebDriverWait(self.driver, 10)
            print("✅ Chrome driver setup successfully with research profile")
            return self.driver
            
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {e}")
            raise
    
    def navigate_to(self, url: str) -> bool:
        """Navigate to a URL and wait for page load.
        
        Navigates the browser to the specified URL and waits for the page
        to fully load before returning.
        
        Args:
            url (str): Target URL to navigate to
        
        Returns:
            bool: True if navigation successful, False otherwise
        
        Example:
            >>> success = browser.navigate_to("https://tokopedia.com")
            >>> if success:
            ...     print("Page loaded successfully")
        
        Note:
            Includes automatic page load waiting with timeout handling.
        """
        if not self.driver:
            print("❌ Driver not initialized")
            return False
        self.driver.get(url)
        return self.wait_for_page_load()
    
    def wait_for_page_load(self, timeout: int = 5) -> bool:
        """Wait for page to fully load with reduced timing"""
        if not self.wait:
            print("❌ WebDriverWait not initialized")
            return False
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(0.5)  # Reduced wait for dynamic content
            return True
        except:
            print("⚠️ Page load timeout")
            return False
    
    def auto_login(self, site_name: str) -> bool:
        """Automatically login to specified site using saved credentials.
        
        Performs automated login using credentials stored in the configuration file.
        Supports tokopedia, shopee, and bukalapak with site-specific login flows.
        
        Args:
            site_name (str): Name of the site to login to ('tokopedia', 'shopee', 'bukalapak')
        
        Returns:
            bool: True if login successful, False otherwise
        
        Raises:
            Exception: Various login-related errors including missing credentials
        
        Example:
            >>> if browser.auto_login("tokopedia"):
            ...     print("Successfully logged in to Tokopedia")
            ... else:
            ...     print("Login failed")
        
        Note:
            Requires valid credentials in config/login_credentials.json.
            Creates template file if it doesn't exist.
        """
        if not self.driver or not self.wait:
            print("❌ Browser not properly initialized")
            return False
            
        credentials = self._load_credentials()
        
        if site_name not in credentials:
            print(f"❌ No credentials found for {site_name}")
            return False
        
        site_creds = credentials[site_name]
        if not site_creds.get('email') or not site_creds.get('password'):
            print(f"⚠️ Empty credentials for {site_name}. Please update {self.credentials_file}")
            return False
        
        try:
            print(f"🔐 Attempting auto-login to {site_name}...")
            self.navigate_to(site_creds['login_url'])
            
            if site_name == 'tokopedia':
                return self._login_tokopedia(site_creds)
            elif site_name == 'shopee':
                return self._login_shopee(site_creds)
            elif site_name == 'bukalapak':
                return self._login_bukalapak(site_creds)
            else:
                print(f"❌ Auto-login not implemented for {site_name}")
                return False
                
        except Exception as e:
            print(f"❌ Auto-login failed for {site_name}: {e}")
            return False
    
    def _login_tokopedia(self, creds: Dict[str, Any]) -> bool:
        """Login to Tokopedia"""
        if not self.driver or not self.wait:
            print("❌ Browser not properly initialized")
            return False
            
        try:
            # Wait for email input
            email_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='email'], input[placeholder*='email']"))
            )
            email_input.clear()
            email_input.send_keys(creds['email'])
            time.sleep(1)
            
            # Find and click continue/next button
            continue_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], button[data-testid*='submit'], .btn-primary")
            continue_btn.click()
            time.sleep(2)
            
            # Wait for password input
            password_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password'], input[name='password']"))
            )
            password_input.clear()
            password_input.send_keys(creds['password'])
            time.sleep(1)
            
            # Submit login
            login_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], button[data-testid*='submit'], .btn-primary")
            login_btn.click()
            
            # Wait for login success (check for profile or dashboard elements)
            self.wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid*='profile'], .user-menu, .account-menu")),
                    EC.url_contains("tokopedia.com")
                )
            )
            
            print("✅ Tokopedia login successful")
            return True
            
        except TimeoutException:
            print("⚠️ Tokopedia login timeout - may need manual intervention")
            return False
        except Exception as e:
            print(f"❌ Tokopedia login error: {e}")
            return False
    
    def _login_shopee(self, creds: Dict[str, Any]) -> bool:
        """Login to Shopee"""
        if not self.driver or not self.wait:
            print("❌ Browser not properly initialized")
            return False
            
        try:
            # Wait for email/phone input
            email_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='loginKey'], input[placeholder*='email'], input[placeholder*='phone']"))
            )
            email_input.clear()
            email_input.send_keys(creds['email'])
            time.sleep(1)
            
            # Password input
            password_input = self.driver.find_element(By.CSS_SELECTOR, "input[name='password'], input[type='password']")
            password_input.clear()
            password_input.send_keys(creds['password'])
            time.sleep(1)
            
            # Submit login
            login_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit'], .btn-solid-primary")
            login_btn.click()
            
            # Wait for login success
            self.wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".navbar__username, .user-info")),
                    EC.url_contains("shopee.co.id")
                )
            )
            
            print("✅ Shopee login successful")
            return True
            
        except TimeoutException:
            print("⚠️ Shopee login timeout - may need manual intervention")
            return False
        except Exception as e:
            print(f"❌ Shopee login error: {e}")
            return False
    
    def _login_bukalapak(self, creds: Dict[str, Any]) -> bool:
        """Login to Bukalapak"""
        if not self.driver or not self.wait:
            print("❌ Browser not properly initialized")
            return False
            
        try:
            # Wait for email input
            email_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='user[email]'], input[type='email']"))
            )
            email_input.clear()
            email_input.send_keys(creds['email'])
            time.sleep(1)
            
            # Password input
            password_input = self.driver.find_element(By.CSS_SELECTOR, "input[name='user[password]'], input[type='password']")
            password_input.clear()
            password_input.send_keys(creds['password'])
            time.sleep(1)
            
            # Submit login
            login_btn = self.driver.find_element(By.CSS_SELECTOR, "input[type='submit'], button[type='submit']")
            login_btn.click()
            
            # Wait for login success
            self.wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".user-menu, .account-dropdown")),
                    EC.url_contains("bukalapak.com")
                )
            )
            
            print("✅ Bukalapak login successful")
            return True
            
        except TimeoutException:
            print("⚠️ Bukalapak login timeout - may need manual intervention")
            return False
        except Exception as e:
            print(f"❌ Bukalapak login error: {e}")
            return False
    
    def check_login_status(self, site_name: str) -> bool:
        """Check if already logged in to a site"""
        if not self.driver:
            print("❌ Driver not initialized")
            return False
            
        try:
            if site_name == 'tokopedia':
                # Check for Tokopedia login indicators
                login_indicators = [
                    "[data-testid*='profile']",
                    ".user-menu",
                    ".account-menu",
                    "[data-testid*='user-menu']"
                ]
            elif site_name == 'shopee':
                # Check for Shopee login indicators
                login_indicators = [
                    ".navbar__username",
                    ".user-info",
                    ".shopee-avatar"
                ]
            elif site_name == 'bukalapak':
                # Check for Bukalapak login indicators
                login_indicators = [
                    ".user-menu",
                    ".account-dropdown",
                    ".user-avatar"
                ]
            else:
                return False
            
            for indicator in login_indicators:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, indicator)
                    if elements and any(el.is_displayed() for el in elements):
                        print(f"✅ Already logged in to {site_name}")
                        return True
                except:
                    continue
            
            print(f"❌ Not logged in to {site_name}")
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking login status for {site_name}: {e}")
            return False
    
    def ensure_login(self, site_name: str) -> bool:
        """Ensure user is logged in to the specified site"""
        if not self.check_login_status(site_name):
            print(f"🔐 Need to login to {site_name}")
            return self.auto_login(site_name)
        return True
    
    def close(self) -> None:
        """Close the browser and save profile data.
        
        Properly closes the Chrome browser instance and saves any profile
        data for future sessions.
        
        Example:
            >>> browser.close()
            🔧 Closing browser and saving profile...
        
        Note:
            Should always be called when finished with the browser to
            prevent resource leaks and save session data.
        """
        if self.driver:
            print("🔧 Closing browser and saving profile...")
            self.driver.quit()

