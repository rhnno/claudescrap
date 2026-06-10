from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
import json
from pathlib import Path

class BrowserManager:
    """Handles all browser-related operations with persistent profile support"""
    
    def __init__(self, use_profile=True, profile_name="research_profile", headless=False):
        self.driver = None
        self.wait = None
        self.use_profile = use_profile
        self.profile_name = profile_name
        self.headless = headless
        self.profile_path = self._get_profile_path()
        self.credentials_file = "config/login_credentials.json"
    
    def _get_profile_path(self):
        """Get the path from chrome_profiles folder for the given profile name"""
        profile_dir = Path("chrome_profiles")
        profile_dir.mkdir(exist_ok=True)
        return profile_dir / self.profile_name
    
    def _load_credentials(self):
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

    def setup_driver(self):
        """Setup Chrome driver with persistent profile and anti-detection options"""
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
        
        # Enhanced stealth options
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        
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
    
    def navigate_to(self, url):
        """Navigate to a URL and wait for page load"""
        self.driver.get(url)
        return self.wait_for_page_load()
    
    def human_navigation_trick(self) -> None:
        import time
        try:
            # Back via JavaScript
            print('  → Back...')
            self.driver.execute_script('window.history.back()')
            time.sleep(0.5)
            
            # Forward via JavaScript  
            print('  → Forward...')
            self.driver.execute_script('window.history.forward()')
            time.sleep(1)
            print('  ✔ Navigation trick selesai')
            
        except Exception as e:
            print(f'  ⚠ Navigation trick gagal: {e}')
    
    def wait_for_page_load(self, timeout=5):
        """Wait for page to fully load with reduced timing"""
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(0.5)  # Reduced wait for dynamic content
            return True
        except:
            print("⚠️ Page load timeout")
            return False
    
    def auto_login(self, site_name):
        """Automatically login to specified site using saved credentials"""
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
                print(f" [Browser] Auto-login not implemented for {site_name}")
                return False
                
        except Exception as e:
            print(f" [Browser] Auto-login failed for {site_name}: {e}")
            return False
    
    def _login_tokopedia(self, creds):
        """Login to Tokopedia"""
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
            
            print(" [Browser] Tokopedia login successful")
            return True
            
        except TimeoutException:
            print(" [Browser] Tokopedia login timeout - may need manual intervention")
            return False
        except Exception as e:
            print(f" [Browser] Tokopedia login error: {e}")
            return False
    
    def _login_shopee(self, creds):
        """Login to Shopee"""
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
            
            print("[Browser] Shopee login successful")
            return True
            
        except TimeoutException:
            print(" [Browser] Shopee login timeout - may need manual intervention")
            return False
        except Exception as e:
            print(f" [Browser] Shopee login error: {e}")
            return False
    
    def _login_bukalapak(self, creds):
        """Login to Bukalapak"""
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
            
            print(" [Browser] Bukalapak login successful")
            return True
            
        except TimeoutException:
            print(" [Browser] Bukalapak login timeout - may need manual intervention")
            return False
        except Exception as e:
            print(f" [Browser] Bukalapak login error: {e}")
            return False
        
    def save_cookies(self, filepath: str = "condig/tokopedia_cookies.json") -> bool:
        """save cookies after manually login"""
        try:
            cookies = self.driver.get_cookies()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(cookies, f)
            print(f" [Browser] Cookies saved: {len(cookies)} cookies")
            return True
        except Exception as e:
            print(f" [Browser] Failed to save cookies: {e}")
            return False
        
    def load_cookies(self, filepath: str = "config/tokopedia_cookies.json") -> bool:
        """Load Cookies from saved json file -- must be navigated into domain first before load"""
        try:
            if not os.path.exists(filepath):
                print(" [Browser] Cookie files not found")
                return False
        
            self.driver.get("https://www.tokopedia.com")
            import time
            time.sleep(2)
        
            with open(filepath, 'r') as f:
                cookies = json.load(f)

            for cookie in cookies:
                # remove non compatible key ;to be honest idk why;
                cookie.pop('sameSite', None)
                cookie.pop('expiry', None)
                try:
                    self.driver.add_cookie(cookie)
                except Exception:
                    continue
            
            print(f"✅ Loaded cookie : {len(cookies)} cookies")
            return True
        except Exception as e:
            print(f" [Browser] Failed to load cookie: {e}")
            return False
        
    def check_login_status(self, site_name):
        """Check if already logged in to a site"""
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
            
            print(f"[Browser] Not logged in to {site_name}")
            return False
            
        except Exception as e:
            print(f"[Browser] Error checking login status for {site_name}: {e}")
            return False
    
    def ensure_login(self, site_name):
        """Ensure user is logged in to the specified site"""
        if not self.check_login_status(site_name):
            print(f"[Browser] Need to login to {site_name}")
            return self.auto_login(site_name)
        return True
    
    def close(self):
        """Close the browser"""
        if self.driver:
            print(" [Browser] Closing browser and saving profile...")
            self.driver.quit()

