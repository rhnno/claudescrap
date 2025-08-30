from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time

class BrowserManager:
    """Handles all browser-related operations"""
    
    def __init__(self):
        self.driver = None
        self.wait = None
    
    def setup_driver(self):
        """Setup Chrome driver with anti-detection options"""
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', True)
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
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 5)
        return self.driver
    
    def navigate_to(self, url):
        """Navigate to a URL and wait for page load"""
        self.driver.get(url)
        return self.wait_for_page_load()
    
    def wait_for_page_load(self, timeout=5):
        """Wait for page to fully load with reduced timing"""
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(0.5)  # Reduced wait for dynamic content
            return True
        except:
            print("⚠️ Page load timeout")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()

