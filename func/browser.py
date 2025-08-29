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
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 15_6_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 10)
        return self.driver
    
    def navigate_to(self, url):
        """Navigate to a URL and wait for page load"""
        self.driver.get(url)
        return self.wait_for_page_load()
    
    def wait_for_page_load(self, timeout=10):
        """Wait for page to fully load"""
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)  # Additional wait for dynamic content
            return True
        except:
            print("⚠️ Page load timeout")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()

