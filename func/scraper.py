from selenium.webdriver.common.by import By
from .utils import RandomUtils
from .storage import DataStorage
from .analyzer import SmartTokopediaScraper
from .analyzer import TrainingDataCollector
from .analyzer import ConfigurableAnalyzer


import os

class TokopediaScraper:
    """Handles Tokopedia-specific scraping logic"""
    
    def __init__(self, browser_manager):
        self.browser = browser_manager
        self.storage = DataStorage()
        self.utils = RandomUtils()
        self.analyzer = ConfigurableAnalyzer()
    
    def build_search_url(self, query):
        """Build Tokopedia search URL"""
        
        base_url = "https://www.tokopedia.com/search?st&q="
        encoded_query = query.replace(' ', '%20')
        return f"{base_url}{encoded_query}"
    
    def find_next_button(self):
        """Find next page button using multiple strategies"""
        selectors = [
            'button[class*="css-1turmok-unf-btn"][class*="eg8apji0"]',
            'button[class="css-dzvl4q-unf-pagination-item"]',
            'button[aria-label="Laman berikutnya"]'
        ]
        
        for selector in selectors:
            try:
                elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        print(f"✅ Found next button: {selector}")
                        return element
            except Exception as e:
                print(f"❌ Selector failed: {selector} - {e}")
                continue
        
        return None
    
    def scrape_pages(self, query, max_pages=10):
        """Main scraping method"""
        url = self.build_search_url(query)
        results = []
        
        print(f"🚀 Starting scrape for: {query}")
        self.browser.navigate_to(url)
        
        for page_num in range(1, max_pages + 1):
            page_type = self.analyzer.analyze_page_structure(self.browser.driver)
            
            if page_type == "pagination":
                next_btn = self.find_next_button()
                if next_btn: next_btn.click()

            elif page_type == "infinite_scroll":
                self.utils.human_like_scroll(self.browser.driver, scroll_pause_time=2)
                self.analyzer.collect_training_data(
                    self.browser.driver,
                    actual_type="infinite_scroll",
                    site_name="tokopedia",
                    extra_features={"scroll_round": page_num}
                )

            elif page_type == "last_page":
                break
            
            # Wait for page load
            if not self.browser.wait_for_page_load():
                print(f"❌ Page {page_num} failed to load")
                continue
            
            # Save page source
            filename = self.storage.save_html(
                self.browser.driver.page_source, 
                query, 
                page_num
            )
            
            results.append({
                'page': page_num,
                'url': self.browser.driver.current_url,
                'filename': filename
            })
            
            # Check for last page
            if page_num >= max_pages:
                break
            
            # Find next button
            self._scroll_to_pagination()
            next_button = self.find_next_button()
            
            if next_button:
                try:
                    next_button.click()
                    self.utils.random_sleep()
                except Exception as e:
                    print(f"❌ Click failed: {e}")
                    break
            else:
                print("❌ No more pages")
                break

            page_type = self.analyzer(self.browser.driver)
        
            if page_type == 'pagination':
            # Use traditional next button clicking
                success = self.analyzer.click_next_button(self.browser.driver)
            elif page_type == 'infinite_scroll':
            # Use scroll-based loading
                success = self.analyzer.handle_infinite_scroll(self.browser.driver)
            elif page_type == 'last_page':
                # Stop scraping
                break
        
        return results
    
    def _scroll_to_pagination(self):
        """Scroll to pagination area"""
        try:
            self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.utils.random_sleep(0.5, 1)
        except Exception as e:
            print(f"❌ Scroll error: {e}")