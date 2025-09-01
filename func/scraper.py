#!/usr/bin/env python3
"""
Scraper classes for the Enhanced ML-Powered Web Scraper
"""

import os
import re
import time
import random
from datetime import datetime
import csv
import pandas as pd
from selenium.webdriver.common.by import By

from .analyzer import EnhancedConfigurableAnalyzer, ConfigurableAnalyzer
from .browser import BrowserManager
from .storage import DataStorage
from .utils import RandomUtils


class SmartTokopediaScraper:
    """Scraper with ML-powered pagination detection"""
    
    def __init__(self, browser_manager):
        self.browser = browser_manager
        self.analyzer = ConfigurableAnalyzer()
        self.storage = DataStorage()  # Import from your storage module
    
    def smart_scrape(self, query, max_pages=50):
        """Intelligent scraping with ML pagination detection"""
        results = []
        url = f"https://www.tokopedia.com/search?st=product&q={query}"
        
        # Reset analyzer state for new session
        self.analyzer.reset_state()

        self.browser.navigate_to(url)

        page_num = 1
        consecutive_infinite_scroll = 0

        while page_num <= max_pages:
            print(f"\n🔍 Analyzing page {page_num}...")

            # Extract features and predict pagination type
            page_type = self.analyzer.analyze_page_structure(self.browser.driver)

            # Save current page
            filename = self._save_page(query, page_num)
            results.append({
                'page': page_num,
                'type': page_type,
                'filename': filename,
                'url': self.browser.driver.current_url
            })

            # ✅ Always collect training data (pagination, infinite_scroll, last_page)
            self.analyzer.collect_training_data(
                self.browser.driver,
                actual_type=page_type,
                site_name="tokopedia"
            )

            # Handle different page types
            if page_type == 'pagination':
                print("📄 Detected: Traditional pagination")
                if not self._handle_pagination():
                    break

            elif page_type == 'infinite_scroll':
                print("♾️ Detected: Infinite scroll")
                consecutive_infinite_scroll += 1

                if consecutive_infinite_scroll > 5:  # Prevent infinite loops
                    print("⚠️ Too many infinite scroll detections, switching strategy")
                    if not self._handle_pagination():  # Try pagination anyway
                        break
                else:
                    if not self._handle_infinite_scroll():
                        break

            elif page_type == 'last_page':
                print("🏁 Detected: Last page reached")
                break

            page_num += 1
            self._random_delay()

        return results

    def _handle_pagination(self):
        """Handle traditional pagination"""
        next_selectors = [
            "button[aria-label*='next'], button[aria-label*='Next']",
            "a[aria-label*='next'], a[aria-label*='Next']",
            ".pagination .next:not(.disabled)",
            "button[class*='next']:not([disabled])",
            "a[class*='next']:not([disabled])"
        ]
        
        # XPath selectors for text-based matching
        next_xpaths = [
            "//button[contains(text(),'›')]",
            "//a[contains(text(),'›')]",
            "//button[contains(text(),'Next')]",
            "//a[contains(text(),'Next')]"
        ]

        for selector in next_selectors:
            try:
                elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        element.click()
                        return True
            except:
                continue

        # Try XPath selectors
        for xpath in next_xpaths:
            try:
                elements = self.browser.driver.find_elements(By.XPATH, xpath)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        element.click()
                        return True
            except:
                continue
        
        return False
    
    def _handle_infinite_scroll(self, max_rounds=10, delay_range=(0.5, 1.5)):
        """
        Handle infinite scroll properly by scrolling all the way down repeatedly
        and collect training data per round with enhanced features.
        """
        import random, time

        last_height = self.browser.driver.execute_script("return document.body.scrollHeight")
        start_time = time.time()

        for round_num in range(1, max_rounds + 1):
            print(f"📽 Infinite scroll round {round_num}/{max_rounds}")

            # Scroll to bottom
            scroll_start_time = time.time()
            self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(*delay_range))
            scroll_end_time = time.time()

            # Measure new height
            new_height = self.browser.driver.execute_script("return document.body.scrollHeight")
            new_items_loaded = new_height - last_height
            scroll_time = scroll_end_time - scroll_start_time

            # ✅ Enhanced dynamic features
            dynamic_features = {
                "scroll_round": round_num,
                "new_items_loaded": new_items_loaded,
                "scroll_time_taken": scroll_time,
                "cumulative_scroll_time": scroll_end_time - start_time
            }

            # Collect training data with enhanced dynamic info
            self.analyzer.collect_training_data(
                self.browser.driver,
                actual_type="infinite_scroll",
                site_name="tokopedia",
                extra_features=dynamic_features
            )

            if new_height == last_height:
                print("🏁 Reached the end of infinite scroll (no new content).")
                return False
            else:
                print(f"✅ New content loaded (+{new_items_loaded}px in {scroll_time:.2f}s)")
                last_height = new_height

        return True

    def _save_page(self, query, page_num):
        """Save current page HTML safely"""
        try:
            folder = os.path.join(os.getcwd(), "data", "raw_html")
            os.makedirs(folder, exist_ok=True)

            # Sanitize query for filesystem safety
            safe_query = re.sub(r"[^a-zA-Z0-9_-]", "_", query)

            filename = os.path.join(folder, f"page_{safe_query}_{page_num:03d}.html")

            # Fetch page source safely
            try:
                html = self.browser.driver.page_source
            except Exception as e:
                print(f"⚠ Could not fetch page source: {e}")
                return None

            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)

            print(f"💾 Page saved: {filename}")
            return filename

        except Exception as e:
            print(f"⚠ Error saving page {page_num}: {e}")
            return None

    def _random_delay(self, min_sec=0.5, max_sec=2):
        """Random delay between actions with reduced timing"""
        import random
        import time
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)

    def train_on_current_session(self):
        """Train model based on current scraping session"""
        print("\n🧠 Training ML model on collected data...")
        success = self.analyzer.train_model()
        
        if success:
            print("✅ Model training completed!")
            # Show feature summary for the last analyzed page
            if hasattr(self, 'browser') and self.browser.driver:
                self.analyzer.get_feature_summary(self.browser.driver)
        else:
            print("⚠ Model training failed")
        
        return success


class EnhancedTokopediaScraper:
    """Enhanced Tokopedia scraper with ML-powered pagination detection"""

    def __init__(self, use_trained_model=True):
        """Initialize the enhanced scraper"""
        self.browser = None
        self.storage = DataStorage()
        self.utils = RandomUtils()

        # Initialize enhanced analyzer
        self.analyzer = EnhancedConfigurableAnalyzer(
            config_path="config/analyzer_config.yaml",
            template_path="config/enhanced_training_templates.yaml"
        )

        # Load trained model if available
        if use_trained_model:
            self.analyzer.load_model()
            if self.analyzer.is_trained:
                print("✅ Using trained ML model for pagination detection")
            else:
                print("⚠️ No trained model found, using enhanced rule-based analysis")

        print("🚀 Enhanced Tokopedia Scraper initialized")

    def setup_browser(self, headless=False, use_profile=True):
        """Setup browser with enhanced configuration"""
        try:
            self.browser = BrowserManager(
                headless=headless,
                use_profile=use_profile,
                profile_name="research_profile"
            )
            print("✅ Browser setup complete")
            return True
        except Exception as e:
            print(f"❌ Browser setup failed: {e}")
            return False

    def scrape_tokopedia_products(self, search_query, max_pages=10, depth_scroll=3):
        """
        Enhanced scraping with ML-powered pagination detection
        Uses your existing selectors with intelligent page navigation
        """
        if not self.browser:
            print("❌ Browser not initialized. Call setup_browser() first.")
            return []

        print(f"🔍 Starting enhanced scraping for: '{search_query}'")
        print(f"📊 Max pages: {max_pages}, Scroll depth: {depth_scroll}")

        # Prepare search URL (using your existing URL format)
        search_query_url = search_query.replace(' ', '%20')
        url = f"https://www.tokopedia.com/search?navsource=home&ob=5&search_id=20250821154911CAF1A3DFB505FF3D4RJW&source=universe&srp_component_id=04.06.00.00&st=product&q={search_query_url}"

        # Navigate to search page
        self.browser.navigate_to(url)
        time.sleep(3)

        all_products = []
        page_num = 1
        consecutive_no_products = 0

        # Reset analyzer state for new session
        self.analyzer.reset_state()

        while page_num <= max_pages:
            print(f"\n📄 Processing page {page_num}...")
            
            # Scroll to load content (using your existing scroll logic)
            self._perform_scroll(depth_scroll)

            # Extract products from current page
            page_products = self._extract_products_from_page()
            
            if page_products:
                all_products.extend(page_products)
                consecutive_no_products = 0
                print(f"✅ Found {len(page_products)} products on page {page_num}")
                print(f"📊 Total products so far: {len(all_products)}")
            else:
                consecutive_no_products += 1
                print(f"⚠️ No products found on page {page_num}")

                if consecutive_no_products >= 2:
                    print("🏁 No products found on multiple pages, ending scraping")
                    break

            # Use enhanced analyzer to determine page type and next action
            page_type = self.analyzer.analyze_page_structure(
                self.browser.driver,
                site_name='tokopedia'
            )
            
            print(f"🧠 Enhanced Analysis: {page_type}")
            
            # Handle different page types
            if page_type == 'pagination':
                print("📄 Detected: Traditional pagination")
                if not self._handle_pagination():
                    print("🏁 No more pages available")
                    break

            elif page_type == 'infinite_scroll':
                print("♾️ Detected: Infinite scroll")
                if not self._handle_infinite_scroll():
                    print("🏁 Reached end of infinite scroll")
                    break

            elif page_type == 'last_page':
                print("🏁 Detected: Last page reached")
                break
            
            page_num += 1

            # Random delay between pages
            self.utils.random_sleep(1, 3)

        print(f"\n🎉 Scraping completed! Total products: {len(all_products)}")
        return all_products

    def _perform_scroll(self, depth_scroll):
        """Perform scrolling to load content (your existing logic)"""
        try:
            for i in range(depth_scroll):
                print(f"📜 Scrolling {i+1}/{depth_scroll}...")
                self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(1, 2))

                # Check if new content loaded
                new_height = self.browser.driver.execute_script("return document.body.scrollHeight")
                if i > 0:  # Compare with previous height if available
                    print(f"   Page height: {new_height}px")
        except Exception as e:
            print(f"⚠️ Scrolling error: {e}")

    def _extract_products_from_page(self):
        """Extract products using your existing selectors"""
        products = []

        try:
            # Use your existing product container selector
            product_items = self.browser.driver.find_elements(By.XPATH, './/*[contains(@class, "css-5wh65g")]')
            print(f"🔍 Found {len(product_items)} product containers")
            
            for item in product_items:
                try:
                    product_data = self._extract_single_product(item)
                    if product_data:
                        products.append(product_data)
                except Exception as e:
                    print(f"⚠️ Error extracting product: {e}")
                    continue

        except Exception as e:
            print(f"❌ Error finding product containers: {e}")

        return products

    def _extract_single_product(self, item):
        """Extract single product data using selectors from the config file."""
        product_data = {}

        # Get the product detail selectors from the analyzer's templates
        site_template = self.analyzer._get_site_template('tokopedia')
        selectors = site_template.get('selectors', {}).get('product_details', {})

        for confidence, details in selectors.items():
            for selector_info in details:
                name = selector_info.get('name')
                xpath = selector_info.get('xpath')
                selector_type = selector_info.get('type', 'text')
                attribute = selector_info.get('attribute')

                if not name or not xpath:
                    continue

                try:
                    if selector_type == 'text':
                        product_data[name] = item.find_element(By.XPATH, xpath).text
                    elif selector_type == 'attribute' and attribute:
                        product_data[name] = item.find_element(By.XPATH, xpath).get_attribute(attribute)
                except Exception:
                    product_data[name] = "N/A"

        # Special handling for some fields if necessary
        if 'sold' in product_data and product_data['sold'] == "":
            product_data['sold'] = "0 sold"
        elif 'sold' in product_data and product_data['sold'] != "N/A":
            num = product_data['sold'].split()[0]
            product_data['sold'] = f"{num} sold"

        return product_data

    def _handle_pagination(self):
        """Handle traditional pagination with enhanced selectors"""
        # Enhanced selectors from YAML templates + your existing logic
        next_selectors = [
            "button[aria-label*='Laman berikutnya']",  # Tokopedia specific
            "button[aria-label*='next'], button[aria-label*='Next']",
            "a[aria-label*='next'], a[aria-label*='Next']",
            ".pagination .next:not(.disabled)",
            "button[class*='next']:not([disabled])",
            "a[class*='next']:not([disabled])"
        ]
        
        next_xpaths = [
            "//button[contains(@aria-label, 'Laman berikutnya')]",  # Indonesian
            "//button[contains(text(),'›')]",
            "//a[contains(text(),'›')]",
            "//button[contains(text(),'Next')]",
            "//a[contains(text(),'Next')]"
        ]
        
        # Try CSS selectors first
        for selector in next_selectors:
            try:
                elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        element.click()
                        time.sleep(2)
                        return True
            except Exception as e:
                continue

        # Try XPath selectors
        for xpath in next_xpaths:
            try:
                elements = self.browser.driver.find_elements(By.XPATH, xpath)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        element.click()
                        time.sleep(2)
                        return True
            except Exception as e:
                continue

        return False
    
    def _handle_infinite_scroll(self, max_rounds=5):
        """Handle infinite scroll with enhanced monitoring"""
        last_height = self.browser.driver.execute_script("return document.body.scrollHeight")

        for round_num in range(1, max_rounds + 1):
            print(f"📽 Infinite scroll round {round_num}/{max_rounds}")

            # Scroll to bottom
            self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1.5, 3))

            # Check for new content
            new_height = self.browser.driver.execute_script("return document.body.scrollHeight")

            if new_height == last_height:
                print("🏁 No new content loaded")
                return False
            else:
                print(f"✅ New content loaded (+{new_height - last_height}px)")
                last_height = new_height

        return True

    def save_results(self, products, search_query):
        """Save results using your existing format + enhanced storage"""
        if not products:
            print("⚠️ No products to save")
            return

        # Create safe filename (your existing logic)
        def safe_filename(query, extension):
            safename = re.sub(r'[\\/*?:"<>|]', '', query)
            safename = safename.replace(' ', '_')
            return f"enhanced_{safename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"

        csv_filename = safe_filename(search_query, 'csv')
        excel_filename = safe_filename(search_query, 'xlsx')

        # Save to CSV (your existing format)
        fieldnames = ['Product Name', 'Price', 'Sold', 'discount', 'Before Discount Price',
                     'Shop Name', 'location', 'Rating', 'Link Product']

        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(products)
            print(f"✅ CSV saved: {csv_filename}")
        except Exception as e:
            print(f"❌ CSV save failed: {e}")

        # Save to Excel (your existing format)
        try:
            df = pd.DataFrame(products)
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Enhanced Tokopedia Listings')
            print(f"✅ Excel saved: {excel_filename}")
        except Exception as e:
            print(f"❌ Excel save failed: {e}")

        # Enhanced storage using DataStorage class
        try:
            self.storage.save_to_csv(products, f"data_{search_query}")
            print("✅ Enhanced storage completed")
        except Exception as e:
            print(f"⚠️ Enhanced storage failed: {e}")

        return csv_filename, excel_filename

    def print_results_summary(self, products):
        """Print results summary (enhanced version of your existing logic)"""
        if not products:
            print("📊 No products found")
            return

        print(f"\n📊 SCRAPING SUMMARY")
        print("=" * 50)
        print(f"Total products found: {len(products)}")
        print(f"Unique shops: {len(set(p.get('Shop Name', 'N/A') for p in products))}")
        print(f"Products with ratings: {len([p for p in products if p.get('Rating', 'N/A') != 'N/A'])}")
        print(f"Products with discounts: {len([p for p in products if p.get('discount', 'N/A') != 'N/A'])}")

        print(f"\n📋 SAMPLE PRODUCTS:")
        print("=" * 50)

        for index, product in enumerate(products[:5], start=1):  # Show first 5
            print(f"Product {index}:")
            print(f"  Name: {product.get('Product Name', 'N/A')}")
            print(f"  Price: {product.get('Price', 'N/A')}")
            print(f"  Sold: {product.get('Sold', 'N/A')}")
            print(f"  Shop: {product.get('Shop Name', 'N/A')}")
            print(f"  Rating: {product.get('Rating', 'N/A')}")
            print(f"  Link: {product.get('Link Product', 'N/A')[:50]}...")
            print("-" * 30)

        if len(products) > 5:
            print(f"... and {len(products) - 5} more products")

    def close(self):
        """Clean up resources"""
        if self.browser:
            self.browser.close()
            print("✅ Browser closed")
