#!/usr/bin/env python3
"""
Enhanced Tokopedia Scraper with ML-Powered Pagination Detection
Integrates your existing selectors with the enhanced analyzer system
"""

from selenium.webdriver.common.by import By
import time
import random
import pandas as pd
import csv
import re
from datetime import datetime

# Using proper package imports

# Import enhanced analyzer
from func import EnhancedConfigurableAnalyzer, BrowserManager, DataStorage, RandomUtils

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
        """Extract single product data using your existing selectors"""
        product_data = {}
        
        try:
            # Product Link (your existing logic)
            try:
                link_element = item.find_element(By.XPATH, './/a[contains(@href, "/")]')
                href = link_element.get_attribute('href')
                if href and "tokopedia.com" in href:
                    product_data['Link Product'] = href
                else:
                    product_data['Link Product'] = "N/A"
            except:
                product_data['Link Product'] = "N/A"
            
            # Product Name (your existing selector)
            try:
                product_data['Product Name'] = item.find_element(By.XPATH, './/*[contains(@class,"tnoqZhn89")]').text
            except:
                product_data['Product Name'] = "N/A"
            
            # Product Price (your existing selector)
            try:
                product_data['Price'] = item.find_element(By.XPATH, './/*[contains(@class, "urMOIDHH7")]').text
            except:
                product_data['Price'] = "N/A"
            
            # Discount and Before Discount Price (your existing logic)
            try:
                discount_element = item.find_element(By.XPATH, './/*[contains(@class, "discount")]')
                product_data['discount'] = discount_element.text
                
                before_discount_element = item.find_element(By.XPATH, './/*[contains(@class, "before-discount")]')
                product_data['Before Discount Price'] = before_discount_element.text
            except:
                product_data['discount'] = "N/A"
                product_data['Before Discount Price'] = "N/A"
            
            # Product Sold (your existing logic)
            try:
                product_sell = item.find_element(By.XPATH, './/*[contains(text(), "terjual")]').text
                if product_sell == "":
                    product_data['Sold'] = "0 sold"
                else:
                    num = product_sell.split()[0]
                    product_data['Sold'] = f"{num} sold"
            except:
                product_data['Sold'] = "0 Sold"
            
            # Shop Name (your existing selector)
            try:
                product_data['Shop Name'] = item.find_element(By.XPATH, './/*[contains(@class, "si3CNdiG8AR0EaXvf6bFbQ")]').text
            except:
                product_data['Shop Name'] = "N/A"
            
            # Product Rating (your existing selector)
            try:
                product_data['Rating'] = item.find_element(By.XPATH, './/*[contains(@class, "55aCJ8bEsyw")]').text
            except:
                product_data['Rating'] = "N/A"
            
            # Product Location (your existing logic)
            try:
                location_elements = item.find_elements(By.XPATH, './/*[contains(@class, "location")]')
                location_texts = [elem.text for elem in location_elements if elem.text.strip()]
                product_data['location'] = " | ".join(location_texts) if location_texts else "N/A"
            except:
                product_data['location'] = "N/A"
            
            return product_data
            
        except Exception as e:
            print(f"⚠️ Error extracting product data: {e}")
            return None
    
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


def main():
    """Main function to run enhanced scraping"""
    print("🚀 Enhanced Tokopedia Scraper")
    print("=" * 50)
    
    # Configuration
    search_query = input("🔍 Enter search query: ").strip()
    if not search_query:
        search_query = "laptop gaming"  # Default
    
    try:
        max_pages = int(input("📄 Max pages to scrape (default 5): ") or "5")
    except:
        max_pages = 5
    
    try:
        depth_scroll = int(input("📜 Scroll depth per page (default 3): ") or "3")
    except:
        depth_scroll = 3
    
    use_headless = input("🖥️ Run in headless mode? (y/n, default n): ").lower().strip() == 'y'
    
    # Initialize scraper
    scraper = EnhancedTokopediaScraper(use_trained_model=True)
    
    try:
        # Setup browser
        if not scraper.setup_browser(headless=use_headless, use_profile=True):
            print("❌ Failed to setup browser")
            return
        
        # Start scraping
        print(f"\n🔍 Starting enhanced scraping...")
        products = scraper.scrape_tokopedia_products(
            search_query=search_query,
            max_pages=max_pages,
            depth_scroll=depth_scroll
        )
        
        # Save and display results
        if products:
            csv_file, excel_file = scraper.save_results(products, search_query)
            scraper.print_results_summary(products)
            
            print(f"\n✅ Scraping completed successfully!")
            print(f"📁 Files saved: {csv_file}, {excel_file}")
        else:
            print("❌ No products found")
    
    except KeyboardInterrupt:
        print("\n⏹️ Scraping interrupted by user")
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()


if __name__ == "__main__":
    main()