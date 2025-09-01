#!/usr/bin/env python3
"""
Scraping Orchestrator for Enhanced Tokopedia Scraper
Production-ready orchestrator that uses trained ML models for intelligent scraping
"""

import os
import sys
import time
import json
import csv
from datetime import datetime
import pandas as pd

# Using proper package imports

from func import EnhancedConfigurableAnalyzer, BrowserManager, DataStorage, RandomUtils

class ScrapingOrchestrator:
    """Production orchestrator for intelligent scraping with trained models"""
    
    def __init__(self, use_trained_model=True):
        """Initialize the scraping orchestrator"""
        self.browser = None
        self.storage = DataStorage()
        self.utils = RandomUtils()
        
        # Initialize enhanced analyzer with trained model
        self.analyzer = EnhancedConfigurableAnalyzer(
            config_path="config/analyzer_config.yaml",
            template_path="config/enhanced_training_templates.yaml"
        )
        
        # Load trained model
        if use_trained_model:
            if self.analyzer.load_model():
                print("✅ Trained ML model loaded successfully")
                self.model_loaded = True
            else:
                print("⚠️ No trained model found, using enhanced rule-based analysis")
                self.model_loaded = False
        else:
            print("ℹ️ Using rule-based analysis only")
            self.model_loaded = False
        
        # Scraping statistics
        self.session_stats = {
            'start_time': None,
            'total_products': 0,
            'total_pages': 0,
            'sites_scraped': 0,
            'queries_processed': 0,
            'pagination_detected': 0,
            'infinite_scroll_detected': 0,
            'last_page_detected': 0,
            'errors': 0
        }
        
        print("🚀 Scraping Orchestrator initialized")
    
    def setup_browser(self, headless=False, use_profile=True):
        """Setup browser for production scraping"""
        try:
            self.browser = BrowserManager(
                headless=headless,
                use_profile=use_profile,
                profile_name="research_profile"
            )
            self.browser.setup_driver()
            print("✅ Browser setup for production scraping")
            return True
        except Exception as e:
            print(f"❌ Browser setup failed: {e}")
            return False
    
    def run_batch_scraping(self, scraping_config):
        """
        Run batch scraping based on configuration
        
        scraping_config format:
        {
            "sites": [
                {
                    "name": "tokopedia",
                    "queries": ["laptop", "smartphone"],
                    "max_pages": 10,
                    "scroll_depth": 3
                }
            ],
            "output_format": ["csv", "excel", "json"],
            "delay_range": [1, 3]
        }
        """
        print("🚀 Starting batch scraping...")
        self.session_stats['start_time'] = datetime.now()
        
        all_results = {}
        
        for site_config in scraping_config.get('sites', []):
            site_name = site_config['name']
            print(f"\n🌐 Processing site: {site_name}")
            
            site_results = {}
            
            for query in site_config.get('queries', []):
                print(f"\n🔍 Query: {query}")
                
                try:
                    # Scrape products for this query
                    products = self._scrape_query(
                        site_name=site_name,
                        query=query,
                        max_pages=site_config.get('max_pages', 10),
                        scroll_depth=site_config.get('scroll_depth', 3),
                        delay_range=scraping_config.get('delay_range', [1, 3])
                    )
                    
                    if products:
                        site_results[query] = products
                        self.session_stats['total_products'] += len(products)
                        self.session_stats['queries_processed'] += 1
                        print(f"✅ {len(products)} products collected for '{query}'")
                    else:
                        print(f"⚠️ No products found for '{query}'")
                
                except Exception as e:
                    print(f"❌ Error scraping '{query}': {e}")
                    self.session_stats['errors'] += 1
                    continue
            
            if site_results:
                all_results[site_name] = site_results
                self.session_stats['sites_scraped'] += 1
        
        # Save results
        if all_results:
            self._save_batch_results(all_results, scraping_config)
            self._print_session_summary()
        else:
            print("❌ No results to save")
        
        return all_results
    
    def _scrape_query(self, site_name, query, max_pages, scroll_depth, delay_range):
        """Scrape a single query using intelligent pagination detection"""
        
        # Build URL based on site
        if site_name == 'tokopedia':
            search_query_url = query.replace(' ', '%20')
            url = f"https://www.tokopedia.com/search?st=product&q={search_query_url}"
        elif site_name == 'shopee':
            url = f"https://shopee.co.id/search?keyword={query}"
        else:
            print(f"⚠️ Unknown site: {site_name}")
            return []
        
        # Navigate to search page
        self.browser.navigate_to(url)
        time.sleep(3)
        
        # Reset analyzer state
        self.analyzer.reset_state()
        
        all_products = []
        page_num = 1
        consecutive_failures = 0
        
        while page_num <= max_pages:
            print(f"📄 Processing page {page_num}...")
            
            try:
                # Scroll to load content
                self._perform_scroll(scroll_depth)
                
                # Extract products from current page
                page_products = self._extract_products(site_name)
                
                if page_products:
                    all_products.extend(page_products)
                    consecutive_failures = 0
                    print(f"✅ Found {len(page_products)} products on page {page_num}")
                else:
                    consecutive_failures += 1
                    print(f"⚠️ No products found on page {page_num}")
                    
                    if consecutive_failures >= 2:
                        print("🏁 Multiple pages with no products, ending")
                        break
                
                # Use enhanced analyzer for intelligent navigation
                page_type = self.analyzer.analyze_page_structure(
                    self.browser.driver, 
                    site_name=site_name
                )
                
                print(f"🧠 ML Analysis: {page_type}")
                
                # Update statistics
                if page_type == 'pagination':
                    self.session_stats['pagination_detected'] += 1
                elif page_type == 'infinite_scroll':
                    self.session_stats['infinite_scroll_detected'] += 1
                elif page_type == 'last_page':
                    self.session_stats['last_page_detected'] += 1
                
                # Handle navigation based on detected type
                if page_type == 'pagination':
                    if not self._handle_pagination():
                        print("🏁 No more pages available")
                        break
                elif page_type == 'infinite_scroll':
                    if not self._handle_infinite_scroll():
                        print("🏁 Reached end of infinite scroll")
                        break
                elif page_type == 'last_page':
                    print("🏁 Last page detected")
                    break
                
                page_num += 1
                self.session_stats['total_pages'] += 1
                
                # Random delay
                delay = self.utils.random_delay(delay_range[0], delay_range[1])
                time.sleep(delay)
                
            except Exception as e:
                print(f"⚠️ Error on page {page_num}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("❌ Too many consecutive failures, stopping")
                    break
                continue
        
        return all_products
    
    def _perform_scroll(self, depth_scroll):
        """Perform scrolling to load content"""
        try:
            for i in range(depth_scroll):
                self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(self.utils.random_delay(0.5, 1.5))
        except Exception as e:
            print(f"⚠️ Scrolling error: {e}")
    
    def _extract_products(self, site_name):
        """Extract products based on site-specific selectors"""
        products = []
        
        try:
            if site_name == 'tokopedia':
                # Use Tokopedia-specific selectors from your original script
                product_items = self.browser.driver.find_elements(By.XPATH, './/*[contains(@class, "css-5wh65g")]')
                
                for item in product_items:
                    try:
                        product = self._extract_tokopedia_product(item)
                        if product:
                            products.append(product)
                    except Exception as e:
                        continue
                        
            elif site_name == 'shopee':
                # Shopee-specific selectors
                product_items = self.browser.driver.find_elements(By.CSS_SELECTOR, '[data-sqe="item"]')
                
                for item in product_items:
                    try:
                        product = self._extract_shopee_product(item)
                        if product:
                            products.append(product)
                    except Exception as e:
                        continue
            
        except Exception as e:
            print(f"❌ Error extracting products: {e}")
        
        return products
    
    def _extract_tokopedia_product(self, item):
        """Extract Tokopedia product using your existing selectors"""
        from selenium.webdriver.common.by import By
        
        product = {}
        
        try:
            # Product Link
            try:
                link_element = item.find_element(By.XPATH, './/a[contains(@href, "/")]')
                href = link_element.get_attribute('href')
                product['Link Product'] = href if href and "tokopedia.com" in href else "N/A"
            except:
                product['Link Product'] = "N/A"
            
            # Product Name
            try:
                product['Product Name'] = item.find_element(By.XPATH, './/*[contains(@class,"tnoqZhn89")]').text
            except:
                product['Product Name'] = "N/A"
            
            # Product Price
            try:
                product['Price'] = item.find_element(By.XPATH, './/*[contains(@class, "urMOIDHH7")]').text
            except:
                product['Price'] = "N/A"
            
            # Discount and Before Discount Price
            try:
                discount_element = item.find_element(By.XPATH, './/*[contains(@class, "discount")]')
                product['discount'] = discount_element.text
                
                before_discount_element = item.find_element(By.XPATH, './/*[contains(@class, "before-discount")]')
                product['Before Discount Price'] = before_discount_element.text
            except:
                product['discount'] = "N/A"
                product['Before Discount Price'] = "N/A"
            
            # Product Sold
            try:
                product_sell = item.find_element(By.XPATH, './/*[contains(text(), "terjual")]').text
                if product_sell == "":
                    product['Sold'] = "0 sold"
                else:
                    num = product_sell.split()[0]
                    product['Sold'] = f"{num} sold"
            except:
                product['Sold'] = "0 Sold"
            
            # Shop Name
            try:
                product['Shop Name'] = item.find_element(By.XPATH, './/*[contains(@class, "si3CNdiG8AR0EaXvf6bFbQ")]').text
            except:
                product['Shop Name'] = "N/A"
            
            # Product Rating
            try:
                product['Rating'] = item.find_element(By.XPATH, './/*[contains(@class, "55aCJ8bEsyw")]').text
            except:
                product['Rating'] = "N/A"
            
            # Product Location
            try:
                location_elements = item.find_elements(By.XPATH, './/*[contains(@class, "location")]')
                location_texts = [elem.text for elem in location_elements if elem.text.strip()]
                product['location'] = " | ".join(location_texts) if location_texts else "N/A"
            except:
                product['location'] = "N/A"
            
            # Add metadata
            product['scraped_at'] = datetime.now().isoformat()
            product['site'] = 'tokopedia'
            
            return product
            
        except Exception as e:
            return None
    
    def _extract_shopee_product(self, item):
        """Extract Shopee product (basic implementation)"""
        from selenium.webdriver.common.by import By
        
        product = {}
        
        try:
            # Basic Shopee selectors (you can enhance these)
            try:
                product['Product Name'] = item.find_element(By.CSS_SELECTOR, '[data-sqe="name"]').text
            except:
                product['Product Name'] = "N/A"
            
            try:
                product['Price'] = item.find_element(By.CSS_SELECTOR, '[class*="price"]').text
            except:
                product['Price'] = "N/A"
            
            try:
                product['Shop Name'] = item.find_element(By.CSS_SELECTOR, '[class*="shop"]').text
            except:
                product['Shop Name'] = "N/A"
            
            # Add standard fields for consistency
            product['Sold'] = "N/A"
            product['discount'] = "N/A"
            product['Before Discount Price'] = "N/A"
            product['Rating'] = "N/A"
            product['location'] = "N/A"
            product['Link Product'] = "N/A"
            
            # Add metadata
            product['scraped_at'] = datetime.now().isoformat()
            product['site'] = 'shopee'
            
            return product
            
        except Exception as e:
            return None
    
    def _handle_pagination(self):
        """Handle pagination navigation"""
        from selenium.webdriver.common.by import By
        
        next_selectors = [
            "button[aria-label*='Laman berikutnya']",  # Tokopedia Indonesian
            "button[aria-label*='next'], button[aria-label*='Next']",
            "a[aria-label*='next'], a[aria-label*='Next']",
            ".pagination .next:not(.disabled)",
            "button[class*='next']:not([disabled])"
        ]
        
        for selector in next_selectors:
            try:
                elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        element.click()
                        time.sleep(2)
                        return True
            except:
                continue
        
        return False
    
    def _handle_infinite_scroll(self, max_rounds=3):
        """Handle infinite scroll"""
        last_height = self.browser.driver.execute_script("return document.body.scrollHeight")
        
        for round_num in range(max_rounds):
            self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(self.utils.random_delay(1, 2))
            
            new_height = self.browser.driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                return False
            else:
                last_height = new_height
        
        return True
    
    def _save_batch_results(self, results, config):
        """Save batch results in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_formats = config.get('output_format', ['csv', 'excel'])
        
        # Flatten results for saving
        all_products = []
        for site_name, site_data in results.items():
            for query, products in site_data.items():
                for product in products:
                    product['query'] = query
                    product['site'] = site_name
                    all_products.append(product)
        
        if not all_products:
            print("⚠️ No products to save")
            return
        
        # Save in requested formats
        if 'csv' in output_formats:
            csv_file = f"batch_scraping_results_{timestamp}.csv"
            try:
                df = pd.DataFrame(all_products)
                df.to_csv(csv_file, index=False, encoding='utf-8')
                print(f"✅ CSV saved: {csv_file}")
            except Exception as e:
                print(f"❌ CSV save failed: {e}")
        
        if 'excel' in output_formats:
            excel_file = f"batch_scraping_results_{timestamp}.xlsx"
            try:
                df = pd.DataFrame(all_products)
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='All Results')
                    
                    # Create separate sheets for each site
                    for site_name in results.keys():
                        site_products = [p for p in all_products if p['site'] == site_name]
                        if site_products:
                            site_df = pd.DataFrame(site_products)
                            site_df.to_excel(writer, index=False, sheet_name=site_name.title())
                
                print(f"✅ Excel saved: {excel_file}")
            except Exception as e:
                print(f"❌ Excel save failed: {e}")
        
        if 'json' in output_formats:
            json_file = f"batch_scraping_results_{timestamp}.json"
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"✅ JSON saved: {json_file}")
            except Exception as e:
                print(f"❌ JSON save failed: {e}")
    
    def _print_session_summary(self):
        """Print session summary statistics"""
        end_time = datetime.now()
        duration = end_time - self.session_stats['start_time']
        
        print(f"\n📊 SCRAPING SESSION SUMMARY")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Sites scraped: {self.session_stats['sites_scraped']}")
        print(f"Queries processed: {self.session_stats['queries_processed']}")
        print(f"Total pages: {self.session_stats['total_pages']}")
        print(f"Total products: {self.session_stats['total_products']}")
        print(f"Errors: {self.session_stats['errors']}")
        
        print(f"\n🧠 ML ANALYSIS BREAKDOWN:")
        print(f"Pagination detected: {self.session_stats['pagination_detected']}")
        print(f"Infinite scroll detected: {self.session_stats['infinite_scroll_detected']}")
        print(f"Last page detected: {self.session_stats['last_page_detected']}")
        
        if self.session_stats['total_pages'] > 0:
            avg_products_per_page = self.session_stats['total_products'] / self.session_stats['total_pages']
            print(f"Average products per page: {avg_products_per_page:.1f}")
    
    def close(self):
        """Clean up resources"""
        if self.browser:
            self.browser.close()
            print("✅ Browser closed")


def load_scraping_config():
    """Load or create scraping configuration"""
    config_file = "scraping_config.json"
    
    default_config = {
        "sites": [
            {
                "name": "tokopedia",
                "queries": ["laptop", "smartphone", "sepatu"],
                "max_pages": 5,
                "scroll_depth": 3
            }
        ],
        "output_format": ["csv", "excel"],
        "delay_range": [1, 3]
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded from {config_file}")
            return config
        except Exception as e:
            print(f"⚠️ Error loading config: {e}, using default")
    
    # Save default config
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        print(f"✅ Default configuration saved to {config_file}")
    except Exception as e:
        print(f"⚠️ Error saving config: {e}")
    
    return default_config


def main():
    """Main scraping orchestrator function"""
    print("🕷️ Enhanced Scraping Orchestrator")
    print("=" * 60)
    
    # Load configuration
    config = load_scraping_config()
    
    print("📋 Current Configuration:")
    print(f"Sites: {[site['name'] for site in config['sites']]}")
    print(f"Total queries: {sum(len(site['queries']) for site in config['sites'])}")
    print(f"Output formats: {config['output_format']}")
    
    # Ask for confirmation or modification
    proceed = input("\n🚀 Proceed with this configuration? (y/n/edit): ").lower().strip()
    
    if proceed == 'edit':
        print("📝 Edit configuration in scraping_config.json and restart")
        return
    elif proceed != 'y':
        print("👋 Cancelled")
        return
    
    # Additional settings
    use_headless = input("🖥️ Run in headless mode? (y/n, default n): ").lower().strip() == 'y'
    use_trained_model = input("🧠 Use trained ML model? (y/n, default y): ").lower().strip() != 'n'
    
    # Initialize orchestrator
    orchestrator = ScrapingOrchestrator(use_trained_model=use_trained_model)
    
    try:
        # Setup browser
        if not orchestrator.setup_browser(headless=use_headless, use_profile=True):
            print("❌ Failed to setup browser")
            return
        
        # Run batch scraping
        print(f"\n🚀 Starting batch scraping...")
        results = orchestrator.run_batch_scraping(config)
        
        if results:
            print("\n✅ Batch scraping completed successfully!")
            print("📁 Check output files for results")
        else:
            print("\n❌ No results obtained")
    
    except KeyboardInterrupt:
        print("\n⏹️ Scraping interrupted by user")
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()