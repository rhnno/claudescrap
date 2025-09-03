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
from src.utils.browser import BrowserManager
from src.utils.utils import RandomUtils
from selenium.webdriver.common.by import By

class ScrapingOrchestrator:
    """A rule-based orchestrator for scraping e-commerce sites."""
    
    def __init__(self):
        """Initialize the scraping orchestrator"""
        self.browser = None
        self.utils = RandomUtils()
        
        # Scraping statistics
        self.session_stats = {
            'start_time': None,
            'total_products': 0,
            'total_pages': 0,
            'sites_scraped': 0,
            'queries_processed': 0,
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
        """Scrape a single query for a given site."""
        
        # Build URL based on site
        if site_name == 'tokopedia':
            search_query_url = query.replace(' ', '%20')
            url = f"https://www.tokopedia.com/search?st=product&q={search_query_url}&page=1"
        elif site_name == 'shopee':
            url = f"https://shopee.co.id/search?keyword={query}"
        else:
            print(f"⚠️ Unknown site: {site_name}")
            return []
        
        # Navigate to search page
        self.browser.navigate_to(url)
        time.sleep(3)
        
        all_products = []
        page_num = 1
        
        while page_num <= max_pages:
            print(f"📄 Processing page {page_num}...")
            
            try:
                self._perform_scroll(scroll_depth)
                
                # Extract products from current page
                page_products = self._extract_products_from_page()
                
                if page_products:
                    all_products.extend(page_products)
                    valid_products = [p for p in page_products if p.get('name') != "Name not found"]
                    print(f"✅ Found {len(valid_products)} valid products on page {page_num} (total extracted: {len(page_products)})")
                    
                    # Increment page number and stats BEFORE pagination
                    page_num += 1
                    self.session_stats['total_pages'] += 1
                    
                    # Only try pagination if we haven't reached max pages
                    if page_num <= max_pages:
                        if not self._handle_pagination():
                            print("🏁 No more pages available or pagination failed.")
                            break
                        time.sleep(self.utils.random_delay(delay_range[0], delay_range[1]))
                    else:
                        print(f"🏁 Reached maximum pages ({max_pages})")
                        break
                else:
                    print("🏁 No products found. Assuming end of results.")
                    break
                
            except Exception as e:
                print(f"⚠️ Error on page {page_num}: {e}")
                import traceback
                traceback.print_exc()
                break # Exit loop on page error
        
        return all_products
    
    def _perform_scroll(self, depth_scroll):
        """Perform scrolling to load content"""
        try:
            for i in range(depth_scroll):
                self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(self.utils.random_delay(0.5, 1.5))
        except Exception as e:
            print(f"⚠️ Scrolling error: {e}")
    
    def _extract_products_from_page(self):
        """Extract products from current page"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.common.exceptions import NoSuchElementException
            
            products = []
            
            # Updated Tokopedia product selectors based on current structure
            product_selectors = [
                'div[class="css-5wh65g"]'
            ]
            
            product_elements = []
            for selector in product_selectors:
                try:
                    elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        product_elements = elements
                        print(f"✅ Found {len(elements)} products using selector: {selector}")
                        break
                except:
                    continue
            
            if not product_elements:
                print("⚠️ No product elements found with any selector")
                return []
            
            for element in product_elements[:60]:  # Limit to first 60 products per page
                try:
                    product = {}
                    
                    # Extract product name with multiple selector fallbacks
                    name_selectors = [
                        'span[class="+tnoqZhn89+NHUA43BpiJg=="]'
                    ]
                    
                    product['name'] = "Name not found"
                    for name_selector in name_selectors:
                        try:
                            name_elem = element.find_element(By.CSS_SELECTOR, name_selector)
                            if name_elem.text.strip():
                                product['name'] = name_elem.text.strip()
                                break
                        except:
                            continue
                    
                    # Extract price with multiple selector fallbacks
                    price_selectors = [
                        'div[class="urMOIDHH7I0Iy1Dv2oFaNw=="]'
                    ]
                    
                    product['price'] = "Price not found"
                    for price_selector in price_selectors:
                        try:
                            price_elem = element.find_element(By.CSS_SELECTOR, price_selector)
                            if price_elem.text.strip():
                                product['price'] = price_elem.text.strip()
                                break
                        except:
                            continue
                    
                    # Extract product URL with multiple approaches
                    try:
                        # Try to find the main product link
                        link_elem = element.find_element(By.CSS_SELECTOR, 'a[href*="/p/"]')
                        product['url'] = link_elem.get_attribute('href')
                    except:
                        try:
                            # Fallback to any link
                            link_elem = element.find_element(By.CSS_SELECTOR, 'a')
                            href = link_elem.get_attribute('href')
                            if href and 'tokopedia.com' in href:
                                product['url'] = href
                            else:
                                product['url'] = "URL not found"
                        except:
                            product['url'] = "URL not found"
                    
                    # Only add products with valid names
                    if product['name'] != "Name not found":
                        products.append(product)
                
                except Exception as e:
                    print(f"⚠️ Error extracting product: {e}")
                    continue
            
            print(f"✅ Extracted {len(products)} valid products from page")
            return products
            
        except Exception as e:
            print(f"❌ Product extraction failed: {e}")
            return []
    
    
    def _handle_pagination(self):
        """Handle pagination navigation using URL-based approach only (safer from bot detection)"""
        
        # URL-based pagination only - no clicking to avoid honeypots
        try:
            current_url = self.browser.driver.current_url
            print(f"🔍 Current URL: {current_url}")
            
            next_url = self._get_next_page_url(current_url)
            print(f"🔍 Next URL: {next_url}")
            
            if next_url != current_url:  # Ensure URL actually changed
                self.browser.navigate_to(next_url)
                time.sleep(3)  # Longer wait for page load
                
                # Verify we actually navigated to the new page
                final_url = self.browser.driver.current_url
                print(f"🔍 Final URL after navigation: {final_url}")
                
                if final_url != current_url:
                    print("✅ Pagination via URL manipulation successful")
                    return True
                else:
                    print("⚠️ URL navigation failed - browser stayed on same page")
                    return False
            else:
                print("⚠️ URL pagination failed - no page parameter increment possible")
                return False
                
        except Exception as e:
            print(f"⚠️ URL pagination failed: {e}")
            return False
    
    def _save_batch_results(self, results, config):
        """Save batch results in multiple formats with organized directory structure"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_formats = config.get('output_format', ['csv', 'excel'])
        
        # **CUSTOMIZE OUTPUT NAME HERE** - Change this line to modify the base filename
        # Extract query names from results to include in filename
        query_names = []
        for site_name, site_data in results.items():
            for query in site_data.keys():
                query_names.append(query)
        
        # Create filename with query names (limit to first 3 queries to avoid long filenames)
        if query_names:
            query_part = "_".join(query_names[:3]).replace(" ", "_")
            base_filename = f"{query_part}_{timestamp}"
        else:
            base_filename = f"batch_scraping_results_{timestamp}"
        
        # Create directories if they don't exist
        directories = {
            'csv': 'data/raw/product_csv',
            'excel': 'data/raw/product_excel', 
            'json': 'data/raw/product_json'
        }
        
        for format_type, directory in directories.items():
            if format_type in output_formats:
                os.makedirs(directory, exist_ok=True)
        
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
        
        # Save in requested formats with organized paths
        if 'csv' in output_formats:
            csv_file = f"raw/product_csv/{base_filename}.csv"
            try:
                df = pd.DataFrame(all_products)
                df.to_csv(csv_file, index=False, encoding='utf-8')
                print(f"✅ CSV saved: {csv_file}")
            except Exception as e:
                print(f"❌ CSV save failed: {e}")
        
        if 'excel' in output_formats:
            excel_file = f"raw/product_excel/{base_filename}.xlsx"
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
            json_file = f"raw/product_json/{base_filename}.json"
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
        
        if self.session_stats['total_pages'] > 0:
            avg_products_per_page = self.session_stats['total_products'] / self.session_stats['total_pages']
            print(f"Average products per page: {avg_products_per_page:.1f}")
    
    def close(self):
        """Clean up resources"""
        if self.browser:
            self.browser.close()
            print("✅ Browser closed")

    def _get_next_page_url(self, current_url: str) -> str:
        """
        Intelligently increments the page number in a URL.
        Handles 'page=X', 'p=X', and the non-standard '{page=X}'.
        If no page parameter is found, it adds '&page=2'.
        """
        import re
        
        # Regex to find 'page=...' or 'p=...'
        match = re.search(r'(page|p)=(\d+)', current_url)
        
        if match:
            param_name = match.group(1)
            page_num = int(match.group(2))
            next_page_num = page_num + 1
            
            # Replace the old page number with the new one
            old_param = f"{param_name}={page_num}"
            new_param = f"{param_name}={next_page_num}"
            return current_url.replace(old_param, new_param, 1)
        
        # Handle the non-standard {page=...} format
        match_special = re.search(r'{page=(\d+)}', current_url)
        if match_special:
            page_num = int(match_special.group(1))
            next_page_num = page_num + 1

            old_param = f"{{page={page_num}}}"
            new_param = f"{{page={next_page_num}}}"
            return current_url.replace(old_param, new_param, 1)

        # If no page parameter is found, assume page 1 and add page=2
        if '?' in current_url:
            return f"{current_url}&page=2"
        else:
            return f"{current_url}?page=2"


def load_scraping_config():
    """Load or create scraping configuration"""
    config_file = "config/scraping_config.json"
    
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
    print("🕷️ Simple Scraping Orchestrator")
    print("=" * 60)
    
    # Load configuration
    config = load_scraping_config()
    
    print("📋 Current Configuration:")
    print(f"Sites: {[site['name'] for site in config['sites']]}")
    print(f"Total queries: {sum(len(site['queries']) for site in config['sites'])}")
    print(f"Output formats: {config['output_format']}")
    
    # Ask for confirmation or modification
    proceed = input("\n🚀 Proceed with this configuration? (y/n): ").lower().strip()
    
    if proceed != 'y':
        print("👋 Cancelled")
        return
    
    # Additional settings
    use_headless = input("🖥️ Run in headless mode? (y/n, default n): ").lower().strip() == 'y'
    
    # Initialize orchestrator
    orchestrator = ScrapingOrchestrator()
    
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