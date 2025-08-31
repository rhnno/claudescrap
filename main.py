#!/usr/bin/env python3
"""
Enhanced ML-Powered Web Scraper - Main Application
Docker-compatible version with configurable analyzer
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path

# Add the func directory to Python path
# Using proper package imports

try:
    from func import BrowserManager, ConfigurableAnalyzer, EnhancedConfigurableAnalyzer, SmartTokopediaScraper, DataStorage, RandomUtils
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Make sure all required modules are in the func/ directory")
    sys.exit(1)

# Configure logging with proper path handling
import os
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Configure logging with cross-platform path
log_file = log_dir / 'analyzer.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedScraperApp:
    """Main application class for the enhanced ML-powered scraper"""
    
    def __init__(self):
        self.browser = None
        self.analyzer = None
        self.scraper = None
        self.running = True
        
        # Load configuration from environment
        self.config = self._load_config()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _load_config(self):
        """Load configuration from environment variables"""
        return {
            'max_pages': int(os.getenv('MAX_PAGES', 999999)),  # No limit by default
            'scrape_interval': int(os.getenv('SCRAPE_INTERVAL', 7200)),
            'training_interval': int(os.getenv('TRAINING_INTERVAL', 86400)),
            'default_site': os.getenv('DEFAULT_SITE', 'tokopedia'),
            'enable_multi_site': os.getenv('ENABLE_MULTI_SITE', 'true').lower() == 'true',
            'auto_train_model': os.getenv('AUTO_TRAIN_MODEL', 'true').lower() == 'true',
            'collect_training_data': os.getenv('COLLECT_TRAINING_DATA', 'true').lower() == 'true',
            'analyzer_config_path': os.getenv('ANALYZER_CONFIG_PATH', 'config/enhanced_analyzer_config.yaml'),
            'debug_level': os.getenv('ANALYZER_DEBUG_LEVEL', 'INFO'),
            'enable_performance_monitoring': os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true',
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Enhanced ML-Powered Web Scraper...")
        
        try:
            # Create necessary directories first
            logger.info("📁 Creating necessary directories...")
            directories = ['logs', 'data', 'data/raw_html', 'data/processed', 'models', 'config']
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize browser manager
            logger.info("🔧 Setting up browser...")
            self.browser = BrowserManager()
            self.browser.setup_driver()
            
            # Initialize analyzer with enhanced configuration
            logger.info("🧠 Initializing ML analyzer...")
            config_path = self.config['analyzer_config_path']
            if not os.path.exists(config_path):
                logger.warning(f"⚠️ Config file not found: {config_path}, using defaults")
                config_path = None
            
            self.analyzer = ConfigurableAnalyzer(config_path)
            
            # Initialize smart scraper
            logger.info("🕷️ Setting up smart scraper...")
            self.scraper = SmartTokopediaScraper(self.browser)
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def run_single_scrape(self, query="laptop", site_name=None):
        """Run a single scraping session"""
        if not site_name:
            site_name = self.config['default_site']
        
        logger.info(f"🔍 Starting scrape session: query='{query}', site='{site_name}'")
        
        try:
            # Reset analyzer state for new session
            self.analyzer.reset_state()
            
            # Perform smart scraping with ML-powered pagination detection
            results = self.scraper.smart_scrape(
                query=query, 
                max_pages=self.config['max_pages']
            )
            
            logger.info(f"📊 Scraping completed: {len(results)} pages processed")
            
            # Train model on session data if enabled
            if self.config['auto_train_model']:
                logger.info("🧠 Training model on session data...")
                success = self.scraper.train_on_current_session()
                if success:
                    logger.info("✅ Model training completed")
                else:
                    logger.warning("⚠️ Model training failed")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Scraping session failed: {e}")
            return []
    
    def run_continuous(self):
        """Run continuous scraping with intervals"""
        logger.info("🔄 Starting continuous scraping mode...")
        
        queries = ["laptop", "smartphone", "headphone", "camera", "tablet"]
        last_training_time = time.time()
        
        while self.running:
            try:
                for query in queries:
                    if not self.running:
                        break
                    
                    logger.info(f"🔍 Processing query: {query}")
                    results = self.run_single_scrape(query)
                    
                    if results:
                        logger.info(f"✅ Successfully processed {len(results)} pages for '{query}'")
                    else:
                        logger.warning(f"⚠️ No results for query '{query}'")
                    
                    # Check if it's time for model retraining
                    current_time = time.time()
                    if (current_time - last_training_time) >= self.config['training_interval']:
                        logger.info("🧠 Performing scheduled model retraining...")
                        if self.analyzer.train_model():
                            logger.info("✅ Scheduled training completed")
                            last_training_time = current_time
                        else:
                            logger.warning("⚠️ Scheduled training failed")
                    
                    # Random delay between queries
                    if self.running:
                        delay = RandomUtils.random_delay(5, 15)
                        logger.info(f"😴 Waiting {delay:.1f}s before next query...")
                        time.sleep(delay)
                
                if self.running:
                    logger.info(f"🕐 Waiting {self.config['scrape_interval']}s before next cycle...")
                    time.sleep(self.config['scrape_interval'])
                
            except KeyboardInterrupt:
                logger.info("🛑 Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"❌ Error in continuous mode: {e}")
                if self.running:
                    logger.info("⏳ Waiting 60s before retry...")
                    time.sleep(60)
    
    def run_development_mode(self):
        """Run in development mode with interactive features"""
        logger.info("🔧 Starting development mode...")
        
        # Run a few test scrapes
        test_queries = ["test laptop", "test phone"]
        
        for query in test_queries:
            logger.info(f"🧪 Testing with query: {query}")
            results = self.run_single_scrape(query)
            
            if results:
                # Show feature analysis
                logger.info("📊 Analyzing features...")
                features = self.analyzer.get_feature_summary(self.browser.driver)
                logger.info(f"📈 Feature completeness: {features.get('_feature_completeness', 0):.2f}")
            
            time.sleep(5)  # Reduced delay between tests
        
        logger.info("✅ Development mode testing completed")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🔧 Cleaning up resources...")
        
        try:
            if self.browser:
                self.browser.close()
                logger.info("✅ Browser closed")
            
            if self.analyzer:
                # Save any pending training data
                logger.info("💾 Saving final state...")
            
        except Exception as e:
            logger.error(f"⚠️ Error during cleanup: {e}")
        
        logger.info("✅ Cleanup completed")


class EnhancedTokopediaScraper:
    """Enhanced scraper using the new YAML template-based analyzer"""
    
    def __init__(self, browser_manager):
        self.browser = browser_manager
        self.analyzer = EnhancedConfigurableAnalyzer(
            config_path="config/analyzer_config.yaml",
            template_path="config/enhanced_training_templates.yaml"
        )
        self.storage = DataStorage()
        self.utils = RandomUtils()
    
    def smart_scrape(self, query, max_pages=50):
        """Intelligent scraping with enhanced ML pagination detection"""
        results = []
        url = f"https://www.tokopedia.com/search?st=product&q={query}"
        
        # Reset analyzer state for new session
        self.analyzer.reset_state()
        
        self.browser.navigate_to(url)
        
        page_num = 1
        consecutive_infinite_scroll = 0
        
        while page_num <= max_pages:
            print(f"\n🔍 Analyzing page {page_num} with Enhanced Analyzer...")
            
            # Extract features and predict pagination type using enhanced analyzer
            page_type = self.analyzer.analyze_page_structure(
                self.browser.driver, 
                site_name='tokopedia'  # Site-specific analysis
            )
            
            # Save current page
            filename = self._save_page(query, page_num)
            results.append({
                'page': page_num,
                'type': page_type,
                'filename': filename,
                'url': self.browser.driver.current_url
            })
            
            print(f"📊 Enhanced Analysis Result: {page_type}")
            
            # Handle different page types
            if page_type == 'pagination':
                print("📄 Detected: Traditional pagination")
                if not self._handle_pagination():
                    break
                    
            elif page_type == 'infinite_scroll':
                print("♾️ Detected: Infinite scroll")
                consecutive_infinite_scroll += 1
                
                if consecutive_infinite_scroll > 5:
                    print("⚠️ Too many infinite scroll detections, switching strategy")
                    if not self._handle_pagination():
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
        from selenium.webdriver.common.by import By
        
        next_selectors = [
            "button[aria-label*='next'], button[aria-label*='Next']",
            "a[aria-label*='next'], a[aria-label*='Next']", 
            ".pagination .next:not(.disabled)",
            "button[class*='next']:not([disabled])",
            "a[class*='next']:not([disabled])"
        ]
        
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
        """Handle infinite scroll with enhanced monitoring"""
        import random, time

        last_height = self.browser.driver.execute_script("return document.body.scrollHeight")

        for round_num in range(1, max_rounds + 1):
            print(f"📽 Enhanced infinite scroll round {round_num}/{max_rounds}")

            self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(*delay_range))

            new_height = self.browser.driver.execute_script("return document.body.scrollHeight")
            new_items_loaded = new_height - last_height

            if new_height == last_height:
                print("🏁 Reached the end of infinite scroll (no new content).")
                return False
            else:
                print(f"✅ New content loaded (+{new_items_loaded}px)")
                last_height = new_height

        return True

    def _save_page(self, query, page_num):
        """Save current page HTML safely"""
        import re
        
        try:
            folder = os.path.join(os.getcwd(), "data", "raw_html")
            os.makedirs(folder, exist_ok=True)

            safe_query = re.sub(r"[^a-zA-Z0-9_-]", "_", query)
            filename = os.path.join(folder, f"enhanced_page_{safe_query}_{page_num:03d}.html")

            try:
                html = self.browser.driver.page_source
            except Exception as e:
                print(f"⚠ Could not fetch page source: {e}")
                return None

            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)

            print(f"💾 Enhanced page saved: {filename}")
            return filename

        except Exception as e:
            print(f"⚠ Error saving page {page_num}: {e}")
            return None
    
    def _random_delay(self, min_sec=0.5, max_sec=2):
        """Random delay between actions"""
        import random
        import time
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)


def main():
    """Main entry point"""
    logger.info("🚀 Enhanced ML-Powered Web Scraper Starting...")
    logger.info(f"📅 Started at: {datetime.now().isoformat()}")
    
    # Create application instance
    app = EnhancedScraperApp()
    
    try:
        # Initialize components
        if not app.initialize():
            logger.error("❌ Failed to initialize application")
            sys.exit(1)
        
        # Determine run mode
        run_mode = os.getenv('RUN_MODE', 'continuous')
        
        if run_mode == 'development':
            app.run_development_mode()
        elif run_mode == 'single':
            query = os.getenv('SCRAPE_QUERY', 'laptop')
            results = app.run_single_scrape(query)
            logger.info(f"🎉 Single scrape completed: {len(results)} pages")
        else:
            app.run_continuous()
    
    except Exception as e:
        logger.error(f"💥 Application error: {e}")
        sys.exit(1)
    
    finally:
        app.cleanup()
        logger.info("👋 Enhanced ML-Powered Web Scraper Shutdown Complete")

if __name__ == "__main__":
    main()