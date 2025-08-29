from func import BrowserManager, TokopediaScraper, ProductParser, DataStorage

def main():
    """Main orchestrator function"""
    query = "Kopi Bubuk"
    max_pages = 10
    
    # Initialize components
    browser = BrowserManager()
    storage = DataStorage()
    
    try:
        # Phase 1: Setup Browser
        print("🔧 Setting up browser...")
        browser.setup_driver()
        
        # Phase 2: Scrape Data  
        print("🕷️ Starting scraping...")
        scraper = TokopediaScraper(browser)
        results = scraper.scrape_pages(query, max_pages)
        
        # Phase 3: Parse Data
        print("📝 Parsing HTML files...")
        parser = ProductParser()
        all_products = parser.parse_all_files()
        
        # Phase 4: Save Data
        print("💾 Saving to CSV...")
        csv_file = storage.save_to_csv(all_products, query)
        
        print(f"🎉 Complete! Found {len(all_products)} products")
        print(f"📊 Data saved to: {csv_file}")
        
    except Exception as e:
        print(f"💥 Error: {e}")
    
    finally:
        print("🔧 Cleaning up...")
        browser.close()

if __name__ == "__main__":
    main()