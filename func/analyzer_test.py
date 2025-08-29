import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import re
import pickle
import os
from datetime import datetime
import time

class PaginationAnalyzer:
    """
    ML-powered pagination pattern analyzer
    Determines if a page uses pagination, infinite scroll, or is the last page
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        self.is_trained = False
        self.features = []
        self.model_path = "models/pagination_model.pkl"
        self.vectorizer_path = "models/pagination_vectorizer.pkl"
        
        # Store state for dynamic features
        self._last_scroll_time = None
        self._last_page_height = None
        self._last_dom_count = None
        
        # Ensure model directory exists
        os.makedirs("models", exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def analyze_page_structure(self, driver):
        """
        Extract features from current page to determine pagination type
        Returns: 'pagination', 'infinite_scroll', 'last_page'
        """
        features = self._extract_features(driver)
        
        if not self.is_trained:
            print("⚠️ Model not trained, using rule-based fallback")
            return self._rule_based_analysis(driver)
        
        # Use ML model to predict
        feature_vector = self._features_to_vector(features)
        prediction = self.model.predict([feature_vector])[0]
        confidence = max(self.model.predict_proba([feature_vector])[0])
        
        print(f"🤖 ML Prediction: {prediction} (confidence: {confidence:.2f})")
        
        # Fallback to rules if confidence is low
        if confidence < 0.7:
            print("🔄 Low confidence, using rule-based fallback")
            return self._rule_based_analysis(driver)
        
        return prediction
    
    def _extract_features(self, driver):
        """Extract robust features from the page"""
        features = {}

        def safe_count(find_fn, locator, desc):
            """Helper to safely count elements"""
            try:
                return len(find_fn(locator))
            except Exception as e:
                print(f"⚠️ Failed to extract {desc}: {e}")
                return 0

        def safe_execute_script(script, desc, default=0):
            """Helper to safely execute JavaScript"""
            try:
                return driver.execute_script(script)
            except Exception as e:
                print(f"⚠️ Failed to execute {desc}: {e}")
                return default

        try:
            # === Original Features ===
            # 1. Button/Link Analysis
            features['pagination_buttons'] = safe_count(
                lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                "button[class*='pagination'], a[class*='pagination'], .pagination button, .pagination a",
                "pagination_buttons"
            )

            features['next_button'] = safe_count(
                lambda loc: driver.find_elements(By.XPATH, loc),
                "//*[contains(text(), 'Next') or contains(text(), 'next') or contains(text(), '›') or contains(text(), '→')]",
                "next_button"
            )

            # Instead of XPath regex, check in Python
            buttons = driver.find_elements(By.XPATH, "//button | //a")
            features['numbered_buttons'] = sum(
                1 for b in buttons if re.fullmatch(r"\d+", b.text.strip())
            )

            # 2. Infinite Scroll Indicators
            features['load_more_buttons'] = safe_count(
                lambda loc: driver.find_elements(By.XPATH, loc),
                "//*[contains(text(), 'Load More') or contains(text(), 'Show More') or contains(text(), 'Muat')]",
                "load_more_buttons"
            )

            features['lazy_load_elements'] = safe_count(
                lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                "[data-lazy], [loading='lazy'], .lazy, .skeleton, .shimmer",
                "lazy_load_elements"
            )

            # 3. Scroll Listeners (skip unreliable window.getEventListeners)
            features['scroll_listeners'] = 0

            # 4. Page Content Analysis
            try:
                page_height = driver.execute_script("return document.body.scrollHeight")
                viewport_height = driver.execute_script("return window.innerHeight")
                features['height_ratio'] = page_height / viewport_height if viewport_height > 0 else 0
            except Exception as e:
                print(f"⚠️ Failed to measure height_ratio: {e}")
                features['height_ratio'] = 0

            features['total_products'] = safe_count(
                lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                "[class*='product'], [class*='item'], [data-testid*='product']",
                "total_products"
            )

            try:
                page_text = driver.find_element(By.TAG_NAME, "body").text.lower()
                features['page_text_length'] = len(page_text)
                features['contains_page_numbers'] = 1 if re.search(r'page\s+\d+|halaman\s+\d+', page_text) else 0
                features['contains_total_results'] = 1 if re.search(r'total.*?\d+|hasil.*?\d+', page_text) else 0

                # Instead of embedding TF-IDF here, just store raw text
                features['page_text_raw'] = page_text  

            except Exception as e:
                print(f"⚠️ Failed to extract text content: {e}")
                features['page_text_length'] = 0
                features['contains_page_numbers'] = 0
                features['contains_total_results'] = 0
                features['page_text_raw'] = ""


            # 5. URL Analysis
            try:
                current_url = driver.current_url.lower()
                features['url_has_page_param'] = 1 if any(param in current_url for param in ['page=', 'p=', 'offset=']) else 0
            except Exception as e:
                print(f"⚠️ Failed to analyze URL: {e}")
                features['url_has_page_param'] = 0

            # 6. Dynamic content
            features['dynamic_content'] = safe_count(
                lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                "[class*='loading'], [class*='spinner'], [class*='skeleton']",
                "dynamic_content"
            )

            # === NEW ENHANCED FEATURES ===
            
            # 1. Scroll Velocity
            current_height = safe_execute_script("return document.body.scrollHeight", "current_height")
            current_time = time.time()
            
            if self._last_page_height is not None and self._last_scroll_time is not None:
                height_diff = current_height - self._last_page_height
                time_diff = current_time - self._last_scroll_time
                features['scroll_velocity'] = height_diff / max(0.1, time_diff)  # pixels per second
            else:
                features['scroll_velocity'] = 0
            
            # Update state for next call
            self._last_page_height = current_height
            self._last_scroll_time = current_time

            # 2. Content Density
            features['content_density'] = features['total_products'] / max(1, features['page_text_length'])

            # 3. New DOM Nodes
            current_dom_count = safe_execute_script(
                "return document.querySelectorAll('*').length", 
                "dom_count", 
                0
            )
            
            if self._last_dom_count is not None:
                features['new_dom_nodes'] = current_dom_count - self._last_dom_count
            else:
                features['new_dom_nodes'] = 0
            
            self._last_dom_count = current_dom_count

            # 4. End of Results Text Detection
            end_of_results_patterns = [
                r'no\s+more\s+results',
                r'end\s+of\s+results', 
                r'end\s+of\s+page',
                r'tidak\s+ada\s+hasil\s+lagi',  # Indonesian
                r'hasil\s+pencarian\s+habis',   # Indonesian
                r'semua\s+produk\s+telah\s+ditampilkan'  # Indonesian
            ]
            
            features['has_end_of_results_text'] = 0
            if page_text:  # Use page_text from earlier extraction
                for pattern in end_of_results_patterns:
                    if re.search(pattern, page_text, re.IGNORECASE):
                        features['has_end_of_results_text'] = 1
                        break

            # 5. Pagination at Footer
            features['pagination_at_footer'] = 0
            try:
                total_page_height = safe_execute_script("return document.body.scrollHeight", "total_page_height")
                footer_threshold = total_page_height * 0.8  # Last 20%
                
                pagination_elements = driver.find_elements(
                    By.CSS_SELECTOR,
                    "button[class*='pagination'], a[class*='pagination'], .pagination, [class*='pager']"
                )
                
                for element in pagination_elements:
                    try:
                        element_y = safe_execute_script(
                            f"return arguments[0].getBoundingClientRect().top + window.pageYOffset",
                            "element_y",
                            0
                        )
                        if element_y >= footer_threshold:
                            features['pagination_at_footer'] = 1
                            break
                    except:
                        continue
            
                        
            except Exception as e:
                print(f"⚠️ Failed to detect footer pagination: {e}")

            # 6. Average Product Height
            if features['total_products'] > 0 and current_height > 0:
                features['avg_product_height'] = current_height / features['total_products']
            else:
                features['avg_product_height'] = 0

            # 7. XHR Request Count (from browser logs)
            features['xhr_request_count'] = 0
            try:
                # Get browser logs (performance entries)
                performance_entries = safe_execute_script(
                    """
                    var entries = performance.getEntriesByType('resource');
                    var recent_requests = entries.filter(entry => 
                        entry.startTime > (performance.now() - 5000) && 
                        (entry.initiatorType === 'xmlhttprequest' || entry.initiatorType === 'fetch')
                    );
                    return recent_requests.length;
                    """,
                    "xhr_count",
                    0
                )
                features['xhr_request_count'] = performance_entries
                
            except Exception as e:
                print(f"⚠️ Failed to count XHR requests: {e}")

        except Exception as e:
            print(f"⚠ General extraction error: {e}")

        # Ensure all expected keys exist (backward compatibility)
        expected_keys = [
            # Original features
            'pagination_buttons', 'next_button', 'numbered_buttons',
            'load_more_buttons', 'lazy_load_elements', 'scroll_listeners',
            'height_ratio', 'total_products', 'page_text_length',
            'contains_page_numbers', 'contains_total_results', 'url_has_page_param',
            'dynamic_content',
            # New enhanced features
            'scroll_velocity', 'content_density', 'new_dom_nodes',
            'has_end_of_results_text', 'pagination_at_footer', 'avg_product_height',
            'xhr_request_count'
        ]
        
        for key in expected_keys:
            features.setdefault(key, 0)

        return features

    def _rule_based_analysis(self, driver):
        """Fallback rule-based analysis"""
        try:
            # Check for obvious pagination buttons
            next_buttons = driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Next')] | //a[contains(text(), 'Next')] | //*[contains(text(), '›')]")
            
            if next_buttons and any(btn.is_enabled() for btn in next_buttons):
                return 'pagination'
            
            # Check for load more buttons
            load_more = driver.find_elements(By.XPATH, 
                "//*[contains(text(), 'Load More') or contains(text(), 'Show More')]")
            
            if load_more:
                return 'infinite_scroll'
            
            # Enhanced: Check for end of results text
            try:
                page_text = driver.find_element(By.TAG_NAME, "body").text.lower()
                end_patterns = [
                    r'no\s+more\s+results', r'end\s+of\s+results', r'end\s+of\s+page',
                    r'tidak\s+ada\s+hasil\s+lagi', r'hasil\s+pencarian\s+habis'
                ]
                for pattern in end_patterns:
                    if re.search(pattern, page_text, re.IGNORECASE):
                        return 'last_page'
            except:
                pass
            
            # Check if we can scroll more
            current_height = driver.execute_script("return window.pageYOffset + window.innerHeight")
            total_height = driver.execute_script("return document.body.scrollHeight")
            
            if current_height >= total_height * 0.95:  # Near bottom
                return 'last_page'
            
            return 'infinite_scroll'  # Default assumption
            
        except Exception as e:
            print(f"⚠ Rule-based analysis error: {e}")
            return 'last_page'  # Conservative fallback
    
    def collect_training_data(self, driver, actual_type, site_name="unknown", extra_features=None):
        """Collect training data for the ML model (supports extra dynamic features)"""
        features = self._extract_features(driver)

        # Merge extra dynamic features if provided
        if extra_features:
            features.update(extra_features)

        # Add metadata
        features['site_name'] = site_name
        features['timestamp'] = datetime.now().isoformat()
        features['actual_type'] = actual_type
        features['url'] = driver.current_url

        # Save to training data file
        self._save_training_sample(features)
        print(f"📊 Training data collected: {actual_type} (extra: {extra_features})")

    def train_model(self, training_data_file="models/training_data.csv"):
        """Train the ML model on collected data"""
        try:
            import pandas as pd
            
            if not os.path.exists(training_data_file):
                print("⚠ No training data found. Collect some data first!")
                return False
            
            df = pd.read_csv(training_data_file)

            # --- TF-IDF Processing ---
            if 'page_text_raw' in df.columns:
                text_corpus = df['page_text_raw'].fillna("")
                self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
                tfidf_matrix = self.vectorizer.fit_transform(text_corpus)
                
                # Convert TF-IDF matrix to DataFrame
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(), 
                    columns=[f"tfidf_{w}" for w in self.vectorizer.get_feature_names_out()]
                )
                
                # Concatenate with original DataFrame
                df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

                # Save vectorizer
                with open(self.vectorizer_path, "wb") as f:
                    pickle.dump(self.vectorizer, f)
                print(f"💾 TF-IDF vectorizer saved to {self.vectorizer_path}")

            # --- Feature Selection ---
            feature_columns = [
                # Static features
                'pagination_buttons', 'next_button', 'numbered_buttons',
                'load_more_buttons', 'lazy_load_elements', 'scroll_listeners',
                'height_ratio', 'total_products', 'page_text_length',
                'contains_page_numbers', 'contains_total_results', 'url_has_page_param',
                'dynamic_content',
                # Dynamic features
                'scroll_round', 'new_items_loaded',
                'scroll_velocity', 'content_density', 'new_dom_nodes',
                'has_end_of_results_text', 'pagination_at_footer', 'avg_product_height',
                'xhr_request_count',
                # TF-IDF (all matching columns)
            ] + [col for col in df.columns if col.startswith("tfidf_")]

            # Ensure missing cols are added as 0
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0

            X = df[feature_columns]
            y = df['actual_type']

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            accuracy = self.model.score(X_test, y_test)
            print(f"🎯 Model accuracy: {accuracy:.2f}")

            # Save model
            self.save_model()
            self.is_trained = True
            return True

        except Exception as e:
            print(f"⚠ Training error: {e}")
            return False

    
    def _features_to_vector(self, features):
        """Convert features dict to numpy array"""
        feature_order = [
            # Original features
            'pagination_buttons', 'next_button', 'numbered_buttons',
            'load_more_buttons', 'lazy_load_elements', 'scroll_listeners',
            'height_ratio', 'total_products', 'page_text_length',
            'contains_page_numbers', 'contains_total_results', 'url_has_page_param',
            'dynamic_content',
            # New enhanced features
            'scroll_velocity', 'content_density', 'new_dom_nodes',
            'has_end_of_results_text', 'pagination_at_footer', 'avg_product_height',
            'xhr_request_count'
        ]
        
        return np.array([features.get(key, 0) for key in feature_order])
    
    def _save_training_sample(self, features):
        """Save a single training sample"""
        import csv
        
        file_path = "models/training_data.csv"
        file_exists = os.path.exists(file_path)
        
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = list(features.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(features)
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"💾 Model saved to {self.model_path}")
        except Exception as e:
            print(f"⚠ Error saving model: {e}")
    
    def load_model(self):
        """Load trained model and vectorizer from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                print(f"✅ Model loaded from {self.model_path}")

            if os.path.exists(self.vectorizer_path):
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print(f"✅ Vectorizer loaded from {self.vectorizer_path}")
                return True

        except Exception as e:
            print(f"⚠ Error loading model/vectorizer: {e}")

        return False

    def reset_state(self):
        """Reset internal state for new scraping session"""
        self._last_scroll_time = None
        self._last_page_height = None
        self._last_dom_count = None
        print("🔄 Analyzer state reset")

    def get_feature_summary(self, driver):
        """Get a summary of extracted features for debugging"""
        features = self._extract_features(driver)
        
        print("\n📊 Feature Summary:")
        print("-" * 50)
        
        # Group features by category
        categories = {
            "Navigation": ['pagination_buttons', 'next_button', 'numbered_buttons', 'pagination_at_footer'],
            "Infinite Scroll": ['load_more_buttons', 'lazy_load_elements', 'scroll_velocity', 'xhr_request_count'],
            "Content": ['total_products', 'content_density', 'avg_product_height', 'new_dom_nodes'],
            "Page Analysis": ['height_ratio', 'page_text_length', 'has_end_of_results_text'],
            "Metadata": ['contains_page_numbers', 'contains_total_results', 'url_has_page_param', 'dynamic_content']
        }
        
        for category, feature_list in categories.items():
            print(f"\n{category}:")
            for feature in feature_list:
                value = features.get(feature, 0)
                print(f"  {feature}: {value}")
        
        return features

# Enhanced Scraper with ML Analysis
class SmartTokopediaScraper:
    """Scraper with ML-powered pagination detection"""
    
    def __init__(self, browser_manager):
        self.browser = browser_manager
        self.analyzer = PaginationAnalyzer()
        self.storage = None  # Import from your storage module
    
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
            "button[contains(text(),'›')],"
            "a[contains(text(),'›')]",
            ".pagination .next:not(.disabled)",
            "button[class*='next']:not([disabled])",
            "a[class*='next']:not([disabled])"
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
        
        return False
    
    def _handle_infinite_scroll(self, max_rounds=10, delay_range=(1, 3)):
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
    
    def _random_delay(self, min_sec=2, max_sec=6):
        """Random delay between actions"""
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

# Training Data Collector
class TrainingDataCollector:
    """Helper class to collect training data for the ML model"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.sites = {
            'tokopedia': 'https://www.tokopedia.com/search?q=laptop',
            'amazon': 'https://www.amazon.com/s?k=laptop',
            'youtube': 'https://www.youtube.com/results?search_query=cina'
        }
    
    def collect_diverse_data(self, browser_manager, samples_per_site=20):
        """Collect training data from multiple sites"""
        print("📊 Starting training data collection...")
        
        for site_name, base_url in self.sites.items():
            print(f"\n🌐 Collecting data from {site_name}")
            
            # Reset analyzer state for each site
            self.analyzer.reset_state()
            
            try:
                browser_manager.navigate_to(base_url)
                
                for i in range(samples_per_site):
                    # Manual labeling prompt (in real scenario)
                    print(f"\n📄 Page {i+1} on {site_name}")
                    print("Current URL:", browser_manager.driver.current_url)
                    
                    # In practice, you would manually check and label
                    # For demo, we'll use rule-based labeling
                    actual_type = self._auto_label_page(browser_manager.driver)
                    
                    # Enhanced: Show feature summary for manual verification
                    features_summary = self.analyzer.get_feature_summary(browser_manager.driver)
                    
                    # Collect features
                    self.analyzer.collect_training_data(
                        browser_manager.driver, 
                        actual_type, 
                        site_name
                    )
                    
                    # Try to navigate to next page for variety
                    self._navigate_next_page(browser_manager.driver)
                    
            except Exception as e:
                print(f"⚠ Error collecting data from {site_name}: {e}")
    
    def _auto_label_page(self, driver):
        """Automatically label page type with richer rules"""
        try:
            # 1. Check for traditional pagination controls
            pagination_buttons = driver.find_elements(
                By.CSS_SELECTOR,
                "button[class*='pagination'], a[class*='pagination'], .pagination button, .pagination a"
            )
            next_buttons = driver.find_elements(
                By.XPATH,
                "//*[contains(text(), 'Next') or contains(text(), '›') or contains(text(), '→')]"
            )

            if pagination_buttons or next_buttons:
                return "pagination"

            # 2. Look for infinite scroll indicators
            load_more = driver.find_elements(
                By.XPATH,
                "//*[contains(text(), 'Load More') or contains(text(), 'Show More') or contains(text(), 'Muat')]"
            )
            lazy_elements = driver.find_elements(
                By.CSS_SELECTOR,
                "[data-lazy], [loading='lazy'], .lazy, .skeleton, .shimmer"
            )

            if load_more or lazy_elements:
                return "infinite_scroll"

            # 3. URL pattern check (common for pagination)
            current_url = driver.current_url.lower()
            if any(param in current_url for param in ["page=", "p=", "offset="]):
                return "pagination"

            # 4. Scroll height check (heuristic for infinite scroll vs last page)
            page_height = driver.execute_script("return document.body.scrollHeight")
            viewport_height = driver.execute_script("return window.innerHeight")
            if page_height > viewport_height * 2:  # very tall page
                return "infinite_scroll"

            # 5. Text cues in body
            body_text = driver.find_element(By.TAG_NAME, "body").text.lower()
            if "page" in body_text or "halaman" in body_text:  # English/Indonesian
                return "pagination"

            # Default assumption
            return "last_page"

        except Exception as e:
            print(f"⚠️ Auto-labeling failed: {e}")
            return "last_page"

    
    def _navigate_next_page(self, driver):
        """Try to navigate to next page"""
        try:
            next_btn = driver.find_element(By.XPATH, "//*[contains(text(), 'Next') or contains(text(), '›')]")
            if next_btn.is_enabled():
                next_btn.click()
                time.sleep(3)
        except:
            pass

# Usage Example
def main_with_ml():
    """Main function using ML-powered scraper"""
    from browser import BrowserManager  # Your existing browser manager
    
    # Setup
    browser = BrowserManager()
    browser.setup_driver()
    
    try:
        # Option 1: Train model first (do this once)
        collector = TrainingDataCollector(PaginationAnalyzer())
        collector.collect_diverse_data(browser, samples_per_site=10)
        
        # Option 2: Use smart scraper
        scraper = SmartTokopediaScraper(browser)
        results = scraper.smart_scrape("laptop", max_pages=50)
        
        print(f"\n🎉 Scraped {len(results)} pages intelligently!")
        
        # Train model on session data
        scraper.train_on_current_session()
        
    finally:
        browser.close()

if __name__ == "__main__":
    main_with_ml()