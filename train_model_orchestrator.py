#!/usr/bin/env python3
"""
Training Orchestrator for Enhanced Tokopedia Scraper
Collects training data and trains the ML model for pagination detection
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json
import csv
from selenium.webdriver.common.by import By

# Using proper package imports

from func import EnhancedConfigurableAnalyzer, BrowserManager, DataStorage, RandomUtils

class TrainingOrchestrator:
    """Orchestrates the training process for the enhanced analyzer"""
    
    def __init__(self):
        """Initialize the training orchestrator"""
        self.browser = None
        self.analyzer = EnhancedConfigurableAnalyzer(
            config_path="config/analyzer_config.yaml",
            template_path="config/enhanced_training_templates.yaml"
        )
        self.storage = DataStorage()
        self.utils = RandomUtils()
        
        # Training configuration
        self.training_sites = [
            {
                'name': 'tokopedia',
                'base_url': 'https://www.tokopedia.com/search?q={query}',
                'queries': ['laptop', 'smartphone', 'sepatu', 'baju', 'tas']
            },
            {
                'name': 'shopee',
                'base_url': 'https://shopee.co.id/search?keyword={query}',
                'queries': ['laptop', 'hp', 'fashion', 'elektronik', 'rumah']
            }
        ]
        
        self.training_data = []
        self.training_log = []
        
        print("🧠 Training Orchestrator initialized")
    
    def setup_browser(self, headless=True):
        """Setup browser for training data collection"""
        try:
            self.browser = BrowserManager(
                headless=headless,
                use_profile=True,
                profile_name="research_profile"
            )
            # Actually setup the driver - this was missing!
            self.browser.setup_driver()
            print("✅ Browser setup for training")
            return True
        except Exception as e:
            print(f"❌ Browser setup failed: {e}")
            return False
    
    def collect_training_data(self, max_pages_per_query=10, max_samples=500):
        """
        Collect training data from multiple sites and scenarios
        """
        print(f"📊 Starting training data collection...")
        print(f"🎯 Target: {max_samples} samples, {max_pages_per_query} pages per query")
        
        total_samples = 0
        
        for site_config in self.training_sites:
            if total_samples >= max_samples:
                break
                
            site_name = site_config['name']
            print(f"\n🌐 Collecting data from {site_name}...")
            
            for query in site_config['queries']:
                if total_samples >= max_samples:
                    break
                    
                print(f"🔍 Query: {query}")
                
                # Navigate to search page
                url = site_config['base_url'].format(query=query)
                self.browser.navigate_to(url)
                time.sleep(3)
                
                # Reset analyzer state
                self.analyzer.reset_state()
                
                page_num = 1
                while page_num <= max_pages_per_query and total_samples < max_samples:
                    print(f"📄 Page {page_num}...")
                    
                    # Wait for page to load
                    time.sleep(2)
                    
                    # Collect training sample
                    sample = self._collect_training_sample(site_name, query, page_num, url)
                    
                    if sample:
                        self.training_data.append(sample)
                        total_samples += 1
                        print(f"✅ Sample {total_samples}: {sample['page_type']}")
                        
                        # Log the sample
                        self.training_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'site': site_name,
                            'query': query,
                            'page': page_num,
                            'type': sample['page_type'],
                            'confidence': sample.get('confidence', 0)
                        })
                    
                    # Try to navigate to next page for training
                    if not self._navigate_to_next_page():
                        print("🏁 No more pages available")
                        break
                    
                    page_num += 1
                    self.utils.random_sleep(1, 3)
        
        print(f"\n📊 Training data collection completed!")
        print(f"✅ Total samples collected: {len(self.training_data)}")
        return self.training_data
    
    def _collect_training_sample(self, site_name, query, page_num, url):
        """Collect a single training sample"""
        try:
            # Extract enhanced features
            features = self.analyzer._extract_enhanced_features(self.browser.driver, site_name)
            
            # Determine the actual page type (ground truth)
            actual_page_type = self._determine_actual_page_type()
            
            # Create training sample
            sample = {
                'site_name': site_name,
                'query': query,
                'page_num': page_num,
                'url': url,
                'page_type': actual_page_type,
                'timestamp': datetime.now().isoformat(),
                'features': features
            }
            
            # Add confidence score if available
            if self.analyzer.is_trained:
                prediction, confidence = self.analyzer._ml_prediction(features)
                sample['ml_prediction'] = prediction
                sample['confidence'] = confidence
            
            return sample
            
        except Exception as e:
            print(f"⚠️ Error collecting sample: {e}")
            return None
    
    def _determine_actual_page_type(self):
        """
        Determine the actual page type through multiple checks
        This is the ground truth for training
        """
        try:
            # Check for pagination indicators
            pagination_indicators = [
                "button[aria-label*='next']",
                "a[aria-label*='next']",
                ".pagination",
                "[class*='pagination']",
                "button[class*='next']"
            ]
            
            pagination_count = 0
            for selector in pagination_indicators:
                try:
                    elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    pagination_count += len([e for e in elements if e.is_displayed()])
                except:
                    continue
            
            # Check for infinite scroll indicators
            infinite_scroll_indicators = [
                "[data-testid*='infinite']",
                "[class*='infinite']",
                "[class*='loading']",
                "[class*='skeleton']"
            ]
            
            infinite_count = 0
            for selector in infinite_scroll_indicators:
                try:
                    elements = self.browser.driver.find_elements(By.CSS_SELECTOR, selector)
                    infinite_count += len([e for e in elements if e.is_displayed()])
                except:
                    continue
            
            # Check for end-of-results text
            end_text_patterns = [
                "tidak ada produk lagi",
                "tidak ada hasil lagi",
                "no more results",
                "end of results",
                "semua produk telah ditampilkan"
            ]
            
            page_text = self.browser.driver.find_element(By.TAG_NAME, "body").text.lower()
            has_end_text = any(pattern in page_text for pattern in end_text_patterns)
            
            # Determine page type based on indicators
            if has_end_text:
                return 'last_page'
            elif pagination_count > infinite_count and pagination_count > 0:
                return 'pagination'
            elif infinite_count > 0:
                return 'infinite_scroll'
            else:
                # Default based on page position
                current_url = self.browser.driver.current_url
                if 'page=' in current_url or 'p=' in current_url:
                    return 'pagination'
                else:
                    return 'infinite_scroll'
                    
        except Exception as e:
            print(f"⚠️ Error determining page type: {e}")
            return 'unknown'
    
    def _navigate_to_next_page(self):
        """Try to navigate to next page for training purposes"""
        try:
            # Use the enhanced analyzer's navigation methods
            if self.analyzer._enhanced_rule_based_analysis(self.browser.driver, 'tokopedia', {}) == 'pagination':
                return self._handle_pagination()
            else:
                return self._handle_infinite_scroll()
        except:
            return False
    
    def _handle_pagination(self):
        """Handle pagination navigation"""
        next_selectors = [
            "button[aria-label*='next']",
            "a[aria-label*='next']",
            ".pagination .next:not(.disabled)",
            "button[class*='next']:not([disabled])"
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
    
    def _handle_infinite_scroll(self):
        """Handle infinite scroll"""
        try:
            last_height = self.browser.driver.execute_script("return document.body.scrollHeight")
            self.browser.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.browser.driver.execute_script("return document.body.scrollHeight")
            return new_height > last_height
        except:
            return False
    
    def prepare_training_dataset(self):
        """Prepare the dataset for ML training"""
        print("📊 Preparing training dataset...")
        
        if not self.training_data:
            print("❌ No training data available")
            return None, None
        
        # Convert to DataFrame
        df_data = []
        for sample in self.training_data:
            row = {
                'site_name': sample['site_name'],
                'page_type': sample['page_type'],
                'page_num': sample['page_num']
            }
            
            # Add features
            features = sample.get('features', {})
            for key, value in features.items():
                if not key.startswith('_') and isinstance(value, (int, float)):
                    row[f'feature_{key}'] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Prepare features and labels
        feature_columns = [col for col in df.columns if col.startswith('feature_')]
        X = df[feature_columns].fillna(0)
        y = df['page_type']
        
        print(f"✅ Dataset prepared: {len(X)} samples, {len(feature_columns)} features")
        print(f"📊 Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, test_size=0.2):
        """Train the ML model"""
        print("🧠 Training ML model...")
        
        # Prepare dataset
        X, y = self.prepare_training_dataset()
        if X is None:
            return False
        
        # Train using the analyzer's training method
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, accuracy_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Model trained successfully!")
            print(f"📊 Accuracy: {accuracy:.3f}")
            print("\n📋 Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Save model
            self.analyzer.model = model
            self.analyzer.is_trained = True
            self.analyzer.save_model()
            
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_training_data(self):
        """Save training data and logs"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save training data
        training_file = f"models/training_data_{timestamp}.json"
        try:
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Training data saved: {training_file}")
        except Exception as e:
            print(f"❌ Failed to save training data: {e}")
        
        # Save training log
        log_file = f"logs/training_log_{timestamp}.csv"
        try:
            os.makedirs("logs", exist_ok=True)
            with open(log_file, 'w', newline='', encoding='utf-8') as f:
                if self.training_log:
                    writer = csv.DictWriter(f, fieldnames=self.training_log[0].keys())
                    writer.writeheader()
                    writer.writerows(self.training_log)
            print(f"✅ Training log saved: {log_file}")
        except Exception as e:
            print(f"❌ Failed to save training log: {e}")
    
    def close(self):
        """Clean up resources"""
        if self.browser:
            self.browser.close()
            print("✅ Browser closed")


def main():
    """Main training orchestrator function"""
    print("🧠 Enhanced Analyzer Training Orchestrator")
    print("=" * 60)
    
    # Configuration
    print("📋 Training Configuration:")
    try:
        max_samples = int(input("📊 Max training samples (default 100): ") or "100")
    except:
        max_samples = 100
    
    try:
        max_pages = int(input("📄 Max pages per query (default 5): ") or "5")
    except:
        max_pages = 5
    
    use_headless = input("🖥️ Run in headless mode? (y/n, default y): ").lower().strip() != 'n'
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator()
    
    try:
        # Setup browser
        if not orchestrator.setup_browser(headless=use_headless):
            print("❌ Failed to setup browser")
            return
        
        print(f"\n🚀 Starting training process...")
        print(f"🎯 Target: {max_samples} samples from multiple sites")
        
        # Collect training data
        training_data = orchestrator.collect_training_data(
            max_pages_per_query=max_pages,
            max_samples=max_samples
        )
        
        if not training_data:
            print("❌ No training data collected")
            return
        
        # Save training data
        orchestrator.save_training_data()
        
        # Train model
        print(f"\n🧠 Training ML model with {len(training_data)} samples...")
        if orchestrator.train_model():
            print("✅ Model training completed successfully!")
            
            # Test the trained model
            print("\n🧪 Testing trained model...")
            test_analyzer = EnhancedConfigurableAnalyzer(
                template_path="config/enhanced_training_templates.yaml"
            )
            test_analyzer.load_model()
            
            if test_analyzer.is_trained:
                print("✅ Trained model loaded successfully!")
                print("🎉 Training orchestrator completed!")
                print("\n📝 Next steps:")
                print("   1. Use the trained model in enhanced_tokoscrape.py")
                print("   2. Run scraping_orchestrator.py for production scraping")
                print("   3. Monitor model performance and retrain if needed")
            else:
                print("⚠️ Model loading test failed")
        else:
            print("❌ Model training failed")
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()