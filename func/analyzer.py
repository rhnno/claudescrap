import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import re
import pickle
import os
import yaml
from datetime import datetime
import time
import csv
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from .utils import RandomUtils
from .storage import DataStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurableAnalyzer:
    """
    ML-powered pagination pattern analyzer with YAML configuration support
    Determines if a page uses pagination, infinite scroll, or is the last page
    """
    
    def __init__(self, config_path=None):
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
        
        # Load configuration
        self.config = self._load_config(config_path)
        self._setup_paths()
        
        # Try to load existing model
        self.load_model()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        default_config = {
            'paths': {
                'model_dir': 'models',
                'training_data_file': 'models/training_data.csv',
                'model_file': 'pagination_model.pkl',
                'vectorizer_file': 'pagination_vectorizer.pkl'
            },
            'features': {
                'text_analysis': {
                    'max_tfidf_features': 100,
                    'stop_words': 'english',
                    'end_of_results_patterns': [
                        r'no\s+more\s+results',
                        r'end\s+of\s+results',
                        r'end\s+of\s+page',
                        r'tidak\s+ada\s+hasil\s+lagi',
                        r'hasil\s+pencarian\s+habis',
                        r'semua\s+produk\s+telah\s+ditampilkan',
                        r'no\s+more\s+items',
                        r'that\'s\s+all\s+folks'
                    ]
                },
                'selectors': {
                    'pagination_buttons': [
                        "button[class*='pagination']",
                        "a[class*='pagination']",
                        ".pagination button",
                        ".pagination a",
                        "[aria-label*='pagination']"
                    ],
                    'next_buttons': [
                        "//*[contains(text(), 'Next')]",
                        "//*[contains(text(), 'next')]",
                        "//*[contains(text(), '›')]",
                        "//*[contains(text(), '→')]",
                        "//*[contains(@aria-label, 'next')]"
                    ],
                    'load_more_buttons': [
                        "//*[contains(text(), 'Load More')]",
                        "//*[contains(text(), 'Show More')]",
                        "//*[contains(text(), 'Muat')]",
                        "//*[contains(text(), 'Tampilkan Lebih')]"
                    ],
                    'lazy_load_elements': [
                        "[data-lazy]",
                        "[loading='lazy']",
                        ".lazy",
                        ".skeleton",
                        ".shimmer",
                        ".loading-placeholder"
                    ],
                    'product_elements': [
                        "[class*='product']",
                        "[class*='item']",
                        "[data-testid*='product']",
                        ".product-card",
                        ".item-card"
                    ]
                },
                'thresholds': {
                    'confidence_threshold': 0.7,
                    'footer_threshold_ratio': 0.8,
                    'scroll_bottom_threshold': 0.95,
                    'max_infinite_scroll_rounds': 10,
                    'consecutive_infinite_limit': 5
                }
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'n_estimators': 100,
                'auto_collect_samples': True,
                'samples_per_site': 20
            },
            'sites': {
                'tokopedia': {
                    'base_url': 'https://www.tokopedia.com/search?q={query}',
                    'language': 'indonesian',
                    'patterns': {
                        'page_text': ['halaman', 'produk'],
                        'end_results': ['tidak ada hasil lagi', 'hasil pencarian habis']
                    }
                },
                'amazon': {
                    'base_url': 'https://www.amazon.com/s?k={query}',
                    'language': 'english',
                    'patterns': {
                        'page_text': ['page', 'results'],
                        'end_results': ['no more results', 'end of results']
                    }
                }
            },
            'debug': {
                'verbose': True,
                'save_features': True,
                'feature_summary': True
            }
        }
        
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                # Merge with default config
                config = self._deep_merge(default_config, loaded_config)
                print(f"✅ Configuration loaded from {config_path}")
            elif config_path:
                config = default_config
                # Create default config file
                os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, default_flow_style=False, indent=2)
                print(f"📄 Default configuration created at {config_path}")
            else:
                # No config path provided, use defaults without creating file
                config = default_config
                print("📄 Using default configuration (no config file specified)")
        except Exception as e:
            print(f"⚠️ Error loading config, using defaults: {e}")
            config = default_config
        
        return config
    
    def _deep_merge(self, base, override):
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _setup_paths(self):
        """Setup file paths from configuration"""
        try:
            model_dir = self.config['paths']['model_dir']
            os.makedirs(model_dir, exist_ok=True)
            
            self.model_path = os.path.join(model_dir, self.config['paths']['model_file'])
            self.vectorizer_path = os.path.join(model_dir, self.config['paths']['vectorizer_file'])
            
            # Update TF-IDF parameters from config
            text_config = self.config['features']['text_analysis']
            self.vectorizer = TfidfVectorizer(
                max_features=text_config['max_tfidf_features'],
                stop_words=text_config['stop_words']
            )
            
            # Ensure all necessary directories exist
            additional_dirs = ['data', 'data/raw_html', 'data/processed', 'logs', 'config']
            for directory in additional_dirs:
                os.makedirs(directory, exist_ok=True)
                
        except Exception as e:
            print(f"⚠️ Error setting up paths: {e}")
            # Fallback to default paths
            self.model_path = "models/pagination_model.pkl"
            self.vectorizer_path = "models/pagination_vectorizer.pkl"
            os.makedirs("models", exist_ok=True)
    
    def analyze_page_structure(self, driver):
        """
        Extract features from current page to determine pagination type
        Returns: 'pagination', 'infinite_scroll', 'last_page'
        """
        features = self._extract_features(driver)
        
        if not self.is_trained:
            if self.config['debug']['verbose']:
                print("⚠️ Model not trained, using rule-based fallback")
            return self._rule_based_analysis(driver)
        
        # Use ML model to predict
        feature_vector = self._features_to_vector(features)
        prediction = self.model.predict([feature_vector])[0]
        confidence = max(self.model.predict_proba([feature_vector])[0])
        
        if self.config['debug']['verbose']:
            print(f"🤖 ML Prediction: {prediction} (confidence: {confidence:.2f})")
        
        # Fallback to rules if confidence is low
        confidence_threshold = self.config['features']['thresholds']['confidence_threshold']
        if confidence < confidence_threshold:
            if self.config['debug']['verbose']:
                print("🔄 Low confidence, using rule-based fallback")
            return self._rule_based_analysis(driver)
        
        return prediction
    
    def _extract_features(self, driver):
        """Extract robust features from the page using enhanced configuration"""
        features = {}
        # Handle both old and new config structure
        if 'features' in self.config and 'selectors' in self.config['features']:
            selectors = self.config['features']['selectors']
        elif 'selectors' in self.config:
            selectors = self.config['selectors']
        else:
            # Fallback to default selectors
            selectors = {
                'pagination_buttons': ["button[class*='pagination']", ".pagination button"],
                'next_buttons': ["//*[contains(text(), 'Next')]", "//*[contains(text(), '›')]"],
                'load_more_buttons': ["//*[contains(text(), 'Load More')]"],
                'lazy_load_elements': ["[data-lazy]", "[loading='lazy']"]
            }
        
        # Performance monitoring
        import time
        start_time = time.time()

        def safe_count(find_fn, locator, desc):
            """Helper to safely count elements"""
            try:
                return len(find_fn(locator))
            except Exception as e:
                if self.config['debug']['verbose']:
                    print(f"⚠️ Failed to extract {desc}: {e}")
                return 0

        def safe_execute_script(script, desc, default=0):
            """Helper to safely execute JavaScript"""
            try:
                return driver.execute_script(script)
            except Exception as e:
                if self.config['debug']['verbose']:
                    print(f"⚠️ Failed to execute {desc}: {e}")
                return default
        
        def extract_weighted_selectors(selector_config, feature_prefix):
            """Extract features from weighted selector configuration"""
            total_count = 0
            weighted_score = 0
            context_features = {}
            
            for priority_level, selectors_list in selector_config.items():
                for selector_info in selectors_list:
                    if isinstance(selector_info, dict):
                        selector = selector_info.get('selector') or selector_info.get('xpath')
                        priority = selector_info.get('priority', 1)
                        context = selector_info.get('context', 'unknown')
                        
                        if selector:
                            if 'xpath' in selector_info:
                                count = safe_count(
                                    lambda loc: driver.find_elements(By.XPATH, loc),
                                    selector,
                                    f"{feature_prefix}_{context}"
                                )
                            else:
                                count = safe_count(
                                    lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                                    selector,
                                    f"{feature_prefix}_{context}"
                                )
                            
                            total_count += count
                            weighted_score += count * (priority / 10.0)
                            context_features[f"{feature_prefix}_{context}"] = count
                    else:
                        # Backward compatibility with simple selectors
                        count = safe_count(
                            lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                            selector_info,
                            f"{feature_prefix}_simple"
                        )
                        total_count += count
                        weighted_score += count
            
            return total_count, weighted_score, context_features

        try:
            # === Enhanced Button/Link Analysis ===
            # Check if using enhanced configuration format
            if isinstance(selectors.get('pagination_buttons'), dict):
                # Enhanced weighted selector extraction
                pagination_count, pagination_weighted, pagination_contexts = extract_weighted_selectors(
                    selectors['pagination_buttons'], 'pagination'
                )
                features['pagination_buttons'] = pagination_count
                features['pagination_weighted_score'] = pagination_weighted
                features.update(pagination_contexts)
                
                next_count, next_weighted, next_contexts = extract_weighted_selectors(
                    selectors['next_buttons'], 'next'
                )
                features['next_button'] = next_count
                features['next_weighted_score'] = next_weighted
                features.update(next_contexts)
                
            else:
                # Backward compatibility with simple selectors
                pagination_count = 0
                for selector in selectors['pagination_buttons']:
                    pagination_count += safe_count(
                        lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                        selector,
                        f"pagination_buttons ({selector})"
                    )
                features['pagination_buttons'] = pagination_count

                next_count = 0
                for xpath in selectors['next_buttons']:
                    next_count += safe_count(
                        lambda loc: driver.find_elements(By.XPATH, loc),
                        xpath,
                        f"next_button ({xpath})"
                    )
                features['next_button'] = next_count

            # Numbered buttons (improved regex matching)
            buttons = driver.find_elements(By.XPATH, "//button | //a")
            features['numbered_buttons'] = sum(
                1 for b in buttons 
                if b.text.strip() and re.fullmatch(r"\d+", b.text.strip())
            )

            # Load more buttons
            load_more_count = 0
            for xpath in selectors['load_more_buttons']:
                load_more_count += safe_count(
                    lambda loc: driver.find_elements(By.XPATH, loc),
                    xpath,
                    f"load_more_buttons ({xpath})"
                )
            features['load_more_buttons'] = load_more_count

            # Lazy load elements
            lazy_count = 0
            for selector in selectors['lazy_load_elements']:
                lazy_count += safe_count(
                    lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                    selector,
                    f"lazy_load_elements ({selector})"
                )
            features['lazy_load_elements'] = lazy_count

            # === Page Analysis ===
            try:
                page_height = safe_execute_script("return document.body.scrollHeight", "page_height")
                viewport_height = safe_execute_script("return window.innerHeight", "viewport_height")
                features['height_ratio'] = page_height / viewport_height if viewport_height > 0 else 0
            except Exception as e:
                if self.config['debug']['verbose']:
                    print(f"⚠️ Failed to measure height_ratio: {e}")
                features['height_ratio'] = 0

            # Product count (using configurable selectors)
            product_count = 0
            for selector in selectors['product_elements']:
                product_count += safe_count(
                    lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                    selector,
                    f"product_elements ({selector})"
                )
            features['total_products'] = product_count

            # === Enhanced Text Analysis ===
            try:
                page_text = driver.find_element(By.TAG_NAME, "body").text.lower()
                features['page_text_length'] = len(page_text)
                features['page_text_raw'] = page_text
                
                # Enhanced pattern matching with weights
                text_config = self.config['features'].get('text_analysis', {})
                
                # End of results patterns with confidence scoring
                end_patterns = text_config.get('end_of_results_patterns', {})
                features['end_of_results_confidence'] = 0
                features['end_of_results_pattern_matches'] = 0
                
                if isinstance(end_patterns, dict):
                    for confidence_level, patterns in end_patterns.items():
                        for pattern_info in patterns:
                            if isinstance(pattern_info, dict):
                                pattern = pattern_info.get('pattern', '')
                                weight = pattern_info.get('weight', 0.5)
                                if re.search(pattern, page_text, re.IGNORECASE):
                                    features['end_of_results_confidence'] = max(
                                        features['end_of_results_confidence'], weight
                                    )
                                    features['end_of_results_pattern_matches'] += 1
                                    features[f'end_pattern_{confidence_level}'] = 1
                
                # Pagination text patterns
                pagination_patterns = text_config.get('pagination_text_patterns', {})
                features['pagination_text_confidence'] = 0
                
                if isinstance(pagination_patterns, dict):
                    for indicator_level, patterns in pagination_patterns.items():
                        for pattern_info in patterns:
                            if isinstance(pattern_info, dict):
                                pattern = pattern_info.get('pattern', '')
                                weight = pattern_info.get('weight', 0.5)
                                feature_name = pattern_info.get('feature_name', 'pagination_text')
                                
                                if re.search(pattern, page_text, re.IGNORECASE):
                                    features['pagination_text_confidence'] = max(
                                        features['pagination_text_confidence'], weight
                                    )
                                    features[feature_name] = 1
                
                # Backward compatibility
                features['contains_page_numbers'] = 1 if re.search(r'page\s+\d+|halaman\s+\d+', page_text) else 0
                features['contains_total_results'] = 1 if re.search(r'total.*?\d+|hasil.*?\d+', page_text) else 0
                
            except Exception as e:
                if self.config['debug']['verbose']:
                    print(f"⚠️ Failed to extract text content: {e}")
                features['page_text_length'] = 0
                features['contains_page_numbers'] = 0
                features['contains_total_results'] = 0
                features['page_text_raw'] = ""
                features['end_of_results_confidence'] = 0
                features['pagination_text_confidence'] = 0

            # URL Analysis
            try:
                current_url = driver.current_url.lower()
                features['url_has_page_param'] = 1 if any(param in current_url for param in ['page=', 'p=', 'offset=']) else 0
            except Exception as e:
                if self.config['debug']['verbose']:
                    print(f"⚠️ Failed to analyze URL: {e}")
                features['url_has_page_param'] = 0

            # Dynamic content
            features['dynamic_content'] = safe_count(
                lambda loc: driver.find_elements(By.CSS_SELECTOR, loc),
                "[class*='loading'], [class*='spinner'], [class*='skeleton']",
                "dynamic_content"
            )

            # === Enhanced Dynamic Features ===
            dynamic_config = self.config['features'].get('dynamic_features', {})
            
            # Enhanced scroll behavior tracking
            current_height = safe_execute_script("return document.body.scrollHeight", "current_height")
            current_time = time.time()
            
            if dynamic_config.get('scroll_behavior', {}).get('track_velocity', True):
                if self._last_page_height is not None and self._last_scroll_time is not None:
                    height_diff = current_height - self._last_page_height
                    time_diff = current_time - self._last_scroll_time
                    features['scroll_velocity'] = height_diff / max(0.1, time_diff)
                    
                    # Track acceleration if enabled
                    if dynamic_config.get('scroll_behavior', {}).get('track_acceleration', False):
                        if hasattr(self, '_last_scroll_velocity'):
                            velocity_diff = features['scroll_velocity'] - self._last_scroll_velocity
                            features['scroll_acceleration'] = velocity_diff / max(0.1, time_diff)
                        else:
                            features['scroll_acceleration'] = 0
                        self._last_scroll_velocity = features['scroll_velocity']
                else:
                    features['scroll_velocity'] = 0
                    features['scroll_acceleration'] = 0
            
            self._last_page_height = current_height
            self._last_scroll_time = current_time

            # Enhanced content analysis
            features['content_density'] = features['total_products'] / max(1, features['page_text_length'])
            
            # Viewport analysis
            viewport_height = safe_execute_script("return window.innerHeight", "viewport_height")
            scroll_position = safe_execute_script("return window.pageYOffset", "scroll_position")
            
            if current_height > 0 and viewport_height > 0:
                features['scroll_percentage'] = min(1.0, (scroll_position + viewport_height) / current_height)
                features['content_above_fold'] = features['total_products'] * (viewport_height / current_height)
            else:
                features['scroll_percentage'] = 0
                features['content_above_fold'] = 0

            # Enhanced DOM monitoring
            if dynamic_config.get('dom_monitoring', {}).get('track_node_additions', True):
                current_dom_count = safe_execute_script(
                    "return document.querySelectorAll('*').length", 
                    "dom_count", 
                    0
                )
                
                if self._last_dom_count is not None:
                    features['new_dom_nodes'] = current_dom_count - self._last_dom_count
                    
                    # Significant change detection
                    threshold = dynamic_config.get('dom_monitoring', {}).get('significant_change_threshold', 10)
                    features['significant_dom_change'] = 1 if abs(features['new_dom_nodes']) >= threshold else 0
                else:
                    features['new_dom_nodes'] = 0
                    features['significant_dom_change'] = 0
                
                self._last_dom_count = current_dom_count

            # Enhanced network activity monitoring
            if dynamic_config.get('network_activity', {}).get('monitor_xhr_requests', True):
                timeout_window = dynamic_config.get('network_activity', {}).get('request_timeout_window', 5000)
                features['xhr_request_count'] = safe_execute_script(
                    f"""
                    var entries = performance.getEntriesByType('resource');
                    var recent_requests = entries.filter(entry => 
                        entry.startTime > (performance.now() - {timeout_window}) && 
                        (entry.initiatorType === 'xmlhttprequest' || entry.initiatorType === 'fetch')
                    );
                    return recent_requests.length;
                    """,
                    "xhr_count",
                    0
                )
                
                # Track image loading for lazy loading detection
                if dynamic_config.get('network_activity', {}).get('track_image_loads', True):
                    features['recent_image_loads'] = safe_execute_script(
                        f"""
                        var entries = performance.getEntriesByType('resource');
                        var recent_images = entries.filter(entry => 
                            entry.startTime > (performance.now() - {timeout_window}) && 
                            entry.initiatorType === 'img'
                        );
                        return recent_images.length;
                        """,
                        "image_loads",
                        0
                    )

            # End of results text (using configurable patterns)
            features['has_end_of_results_text'] = 0
            if features['page_text_raw']:
                end_patterns = self.config['features']['text_analysis']['end_of_results_patterns']
                for pattern in end_patterns:
                    if re.search(pattern, features['page_text_raw'], re.IGNORECASE):
                        features['has_end_of_results_text'] = 1
                        break

            # Pagination at footer (using configurable threshold)
            features['pagination_at_footer'] = 0
            try:
                total_page_height = safe_execute_script("return document.body.scrollHeight", "total_page_height")
                footer_threshold_ratio = self.config['features']['thresholds']['footer_threshold_ratio']
                footer_threshold = total_page_height * footer_threshold_ratio
                
                # Check all pagination selectors
                for selector in selectors['pagination_buttons']:
                    pagination_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in pagination_elements:
                        try:
                            element_y = driver.execute_script(
                                "return arguments[0].getBoundingClientRect().top + window.pageYOffset",
                                element
                            )
                            if element_y >= footer_threshold:
                                features['pagination_at_footer'] = 1
                                break
                        except:
                            continue
                    if features['pagination_at_footer']:
                        break
                        
            except Exception as e:
                if self.config['debug']['verbose']:
                    print(f"⚠️ Failed to detect footer pagination: {e}")

            # Average product height
            if features['total_products'] > 0 and current_height > 0:
                features['avg_product_height'] = current_height / features['total_products']
            else:
                features['avg_product_height'] = 0

            # XHR Request Count
            features['xhr_request_count'] = safe_execute_script(
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

            # Enhanced scroll listener detection
            if dynamic_config.get('user_interaction', {}).get('monitor_scroll_events', True):
                features['scroll_listeners'] = safe_execute_script(
                    """
                    var scrollListeners = 0;
                    try {
                        // Check for scroll event listeners on window and document
                        if (window.getEventListeners) {
                            var windowListeners = getEventListeners(window);
                            var documentListeners = getEventListeners(document);
                            scrollListeners = (windowListeners.scroll || []).length + 
                                            (documentListeners.scroll || []).length;
                        }
                    } catch(e) {
                        // Fallback: check for common infinite scroll libraries
                        scrollListeners = document.querySelectorAll('[data-infinite-scroll], .infinite-scroll').length;
                    }
                    return scrollListeners;
                    """,
                    "scroll_listeners",
                    0
                )
            else:
                features['scroll_listeners'] = 0

        except Exception as e:
            if self.config['debug']['verbose']:
                print(f"⚠️ General extraction error: {e}")

        # Performance monitoring
        extraction_time = time.time() - start_time
        if self.config.get('debug', {}).get('performance_monitoring', {}).get('track_extraction_time', False):
            features['_extraction_time'] = extraction_time
            if self.config['debug']['verbose']:
                print(f"⏱️ Feature extraction took {extraction_time:.3f}s")

        # Enhanced feature completeness check
        base_expected_keys = [
            'pagination_buttons', 'next_button', 'numbered_buttons',
            'load_more_buttons', 'lazy_load_elements', 'scroll_listeners',
            'height_ratio', 'total_products', 'page_text_length',
            'contains_page_numbers', 'contains_total_results', 'url_has_page_param',
            'dynamic_content', 'scroll_velocity', 'content_density', 'new_dom_nodes',
            'has_end_of_results_text', 'pagination_at_footer', 'avg_product_height',
            'xhr_request_count'
        ]
        
        # Add enhanced feature keys
        enhanced_keys = [
            'pagination_weighted_score', 'next_weighted_score',
            'end_of_results_confidence', 'pagination_text_confidence',
            'scroll_percentage', 'content_above_fold', 'significant_dom_change'
        ]
        
        all_expected_keys = base_expected_keys + enhanced_keys
        
        for key in all_expected_keys:
            features.setdefault(key, 0)

        # Feature quality metrics
        features['_feature_completeness'] = sum(1 for key in all_expected_keys if features.get(key, 0) != 0) / len(all_expected_keys)
        features['_total_features_extracted'] = len([k for k, v in features.items() if not k.startswith('_')])

        return features

    def _rule_based_analysis(self, driver):
        """Enhanced rule-based analysis using configuration"""
        try:
            selectors = self.config['features']['selectors']
            thresholds = self.config['features']['thresholds']
            
            # Check for pagination buttons (using configurable selectors)
            has_pagination = False
            for selector in selectors['pagination_buttons']:
                buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                if buttons and any(btn.is_displayed() and btn.is_enabled() for btn in buttons):
                    has_pagination = True
                    break
            
            # Check for next buttons
            if not has_pagination:
                for xpath in selectors['next_buttons']:
                    next_buttons = driver.find_elements(By.XPATH, xpath)
                    if next_buttons and any(btn.is_enabled() for btn in next_buttons):
                        has_pagination = True
                        break
            
            if has_pagination:
                return 'pagination'
            
            # Check for infinite scroll indicators
            has_infinite_scroll = False
            for xpath in selectors['load_more_buttons']:
                load_more = driver.find_elements(By.XPATH, xpath)
                if load_more:
                    has_infinite_scroll = True
                    break
            
            if not has_infinite_scroll:
                for selector in selectors['lazy_load_elements']:
                    lazy_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if lazy_elements:
                        has_infinite_scroll = True
                        break
            
            if has_infinite_scroll:
                return 'infinite_scroll'
            
            # Check for end of results text
            try:
                page_text = driver.find_element(By.TAG_NAME, "body").text.lower()
                end_patterns = self.config['features']['text_analysis']['end_of_results_patterns']
                for pattern in end_patterns:
                    if re.search(pattern, page_text, re.IGNORECASE):
                        return 'last_page'
            except:
                pass
            
            # Check scroll position
            current_height = driver.execute_script("return window.pageYOffset + window.innerHeight")
            total_height = driver.execute_script("return document.body.scrollHeight")
            scroll_threshold = thresholds['scroll_bottom_threshold']
            
            if current_height >= total_height * scroll_threshold:
                return 'last_page'
            
            return 'infinite_scroll'  # Default assumption
            
        except Exception as e:
            if self.config['debug']['verbose']:
                print(f"⚠️ Rule-based analysis error: {e}")
            return 'last_page'
    
    def collect_training_data(self, driver, actual_type, site_name="unknown", extra_features=None):
        """Enhanced training data collection with site-specific features and weighting"""
        features = self._extract_features(driver)

        # Merge extra dynamic features
        if extra_features:
            features.update(extra_features)

        # Enhanced site-specific feature processing
        if site_name in self.config['sites']:
            site_config = self.config['sites'][site_name]
            features['site_language'] = site_config.get('language', 'unknown')
            features['site_encoding'] = site_config.get('encoding', 'utf-8')
            
            # Apply site-specific feature weights
            feature_weights = site_config.get('feature_weights', {})
            for feature_name, weight in feature_weights.items():
                if feature_name in features and isinstance(features[feature_name], (int, float)):
                    features[f'{feature_name}_weighted'] = features[feature_name] * weight
            
            # Enhanced pattern matching with regex
            page_text = features.get('page_text_raw', '')
            if page_text:
                site_patterns = site_config.get('patterns', {})
                for pattern_type, patterns in site_patterns.items():
                    feature_name = f'site_{pattern_type}_match'
                    features[feature_name] = 0
                    features[f'{feature_name}_count'] = 0
                    features[f'{feature_name}_confidence'] = 0
                    
                    match_count = 0
                    max_confidence = 0
                    
                    for pattern in patterns:
                        # Support both simple string matching and regex
                        if pattern.startswith('\\') or '(' in pattern:
                            # Regex pattern
                            matches = re.findall(pattern, page_text, re.IGNORECASE)
                            if matches:
                                match_count += len(matches)
                                max_confidence = max(max_confidence, 0.9)
                                features[feature_name] = 1
                        else:
                            # Simple string matching
                            if pattern.lower() in page_text:
                                match_count += 1
                                max_confidence = max(max_confidence, 0.7)
                                features[feature_name] = 1
                    
                    features[f'{feature_name}_count'] = match_count
                    features[f'{feature_name}_confidence'] = max_confidence
            
            # Custom selector analysis for site
            custom_selectors = site_config.get('custom_selectors', {})
            for selector_name, selector in custom_selectors.items():
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    features[f'site_custom_{selector_name}'] = len(elements)
                    features[f'site_custom_{selector_name}_present'] = 1 if elements else 0
                except:
                    features[f'site_custom_{selector_name}'] = 0
                    features[f'site_custom_{selector_name}_present'] = 0

        # Enhanced metadata
        features['site_name'] = site_name
        features['timestamp'] = datetime.now().isoformat()
        features['actual_type'] = actual_type
        features['url'] = driver.current_url
        features['user_agent'] = driver.execute_script("return navigator.userAgent") if driver else "unknown"
        features['viewport_width'] = driver.execute_script("return window.innerWidth") if driver else 0
        features['viewport_height'] = driver.execute_script("return window.innerHeight") if driver else 0
        
        # Data quality indicators
        features['_data_quality_score'] = self._calculate_data_quality_score(features)
        features['_collection_method'] = 'enhanced_automated'

        # Save training sample with enhanced format
        self._save_training_sample(features)
        
        # Enhanced logging
        if self.config['debug']['verbose']:
            extra_info = f" (extra: {len(extra_features)} features)" if extra_features else ""
            quality_score = features.get('_data_quality_score', 0)
            print(f"📊 Training data collected: {actual_type}{extra_info} (quality: {quality_score:.2f})")
            
            if self.config['debug'].get('feature_summary', False):
                feature_count = len([k for k, v in features.items() if not k.startswith('_')])
                print(f"   📈 {feature_count} features extracted, completeness: {features.get('_feature_completeness', 0):.2f}")
    
    def _calculate_data_quality_score(self, features):
        """Calculate a quality score for the collected data"""
        score = 0.0
        
        # Basic feature presence (40% of score)
        basic_features = ['pagination_buttons', 'next_button', 'total_products', 'page_text_length']
        present_basic = sum(1 for f in basic_features if features.get(f, 0) > 0)
        score += (present_basic / len(basic_features)) * 0.4
        
        # Text content quality (30% of score)
        text_length = features.get('page_text_length', 0)
        if text_length > 1000:
            score += 0.3
        elif text_length > 500:
            score += 0.2
        elif text_length > 100:
            score += 0.1
        
        # Feature diversity (20% of score)
        non_zero_features = len([k for k, v in features.items() if not k.startswith('_') and v != 0])
        if non_zero_features > 20:
            score += 0.2
        elif non_zero_features > 15:
            score += 0.15
        elif non_zero_features > 10:
            score += 0.1
        
        # Site-specific features (10% of score)
        site_features = len([k for k in features.keys() if k.startswith('site_')])
        if site_features > 5:
            score += 0.1
        elif site_features > 2:
            score += 0.05
        
        return min(1.0, score)

    def train_model(self, training_data_file=None):
        """Train the ML model with configurable parameters"""
        if training_data_file is None:
            training_data_file = self.config['paths']['training_data_file']
            
        try:
            if not os.path.exists(training_data_file):
                print("⚠️ No training data found. Collect some data first!")
                return False
            
            df = pd.read_csv(training_data_file)
            print(f"📚 Loaded {len(df)} training samples")

            # TF-IDF Processing
            if 'page_text_raw' in df.columns:
                text_corpus = df['page_text_raw'].fillna("")
                text_config = self.config['features']['text_analysis']
                
                self.vectorizer = TfidfVectorizer(
                    max_features=text_config['max_tfidf_features'],
                    stop_words=text_config['stop_words']
                )
                
                tfidf_matrix = self.vectorizer.fit_transform(text_corpus)
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(), 
                    columns=[f"tfidf_{w}" for w in self.vectorizer.get_feature_names_out()]
                )
                
                df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

                # Save vectorizer
                with open(self.vectorizer_path, "wb") as f:
                    pickle.dump(self.vectorizer, f)
                print(f"💾 TF-IDF vectorizer saved to {self.vectorizer_path}")

            # Feature selection with site-specific features
            base_features = [
                'pagination_buttons', 'next_button', 'numbered_buttons',
                'load_more_buttons', 'lazy_load_elements', 'scroll_listeners',
                'height_ratio', 'total_products', 'page_text_length',
                'contains_page_numbers', 'contains_total_results', 'url_has_page_param',
                'dynamic_content', 'scroll_velocity', 'content_density', 'new_dom_nodes',
                'has_end_of_results_text', 'pagination_at_footer', 'avg_product_height',
                'xhr_request_count'
            ]
            
            # Add dynamic training features
            dynamic_features = [col for col in df.columns if col.startswith(('scroll_round', 'new_items_loaded'))]
            
            # Add site-specific features
            site_features = [col for col in df.columns if col.startswith('site_')]
            
            # Add TF-IDF features
            tfidf_features = [col for col in df.columns if col.startswith("tfidf_")]

            feature_columns = base_features + dynamic_features + site_features + tfidf_features

            # Ensure missing columns are added as 0
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0

            X = df[feature_columns]
            y = df['actual_type']

            # Training configuration
            train_config = self.config['training']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=train_config['test_size'], 
                random_state=train_config['random_state']
            )

            # Train model
            self.model = RandomForestClassifier(
                n_estimators=train_config['n_estimators'], 
                random_state=train_config['random_state']
            )
            self.model.fit(X_train, y_train)

            # Evaluate
            accuracy = self.model.score(X_test, y_test)
            print(f"🎯 Model accuracy: {accuracy:.2f}")

            # Feature importance analysis
            if self.config['debug']['feature_summary']:
                self._print_feature_importance(feature_columns)

            # Save model
            self.save_model()
            self.is_trained = True
            return True

        except Exception as e:
            print(f"⚠️ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _print_feature_importance(self, feature_columns):
        """Print top feature importances"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = list(zip(feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\n📈 Top 10 Feature Importances:")
            print("-" * 40)
            for feature, importance in feature_importance[:10]:
                print(f"{feature}: {importance:.4f}")
    
    def _features_to_vector(self, features):
        """Convert features dict to numpy array for prediction"""
        base_features = [
            'pagination_buttons', 'next_button', 'numbered_buttons',
            'load_more_buttons', 'lazy_load_elements', 'scroll_listeners',
            'height_ratio', 'total_products', 'page_text_length',
            'contains_page_numbers', 'contains_total_results', 'url_has_page_param',
            'dynamic_content', 'scroll_velocity', 'content_density', 'new_dom_nodes',
            'has_end_of_results_text', 'pagination_at_footer', 'avg_product_height',
            'xhr_request_count'
        ]
        
        # Add dynamic features if they exist
        dynamic_features = [key for key in features.keys() if key.startswith(('scroll_round', 'new_items_loaded'))]
        
        # Add site-specific features if they exist
        site_features = [key for key in features.keys() if key.startswith('site_')]
        
        # Combine all feature types
        all_features = base_features + dynamic_features + site_features
        
        # Get text features if vectorizer is available
        text_features = []
        if self.vectorizer and 'page_text_raw' in features:
            try:
                tfidf_vector = self.vectorizer.transform([features['page_text_raw']])
                text_features = tfidf_vector.toarray()[0].tolist()
            except Exception as e:
                if self.config['debug']['verbose']:
                    print(f"⚠️ TF-IDF transform failed: {e}")
                text_features = []
        
        # Build feature vector
        feature_vector = [features.get(key, 0) for key in all_features]
        
        # Add TF-IDF features
        if text_features:
            feature_vector.extend(text_features)
        
        return np.array(feature_vector)
    
    def _save_training_sample(self, features):
        """Save a single training sample with proper encoding"""
        file_path = self.config['paths']['training_data_file']
        file_exists = os.path.exists(file_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                fieldnames = list(features.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(features)
        except Exception as e:
            print(f"⚠️ Error saving training sample: {e}")
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"💾 Model saved to {self.model_path}")
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
    
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
            print(f"⚠️ Error loading model/vectorizer: {e}")
            self.is_trained = False

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

# Training Data Collector

class TrainingDataCollector:
    """Helper class to collect training data for the ML model with YAML config support"""
    
    def __init__(self, analyzer, config_path="config/analyzer_config.yaml"):
        self.analyzer = analyzer
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        """Load YAML config for sites and labeling rules"""
        default_config = {
            'paths': {
                'model_dir': 'models',
                'training_data_file': 'models/training_data.csv',
                'model_file': 'models/pagination_model.pkl',
                'vectorizer_file': 'models/pagination_vectorizer.pkl'
            },
            'features': {
                'selectors': {
                    'pagination_buttons': [
                        "button[class*='pagination']",
                        "a[class*='pagination']",
                        ".pagination button",
                        ".pagination a"
                    ],
                    'next_buttons': [
                        "//*[contains(text(), 'Next')]",
                        "//*[contains(text(), 'next')]",
                        "//*[contains(text(), '›')]"
                    ],
                    'load_more_buttons': [
                        "//*[contains(text(), 'Load More')]",
                        "//*[contains(text(), 'Show More')]"
                    ],
                    'lazy_load_elements': [
                        "[data-lazy]",
                        "[loading='lazy']",
                        ".lazy"
                    ]
                },
                'thresholds': {
                    'confidence_threshold': 0.7,
                    'footer_threshold_ratio': 0.8,
                    'scroll_bottom_threshold': 0.95
                }
            },
            'sites': {
                'tokopedia': {
                    'url': 'https://www.tokopedia.com/search?q=laptop',
                    'language': 'indonesian'
                },
                'amazon': {
                    'url': 'https://www.amazon.com/s?k=laptop',
                    'language': 'english'
                }
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 10
            },
            'debug': {
                'verbose': True,
                'save_features': True
            },
            'heuristics': {
                'url_pagination_params': ['page=', 'p=', 'offset='],
                'scroll_height_ratio': 2.0,
                'text_pagination_keywords': ['page', 'next', 'previous']
            }
        }

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            # merge with defaults (so missing keys don’t break things)
            return {**default_config, **(loaded or {})}
        else:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(default_config, f, indent=2)
            return default_config

    def collect_diverse_data(self, browser_manager, samples_per_site=20):
        """Collect training data from multiple sites"""
        print("📊 Starting training data collection...")

        for site_name, site_info in self.config["sites"].items():
            base_url = site_info["url"]
            print(f"\n🌐 Collecting data from {site_name}")

            self.analyzer.reset_state()

            try:
                browser_manager.navigate_to(base_url)

                for i in range(samples_per_site):
                    print(f"\n📄 Page {i+1} on {site_name}")
                    print("Current URL:", browser_manager.driver.current_url)

                    actual_type = self._auto_label_page(browser_manager.driver)

                    features_summary = self.analyzer.get_feature_summary(browser_manager.driver)

                    self.analyzer.collect_training_data(
                        browser_manager.driver,
                        actual_type,
                        site_name
                    )

                    self._navigate_next_page(browser_manager.driver)

            except Exception as e:
                print(f"⚠ Error collecting data from {site_name}: {e}")

    def _auto_label_page(self, driver):
        """Automatically label page type using YAML-configured rules"""
        try:
            selectors = self.config["selectors"]
            heuristics = self.config["heuristics"]

            # 1. Pagination buttons
            for selector in selectors["pagination_buttons"]:
                if driver.find_elements(By.CSS_SELECTOR, selector):
                    return "pagination"
            for xpath in selectors["next_buttons"]:
                if driver.find_elements(By.XPATH, xpath):
                    return "pagination"

            # 2. Infinite scroll indicators
            for xpath in selectors["load_more_buttons"]:
                if driver.find_elements(By.XPATH, xpath):
                    return "infinite_scroll"
            for selector in selectors["lazy_load_elements"]:
                if driver.find_elements(By.CSS_SELECTOR, selector):
                    return "infinite_scroll"

            # 3. URL pattern
            current_url = driver.current_url.lower()
            if any(param in current_url for param in heuristics["url_pagination_params"]):
                return "pagination"

            # 4. Scroll height check
            page_height = driver.execute_script("return document.body.scrollHeight")
            viewport_height = driver.execute_script("return window.innerHeight")
            if page_height > viewport_height * heuristics["scroll_height_ratio"]:
                return "infinite_scroll"

            # 5. Text cues
            body_text = driver.find_element(By.TAG_NAME, "body").text.lower()
            if any(keyword in body_text for keyword in heuristics["text_pagination_keywords"]):
                return "pagination"

            return "last_page"

        except Exception as e:
            print(f"⚠️ Auto-labeling failed: {e}")
            return "last_page"

    def _navigate_next_page(self, driver):
        """Try to move to next page (best-effort)"""
        
        try:
            # Try CSS selectors first
            for selector in self.config["selectors"]["pagination_buttons"]:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        element.click()
                        return True
            
            # Try XPath selectors
            for xpath in self.config["selectors"]["next_buttons"]:
                elements = driver.find_elements(By.XPATH, xpath)
                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                        element.click()
                        return True
            
            return False
        except Exception:
            return False

# Usage Example
def main_with_ml():
    """Main function using ML-powered scraper"""
    from browser import BrowserManager  # Your existing browser manager
    
    # Setup
    browser = BrowserManager()
    browser.setup_driver()
    
    try:
        # Option 1: Train model first (do this once)
        collector = TrainingDataCollector(ConfigurableAnalyzer())
        collector.collect_diverse_data(browser, samples_per_site=10)
        
        # Option 2: Use smart scraper
        scraper = SmartTokopediaScraper(browser)
        results = scraper.smart_scrape("laptop", max_pages=50)
        
        print(f"\n🎉 Scraped {len(results)} pages intelligently!")
        
        # Train model on session data
        scraper.train_on_current_session()
        
    finally:
        browser.close()

def test_analyzer_standalone():
    """Standalone test function for the analyzer"""
    print("🧪 Testing ConfigurableAnalyzer in standalone mode...")
    
    try:
        # Test basic initialization
        print("🔧 Initializing analyzer...")
        analyzer = ConfigurableAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Test configuration
        print("⚙️ Testing configuration...")
        config_sections = list(analyzer.config.keys())
        print(f"✅ Config sections: {config_sections}")
        
        # Test feature extraction with mock driver
        print("🔍 Testing feature extraction with mock data...")
        
        class MockDriver:
            def find_elements(self, by, selector):
                # Return some mock elements for testing
                if "pagination" in selector.lower():
                    return [MockElement(), MockElement()]  # 2 pagination buttons
                elif "next" in selector.lower():
                    return [MockElement()]  # 1 next button
                elif "product" in selector.lower():
                    return [MockElement() for _ in range(20)]  # 20 products
                else:
                    return []
            
            def find_element(self, by, selector):
                class MockBodyElement:
                    text = "Sample page text with pagination page 1 of 10 showing 20 products"
                return MockBodyElement()
            
            def execute_script(self, script):
                if "scrollHeight" in script:
                    return 2000
                elif "innerHeight" in script:
                    return 800
                elif "pageYOffset" in script:
                    return 0
                elif "querySelectorAll" in script:
                    return 50
                elif "performance" in script:
                    return 2
                elif "userAgent" in script:
                    return "Mozilla/5.0 Test Browser"
                elif "innerWidth" in script:
                    return 1920
                else:
                    return 1
            
            @property
            def current_url(self):
                return "https://example.com/search?page=1"
        
        class MockElement:
            def __init__(self):
                self.text = "Next"
            
            def is_displayed(self):
                return True
            
            def is_enabled(self):
                return True
        
        mock_driver = MockDriver()
        
        # Test feature extraction
        features = analyzer._extract_features(mock_driver)
        print(f"✅ Feature extraction successful: {len(features)} features")
        
        # Show key features
        key_features = ['pagination_buttons', 'next_button', 'total_products', 'page_text_length']
        for feature in key_features:
            if feature in features:
                print(f"   - {feature}: {features[feature]}")
        
        # Test rule-based analysis
        result = analyzer._rule_based_analysis(mock_driver)
        print(f"✅ Rule-based analysis result: {result}")
        
        # Test feature vector conversion
        feature_vector = analyzer._features_to_vector(features)
        print(f"✅ Feature vector created: shape {feature_vector.shape}")
        
        # Test training data collection
        analyzer.collect_training_data(mock_driver, "pagination", "test_site")
        print("✅ Training data collection successful")
        
        print("\n🎉 All standalone tests passed!")
        print("📋 Summary:")
        print(f"   - Configuration loaded: {len(config_sections)} sections")
        print(f"   - Features extracted: {len(features)}")
        print(f"   - Feature vector size: {len(feature_vector)}")
        print(f"   - Analysis result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Enhanced Analyzer with YAML Template Support
class EnhancedConfigurableAnalyzer:
    """
    Advanced ML-powered pagination analyzer with rich YAML template support
    Features:
    - Weighted selector configurations
    - Site-specific training scenarios
    - Advanced feature extraction
    - Multiple ML algorithms
    - Quality control and validation
    """
    
    def __init__(self, config_path: Optional[str] = None, template_path: Optional[str] = None):
        """Initialize the enhanced analyzer"""
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.features = []
        
        # Load configurations
        self.config = self._load_basic_config(config_path)
        self.templates = self._load_training_templates(template_path)
        
        # Setup paths and initialize components
        self._setup_paths()
        self._initialize_vectorizer()
        
        # State tracking for dynamic features
        self._reset_state()
        
        # Try to load existing model
        self.load_model()
        
        logger.info("Enhanced Configurable Analyzer initialized")
    
    def _load_basic_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load basic configuration with fallback to defaults"""
        default_config = {
            'paths': {
                'model_dir': 'models',
                'training_data_file': 'models/enhanced_training_data.csv',
                'model_file': 'enhanced_pagination_model.pkl',
                'vectorizer_file': 'enhanced_pagination_vectorizer.pkl'
            },
            'features': {
                'text_analysis': {
                    'max_tfidf_features': 150,
                    'stop_words': 'english'
                },
                'thresholds': {
                    'confidence_threshold': 0.75,
                    'footer_threshold_ratio': 0.8,
                    'scroll_bottom_threshold': 0.95
                }
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5
            },
            'debug': {
                'verbose': True,
                'save_features': True,
                'feature_summary': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                config = self._deep_merge(default_config, loaded_config)
                logger.info(f"Basic configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading basic config: {e}, using defaults")
                config = default_config
        else:
            config = default_config
            logger.info("Using default basic configuration")
            
        return config
    
    def _load_training_templates(self, template_path: Optional[str]) -> Dict[str, Any]:
        """Load rich training templates"""
        if template_path is None:
            template_path = "config/enhanced_training_templates.yaml"
            
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    templates = yaml.safe_load(f)
                logger.info(f"Training templates loaded from {template_path}")
                return templates
            except Exception as e:
                logger.warning(f"Error loading training templates: {e}")
                
        # Return minimal templates if file doesn't exist
        logger.info("Using minimal training templates")
        return {
            'training_templates': {
                'ecommerce_sites': {},
                'generic_ecommerce': {}
            }
        }
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _setup_paths(self):
        """Setup file paths and directories"""
        try:
            model_dir = self.config['paths']['model_dir']
            os.makedirs(model_dir, exist_ok=True)
            
            self.model_path = os.path.join(model_dir, self.config['paths']['model_file'])
            self.vectorizer_path = os.path.join(model_dir, self.config['paths']['vectorizer_file'])
            self.training_data_path = self.config['paths']['training_data_file']
            
            # Ensure all necessary directories exist
            for directory in ['data', 'data/raw_html', 'data/processed', 'logs', 'config']:
                os.makedirs(directory, exist_ok=True)
                
        except Exception as e:
            logger.error(f"Error setting up paths: {e}")
            # Fallback to default paths
            self.model_path = "models/enhanced_pagination_model.pkl"
            self.vectorizer_path = "models/enhanced_pagination_vectorizer.pkl"
            self.training_data_path = "models/enhanced_training_data.csv"
            os.makedirs("models", exist_ok=True)
    
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer from configuration"""
        text_config = self.config['features']['text_analysis']
        self.vectorizer = TfidfVectorizer(
            max_features=text_config['max_tfidf_features'],
            stop_words=text_config['stop_words']
        )
    
    def _reset_state(self):
        """Reset internal state for new scraping session"""
        self._last_scroll_time = None
        self._last_page_height = None
        self._last_dom_count = None
        self._last_scroll_velocity = 0
        self._session_start_time = time.time()
        logger.debug("Analyzer state reset")
    
    def analyze_page_structure(self, driver, site_name: str = "unknown") -> str:
        """Enhanced page structure analysis using templates and weighted features"""
        start_time = time.time()
        
        # Extract enhanced features using templates
        features = self._extract_enhanced_features(driver, site_name)
        
        # Use ML model if trained and confident
        if self.is_trained:
            prediction, confidence = self._ml_prediction(features)
            
            confidence_threshold = self.config['features']['thresholds']['confidence_threshold']
            if confidence >= confidence_threshold:
                if self.config['debug']['verbose']:
                    logger.info(f"ML Prediction: {prediction} (confidence: {confidence:.3f})")
                return prediction
            else:
                if self.config['debug']['verbose']:
                    logger.info(f"Low ML confidence ({confidence:.3f}), using enhanced rules")
        
        # Enhanced rule-based analysis with templates
        result = self._enhanced_rule_based_analysis(driver, site_name, features)
        
        analysis_time = time.time() - start_time
        if self.config['debug']['verbose']:
            logger.info(f"Page analysis completed in {analysis_time:.3f}s: {result}")
            
        return result
    
    def _extract_enhanced_features(self, driver, site_name: str) -> Dict[str, Any]:
        """Extract enhanced features using YAML templates and weighted selectors"""
        features = {}
        start_time = time.time()
        
        # Get site-specific template
        site_template = self._get_site_template(site_name)
        
        # Extract weighted selector features
        features.update(self._extract_weighted_selector_features(driver, site_template))
        
        # Extract enhanced text features
        features.update(self._extract_enhanced_text_features(driver, site_template))
        
        # Extract dynamic content features
        features.update(self._extract_dynamic_features(driver))
        
        # Extract performance features
        features.update(self._extract_performance_features(driver))
        
        # Extract URL and metadata features
        features.update(self._extract_metadata_features(driver))
        
        # Calculate derived features
        features.update(self._calculate_derived_features(features))
        
        # Add quality metrics
        features['_extraction_time'] = time.time() - start_time
        features['_feature_count'] = len([k for k in features.keys() if not k.startswith('_')])
        features['_site_name'] = site_name
        features['_timestamp'] = datetime.now().isoformat()
        
        return features
    
    def _get_site_template(self, site_name: str) -> Dict[str, Any]:
        """Get site-specific template or fallback to generic"""
        templates = self.templates.get('training_templates', {})
        ecommerce_sites = templates.get('ecommerce_sites', {})
        
        if site_name in ecommerce_sites:
            return ecommerce_sites[site_name]
        
        # Fallback to generic template
        return templates.get('generic_ecommerce', {})
    
    def _extract_weighted_selector_features(self, driver, site_template: Dict) -> Dict[str, Any]:
        """Extract features using weighted selectors from templates"""
        features = {}
        selectors = site_template.get('selectors', {})
        
        # Process each selector category
        for category, selector_config in selectors.items():
            if isinstance(selector_config, dict):
                # Weighted selector configuration
                total_count = 0
                weighted_score = 0.0
                max_confidence = 0.0
                
                for confidence_level, selector_list in selector_config.items():
                    for selector_info in selector_list:
                        count, confidence = self._count_elements_with_confidence(driver, selector_info)
                        total_count += count
                        weighted_score += count * confidence
                        max_confidence = max(max_confidence, confidence if count > 0 else 0)
                        
                        # Individual selector features
                        context = selector_info.get('context', 'unknown')
                        features[f"{category}_{context}"] = count
                        features[f"{category}_{context}_confidence"] = confidence if count > 0 else 0
                
                # Aggregate features
                features[f"{category}_total"] = total_count
                features[f"{category}_weighted_score"] = weighted_score
                features[f"{category}_max_confidence"] = max_confidence
                
            else:
                # Simple selector list (backward compatibility)
                total_count = 0
                for selector in selector_config:
                    count = self._count_elements_simple(driver, selector)
                    total_count += count
                features[f"{category}_total"] = total_count
        
        return features
    
    def _count_elements_with_confidence(self, driver, selector_info: Dict) -> Tuple[int, float]:
        """Count elements and return confidence based on selector info"""
        try:
            selector = selector_info.get('selector')
            xpath = selector_info.get('xpath')
            weight = selector_info.get('weight', 0.5)
            
            if xpath:
                elements = driver.find_elements(By.XPATH, xpath)
            elif selector:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
            else:
                return 0, 0.0
            
            # Filter for visible and enabled elements
            visible_elements = [e for e in elements if e.is_displayed()]
            enabled_elements = [e for e in visible_elements if e.is_enabled()]
            
            return len(enabled_elements), weight
            
        except Exception as e:
            if self.config['debug']['verbose']:
                logger.debug(f"Selector failed: {selector_info}: {e}")
            return 0, 0.0
    
    def _count_elements_simple(self, driver, selector: str) -> int:
        """Simple element counting for backward compatibility"""
        try:
            if selector.startswith('//'):
                elements = driver.find_elements(By.XPATH, selector)
            else:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
            return len([e for e in elements if e.is_displayed()])
        except:
            return 0
    
    def _extract_enhanced_text_features(self, driver, site_template: Dict) -> Dict[str, Any]:
        """Extract enhanced text features using template patterns"""
        features = {}
        
        try:
            page_text = driver.find_element(By.TAG_NAME, "body").text.lower()
            features['page_text_length'] = len(page_text)
            features['page_text_raw'] = page_text
            
            # Process text patterns from template
            text_patterns = site_template.get('text_patterns', {})
            
            for pattern_category, pattern_config in text_patterns.items():
                category_confidence = 0.0
                category_matches = 0
                
                if isinstance(pattern_config, dict):
                    for confidence_level, patterns in pattern_config.items():
                        for pattern_info in patterns:
                            if isinstance(pattern_info, dict):
                                pattern = pattern_info.get('pattern', '')
                                weight = pattern_info.get('weight', 0.5)
                                feature_name = pattern_info.get('feature_name', f'{pattern_category}_match')
                                
                                matches = re.findall(pattern, page_text, re.IGNORECASE)
                                if matches:
                                    category_matches += len(matches)
                                    category_confidence = max(category_confidence, weight)
                                    features[feature_name] = 1
                                    features[f'{feature_name}_count'] = len(matches)
                                else:
                                    features[feature_name] = 0
                                    features[f'{feature_name}_count'] = 0
                
                # Aggregate category features
                features[f'{pattern_category}_confidence'] = category_confidence
                features[f'{pattern_category}_total_matches'] = category_matches
            
            # Additional text analysis
            features['word_count'] = len(page_text.split())
            features['sentence_count'] = len(re.split(r'[.!?]+', page_text))
            features['avg_word_length'] = np.mean([len(word) for word in page_text.split()]) if page_text else 0
            
        except Exception as e:
            logger.debug(f"Text extraction failed: {e}")
            features.update({
                'page_text_length': 0,
                'page_text_raw': '',
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0
            })
        
        return features
    
    def _extract_dynamic_features(self, driver) -> Dict[str, Any]:
        """Extract dynamic content and behavior features"""
        features = {}
        current_time = time.time()
        
        try:
            # Page dimensions and scroll
            page_height = driver.execute_script("return document.body.scrollHeight")
            viewport_height = driver.execute_script("return window.innerHeight")
            scroll_position = driver.execute_script("return window.pageYOffset")
            
            features['page_height'] = page_height
            features['viewport_height'] = viewport_height
            features['scroll_position'] = scroll_position
            features['height_ratio'] = page_height / viewport_height if viewport_height > 0 else 0
            features['scroll_percentage'] = min(1.0, (scroll_position + viewport_height) / page_height) if page_height > 0 else 0
            
            # Scroll velocity tracking
            if self._last_page_height is not None and self._last_scroll_time is not None:
                height_diff = page_height - self._last_page_height
                time_diff = current_time - self._last_scroll_time
                features['scroll_velocity'] = height_diff / max(0.1, time_diff)
                
                # Scroll acceleration
                velocity_diff = features['scroll_velocity'] - self._last_scroll_velocity
                features['scroll_acceleration'] = velocity_diff / max(0.1, time_diff)
                self._last_scroll_velocity = features['scroll_velocity']
            else:
                features['scroll_velocity'] = 0
                features['scroll_acceleration'] = 0
            
            self._last_page_height = page_height
            self._last_scroll_time = current_time
            
            # DOM node count
            dom_count = driver.execute_script("return document.querySelectorAll('*').length")
            features['dom_node_count'] = dom_count
            
            if self._last_dom_count is not None:
                features['new_dom_nodes'] = dom_count - self._last_dom_count
                features['significant_dom_change'] = 1 if abs(features['new_dom_nodes']) >= 10 else 0
            else:
                features['new_dom_nodes'] = 0
                features['significant_dom_change'] = 0
            
            self._last_dom_count = dom_count
            
        except Exception as e:
            logger.debug(f"Dynamic feature extraction failed: {e}")
            features.update({
                'page_height': 0, 'viewport_height': 0, 'scroll_position': 0,
                'height_ratio': 0, 'scroll_percentage': 0, 'scroll_velocity': 0,
                'scroll_acceleration': 0, 'dom_node_count': 0, 'new_dom_nodes': 0,
                'significant_dom_change': 0
            })
        
        return features
    
    def _extract_performance_features(self, driver) -> Dict[str, Any]:
        """Extract performance-related features"""
        features = {}
        
        try:
            # Page load timing
            timing = driver.execute_script("""
                var timing = performance.timing;
                return {
                    loadTime: timing.loadEventEnd - timing.navigationStart,
                    domReady: timing.domContentLoadedEventEnd - timing.navigationStart
                };
            """)
            
            features['page_load_time'] = timing.get('loadTime', 0)
            features['dom_ready_time'] = timing.get('domReady', 0)
            
        except Exception as e:
            logger.debug(f"Performance feature extraction failed: {e}")
            features.update({
                'page_load_time': 0, 'dom_ready_time': 0
            })
        
        return features
    
    def _extract_metadata_features(self, driver) -> Dict[str, Any]:
        """Extract URL and metadata features"""
        features = {}
        
        try:
            current_url = driver.current_url.lower()
            features['url'] = current_url
            features['url_length'] = len(current_url)
            
            # URL parameter analysis
            url_params = ['page=', 'p=', 'offset=', 'start=', 'from=']
            features['url_has_page_param'] = 1 if any(param in current_url for param in url_params) else 0
            
            # Extract page numbers from URL
            page_matches = re.findall(r'(?:page|p)=(\d+)', current_url)
            if page_matches:
                features['url_page_number'] = int(page_matches[-1])
            else:
                features['url_page_number'] = 0
            
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
            features.update({
                'url': '', 'url_length': 0, 'url_has_page_param': 0,
                'url_page_number': 0
            })
        
        return features
    
    def _calculate_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived features from base features"""
        derived = {}
        
        try:
            # Content density
            total_products = sum(v for k, v in features.items() if 'product' in k and isinstance(v, int))
            page_text_length = features.get('page_text_length', 1)
            derived['content_density'] = total_products / max(1, page_text_length)
            
            # Navigation confidence score
            pagination_score = features.get('pagination_buttons_weighted_score', 0)
            next_score = features.get('next_buttons_weighted_score', 0)
            derived['navigation_confidence'] = (pagination_score + next_score) / 2
            
            # Infinite scroll confidence score
            infinite_score = features.get('infinite_scroll_indicators_weighted_score', 0)
            scroll_velocity = features.get('scroll_velocity', 0)
            derived['infinite_scroll_confidence'] = infinite_score + (scroll_velocity / 100)
            
            # Page completeness score
            end_confidence = features.get('end_of_results_confidence', 0)
            scroll_percentage = features.get('scroll_percentage', 0)
            derived['page_completeness'] = (end_confidence + scroll_percentage) / 2
            
        except Exception as e:
            logger.debug(f"Derived feature calculation failed: {e}")
            derived.update({
                'content_density': 0, 'navigation_confidence': 0,
                'infinite_scroll_confidence': 0, 'page_completeness': 0
            })
        
        return derived
    
    def _ml_prediction(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """Make ML prediction with confidence score"""
        try:
            if not self.model:
                return "unknown", 0.0
            
            feature_vector = self._features_to_vector(features)
            prediction = self.model.predict([feature_vector])[0]
            confidence = max(self.model.predict_proba([feature_vector])[0])
            
            return prediction, confidence
            
        except Exception as e:
            logger.debug(f"ML prediction failed: {e}")
            return "unknown", 0.0
    
    def _enhanced_rule_based_analysis(self, driver, site_name: str, features: Dict[str, Any]) -> str:
        """Enhanced rule-based analysis using templates and features"""
        try:
            # Use extracted features for analysis
            pagination_confidence = features.get('navigation_confidence', 0)
            infinite_confidence = features.get('infinite_scroll_confidence', 0)
            page_completeness = features.get('page_completeness', 0)
            
            # High confidence thresholds
            if pagination_confidence > 0.7:
                return 'pagination'
            elif infinite_confidence > 0.7:
                return 'infinite_scroll'
            elif page_completeness > 0.8:
                return 'last_page'
            
            # Fallback to basic checks
            pagination_total = features.get('pagination_buttons_total', 0)
            next_total = features.get('next_buttons_total', 0)
            
            if pagination_total > 0 or next_total > 0:
                return 'pagination'
            
            infinite_total = features.get('infinite_scroll_indicators_total', 0)
            if infinite_total > 0:
                return 'infinite_scroll'
            
            return 'last_page'
            
        except Exception as e:
            logger.debug(f"Enhanced rule-based analysis failed: {e}")
            return 'last_page'
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features dict to numpy array for ML prediction"""
        # Basic feature set for compatibility
        base_features = [
            'pagination_buttons_total', 'next_buttons_total', 'infinite_scroll_indicators_total',
            'page_text_length', 'word_count', 'height_ratio', 'scroll_velocity',
            'dom_node_count', 'navigation_confidence',
            'infinite_scroll_confidence', 'page_completeness', 'content_density'
        ]
        
        feature_vector = [features.get(key, 0) for key in base_features]
        return np.array(feature_vector, dtype=float)
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"Model loaded from {self.model_path}")
                return True
        except Exception as e:
            logger.debug(f"Model loading failed: {e}")
            self.is_trained = False
        return False
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    def reset_state(self):
        """Reset internal state for new scraping session"""
        self._reset_state()
        logger.info("Analyzer state reset for new session")


if __name__ == "__main__":
    print("🚀 Enhanced ML-Powered Analyzer - Standalone Test")
    print("=" * 50)
    
    # Run standalone test
    success = test_analyzer_standalone()
    
    if success:
        print("\n✅ Analyzer is working correctly!")
        print("\n📝 Next steps:")
        print("   1. Use with browser: analyzer.analyze_page_structure(driver)")
        print("   2. Collect training data: analyzer.collect_training_data(driver, 'pagination', 'site')")
        print("   3. Train model: analyzer.train_model()")
        print("   4. Run full application: python main.py")