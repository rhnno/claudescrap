#!/usr/bin/env python3
"""
Enhanced ML-Powered Analyzer with Rich YAML Template Support
Supports configurable feature extraction, weighted selectors, and training scenarios
"""

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """
        Initialize the enhanced analyzer
        
        Args:
            config_path: Path to basic configuration file
            template_path: Path to rich training templates file
        """
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
        """
        Enhanced page structure analysis using templates and weighted features
        
        Args:
            driver: Selenium WebDriver instance
            site_name: Name of the site being analyzed (for template selection)
            
        Returns:
            str: 'pagination', 'infinite_scroll', or 'last_page'
        """
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
        """
        Extract enhanced features using YAML templates and weighted selectors
        """
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
                                extract_numbers = pattern_info.get('extract_numbers', False)
                                
                                matches = re.findall(pattern, page_text, re.IGNORECASE)
                                if matches:
                                    category_matches += len(matches)
                                    category_confidence = max(category_confidence, weight)
                                    features[feature_name] = 1
                                    features[f'{feature_name}_count'] = len(matches)
                                    
                                    if extract_numbers and matches:
                                        # Extract numeric values from matches
                                        numbers = []
                                        for match in matches:
                                            if isinstance(match, tuple):
                                                numbers.extend([int(n) for n in match if n.isdigit()])
                                            elif match.isdigit():
                                                numbers.append(int(match))
                                        
                                        if numbers:
                                            features[f'{feature_name}_numbers'] = numbers
                                            features[f'{feature_name}_max_number'] = max(numbers)
                                            features[f'{feature_name}_min_number'] = min(numbers)
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
            
            # Network activity
            features['xhr_request_count'] = driver.execute_script("""
                var entries = performance.getEntriesByType('resource');
                var recent_requests = entries.filter(entry => 
                    entry.startTime > (performance.now() - 5000) && 
                    (entry.initiatorType === 'xmlhttprequest' || entry.initiatorType === 'fetch')
                );
                return recent_requests.length;
            """)
            
        except Exception as e:
            logger.debug(f"Dynamic feature extraction failed: {e}")
            features.update({
                'page_height': 0, 'viewport_height': 0, 'scroll_position': 0,
                'height_ratio': 0, 'scroll_percentage': 0, 'scroll_velocity': 0,
                'scroll_acceleration': 0, 'dom_node_count': 0, 'new_dom_nodes': 0,
                'significant_dom_change': 0, 'xhr_request_count': 0
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
                    domReady: timing.domContentLoadedEventEnd - timing.navigationStart,
                    firstPaint: performance.getEntriesByType('paint')[0] ? 
                        performance.getEntriesByType('paint')[0].startTime : 0
                };
            """)
            
            features['page_load_time'] = timing.get('loadTime', 0)
            features['dom_ready_time'] = timing.get('domReady', 0)
            features['first_paint_time'] = timing.get('firstPaint', 0)
            
            # Resource counts
            resource_counts = driver.execute_script("""
                var entries = performance.getEntriesByType('resource');
                var counts = {
                    images: 0, scripts: 0, stylesheets: 0, xhr: 0, other: 0
                };
                entries.forEach(function(entry) {
                    switch(entry.initiatorType) {
                        case 'img': counts.images++; break;
                        case 'script': counts.scripts++; break;
                        case 'link': counts.stylesheets++; break;
                        case 'xmlhttprequest':
                        case 'fetch': counts.xhr++; break;
                        default: counts.other++; break;
                    }
                });
                return counts;
            """)
            
            features.update({
                'resource_images': resource_counts.get('images', 0),
                'resource_scripts': resource_counts.get('scripts', 0),
                'resource_stylesheets': resource_counts.get('stylesheets', 0),
                'resource_xhr': resource_counts.get('xhr', 0),
                'resource_other': resource_counts.get('other', 0)
            })
            
        except Exception as e:
            logger.debug(f"Performance feature extraction failed: {e}")
            features.update({
                'page_load_time': 0, 'dom_ready_time': 0, 'first_paint_time': 0,
                'resource_images': 0, 'resource_scripts': 0, 'resource_stylesheets': 0,
                'resource_xhr': 0, 'resource_other': 0
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
            
            # Browser metadata
            features['user_agent'] = driver.execute_script("return navigator.userAgent")
            features['viewport_width'] = driver.execute_script("return window.innerWidth")
            features['screen_width'] = driver.execute_script("return screen.width")
            features['screen_height'] = driver.execute_script("return screen.height")
            
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
            features.update({
                'url': '', 'url_length': 0, 'url_has_page_param': 0,
                'url_page_number': 0, 'user_agent': '', 'viewport_width': 0,
                'screen_width': 0, 'screen_height': 0
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
            'dom_node_count', 'xhr_request_count', 'navigation_confidence',
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