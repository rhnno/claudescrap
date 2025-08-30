# Enhanced Analyzer Quick Reference Guide

## Table of Contents

1. [Quick Start Commands](#quick-start-commands)
2. [Configuration Cheat Sheet](#configuration-cheat-sheet)
3. [Common Patterns](#common-patterns)
4. [Troubleshooting Quick Fixes](#troubleshooting-quick-fixes)
5. [Performance Tuning](#performance-tuning)
6. [Template Usage](#template-usage)

## Quick Start Commands

### Basic Initialization

```python
# Basic usage with default config
from analyzer import ConfigurableAnalyzer
analyzer = ConfigurableAnalyzer()

# With custom config
analyzer = ConfigurableAnalyzer("config/enhanced_analyzer_config.yaml")

# With feature template
analyzer = ConfigurableAnalyzer("config/feature_templates/ecommerce_template.yaml")
```

### Common Operations

```python
# Analyze page structure
page_type = analyzer.analyze_page_structure(driver)
print(f"Detected: {page_type}")  # 'pagination', 'infinite_scroll', 'last_page'

# Collect training data
analyzer.collect_training_data(driver, "pagination", "amazon")

# Get feature summary for debugging
features = analyzer.get_feature_summary(driver)

# Train model
success = analyzer.train_model()

# Reset state for new session
analyzer.reset_state()
```

### Smart Scraper Usage

```python
from analyzer import SmartTokopediaScraper
from browser import BrowserManager

browser = BrowserManager()
browser.setup_driver()

scraper = SmartTokopediaScraper(browser)
results = scraper.smart_scrape("laptop", max_pages=20)

# Train on session data
scraper.train_on_current_session()
```

## Configuration Cheat Sheet

### Essential Configuration Sections

```yaml
# Minimal working configuration
paths:
  model_dir: models
  training_data_file: models/training_data.csv

features:
  selectors:
    pagination_buttons:
      primary:
        - selector: ".pagination button"
          priority: 10
          context: "basic_pagination"

sites:
  mysite:
    base_url: "https://example.com/search?q={query}"
    language: "english"

debug:
  verbose: true
```

### Selector Configuration Patterns

```yaml
# Priority-based selectors
selectors:
  element_type:
    primary:      # Priority 8-10 (highest)
      - selector: "specific-selector"
        priority: 10
        context: "descriptive_name"
    
    secondary:    # Priority 5-7 (medium)
      - selector: "fallback-selector"
        priority: 6
        context: "fallback_name"
    
    fallback:     # Priority 1-4 (lowest)
      - selector: "generic-selector"
        priority: 2
        context: "generic_name"
```

### Text Pattern Configuration

```yaml
# Pattern matching with confidence
text_analysis:
  end_of_results_patterns:
    high_confidence:
      - pattern: "no\\s+more\\s+results"
        weight: 0.9
        languages: ["english"]
    
    medium_confidence:
      - pattern: "end\\s+of\\s+page"
        weight: 0.6
        languages: ["english"]
```

### Site-Specific Configuration

```yaml
sites:
  site_name:
    base_url: "https://site.com/search?q={query}"
    language: "english"
    
    # Boost/reduce feature importance
    feature_weights:
      pagination_buttons: 1.2    # 20% more important
      lazy_load_elements: 0.8    # 20% less important
    
    # Site-specific patterns
    patterns:
      page_text: ["page", "results"]
      end_results: ["no more results"]
    
    # Custom selectors
    custom_selectors:
      product_container: ".product-list"
      next_button: ".next-page-btn"
```

## Common Patterns

### E-commerce Site Pattern

```yaml
# Copy this for e-commerce sites
feature_weights:
  pagination_buttons: 1.2
  product_elements: 1.5
  lazy_load_elements: 1.3

selectors:
  product_elements:
    primary:
      - selector: ".product-card, .product-item"
        priority: 10
        context: "ecommerce_products"

patterns:
  end_results: 
    - "no more products"
    - "end of catalog"
```

### News/Blog Site Pattern

```yaml
# Copy this for news/blog sites
feature_weights:
  pagination_buttons: 1.4
  next_button: 1.3
  text_patterns: 1.3

selectors:
  product_elements:  # Articles
    primary:
      - selector: "article, .post, .entry"
        priority: 10
        context: "news_articles"

patterns:
  end_results:
    - "no more posts"
    - "end of archive"
```

### Infinite Scroll Pattern

```yaml
# Copy this for infinite scroll sites
feature_weights:
  lazy_load_elements: 2.0
  scroll_velocity: 1.8
  xhr_request_count: 1.7
  pagination_buttons: 0.3  # Very low

dynamic_features:
  scroll_behavior:
    track_velocity: true
    track_acceleration: true
  
  network_activity:
    monitor_xhr_requests: true
    request_timeout_window: 2000
```

## Troubleshooting Quick Fixes

### Problem: Selectors Not Working

```yaml
# Add more fallback selectors
selectors:
  pagination_buttons:
    primary:
      - selector: ".pagination button"
        priority: 10
    fallback:
      - selector: "button"  # Very generic fallback
        priority: 1
```

### Problem: Poor Accuracy

```yaml
# Increase training data and adjust weights
training:
  samples_per_site: 50  # Increase from default 30
  
sites:
  problematic_site:
    feature_weights:
      # Boost important features for this site
      pagination_buttons: 1.5
      text_patterns: 1.3
```

### Problem: Slow Performance

```yaml
# Disable expensive features
features:
  dynamic_features:
    scroll_behavior:
      track_acceleration: false
    dom_monitoring:
      monitor_attribute_changes: false
    network_activity:
      track_image_loads: false

# Reduce text analysis
text_analysis:
  max_tfidf_features: 50  # Reduce from 200
  ngram_range: [1, 1]     # Only unigrams
```

### Problem: False Positives

```yaml
# Increase confidence threshold
features:
  thresholds:
    confidence_threshold: 0.8  # Increase from 0.7

# Add more specific patterns
text_analysis:
  end_of_results_patterns:
    high_confidence:
      - pattern: "very\\s+specific\\s+end\\s+pattern"
        weight: 0.95
```

## Performance Tuning

### Speed Optimization

```yaml
# Fast configuration
features:
  selectors:
    # Only primary selectors
    pagination_buttons:
      primary:
        - selector: ".pagination button"
          priority: 10
  
  dynamic_features:
    # Minimal dynamic features
    scroll_behavior:
      track_velocity: true
      track_acceleration: false
    dom_monitoring:
      track_node_additions: true
      track_node_removals: false

debug:
  verbose: false  # Disable verbose logging
```

### Accuracy Optimization

```yaml
# Comprehensive configuration
features:
  text_analysis:
    max_tfidf_features: 500  # More text features
    ngram_range: [1, 3]      # Include trigrams
  
  selectors:
    # All selector levels
    pagination_buttons:
      primary: [...]
      secondary: [...]
      tertiary: [...]
      fallback: [...]
  
  dynamic_features:
    # All dynamic features enabled
    scroll_behavior:
      track_velocity: true
      track_acceleration: true
    dom_monitoring:
      track_node_additions: true
      track_node_removals: true
      monitor_attribute_changes: true

training:
  ensemble:
    enable: true  # Use ensemble learning
```

### Memory Optimization

```yaml
# Memory-efficient configuration
features:
  text_analysis:
    max_tfidf_features: 100  # Limit text features
  
training:
  max_training_samples: 500  # Limit training data size

debug:
  performance_monitoring:
    track_memory_usage: true  # Monitor memory
  feature_analysis:
    generate_feature_distribution_plots: false  # Disable plots
```

## Template Usage

### Using Pre-built Templates

```python
# E-commerce template
analyzer = ConfigurableAnalyzer("config/feature_templates/ecommerce_template.yaml")

# News/blog template
analyzer = ConfigurableAnalyzer("config/feature_templates/news_blog_template.yaml")

# Infinite scroll template
analyzer = ConfigurableAnalyzer("config/feature_templates/infinite_scroll_template.yaml")
```

### Combining Templates with Custom Config

```python
# Load template first, then override with custom config
import yaml

# Load template
with open("config/feature_templates/ecommerce_template.yaml") as f:
    config = yaml.safe_load(f)

# Override specific settings
config['sites']['mysite'] = {
    'base_url': 'https://mysite.com/search?q={query}',
    'language': 'english'
}

# Save custom config
with open("config/my_custom_config.yaml", "w") as f:
    yaml.dump(config, f)

# Use custom config
analyzer = ConfigurableAnalyzer("config/my_custom_config.yaml")
```

### Creating Custom Templates

```yaml
# custom_template.yaml
template_name: "my_custom_template"
description: "Custom template for my specific use case"
version: "1.0"

# Include only the sections you need
selectors:
  pagination_buttons:
    primary:
      - selector: ".my-pagination"
        priority: 10
        context: "custom_pagination"

feature_weights:
  pagination_buttons: 1.5
  # ... other weights

# Template can be incomplete - missing sections will use defaults
```

## Command Line Usage

### Validation Commands

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Test configuration loading
python -c "
from analyzer import ConfigurableAnalyzer
analyzer = ConfigurableAnalyzer('config.yaml')
print('Config loaded successfully')
"
```

### Quick Testing

```python
# Test feature extraction without browser
from analyzer import ConfigurableAnalyzer

analyzer = ConfigurableAnalyzer()

# Mock driver for testing
class MockDriver:
    def find_elements(self, by, selector):
        return []  # Return empty list
    def find_element(self, by, selector):
        class MockElement:
            text = "sample page text"
        return MockElement()
    def execute_script(self, script):
        return 1000  # Mock return value
    current_url = "https://example.com"

mock_driver = MockDriver()
features = analyzer._extract_features(mock_driver)
print(f"Extracted {len(features)} features")
```

## Environment Variables

```bash
# Set configuration path
export ANALYZER_CONFIG_PATH="config/production_config.yaml"

# Set debug level
export ANALYZER_DEBUG_LEVEL="INFO"

# Set model directory
export ANALYZER_MODEL_DIR="models/production"
```

```python
# Use environment variables in code
import os
config_path = os.getenv('ANALYZER_CONFIG_PATH', 'config/analyzer_config.yaml')
analyzer = ConfigurableAnalyzer(config_path)
```

## Common Configuration Snippets

### Multi-language Support

```yaml
sites:
  multilingual_site:
    language_patterns:
      english:
        end_results: ["no more results", "end of results"]
      spanish:
        end_results: ["no más resultados", "fin de resultados"]
      french:
        end_results: ["plus de résultats", "fin des résultats"]
```

### Mobile-Specific Configuration

```yaml
features:
  selectors:
    pagination_buttons:
      primary:
        - selector: "button[role='button'][aria-label*='page']"
          priority: 10
          context: "mobile_accessible"

  thresholds:
    scroll_bottom_threshold: 0.9  # Higher for mobile
    confidence_threshold: 0.6     # Lower for mobile layouts
```

### Development vs Production

```yaml
# development_config.yaml
debug:
  verbose: true
  performance_monitoring:
    enable: true

training:
  samples_per_site: 10  # Fewer samples for faster development

---
# production_config.yaml
debug:
  verbose: false
  performance_monitoring:
    enable: false

training:
  samples_per_site: 50  # More samples for better accuracy
```

This quick reference guide provides the most commonly needed information for working with the Enhanced Analyzer Configuration System. For detailed explanations, refer to the full documentation guides.