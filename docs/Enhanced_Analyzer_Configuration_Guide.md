# Enhanced Analyzer Configuration Guide

## Overview

The Enhanced Analyzer Configuration System provides a powerful, YAML-driven approach to configure ML-powered pagination detection with rich feature extraction capabilities. This system allows you to create sophisticated training examples and fine-tune the analyzer for different websites and scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Structure](#configuration-structure)
3. [Feature Extraction Configuration](#feature-extraction-configuration)
4. [Site-Specific Configuration](#site-specific-configuration)
5. [Training Configuration](#training-configuration)
6. [Debug and Monitoring](#debug-and-monitoring)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Usage

```python
from analyzer import ConfigurableAnalyzer

# Initialize with enhanced configuration
analyzer = ConfigurableAnalyzer("config/enhanced_analyzer_config.yaml")

# Use with your browser driver
page_type = analyzer.analyze_page_structure(driver)
print(f"Detected pagination type: {page_type}")

# Collect training data with rich features
analyzer.collect_training_data(driver, "pagination", "tokopedia")
```

### Minimal Configuration

```yaml
# minimal_config.yaml
features:
  selectors:
    pagination_buttons:
      primary:
        - selector: ".pagination button"
          priority: 10
          context: "bootstrap_style"
    
sites:
  mysite:
    base_url: "https://example.com/search?q={query}"
    language: "english"
```

## Configuration Structure

### Root Level Sections

| Section | Purpose | Required |
|---------|---------|----------|
| `paths` | File paths and directories | Yes |
| `features` | Feature extraction configuration | Yes |
| `sites` | Site-specific configurations | No |
| `training` | ML training parameters | No |
| `debug` | Debug and monitoring settings | No |
| `export` | Export and integration settings | No |

### Paths Configuration

```yaml
paths:
  model_dir: models                           # Directory for model files
  training_data_file: models/training_data.csv # Training data CSV
  model_file: pagination_model.pkl            # Trained model file
  vectorizer_file: pagination_vectorizer.pkl  # TF-IDF vectorizer
  feature_templates_dir: config/feature_templates # Feature templates
```

## Feature Extraction Configuration

### Text Analysis

#### End of Results Patterns

Configure patterns with confidence weights to detect when pagination ends:

```yaml
features:
  text_analysis:
    end_of_results_patterns:
      high_confidence:
        - pattern: "no\\s+more\\s+results"
          weight: 0.9
          languages: ["english"]
        - pattern: "tidak\\s+ada\\s+hasil\\s+lagi"
          weight: 0.9
          languages: ["indonesian"]
      
      medium_confidence:
        - pattern: "no\\s+more\\s+items"
          weight: 0.7
          languages: ["english"]
      
      low_confidence:
        - pattern: "end\\s+of\\s+page"
          weight: 0.4
          languages: ["english", "indonesian"]
```

**Pattern Configuration:**
- `pattern`: Regex pattern to match
- `weight`: Confidence score (0.0-1.0)
- `languages`: Applicable languages

#### Pagination Text Patterns

Detect pagination indicators in page text:

```yaml
features:
  text_analysis:
    pagination_text_patterns:
      strong_indicators:
        - pattern: "page\\s+\\d+\\s+of\\s+\\d+"
          weight: 0.95
          feature_name: "strong_pagination_text"
        - pattern: "showing\\s+\\d+\\s*-\\s*\\d+\\s+of\\s+\\d+"
          weight: 0.9
          feature_name: "results_range_text"
```

### Enhanced Selectors

#### Priority-Based Selector Configuration

Configure selectors with priority and context for better feature extraction:

```yaml
features:
  selectors:
    pagination_buttons:
      primary:                                    # High priority selectors
        - selector: "nav[aria-label*='pagination'] button"
          priority: 10                           # Priority score (1-10)
          context: "semantic_navigation"         # Context identifier
        - selector: ".pagination button:not([disabled])"
          priority: 9
          context: "bootstrap_style"
      
      secondary:                                 # Medium priority selectors
        - selector: "a[class*='pagination']"
          priority: 7
          context: "link_based"
      
      fallback:                                  # Low priority selectors
        - selector: "button[aria-label*='page']"
          priority: 3
          context: "aria_fallback"
```

**Selector Configuration:**
- `selector`: CSS selector or `xpath`: XPath expression
- `priority`: Priority score (1-10, higher = more important)
- `context`: Descriptive context for feature naming

#### Supported Selector Types

| Type | Description | Example |
|------|-------------|---------|
| `pagination_buttons` | Traditional pagination buttons | `.pagination button` |
| `next_buttons` | Next page buttons/links | `//button[contains(text(), 'Next')]` |
| `load_more_buttons` | Load more/show more buttons | `//button[contains(text(), 'Load More')]` |
| `lazy_load_elements` | Lazy loading indicators | `img[loading='lazy']` |
| `product_elements` | Product/item containers | `[data-testid*='product']` |

### Dynamic Features

#### Scroll Behavior Tracking

```yaml
features:
  dynamic_features:
    scroll_behavior:
      track_velocity: true                      # Track scroll speed
      track_acceleration: true                  # Track scroll acceleration
      velocity_window: 5                        # Measurements for calculation
      detect_smooth_scrolling: true             # Detect smooth scrolling
```

#### DOM Monitoring

```yaml
features:
  dynamic_features:
    dom_monitoring:
      track_node_additions: true                # Monitor new DOM nodes
      track_node_removals: true                 # Monitor removed nodes
      monitor_attribute_changes: true           # Monitor attribute changes
      significant_change_threshold: 10          # Minimum nodes for significance
```

#### Network Activity

```yaml
features:
  dynamic_features:
    network_activity:
      monitor_xhr_requests: true                # Track AJAX requests
      monitor_fetch_requests: true              # Track fetch API calls
      track_image_loads: true                   # Track image loading
      request_timeout_window: 5000              # Time window in milliseconds
```

### Adaptive Thresholds

```yaml
features:
  thresholds:
    confidence_threshold: 0.7                   # Base confidence threshold
    adaptive_thresholds:
      enable: true                              # Enable adaptive behavior
      confidence_decay: 0.05                    # Reduce threshold over time
      min_confidence: 0.5                       # Minimum threshold
      scroll_patience: 3                        # Rounds before declaring end
```

## Site-Specific Configuration

### Basic Site Configuration

```yaml
sites:
  tokopedia:
    base_url: "https://www.tokopedia.com/search?q={query}"
    language: "indonesian"
    encoding: "utf-8"
    
    # Site-specific feature weights
    feature_weights:
      pagination_buttons: 1.2                   # Boost pagination button importance
      lazy_load_elements: 1.5                   # Boost lazy loading detection
      text_patterns: 1.1                        # Boost text pattern matching
```

### Advanced Site Configuration

```yaml
sites:
  amazon:
    base_url: "https://www.amazon.com/s?k={query}"
    language: "english"
    encoding: "utf-8"
    
    # Enhanced pattern matching
    patterns:
      page_text: 
        - "page"
        - "results"
        - "of over"
      end_results: 
        - "no more results"
        - "end of results"
      pagination_indicators:
        - "page \\d+ of"                        # Regex patterns supported
        - "\\d+-\\d+ of over"
    
    # Site-specific selectors
    custom_selectors:
      product_container: "[data-component-type='s-search-result']"
      pagination_container: ".a-pagination"
      next_button: ".a-pagination .a-last a"
    
    # Performance tuning
    performance:
      page_load_timeout: 25
      element_wait_timeout: 8
      scroll_pause_time: 1.5
```

### Feature Weights

Adjust the importance of different features for specific sites:

```yaml
sites:
  shopee:
    feature_weights:
      lazy_load_elements: 2.0                   # Very important for Shopee
      scroll_velocity: 1.5                      # Important for infinite scroll
      xhr_request_count: 1.3                    # Important for AJAX loading
      pagination_buttons: 0.5                   # Less important (infinite scroll site)
```

## Training Configuration

### Basic Training Settings

```yaml
training:
  test_size: 0.2                                # Train/test split ratio
  random_state: 42                              # Reproducibility seed
  n_estimators: 150                             # Random Forest trees
  max_depth: 15                                 # Maximum tree depth
  
  # Data collection
  auto_collect_samples: true                    # Automatic data collection
  samples_per_site: 30                         # Samples per site
  max_training_samples: 1000                    # Maximum total samples
```

### Advanced Training Features

#### Feature Engineering

```yaml
training:
  feature_engineering:
    enable_feature_selection: true              # Automatic feature selection
    feature_selection_k: 50                     # Top K features to select
    create_interaction_features: true           # Create feature interactions
    create_polynomial_features: false           # Create polynomial features
    polynomial_degree: 2                        # Polynomial degree
```

#### Model Ensemble

```yaml
training:
  ensemble:
    enable: true                                # Enable ensemble learning
    models:
      - name: "random_forest"
        params:
          n_estimators: 150
          max_depth: 15
        weight: 0.4                             # Ensemble weight
      - name: "gradient_boosting"
        params:
          n_estimators: 100
          learning_rate: 0.1
        weight: 0.3
      - name: "svm"
        params:
          kernel: "rbf"
          C: 1.0
        weight: 0.3
```

#### Cross-Validation

```yaml
training:
  cross_validation:
    enable: true                                # Enable cross-validation
    cv_folds: 5                                 # Number of folds
    scoring: ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
```

#### Data Quality Controls

```yaml
training:
  data_quality:
    remove_duplicates: true                     # Remove duplicate samples
    handle_missing_values: true                 # Handle missing values
    outlier_detection: true                     # Detect and handle outliers
    balance_classes: true                       # Balance class distribution
```

## Debug and Monitoring

### Basic Debug Settings

```yaml
debug:
  verbose: true                                 # Verbose logging
  save_features: true                           # Save extracted features
  feature_summary: true                         # Show feature summaries
  log_level: "INFO"                            # Logging level
```

### Performance Monitoring

```yaml
debug:
  performance_monitoring:
    enable: true                                # Enable performance tracking
    track_extraction_time: true                 # Track feature extraction time
    track_prediction_time: true                 # Track prediction time
    track_memory_usage: true                    # Track memory usage
```

### Feature Analysis

```yaml
debug:
  feature_analysis:
    enable: true                                # Enable feature analysis
    save_feature_importance: true               # Save feature importance scores
    create_feature_correlation_matrix: true     # Create correlation matrix
    generate_feature_distribution_plots: false  # Generate distribution plots
```

### Model Evaluation

```yaml
debug:
  model_evaluation:
    enable: true                                # Enable model evaluation
    save_confusion_matrix: true                 # Save confusion matrix
    save_classification_report: true            # Save classification report
    save_learning_curves: false                 # Save learning curves
```

## Advanced Features

### Custom Feature Templates

Create reusable feature templates for different scenarios:

```yaml
# config/feature_templates/ecommerce_template.yaml
template_name: "ecommerce_pagination"
description: "Template for e-commerce pagination detection"

selectors:
  pagination_buttons:
    primary:
      - selector: ".pagination button, .pager button"
        priority: 10
        context: "ecommerce_pagination"
  
  product_elements:
    primary:
      - selector: ".product-item, .listing-item"
        priority: 10
        context: "ecommerce_products"

patterns:
  pagination_indicators:
    - "page \\d+ of \\d+"
    - "showing \\d+-\\d+ of \\d+"
  
  end_indicators:
    - "no more products"
    - "end of catalog"
```

### API Integration

```yaml
export:
  api_integration:
    enable: true                                # Enable API integration
    webhook_url: "https://api.example.com/webhook"
    api_key: "your-api-key"
    batch_size: 100                             # Batch size for API calls
```

### Export Formats

```yaml
export:
  formats: ["json", "csv", "pickle"]            # Export formats
  include_metadata: true                        # Include metadata in exports
  compress_exports: true                        # Compress export files
```

## Best Practices

### 1. Configuration Organization

- **Separate environments**: Use different configs for development, testing, and production
- **Modular configuration**: Break large configs into smaller, focused files
- **Version control**: Track configuration changes with version control

### 2. Feature Engineering

- **Start simple**: Begin with basic selectors and gradually add complexity
- **Test incrementally**: Test each new feature addition separately
- **Monitor performance**: Track feature extraction time and model performance

### 3. Site-Specific Tuning

- **Analyze target sites**: Study the HTML structure and behavior patterns
- **Use browser dev tools**: Inspect elements to find reliable selectors
- **Test across pages**: Verify selectors work across different pages

### 4. Training Data Quality

- **Diverse samples**: Collect samples from different page types and states
- **Quality over quantity**: Focus on high-quality, correctly labeled samples
- **Regular updates**: Periodically update training data as sites change

### 5. Performance Optimization

- **Selector efficiency**: Use efficient CSS selectors and XPath expressions
- **Feature selection**: Remove redundant or low-importance features
- **Caching**: Cache model predictions for similar page structures

## Troubleshooting

### Common Issues

#### 1. Configuration Loading Errors

**Problem**: YAML syntax errors or missing required sections

**Solution**:
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check required sections
python -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
required = ['features', 'paths']
missing = [r for r in required if r not in config]
print('Missing sections:', missing)
"
```

#### 2. Selector Not Working

**Problem**: CSS selectors or XPath expressions not finding elements

**Solution**:
```python
# Test selectors in browser console
driver.find_elements(By.CSS_SELECTOR, "your-selector")

# Add fallback selectors
pagination_buttons:
  primary:
    - selector: ".pagination button"
      priority: 10
  fallback:
    - selector: "button[aria-label*='page']"
      priority: 3
```

#### 3. Poor Model Performance

**Problem**: Low accuracy or inconsistent predictions

**Solution**:
- Increase training data diversity
- Adjust feature weights for problematic sites
- Enable cross-validation to identify overfitting
- Review feature importance scores

#### 4. Slow Feature Extraction

**Problem**: Feature extraction takes too long

**Solution**:
```yaml
# Disable expensive features
features:
  dynamic_features:
    network_activity:
      monitor_xhr_requests: false
    
# Reduce selector complexity
selectors:
  pagination_buttons:
    primary:  # Only use primary selectors
      - selector: ".pagination button"
        priority: 10
```

### Debug Techniques

#### 1. Enable Verbose Logging

```yaml
debug:
  verbose: true
  performance_monitoring:
    enable: true
    track_extraction_time: true
```

#### 2. Feature Analysis

```python
# Get feature summary
analyzer = ConfigurableAnalyzer()
features = analyzer.get_feature_summary(driver)

# Check feature completeness
completeness = features.get('_feature_completeness', 0)
print(f"Feature completeness: {completeness:.2f}")
```

#### 3. Model Evaluation

```python
# Train with evaluation
analyzer.train_model()

# Check feature importance
if analyzer.config['debug']['feature_analysis']['enable']:
    # Feature importance will be saved automatically
    pass
```

### Performance Tuning

#### 1. Optimize Selectors

```yaml
# Use specific, efficient selectors
selectors:
  pagination_buttons:
    primary:
      - selector: "nav[role='navigation'] button[aria-label*='page']"
        priority: 10
        context: "semantic_specific"
```

#### 2. Reduce Feature Complexity

```yaml
# Disable expensive features for production
features:
  dynamic_features:
    scroll_behavior:
      track_acceleration: false  # Disable if not needed
    dom_monitoring:
      monitor_attribute_changes: false  # Expensive operation
```

#### 3. Adjust Timeouts

```yaml
sites:
  mysite:
    performance:
      page_load_timeout: 15      # Reduce if site is fast
      element_wait_timeout: 5    # Reduce for faster elements
      scroll_pause_time: 1       # Reduce for faster scrolling
```

## Migration Guide

### From Basic to Enhanced Configuration

#### Step 1: Backup Current Configuration

```bash
cp config/analyzer_config.yaml config/analyzer_config_backup.yaml
```

#### Step 2: Convert Simple Selectors

**Before (Basic)**:
```yaml
selectors:
  pagination_buttons:
    - ".pagination button"
    - "button[class*='pagination']"
```

**After (Enhanced)**:
```yaml
selectors:
  pagination_buttons:
    primary:
      - selector: ".pagination button"
        priority: 10
        context: "bootstrap_style"
      - selector: "button[class*='pagination']"
        priority: 8
        context: "class_based"
```

#### Step 3: Add Site-Specific Configuration

```yaml
sites:
  your_site:
    base_url: "https://yoursite.com/search?q={query}"
    language: "english"
    feature_weights:
      pagination_buttons: 1.2
    patterns:
      page_text: ["page", "results"]
      end_results: ["no more results"]
```

#### Step 4: Enable Enhanced Features

```yaml
features:
  dynamic_features:
    scroll_behavior:
      track_velocity: true
    dom_monitoring:
      track_node_additions: true
    network_activity:
      monitor_xhr_requests: true

debug:
  performance_monitoring:
    enable: true
  feature_analysis:
    enable: true
```

### Testing Migration

```python
# Test with both configurations
basic_analyzer = ConfigurableAnalyzer("config/analyzer_config_backup.yaml")
enhanced_analyzer = ConfigurableAnalyzer("config/enhanced_analyzer_config.yaml")

# Compare feature extraction
basic_features = basic_analyzer._extract_features(driver)
enhanced_features = enhanced_analyzer._extract_features(driver)

print(f"Basic features: {len(basic_features)}")
print(f"Enhanced features: {len(enhanced_features)}")
```

## Conclusion

The Enhanced Analyzer Configuration System provides powerful tools for creating sophisticated pagination detection systems. By leveraging the rich configuration options, you can:

- Create highly accurate, site-specific models
- Extract detailed features for better training
- Monitor and optimize performance
- Scale across multiple websites and languages

Start with the basic configuration and gradually add enhanced features as needed. The system is designed to be backward-compatible while providing advanced capabilities for complex scenarios.

For additional support or questions, refer to the troubleshooting section or check the example configurations in the `config/` directory.