# Enhanced ML-Powered Web Scraper - Developer Documentation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [API Reference](#api-reference)
5. [Configuration System](#configuration-system)
6. [ML Pipeline](#ml-pipeline)
7. [Development Guidelines](#development-guidelines)
8. [Testing Framework](#testing-framework)
9. [Deployment Guide](#deployment-guide)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [Future Roadmap](#future-roadmap)

---

## 🎯 Project Overview

### Purpose
Enhanced ML-powered web scraper with intelligent pagination detection for Indonesian e-commerce sites. Built for academic research on e-commerce trends with production-ready capabilities.

### Key Features
- **ML-Powered Pagination Detection**: 50+ features with site-specific intelligence
- **YAML-Configurable Templates**: No code changes needed for new sites
- **Multi-Site Support**: Tokopedia, Shopee, Amazon with extensible architecture
- **Production-Ready**: Batch processing, error handling, session persistence
- **Research-Focused**: Rich feature extraction for academic analysis

### Technology Stack
- **Backend**: Python 3.8+
- **Web Automation**: Selenium WebDriver
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Configuration**: YAML, JSON
- **Storage**: CSV, Excel, JSON outputs

---

## 🏗️ Architecture

### System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Scraper System                  │
├─────────────────────────────────────────────────────────────┤
│  Training Orchestrator  │  Scraping Orchestrator           │
│  ├─ Data Collection     │  ├─ Batch Processing             │
│  ├─ Model Training      │  ├─ Multi-Site Support           │
│  └─ Validation         │  └─ Output Management            │
├─────────────────────────────────────────────────────────────┤
│                Enhanced Analyzer Core                       │
│  ├─ ML Prediction Engine    ├─ YAML Template System        │
│  ├─ Feature Extraction     ├─ Site-Specific Intelligence   │
│  ├─ Confidence Scoring     └─ Dynamic Content Monitoring   │
├─────────────────────────────────────────────────────────────┤
│              Browser Management Layer                       │
│  ├─ Profile Management     ├─ Session Persistence          │
│  ├─ Anti-Detection        └─ Error Recovery                │
├─────────────────────────────────────────────────────────────┤
│                Storage & Utilities                          │
│  ├─ Data Storage          ├─ Random Utils                  │
│  ├─ File Management       └─ Logging System                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
Input Query → Browser Setup → Page Navigation → Feature Extraction 
     ↓
ML Analysis → Pagination Detection → Navigation Strategy → Data Extraction
     ↓
Quality Validation → Storage → Output Generation → Session Statistics
```

### Module Dependencies
```python
# Core Dependencies
selenium>=4.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
PyYAML>=6.0
openpyxl>=3.0.0

# Optional Dependencies
webdriver-manager>=3.8.0  # Auto ChromeDriver management
beautifulsoup4>=4.10.0    # HTML parsing fallback
requests>=2.28.0          # HTTP requests
```

---

## 🔧 Core Components

### 1. Enhanced Analyzer (`func/analyzer.py`)

#### Purpose
Core ML-powered pagination detection with site-specific intelligence.

#### Key Classes

##### `EnhancedConfigurableAnalyzer`
```python
class EnhancedConfigurableAnalyzer:
    """
    Advanced ML-powered pagination analyzer with rich YAML template support
    """
    
    def __init__(self, config_path=None, template_path=None):
        """Initialize with configuration and templates"""
        
    def analyze_page_structure(self, driver, site_name="unknown") -> str:
        """
        Main analysis method
        Returns: 'pagination', 'infinite_scroll', or 'last_page'
        """
        
    def _extract_enhanced_features(self, driver, site_name) -> Dict[str, Any]:
        """Extract 50+ features using YAML templates"""
        
    def _ml_prediction(self, features) -> Tuple[str, float]:
        """ML prediction with confidence score"""
```

#### Feature Categories
1. **Weighted Selectors**: Site-specific CSS/XPath with confidence weights
2. **Text Analysis**: Regex patterns with language-specific rules
3. **Dynamic Content**: Scroll velocity, DOM changes, network activity
4. **Performance Metrics**: Load times, resource counts
5. **Metadata**: URL analysis, browser information
6. **Derived Features**: Calculated from base features

### 2. Browser Manager (`func/browser.py`)

#### Purpose
Manages browser instances with anti-detection and session persistence.

#### Key Features
- **Profile Management**: Persistent Chrome profiles
- **Anti-Detection**: Remove automation indicators
- **Error Recovery**: Automatic retry and fallback
- **Session Persistence**: Login state preservation

### 3. Training Orchestrator (`train_model_orchestrator.py`)

#### Purpose
Automated training data collection and ML model training.

#### Workflow
1. **Data Collection**: Multi-site automated browsing
2. **Ground Truth Labeling**: Automatic page type detection
3. **Feature Engineering**: Extract training features
4. **Model Training**: Cross-validation and evaluation
5. **Model Persistence**: Save trained models

### 4. Scraping Orchestrator (`scraping_orchestrator.py`)

#### Purpose
Production-ready batch scraping with trained models.

#### Features
- **Batch Processing**: Multiple queries and sites
- **Configuration-Driven**: JSON-based setup
- **Error Handling**: Comprehensive retry logic
- **Output Management**: Multiple formats (CSV, Excel, JSON)
- **Session Monitoring**: Real-time statistics

---

## 📚 API Reference

### Core Classes

#### `EnhancedConfigurableAnalyzer`

```python
from func.analyzer import EnhancedConfigurableAnalyzer

# Initialize
analyzer = EnhancedConfigurableAnalyzer(
    config_path="config/analyzer_config.yaml",
    template_path="config/enhanced_training_templates.yaml"
)

# Main analysis method
result = analyzer.analyze_page_structure(driver, site_name='tokopedia')
# Returns: 'pagination' | 'infinite_scroll' | 'last_page'

# Extract features
features = analyzer._extract_enhanced_features(driver, 'tokopedia')
# Returns: Dict with 50+ features

# Load/Save model
analyzer.load_model()  # Returns: bool
analyzer.save_model()  # Saves to configured path

# Reset state for new session
analyzer.reset_state()
```

#### `BrowserManager`

```python
from func.browser import BrowserManager

# Initialize browser
browser = BrowserManager(
    headless=False,
    use_profile=True,
    profile_name="research_profile"
)

# Navigation
browser.navigate_to(url)
browser.wait_for_element(selector, timeout=10)

# Cleanup
browser.close()
```

#### `EnhancedTokopediaScraper`

```python
from enhanced_tokoscrape import EnhancedTokopediaScraper

# Initialize
scraper = EnhancedTokopediaScraper(use_trained_model=True)

# Setup browser
scraper.setup_browser(headless=False, use_profile=True)

# Scrape products
products = scraper.scrape_tokopedia_products(
    search_query="laptop gaming",
    max_pages=10,
    depth_scroll=3
)

# Save results
csv_file, excel_file = scraper.save_results(products, "laptop_gaming")

# Cleanup
scraper.close()
```

#### `TrainingOrchestrator`

```python
from train_model_orchestrator import TrainingOrchestrator

# Initialize
orchestrator = TrainingOrchestrator()

# Setup browser
orchestrator.setup_browser(headless=True)

# Collect training data
training_data = orchestrator.collect_training_data(
    max_pages_per_query=5,
    max_samples=100
)

# Train model
success = orchestrator.train_model(test_size=0.2)

# Save data and logs
orchestrator.save_training_data()

# Cleanup
orchestrator.close()
```

#### `ScrapingOrchestrator`

```python
from scraping_orchestrator import ScrapingOrchestrator

# Initialize
orchestrator = ScrapingOrchestrator(use_trained_model=True)

# Setup browser
orchestrator.setup_browser(headless=False, use_profile=True)

# Load configuration
config = {
    "sites": [
        {
            "name": "tokopedia",
            "queries": ["laptop", "smartphone"],
            "max_pages": 5,
            "scroll_depth": 3
        }
    ],
    "output_format": ["csv", "excel"],
    "delay_range": [1, 3]
}

# Run batch scraping
results = orchestrator.run_batch_scraping(config)

# Cleanup
orchestrator.close()
```

### Utility Functions

#### `RandomUtils`

```python
from func.utils import RandomUtils

# Random delays
delay = RandomUtils.random_delay(min_delay=0.5, max_delay=3)
RandomUtils.random_sleep(min_sec=1, max_sec=2)

# Human-like scrolling
RandomUtils.human_like_scroll(driver, scroll_pause_time=1)
```

#### `DataStorage`

```python
from func.storage import DataStorage

storage = DataStorage()

# Save HTML
filepath = storage.save_html(html_content, query="laptop", page_num=1)

# Save to CSV
filepath = storage.save_to_csv(listings, query="laptop")
```

---

## ⚙️ Configuration System

### YAML Template Structure (`config/enhanced_training_templates.yaml`)

#### Site Configuration
```yaml
training_templates:
  ecommerce_sites:
    tokopedia:
      base_config:
        name: "Tokopedia Indonesia"
        language: "indonesian"
        encoding: "utf-8"
        base_url: "https://www.tokopedia.com/search?q={query}"
      
      selectors:
        pagination_buttons:
          high_confidence:
            - selector: "button[class*='css-1turmok-unf-btn']"
              priority: 10
              context: "tokopedia_main_pagination"
              weight: 0.95
          medium_confidence:
            - selector: ".pagination button:not([disabled])"
              priority: 6
              context: "generic_pagination"
              weight: 0.7
      
      text_patterns:
        end_of_results:
          high_confidence:
            - pattern: "tidak\\s+ada\\s+produk\\s+lagi"
              weight: 0.95
              feature_name: "tokopedia_no_more_products"
      
      training_scenarios:
        pagination_examples:
          scenario_1:
            description: "First page with clear pagination"
            expected_features:
              pagination_buttons: ">= 3"
              next_button: ">= 1"
            expected_type: "pagination"
            confidence_threshold: 0.9
```

#### Adding New Sites
```yaml
training_templates:
  ecommerce_sites:
    your_new_site:
      base_config:
        name: "Your E-commerce Site"
        language: "english"
        base_url: "https://yoursite.com/search?q={query}"
      
      selectors:
        pagination_buttons:
          high_confidence:
            - selector: ".your-pagination-class"
              priority: 10
              context: "your_site_pagination"
              weight: 0.95
      
      text_patterns:
        end_of_results:
          high_confidence:
            - pattern: "no\\s+more\\s+items"
              weight: 0.9
              feature_name: "your_site_no_more"
```

### JSON Configuration (`scraping_config.json`)

#### Basic Structure
```json
{
  "sites": [
    {
      "name": "tokopedia",
      "queries": ["laptop gaming", "smartphone android"],
      "max_pages": 10,
      "scroll_depth": 3
    },
    {
      "name": "shopee", 
      "queries": ["laptop", "hp samsung"],
      "max_pages": 5,
      "scroll_depth": 2
    }
  ],
  "output_format": ["csv", "excel", "json"],
  "delay_range": [1, 3],
  "settings": {
    "max_retries": 3,
    "timeout": 30,
    "save_html": false,
    "detailed_logging": true
  }
}
```

#### Configuration Options

##### Site Configuration
- **name**: Site identifier (must match YAML template)
- **queries**: List of search terms
- **max_pages**: Maximum pages to scrape per query
- **scroll_depth**: Number of scrolls per page

##### Output Configuration
- **output_format**: Array of formats ["csv", "excel", "json"]
- **delay_range**: [min_seconds, max_seconds] between requests

##### Advanced Settings
- **max_retries**: Retry attempts for failed requests
- **timeout**: Request timeout in seconds
- **save_html**: Save raw HTML files
- **detailed_logging**: Enable verbose logging

### Analyzer Configuration (`config/analyzer_config.yaml`)

```yaml
paths:
  model_dir: 'models'
  training_data_file: 'models/enhanced_training_data.csv'
  model_file: 'enhanced_pagination_model.pkl'
  vectorizer_file: 'enhanced_pagination_vectorizer.pkl'

features:
  text_analysis:
    max_tfidf_features: 150
    stop_words: 'english'
  thresholds:
    confidence_threshold: 0.75
    footer_threshold_ratio: 0.8
    scroll_bottom_threshold: 0.95

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

debug:
  verbose: true
  save_features: true
  feature_summary: true
```

---

## 🧠 ML Pipeline

### Training Pipeline Architecture

```
Data Collection → Feature Engineering → Model Training → Validation → Deployment
      ↓                    ↓                ↓             ↓           ↓
   Multi-Site         50+ Features     Cross-Validation  Metrics   Production
   Browsing          Extraction        Grid Search       Analysis   Model
```

### Feature Engineering

#### Feature Categories (50+ total)

1. **Weighted Selectors** (15-20 features)
   ```python
   # Example features
   pagination_buttons_total: int
   pagination_buttons_weighted_score: float
   pagination_buttons_max_confidence: float
   next_buttons_tokopedia_main_pagination: int
   ```

2. **Text Analysis** (10-15 features)
   ```python
   # Example features
   page_text_length: int
   word_count: int
   tokopedia_no_more_products: int
   end_of_results_confidence: float
   ```

3. **Dynamic Content** (8-12 features)
   ```python
   # Example features
   scroll_velocity: float
   scroll_acceleration: float
   dom_node_count: int
   new_dom_nodes: int
   xhr_request_count: int
   ```

4. **Performance Metrics** (5-8 features)
   ```python
   # Example features
   page_load_time: int
   dom_ready_time: int
   resource_images: int
   resource_scripts: int
   ```

5. **Derived Features** (5-10 features)
   ```python
   # Example features
   content_density: float
   navigation_confidence: float
   infinite_scroll_confidence: float
   page_completeness: float
   ```

### Model Training Process

#### 1. Data Collection
```python
# Automated training data collection
orchestrator = TrainingOrchestrator()
training_data = orchestrator.collect_training_data(
    max_pages_per_query=10,
    max_samples=500
)
```

#### 2. Feature Preparation
```python
# Convert to ML-ready format
X, y = orchestrator.prepare_training_dataset()
# X: Feature matrix (samples × features)
# y: Labels ['pagination', 'infinite_scroll', 'last_page']
```

#### 3. Model Training
```python
# Train with cross-validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_macro'
)
```

#### 4. Model Evaluation
```python
# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
confusion_matrix(y_test, y_pred)

# Feature importance analysis
feature_importance = model.feature_importances_
```

### Model Deployment

#### Model Persistence
```python
# Save trained model
import pickle
with open('models/enhanced_pagination_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load for production
with open('models/enhanced_pagination_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

#### Production Integration
```python
# Use in production
analyzer = EnhancedConfigurableAnalyzer()
analyzer.load_model()  # Loads saved model

# Make predictions
features = analyzer._extract_enhanced_features(driver, 'tokopedia')
prediction, confidence = analyzer._ml_prediction(features)
```

---

## 👨‍💻 Development Guidelines

### Code Style

#### Python Standards
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations for better code clarity
- **Docstrings**: Document all classes and methods
- **Error Handling**: Comprehensive exception handling

#### Example Code Structure
```python
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ExampleClass:
    """
    Example class following project standards
    
    Args:
        config_path: Path to configuration file
        debug: Enable debug mode
    """
    
    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        self.config = self._load_config(config_path)
        self.debug = debug
        logger.info("ExampleClass initialized")
    
    def process_data(self, data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Process input data with error handling
        
        Args:
            data: List of data dictionaries
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Processing logic
            result = self._internal_process(data)
            return True, "Success"
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False, str(e)
    
    def _internal_process(self, data: List[Dict[str, Any]]) -> Any:
        """Internal processing method"""
        # Implementation
        pass
```

### Project Structure

#### Directory Organization
```
project_root/
├── func/                          # Core functionality
│   ├── __init__.py               # Package initialization
│   ├── analyzer.py               # ML analyzer core
│   ├── browser.py                # Browser management
│   ├── storage.py                # Data storage
│   └── utils.py                  # Utility functions
├── config/                       # Configuration files
│   ├── analyzer_config.yaml      # Analyzer settings
│   ├── enhanced_training_templates.yaml  # Site templates
│   └── feature_templates/        # Feature-specific templates
├── models/                       # ML models and training data
│   ├── enhanced_pagination_model.pkl
│   ├── training_data_*.json
│   └── training_logs/
├── data/                         # Scraped data
│   ├── raw_html/                 # Raw HTML files
│   └── processed/                # Processed CSV/Excel files
├── logs/                         # Application logs
├── tests/                        # Test files
└── docs/                         # Documentation
```

#### File Naming Conventions
- **Scripts**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Config files**: `lowercase_with_underscores.yaml`

### Adding New Sites

#### Step 1: Create YAML Template
```yaml
# Add to config/enhanced_training_templates.yaml
training_templates:
  ecommerce_sites:
    new_site:
      base_config:
        name: "New E-commerce Site"
        language: "english"
        base_url: "https://newsite.com/search?q={query}"
      
      selectors:
        pagination_buttons:
          high_confidence:
            - selector: ".pagination-btn"
              priority: 10
              context: "newsite_pagination"
              weight: 0.95
        
        product_containers:
          primary:
            - selector: ".product-card"
              priority: 10
              context: "newsite_product"
              weight: 1.0
      
      text_patterns:
        end_of_results:
          high_confidence:
            - pattern: "no\\s+more\\s+results"
              weight: 0.95
              feature_name: "newsite_no_more"
```

#### Step 2: Add Product Extraction Logic
```python
# In enhanced_tokoscrape.py or scraping_orchestrator.py
def _extract_newsite_product(self, item):
    """Extract product from new site"""
    product = {}
    
    try:
        # Site-specific selectors
        product['Product Name'] = item.find_element(
            By.CSS_SELECTOR, '.product-title'
        ).text
        
        product['Price'] = item.find_element(
            By.CSS_SELECTOR, '.price'
        ).text
        
        # Add standard fields
        product['scraped_at'] = datetime.now().isoformat()
        product['site'] = 'newsite'
        
        return product
    except Exception as e:
        logger.debug(f"Product extraction failed: {e}")
        return None
```

#### Step 3: Update Configuration
```json
// Add to scraping_config.json
{
  "sites": [
    {
      "name": "newsite",
      "queries": ["laptop", "smartphone"],
      "max_pages": 5,
      "scroll_depth": 3
    }
  ]
}
```

#### Step 4: Test Integration
```python
# Test new site
python verify_enhanced_analyzer.py

# Train with new site data
python train_model_orchestrator.py

# Test scraping
python scraping_orchestrator.py
```

---

## 🧪 Testing Framework

### Test Structure

#### Unit Tests
```python
# tests/test_analyzer.py
import unittest
from func.analyzer import EnhancedConfigurableAnalyzer

class TestEnhancedAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = EnhancedConfigurableAnalyzer()
    
    def test_template_loading(self):
        """Test YAML template loading"""
        templates = self.analyzer.templates
        self.assertIn('training_templates', templates)
        
    def test_site_template_retrieval(self):
        """Test site-specific template retrieval"""
        template = self.analyzer._get_site_template('tokopedia')
        self.assertIsNotNone(template)
        
    def test_feature_extraction(self):
        """Test feature extraction with mock driver"""
        # Mock driver implementation
        mock_driver = MockWebDriver()
        features = self.analyzer._extract_enhanced_features(
            mock_driver, 'tokopedia'
        )
        self.assertGreater(len(features), 30)  # Should have 30+ features
```

#### Integration Tests
```python
# tests/test_integration.py
import unittest
from enhanced_tokoscrape import EnhancedTokopediaScraper

class TestIntegration(unittest.TestCase):
    
    def test_full_scraping_workflow(self):
        """Test complete scraping workflow"""
        scraper = EnhancedTokopediaScraper(use_trained_model=False)
        
        # Test with mock browser
        mock_browser = MockBrowserManager()
        scraper.browser = mock_browser
        
        products = scraper.scrape_tokopedia_products(
            search_query="test",
            max_pages=1,
            depth_scroll=1
        )
        
        self.assertIsInstance(products, list)
```

#### Mock Objects
```python
# tests/mocks.py
class MockWebDriver:
    """Mock Selenium WebDriver for testing"""
    
    def find_elements(self, by, selector):
        # Return mock elements based on selector
        if 'pagination' in selector:
            return [MockElement() for _ in range(3)]
        return []
    
    def execute_script(self, script):
        # Return mock values for different scripts
        if "scrollHeight" in script:
            return 2000
        return 1
    
    @property
    def current_url(self):
        return "https://www.tokopedia.com/search?q=test"

class MockElement:
    """Mock Selenium WebElement"""
    
    def is_displayed(self):
        return True
    
    def is_enabled(self):
        return True
    
    @property
    def text(self):
        return "Mock Text"
```

### Running Tests

#### Command Line
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_analyzer.py

# Run with coverage
python -m pytest --cov=func tests/

# Run integration tests only
python -m pytest tests/test_integration.py -v
```

#### Test Configuration
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --verbose --tb=short
```

### Performance Testing

#### Load Testing
```python
# tests/test_performance.py
import time
import unittest
from func.analyzer import EnhancedConfigurableAnalyzer

class TestPerformance(unittest.TestCase):
    
    def test_feature_extraction_speed(self):
        """Test feature extraction performance"""
        analyzer = EnhancedConfigurableAnalyzer()
        mock_driver = MockWebDriver()
        
        start_time = time.time()
        features = analyzer._extract_enhanced_features(mock_driver, 'tokopedia')
        extraction_time = time.time() - start_time
        
        # Should complete within 2 seconds
        self.assertLess(extraction_time, 2.0)
        
    def test_ml_prediction_speed(self):
        """Test ML prediction performance"""
        analyzer = EnhancedConfigurableAnalyzer()
        
        # Mock features
        features = {f'feature_{i}': i for i in range(50)}
        
        start_time = time.time()
        prediction, confidence = analyzer._ml_prediction(features)
        prediction_time = time.time() - start_time
        
        # Should complete within 0.1 seconds
        self.assertLess(prediction_time, 0.1)

---

## 🚀 Deployment Guide

### Production Environment Setup

#### System Requirements
- **Python 3.8+**
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models and data
- **Chrome/Chromium browser**
- **ChromeDriver** (auto-managed with webdriver-manager)

#### Environment Setup
```bash
# 1. Create virtual environment
python -m venv enhanced_scraper_env
source enhanced_scraper_env/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Chrome for headless operation
sudo apt-get install google-chrome-stable  # Linux
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/raw_html data/processed logs models config

ENV PYTHONPATH=/app
ENV SCRAPER_ENV=production

CMD ["python", "scraping_orchestrator.py"]
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Problem: ModuleNotFoundError: No module named 'func'
# Solution:
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
```

#### 2. Browser Issues
```bash
# Problem: ChromeDriver version mismatch
# Solution:
pip install webdriver-manager
```

#### 3. Memory Issues
```python
# Solution: Clear browser cache periodically
if self.processed_count % 50 == 0:
    self.browser.driver.delete_all_cookies()
    self.browser.driver.execute_script("window.localStorage.clear();")
```

#### 4. Site Structure Changes
```yaml
# Solution: Update YAML templates with fallback selectors
training_templates:
  ecommerce_sites:
    tokopedia:
      selectors:
        pagination_buttons:
          fallback:
            - selector: "[class*='pagination']"
              priority: 3
              weight: 0.4
```

---

## 🤝 Contributing

### Development Workflow

1. **Fork and clone repository**
2. **Create development branch**
3. **Install development dependencies**
4. **Follow code standards** (PEP 8, type hints, docstrings)
5. **Write tests** (minimum 80% coverage)
6. **Submit pull request**

### Code Standards
```python
def new_feature(self, param1: str, param2: int = 10) -> bool:
    """
    Brief description of the feature
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 10)
        
    Returns:
        bool: True if successful, False otherwise
    """
    pass
```

---

## 🔮 Future Roadmap

### Short Term (3-6 months)
- **Deep Learning Integration**: CNN for visual pagination detection
- **Multi-language Support**: Extend text patterns
- **Additional Site Support**: Bukalapak, international sites
- **Performance Improvements**: Parallel processing, caching

### Medium Term (6-12 months)
- **Advanced Analytics**: Real-time dashboards, trend analysis
- **Cloud Integration**: AWS/GCP support, Kubernetes
- **Data Quality**: Advanced validation, deduplication

### Long Term (1+ years)
- **AI-Powered Features**: Computer vision, NLP integration
- **Enterprise Features**: Multi-tenant architecture
- **Research Applications**: Academic partnerships

---

## 📞 Support and Contact

### Getting Help
- **Documentation**: Check this comprehensive guide first
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions

### Community
- **Contributing**: See Contributing section above
- **License**: MIT License
- **Code of Conduct**: Be respectful and inclusive

---

**Last Updated**: December 2024  
**Version**: 2.0.0  
**Maintainers**: Enhanced Scraper Development Team
```