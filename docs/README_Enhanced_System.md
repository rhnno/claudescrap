# Enhanced Tokopedia Scraper with ML-Powered Pagination Detection

## 🎯 Overview

This enhanced system integrates your existing Tokoscrape.py selectors with advanced ML-powered pagination detection, providing intelligent scraping capabilities with separate orchestrators for training and production use.

## 📁 File Structure

```
├── Tokscrape.py                    # Your original scraping script
├── enhanced_tokoscrape.py          # Enhanced version with ML integration
├── train_model_orchestrator.py     # Training orchestrator for ML models
├── scraping_orchestrator.py        # Production scraping orchestrator
├── scraping_config.json           # Configuration for batch scraping
├── verify_enhanced_analyzer.py     # Verification script
└── func/
    ├── analyzer.py                 # Enhanced analyzer with YAML templates
    ├── browser.py                  # Browser management
    ├── storage.py                  # Data storage utilities
    └── utils.py                    # Utility functions
```

## 🚀 Quick Start

### 1. **Verify Setup**
```bash
python verify_enhanced_analyzer.py
```
This ensures all components are working correctly.

### 2. **Train ML Model (First Time)**
```bash
python train_model_orchestrator.py
```
- Collects training data from multiple sites
- Trains ML model for pagination detection
- Saves trained model for production use

### 3. **Run Enhanced Scraping**

#### Option A: Single Query (Interactive)
```bash
python enhanced_tokoscrape.py
```

#### Option B: Batch Scraping (Production)
```bash
python scraping_orchestrator.py
```

## 📊 System Components

### 🧠 **Enhanced Analyzer** (`func/analyzer.py`)

**Features:**
- **50+ features per page** vs. previous 20
- **Site-specific intelligence** for Tokopedia, Shopee, Amazon
- **Weighted confidence scoring** for better accuracy
- **YAML configuration** - no code changes needed for new sites
- **Dynamic content monitoring** (scroll velocity, DOM changes)

**Your Original Selectors Integrated:**
```python
# Product containers
product_items = driver.find_elements(By.XPATH, './/*[contains(@class, "css-5wh65g")]')

# Product name
product_name = item.find_element(By.XPATH, './/*[contains(@class,"tnoqZhn89")]').text

# Product price
product_price = item.find_element(By.XPATH, './/*[contains(@class, "urMOIDHH7")]').text

# Shop name
product_shop = item.find_element(By.XPATH, './/*[contains(@class, "si3CNdiG8AR0EaXvf6bFbQ")]').text
```

### 🎓 **Training Orchestrator** (`train_model_orchestrator.py`)

**Purpose:** Collect training data and train ML models

**Features:**
- **Multi-site data collection** (Tokopedia, Shopee)
- **Automated ground truth labeling**
- **Quality control and validation**
- **Model training with cross-validation**
- **Performance metrics and reporting**

**Usage:**
```bash
python train_model_orchestrator.py
```

**Configuration Options:**
- Max training samples (default: 100)
- Max pages per query (default: 5)
- Headless mode (default: yes)

**Output:**
- Trained ML model (`models/enhanced_pagination_model.pkl`)
- Training data (`models/training_data_YYYYMMDD_HHMMSS.json`)
- Training logs (`logs/training_log_YYYYMMDD_HHMMSS.csv`)

### 🕷️ **Scraping Orchestrator** (`scraping_orchestrator.py`)

**Purpose:** Production-ready intelligent scraping

**Features:**
- **Batch processing** multiple queries and sites
- **ML-powered pagination detection**
- **Automatic retry and error handling**
- **Multiple output formats** (CSV, Excel, JSON)
- **Session statistics and monitoring**

**Configuration:** Edit `scraping_config.json`
```json
{
  "sites": [
    {
      "name": "tokopedia",
      "queries": ["laptop gaming", "smartphone android"],
      "max_pages": 10,
      "scroll_depth": 3
    }
  ],
  "output_format": ["csv", "excel", "json"],
  "delay_range": [1, 3]
}
```

### 🔧 **Enhanced Tokoscrape** (`enhanced_tokoscrape.py`)

**Purpose:** Enhanced version of your original script

**Your Selectors Preserved:**
- ✅ Product containers: `css-5wh65g`
- ✅ Product names: `tnoqZhn89`
- ✅ Product prices: `urMOIDHH7`
- ✅ Shop names: `si3CNdiG8AR0EaXvf6bFbQ`
- ✅ Ratings: `55aCJ8bEsyw`
- ✅ Sold count: `terjual` text matching

**Enhanced Features:**
- **ML-powered pagination detection**
- **Intelligent scroll handling**
- **Enhanced error handling**
- **Better data quality**
- **Session persistence**

## 🎯 Workflow Comparison

### **Original Workflow (Tokscrape.py):**
```
1. Manual URL construction
2. Fixed scroll depth
3. Basic pagination detection
4. Simple data extraction
5. CSV/Excel output
```

### **Enhanced Workflow:**
```
1. 🧠 ML-powered page analysis
2. 📊 Intelligent navigation (pagination/infinite scroll)
3. 🎯 Site-specific feature extraction
4. 📈 Quality monitoring and statistics
5. 💾 Multiple output formats with metadata
```

## 📈 Benefits Over Original Script

| Feature | Original | Enhanced |
|---------|----------|----------|
| **Pagination Detection** | Basic rules | ML + 50+ features |
| **Site Support** | Tokopedia only | Multi-site with templates |
| **Navigation** | Fixed scroll | Intelligent adaptation |
| **Data Quality** | Basic extraction | Enhanced validation |
| **Error Handling** | Minimal | Comprehensive retry logic |
| **Monitoring** | None | Session statistics |
| **Scalability** | Single query | Batch processing |
| **Training** | None | Automated model training |

## 🔄 Migration Guide

### **From Your Original Script:**

1. **Keep using your selectors** - they're integrated into the enhanced system
2. **Your data format preserved** - same CSV/Excel structure
3. **Enhanced capabilities** - better pagination detection and error handling

### **Step-by-Step Migration:**

```bash
# 1. Verify setup
python verify_enhanced_analyzer.py

# 2. Train model (one-time)
python train_model_orchestrator.py

# 3. Test enhanced scraping
python enhanced_tokoscrape.py

# 4. Use for production
python scraping_orchestrator.py
```

## 📊 Output Examples

### **Enhanced CSV Output:**
```csv
Product Name,Price,Sold,discount,Before Discount Price,Shop Name,location,Rating,Link Product,scraped_at,site,query
"Laptop Gaming ASUS ROG","Rp15.000.000","100 sold","10%","Rp16.500.000","Tech Store","Jakarta","4.5","https://...","2024-01-15T10:30:00","tokopedia","laptop gaming"
```

### **Session Statistics:**
```
📊 SCRAPING SESSION SUMMARY
Duration: 0:15:30
Sites scraped: 1
Queries processed: 3
Total pages: 25
Total products: 500
Errors: 2

🧠 ML ANALYSIS BREAKDOWN:
Pagination detected: 20
Infinite scroll detected: 3
Last page detected: 2
```

## 🎛️ Configuration Options

### **Enhanced Analyzer Config** (`config/enhanced_training_templates.yaml`)
- Site-specific selectors with confidence weights
- Text patterns for different languages
- Training scenarios and expected outcomes
- Advanced feature extraction rules

### **Scraping Config** (`scraping_config.json`)
- Multiple sites and queries
- Output format preferences
- Delay and retry settings
- Performance tuning options

## 🧪 Testing and Validation

### **Verify System:**
```bash
python verify_enhanced_analyzer.py
```

### **Test Individual Components:**
```python
# Test enhanced analyzer
from func.analyzer import EnhancedConfigurableAnalyzer
analyzer = EnhancedConfigurableAnalyzer()
result = analyzer.analyze_page_structure(driver, 'tokopedia')

# Test enhanced scraper
from enhanced_tokoscrape import EnhancedTokopediaScraper
scraper = EnhancedTokopediaScraper()
products = scraper.scrape_tokopedia_products('laptop', max_pages=3)
```

## 🔧 Troubleshooting

### **Common Issues:**

1. **Import Errors:**
   ```bash
   python verify_enhanced_analyzer.py
   ```

2. **Browser Issues:**
   - Ensure Chrome is installed
   - Check ChromeDriver compatibility
   - Try headless mode

3. **Model Training Fails:**
   - Reduce sample size
   - Check internet connection
   - Verify site accessibility

4. **No Products Found:**
   - Check selectors in YAML templates
   - Verify site structure hasn't changed
   - Test with original script first

### **Performance Optimization:**

1. **Faster Training:**
   - Use headless mode
   - Reduce max_samples
   - Limit pages per query

2. **Better Scraping:**
   - Adjust delay_range in config
   - Use trained model
   - Enable session persistence

## 📝 Next Steps

1. **Customize Templates:** Edit `config/enhanced_training_templates.yaml` for new sites
2. **Expand Queries:** Add more search terms to `scraping_config.json`
3. **Monitor Performance:** Check training logs and session statistics
4. **Scale Up:** Use batch processing for large-scale data collection

## 🎉 Summary

Your enhanced system now provides:
- ✅ **Your original selectors preserved and enhanced**
- ✅ **ML-powered intelligent pagination detection**
- ✅ **Production-ready batch processing**
- ✅ **Comprehensive training and validation**
- ✅ **Better data quality and error handling**
- ✅ **Scalable multi-site architecture**

**Ready to start? Run `python verify_enhanced_analyzer.py` to begin!**