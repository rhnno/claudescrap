# Configuration Examples and Use Cases

## Table of Contents

1. [E-commerce Sites](#e-commerce-sites)
2. [News and Blog Sites](#news-and-blog-sites)
3. [Social Media Platforms](#social-media-platforms)
4. [Search Engines](#search-engines)
5. [Custom Scenarios](#custom-scenarios)
6. [Performance Optimization Examples](#performance-optimization-examples)

## E-commerce Sites

### Amazon Configuration

```yaml
# config/sites/amazon.yaml
sites:
  amazon:
    base_url: "https://www.amazon.com/s?k={query}"
    language: "english"
    encoding: "utf-8"
    
    feature_weights:
      pagination_buttons: 1.4
      numbered_buttons: 1.3
      url_params: 1.2
      text_patterns: 1.1
    
    patterns:
      page_text: 
        - "page"
        - "results"
        - "of over"
        - "\\d+-\\d+ of over \\d+"
      end_results: 
        - "no more results"
        - "end of results"
        - "no results found"
      pagination_indicators:
        - "page \\d+ of \\d+"
        - "\\d+-\\d+ of over \\d+"
        - "results \\d+-\\d+ of"
    
    custom_selectors:
      product_container: "[data-component-type='s-search-result']"
      pagination_container: ".a-pagination"
      next_button: ".a-pagination .a-last a"
      product_title: "[data-cy='title-recipe-title']"
      product_price: ".a-price-whole"
      results_count: "[data-component-type='s-result-info-bar']"
    
    performance:
      page_load_timeout: 25
      element_wait_timeout: 8
      scroll_pause_time: 1.5

features:
  selectors:
    pagination_buttons:
      primary:
        - selector: ".a-pagination .a-normal"
          priority: 10
          context: "amazon_pagination"
        - selector: ".a-pagination .a-selected"
          priority: 9
          context: "amazon_current_page"
      
      secondary:
        - selector: "span.pagnLink"
          priority: 7
          context: "amazon_legacy"
    
    next_buttons:
      primary:
        - xpath: "//a[contains(@aria-label, 'Go to next page')]"
          priority: 10
          context: "amazon_next_aria"
        - xpath: "//a[contains(text(), 'Next')]"
          priority: 9
          context: "amazon_next_text"
    
    product_elements:
      primary:
        - selector: "[data-component-type='s-search-result']"
          priority: 10
          context: "amazon_search_result"
        - selector: ".s-result-item"
          priority: 9
          context: "amazon_result_item"

  text_analysis:
    pagination_text_patterns:
      strong_indicators:
        - pattern: "\\d+-\\d+ of over [\\d,]+ results"
          weight: 0.95
          feature_name: "amazon_results_range"
        - pattern: "page \\d+ of \\d+"
          weight: 0.9
          feature_name: "amazon_page_indicator"
```

### Tokopedia Configuration

```yaml
# config/sites/tokopedia.yaml
sites:
  tokopedia:
    base_url: "https://www.tokopedia.com/search?q={query}"
    language: "indonesian"
    encoding: "utf-8"
    
    feature_weights:
      lazy_load_elements: 1.8
      scroll_velocity: 1.6
      xhr_request_count: 1.4
      pagination_buttons: 0.8  # Less important due to infinite scroll
    
    patterns:
      page_text: 
        - "halaman"
        - "produk"
        - "hasil pencarian"
        - "menampilkan \\d+ dari \\d+"
      end_results: 
        - "tidak ada hasil lagi"
        - "hasil pencarian habis"
        - "produk tidak ditemukan"
        - "semua produk telah ditampilkan"
      infinite_scroll_indicators:
        - "memuat lebih banyak"
        - "loading"
        - "sedang memuat"
    
    custom_selectors:
      product_container: "[data-testid='lstCL2ProductList'] > div"
      infinite_scroll_trigger: "[data-testid='btnShopProductLoadMore']"
      product_card: "[data-testid='divProductWrapper']"
      loading_skeleton: ".css-1q90pod"  # Tokopedia's skeleton loader
    
    performance:
      page_load_timeout: 35
      element_wait_timeout: 12
      scroll_pause_time: 3

features:
  selectors:
    lazy_load_elements:
      images:
        - selector: "img[data-src]:not([src])"
          priority: 10
          context: "tokopedia_lazy_images"
        - selector: "img[loading='lazy']"
          priority: 9
          context: "tokopedia_native_lazy"
      
      content:
        - selector: ".css-1q90pod"  # Skeleton loader
          priority: 10
          context: "tokopedia_skeleton"
        - selector: "[data-testid*='skeleton']"
          priority: 9
          context: "tokopedia_skeleton_testid"
    
    load_more_buttons:
      primary:
        - xpath: "//button[contains(text(), 'Muat Lebih Banyak')]"
          priority: 10
          context: "tokopedia_load_more"
        - selector: "[data-testid='btnShopProductLoadMore']"
          priority: 9
          context: "tokopedia_load_more_testid"

  dynamic_features:
    scroll_behavior:
      track_velocity: true
      track_acceleration: true
      velocity_window: 3
    
    network_activity:
      monitor_xhr_requests: true
      monitor_fetch_requests: true
      request_timeout_window: 3000
```

### Shopee Configuration

```yaml
# config/sites/shopee.yaml
sites:
  shopee:
    base_url: "https://shopee.co.id/search?keyword={query}"
    language: "indonesian"
    encoding: "utf-8"
    
    feature_weights:
      lazy_load_elements: 2.0
      scroll_velocity: 1.8
      xhr_request_count: 1.5
      new_dom_nodes: 1.4
    
    patterns:
      infinite_scroll_indicators:
        - "memuat lebih banyak"
        - "loading"
        - "sedang memuat produk"
      end_results:
        - "tidak ada produk lagi"
        - "akhir hasil pencarian"
        - "semua produk sudah ditampilkan"
    
    custom_selectors:
      product_container: ".shopee-search-item-result__item"
      infinite_scroll_trigger: ".shopee-button-outline"
      loading_spinner: ".shopee-page-loading"
    
    performance:
      page_load_timeout: 30
      element_wait_timeout: 10
      scroll_pause_time: 2.5

features:
  selectors:
    product_elements:
      primary:
        - selector: ".shopee-search-item-result__item"
          priority: 10
          context: "shopee_search_item"
        - selector: "[data-sqe='item']"
          priority: 9
          context: "shopee_data_item"
    
    lazy_load_elements:
      images:
        - selector: "img[data-src]"
          priority: 10
          context: "shopee_lazy_images"
      
      content:
        - selector: ".shopee-skeleton"
          priority: 10
          context: "shopee_skeleton"
        - selector: ".shopee-page-loading"
          priority: 9
          context: "shopee_page_loading"

  dynamic_features:
    scroll_behavior:
      track_velocity: true
      detect_smooth_scrolling: true
    
    dom_monitoring:
      track_node_additions: true
      significant_change_threshold: 5
```

## News and Blog Sites

### WordPress Blog Configuration

```yaml
# config/sites/wordpress_blog.yaml
sites:
  wordpress_blog:
    base_url: "https://example.com/page/{page}"
    language: "english"
    encoding: "utf-8"
    
    feature_weights:
      pagination_buttons: 1.5
      numbered_buttons: 1.4
      next_button: 1.3
    
    patterns:
      page_text:
        - "page \\d+ of \\d+"
        - "posts \\d+-\\d+ of \\d+"
      pagination_indicators:
        - "page \\d+"
        - "older posts"
        - "newer posts"
      end_results:
        - "no more posts"
        - "end of posts"

features:
  selectors:
    pagination_buttons:
      primary:
        - selector: ".wp-pagenavi .pages"
          priority: 10
          context: "wp_pagenavi"
        - selector: ".pagination .page-numbers"
          priority: 9
          context: "wp_pagination"
      
      secondary:
        - selector: ".nav-links .page-numbers"
          priority: 7
          context: "wp_nav_links"
    
    next_buttons:
      primary:
        - xpath: "//a[contains(@class, 'next')]"
          priority: 10
          context: "wp_next_class"
        - xpath: "//a[contains(text(), 'Next')]"
          priority: 9
          context: "wp_next_text"
        - xpath: "//a[contains(text(), 'Older')]"
          priority: 8
          context: "wp_older_posts"
```

### News Site Configuration

```yaml
# config/sites/news_site.yaml
sites:
  news_site:
    base_url: "https://news-site.com/search?q={query}&page={page}"
    language: "english"
    encoding: "utf-8"
    
    feature_weights:
      pagination_buttons: 1.3
      url_params: 1.4
      text_patterns: 1.2
    
    patterns:
      page_text:
        - "page \\d+ of \\d+"
        - "articles \\d+-\\d+"
        - "showing \\d+ articles"
      pagination_indicators:
        - "more articles"
        - "continue reading"
      end_results:
        - "no more articles"
        - "end of search results"

features:
  selectors:
    pagination_buttons:
      primary:
        - selector: ".pagination-container .page-link"
          priority: 10
          context: "news_pagination"
        - selector: ".article-pagination button"
          priority: 9
          context: "news_article_pagination"
    
    product_elements:  # Articles in this case
      primary:
        - selector: ".article-item, .news-item"
          priority: 10
          context: "news_articles"
        - selector: "[data-article-id]"
          priority: 9
          context: "news_data_articles"
```

## Social Media Platforms

### Twitter-like Infinite Scroll

```yaml
# config/sites/social_media.yaml
sites:
  social_platform:
    base_url: "https://social-platform.com/search?q={query}"
    language: "english"
    encoding: "utf-8"
    
    feature_weights:
      lazy_load_elements: 2.0
      scroll_velocity: 1.8
      xhr_request_count: 1.7
      new_dom_nodes: 1.6
    
    patterns:
      infinite_scroll_indicators:
        - "loading more posts"
        - "loading"
        - "fetching more content"
      end_results:
        - "no more posts"
        - "you've reached the end"
        - "that's all for now"

features:
  selectors:
    product_elements:  # Posts in this case
      primary:
        - selector: "[data-testid='tweet']"
          priority: 10
          context: "social_posts"
        - selector: ".post-item, .feed-item"
          priority: 9
          context: "social_feed_items"
    
    lazy_load_elements:
      content:
        - selector: ".loading-spinner"
          priority: 10
          context: "social_loading"
        - selector: "[data-testid='loading']"
          priority: 9
          context: "social_loading_testid"

  dynamic_features:
    scroll_behavior:
      track_velocity: true
      track_acceleration: true
      velocity_window: 5
    
    network_activity:
      monitor_xhr_requests: true
      monitor_fetch_requests: true
      request_timeout_window: 2000
    
    dom_monitoring:
      track_node_additions: true
      significant_change_threshold: 3
```

## Search Engines

### Google Search Results

```yaml
# config/sites/google_search.yaml
sites:
  google:
    base_url: "https://www.google.com/search?q={query}&start={start}"
    language: "english"
    encoding: "utf-8"
    
    feature_weights:
      pagination_buttons: 1.5
      next_button: 1.4
      url_params: 1.3
    
    patterns:
      page_text:
        - "page \\d+ of about"
        - "\\d+ of about [\\d,]+ results"
      pagination_indicators:
        - "Goooooogle"  # Google's pagination style
        - "page \\d+"
      end_results:
        - "no more results"
        - "end of search results"

features:
  selectors:
    pagination_buttons:
      primary:
        - selector: "#pnnext, #pnprev"
          priority: 10
          context: "google_nav_buttons"
        - selector: "a[aria-label*='Page']"
          priority: 9
          context: "google_page_links"
    
    next_buttons:
      primary:
        - selector: "#pnnext"
          priority: 10
          context: "google_next"
        - xpath: "//a[contains(@aria-label, 'Next page')]"
          priority: 9
          context: "google_next_aria"
    
    product_elements:  # Search results
      primary:
        - selector: ".g .yuRUbf"
          priority: 10
          context: "google_search_results"
        - selector: "[data-ved] h3"
          priority: 9
          context: "google_result_titles"
```

## Custom Scenarios

### Multi-language Site

```yaml
# config/sites/multilingual.yaml
sites:
  multilingual_site:
    base_url: "https://example.com/{lang}/search?q={query}"
    language: "auto_detect"  # Special value for auto-detection
    encoding: "utf-8"
    
    # Language-specific patterns
    language_patterns:
      english:
        page_text: ["page", "results", "of"]
        end_results: ["no more results", "end of results"]
        pagination_indicators: ["page \\d+ of \\d+"]
      
      spanish:
        page_text: ["página", "resultados", "de"]
        end_results: ["no más resultados", "fin de resultados"]
        pagination_indicators: ["página \\d+ de \\d+"]
      
      french:
        page_text: ["page", "résultats", "de"]
        end_results: ["plus de résultats", "fin des résultats"]
        pagination_indicators: ["page \\d+ de \\d+"]

features:
  text_analysis:
    # Multi-language end of results patterns
    end_of_results_patterns:
      high_confidence:
        - pattern: "no\\s+more\\s+results|no\\s+más\\s+resultados|plus\\s+de\\s+résultats"
          weight: 0.9
          languages: ["english", "spanish", "french"]
      
      medium_confidence:
        - pattern: "end\\s+of\\s+results|fin\\s+de\\s+resultados|fin\\s+des\\s+résultats"
          weight: 0.7
          languages: ["english", "spanish", "french"]
```

### AJAX-Heavy Application

```yaml
# config/sites/ajax_app.yaml
sites:
  ajax_application:
    base_url: "https://app.example.com/data?query={query}"
    language: "english"
    encoding: "utf-8"
    
    feature_weights:
      xhr_request_count: 2.0
      new_dom_nodes: 1.8
      lazy_load_elements: 1.6
      scroll_velocity: 1.4
    
    patterns:
      infinite_scroll_indicators:
        - "loading"
        - "fetching data"
        - "please wait"
      end_results:
        - "no more data"
        - "end of dataset"

features:
  dynamic_features:
    network_activity:
      monitor_xhr_requests: true
      monitor_fetch_requests: true
      track_image_loads: false  # Not relevant for data app
      request_timeout_window: 1000  # Shorter window for fast app
    
    dom_monitoring:
      track_node_additions: true
      track_node_removals: true
      significant_change_threshold: 1  # Very sensitive
    
    scroll_behavior:
      track_velocity: true
      track_acceleration: true
      velocity_window: 3

  selectors:
    product_elements:  # Data rows
      primary:
        - selector: ".data-row, .table-row"
          priority: 10
          context: "ajax_data_rows"
        - selector: "[data-id]"
          priority: 9
          context: "ajax_data_items"
    
    lazy_load_elements:
      content:
        - selector: ".loading-indicator"
          priority: 10
          context: "ajax_loading"
        - selector: ".spinner, .loader"
          priority: 9
          context: "ajax_spinner"
```

## Performance Optimization Examples

### High-Performance Configuration

```yaml
# config/performance/high_performance.yaml
# Optimized for speed over feature richness

features:
  # Minimal text analysis
  text_analysis:
    max_tfidf_features: 50  # Reduced from default 200
    ngram_range: [1, 1]     # Only unigrams, no bigrams
    
    # Only high-confidence patterns
    end_of_results_patterns:
      high_confidence:
        - pattern: "no\\s+more\\s+results"
          weight: 0.9
          languages: ["english"]
  
  # Simplified selectors
  selectors:
    pagination_buttons:
      primary:  # Only primary selectors
        - selector: ".pagination button"
          priority: 10
          context: "simple_pagination"
    
    next_buttons:
      primary:
        - xpath: "//a[contains(text(), 'Next')]"
          priority: 10
          context: "simple_next"
  
  # Minimal dynamic features
  dynamic_features:
    scroll_behavior:
      track_velocity: true
      track_acceleration: false  # Disabled for performance
    
    dom_monitoring:
      track_node_additions: true
      track_node_removals: false  # Disabled
      monitor_attribute_changes: false  # Disabled
    
    network_activity:
      monitor_xhr_requests: true
      monitor_fetch_requests: false  # Disabled
      track_image_loads: false  # Disabled

# Aggressive timeouts
sites:
  default:
    performance:
      page_load_timeout: 10
      element_wait_timeout: 3
      scroll_pause_time: 0.5

debug:
  verbose: false  # Minimal logging
  performance_monitoring:
    enable: true
    track_extraction_time: true
```

### Comprehensive Analysis Configuration

```yaml
# config/performance/comprehensive.yaml
# Maximum feature extraction for research/analysis

features:
  # Extensive text analysis
  text_analysis:
    max_tfidf_features: 500
    ngram_range: [1, 3]  # Unigrams, bigrams, trigrams
    min_df: 1
    max_df: 0.99
    
    # All confidence levels
    end_of_results_patterns:
      high_confidence:
        - pattern: "no\\s+more\\s+results"
          weight: 0.9
          languages: ["english"]
      medium_confidence:
        - pattern: "no\\s+more\\s+items"
          weight: 0.7
          languages: ["english"]
      low_confidence:
        - pattern: "end\\s+of\\s+page"
          weight: 0.4
          languages: ["english"]
  
  # Comprehensive selectors
  selectors:
    pagination_buttons:
      primary:
        - selector: "nav[aria-label*='pagination'] button"
          priority: 10
          context: "semantic_navigation"
      secondary:
        - selector: ".pagination button"
          priority: 8
          context: "class_based"
      tertiary:
        - selector: "button[aria-label*='page']"
          priority: 6
          context: "aria_based"
      fallback:
        - selector: "button"
          priority: 2
          context: "generic_button"
  
  # All dynamic features enabled
  dynamic_features:
    scroll_behavior:
      track_velocity: true
      track_acceleration: true
      velocity_window: 10
      detect_smooth_scrolling: true
    
    dom_monitoring:
      track_node_additions: true
      track_node_removals: true
      monitor_attribute_changes: true
      significant_change_threshold: 1
    
    network_activity:
      monitor_xhr_requests: true
      monitor_fetch_requests: true
      track_image_loads: true
      request_timeout_window: 10000
    
    user_interaction:
      detect_hover_effects: true
      track_focus_changes: true
      monitor_scroll_events: true

debug:
  verbose: true
  performance_monitoring:
    enable: true
    track_extraction_time: true
    track_prediction_time: true
    track_memory_usage: true
  
  feature_analysis:
    enable: true
    save_feature_importance: true
    create_feature_correlation_matrix: true
    generate_feature_distribution_plots: true
```

### Mobile-Optimized Configuration

```yaml
# config/performance/mobile.yaml
# Optimized for mobile devices and touch interfaces

features:
  selectors:
    pagination_buttons:
      primary:
        - selector: "button[role='button'][aria-label*='page']"
          priority: 10
          context: "mobile_accessible"
        - selector: ".pagination-mobile button"
          priority: 9
          context: "mobile_pagination"
    
    next_buttons:
      primary:
        - xpath: "//button[contains(@aria-label, 'next') or contains(@aria-label, 'Next')]"
          priority: 10
          context: "mobile_next_aria"
        - selector: "button[data-direction='next']"
          priority: 9
          context: "mobile_next_data"
    
    # Mobile-specific lazy loading
    lazy_load_elements:
      images:
        - selector: "img[data-src][loading='lazy']"
          priority: 10
          context: "mobile_lazy_images"
      content:
        - selector: ".mobile-skeleton, .touch-loader"
          priority: 10
          context: "mobile_loading"
  
  # Touch-optimized dynamic features
  dynamic_features:
    scroll_behavior:
      track_velocity: true
      detect_smooth_scrolling: true  # Important for mobile
      velocity_window: 3  # Shorter window for touch scrolling
    
    user_interaction:
      monitor_scroll_events: true
      detect_hover_effects: false  # Not relevant for touch

# Mobile-specific thresholds
thresholds:
  scroll_bottom_threshold: 0.9  # Slightly higher for mobile
  confidence_threshold: 0.6     # Slightly lower due to varied mobile layouts

sites:
  mobile_site:
    performance:
      page_load_timeout: 20  # Longer for mobile networks
      element_wait_timeout: 8
      scroll_pause_time: 2   # Longer for mobile rendering
```

These examples demonstrate how to configure the enhanced analyzer for various scenarios. Each configuration is tailored to the specific characteristics and challenges of different types of websites and use cases.