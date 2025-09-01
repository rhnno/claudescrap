// MongoDB initialization script for Enhanced Analyzer
// This script sets up the database and collections for the ML-powered analyzer

// Switch to analyzer database
db = db.getSiblingDB('analyzer_db');

// Create collections for different data types
db.createCollection('training_data');
db.createCollection('scraping_results');
db.createCollection('model_performance');
db.createCollection('feature_analysis');
db.createCollection('site_configurations');

// Create indexes for better performance
db.training_data.createIndex({ "timestamp": 1 });
db.training_data.createIndex({ "site_name": 1 });
db.training_data.createIndex({ "actual_type": 1 });
db.training_data.createIndex({ "url": 1 });

db.scraping_results.createIndex({ "timestamp": 1 });
db.scraping_results.createIndex({ "site_name": 1 });
db.scraping_results.createIndex({ "query": 1 });

db.model_performance.createIndex({ "timestamp": 1 });
db.model_performance.createIndex({ "model_version": 1 });

db.feature_analysis.createIndex({ "timestamp": 1 });
db.feature_analysis.createIndex({ "feature_name": 1 });

db.site_configurations.createIndex({ "site_name": 1 }, { unique: true });

// Insert default site configurations
db.site_configurations.insertMany([
    {
        "site_name": "tokopedia",
        "base_url": "https://www.tokopedia.com/search?q={query}",
        "language": "indonesian",
        "created_at": new Date(),
        "active": true,
        "feature_weights": {
            "lazy_load_elements": 1.8,
            "scroll_velocity": 1.6,
            "xhr_request_count": 1.4,
            "pagination_buttons": 0.8
        }
    },
    {
        "site_name": "amazon",
        "base_url": "https://www.amazon.com/s?k={query}",
        "language": "english",
        "created_at": new Date(),
        "active": true,
        "feature_weights": {
            "pagination_buttons": 1.4,
            "numbered_buttons": 1.3,
            "url_params": 1.2,
            "text_patterns": 1.1
        }
    },
    {
        "site_name": "shopee",
        "base_url": "https://shopee.co.id/search?keyword={query}",
        "language": "indonesian",
        "created_at": new Date(),
        "active": true,
        "feature_weights": {
            "lazy_load_elements": 2.0,
            "scroll_velocity": 1.8,
            "xhr_request_count": 1.5,
            "new_dom_nodes": 1.4
        }
    }
]);

// Create user for the application
db.createUser({
    user: "analyzer_app",
    pwd: "analyzer_app_password_2024",
    roles: [
        {
            role: "readWrite",
            db: "analyzer_db"
        }
    ]
});

print("✅ MongoDB initialization completed successfully");
print("📊 Created collections: training_data, scraping_results, model_performance, feature_analysis, site_configurations");
print("🔍 Created indexes for optimal query performance");
print("👤 Created application user: analyzer_app");
print("🌐 Inserted default site configurations for Tokopedia, Amazon, and Shopee");