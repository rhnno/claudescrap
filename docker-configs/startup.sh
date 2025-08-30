#!/bin/bash
set -e

echo "🚀 Starting Enhanced ML-Powered Web Scraper..."

# Set environment variables
export DISPLAY=${DISPLAY:-:99}
export PYTHONPATH="/app:$PYTHONPATH"

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Initialize configuration if not exists
if [ ! -f "/app/config/analyzer_config.yaml" ]; then
    echo "📝 Initializing default configuration..."
    python -c "
from func.analyzer import ConfigurableAnalyzer
analyzer = ConfigurableAnalyzer()
print('✅ Configuration initialized')
"
fi

# Download NLTK data if needed (for text processing)
python -c "
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK data downloaded')
except:
    print('ℹ️ NLTK data download skipped')
"

echo "✅ Startup preparation complete"
exec "$@"