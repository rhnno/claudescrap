#!/bin/bash
# Enhanced healthcheck for ML-powered analyzer

# Check if Python processes are running
if ! pgrep -f "python.*main.py" > /dev/null; then
    echo "ERROR: Main analyzer process not running"
    exit 1
fi

# Check if Xvfb is running
if ! pgrep -f "Xvfb" > /dev/null; then
    echo "ERROR: Xvfb not running"
    exit 1
fi

# Check if Chrome can be started (basic test)
if ! timeout 10 google-chrome --version > /dev/null 2>&1; then
    echo "ERROR: Chrome not accessible"
    exit 1
fi

# Check if required directories exist and are writable
for dir in /app/logs /app/data /app/models; do
    if [ ! -d "$dir" ] || [ ! -w "$dir" ]; then
        echo "ERROR: Directory $dir not accessible"
        exit 1
    fi
done

# Check if configuration files exist
if [ ! -f "/app/config/enhanced_analyzer_config.yaml" ]; then
    echo "WARNING: Enhanced config not found, using defaults"
fi

echo "OK: All health checks passed"
exit 0