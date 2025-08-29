#!/bin/bash

set -e  # Exit immediately if any command fails

APP_NAME="tokopedia-scraper"
IMAGE_NAME="tokopedia-scraper:latest"

echo "🔨 Building Docker image for $APP_NAME..."

# Stop any existing containers
docker-compose down

# Build with Docker Compose (uses Dockerfile automatically)
docker-compose build --no-cache

echo "✅ Build complete!"

echo "🚀 Starting $APP_NAME service..."
docker-compose up -d

echo "🔎 Checking container status..."
docker-compose ps

echo "📜 Showing recent logs..."
docker-compose logs -f --tail=50
