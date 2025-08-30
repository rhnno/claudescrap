# Docker Setup Guide - Enhanced ML-Powered Web Scraper

## Overview

This guide covers the complete Docker setup for the Enhanced ML-Powered Web Scraper with configurable pagination detection. The Docker setup includes the main analyzer, supporting services (Redis, MongoDB), and optional development/monitoring tools.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration](#configuration)
4. [Service Profiles](#service-profiles)
5. [Development Setup](#development-setup)
6. [Production Deployment](#production-deployment)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM available
- 10GB free disk space

### Basic Setup

```bash
# Clone the repository
git clone <repository-url>
cd enhanced-ml-scraper

# Copy environment configuration
cp .env.example .env

# Edit configuration (optional)
nano .env

# Build and start the services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f enhanced-analyzer
```

### Verify Installation

```bash
# Check health status
docker-compose exec enhanced-analyzer /usr/local/bin/healthcheck.sh

# Test the analyzer
docker-compose exec enhanced-analyzer python -c "
from func.analyzer import ConfigurableAnalyzer
analyzer = ConfigurableAnalyzer()
print('✅ Analyzer initialized successfully')
"
```

## Architecture Overview

### Services

| Service | Purpose | Port | Required |
|---------|---------|------|----------|
| `enhanced-analyzer` | Main ML-powered scraper | 4000, 8080 | Yes |
| `redis` | Caching and session management | 6379 | Yes |
| `mongodb` | Training data and results storage | 27017 | Yes |
| `jupyter` | Development and analysis | 8888 | Optional |
| `grafana` | Monitoring and dashboards | 3000 | Optional |

### Data Flow

```
Web Pages → Enhanced Analyzer → Feature Extraction → ML Model → Pagination Detection
     ↓              ↓                    ↓              ↓
  Raw HTML    Training Data        Model Cache    Results Storage
     ↓              ↓                    ↓              ↓
 File Storage    MongoDB            Redis Cache     MongoDB
```

### Volume Mounts

```
./data          → /app/data          # Scraped data and HTML files
./logs          → /app/logs          # Application logs
./models        → /app/models        # ML models and training data
./config        → /app/config        # Configuration files
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

#### Key Configuration Sections

**Core Settings:**
```env
ANALYZER_CONFIG_PATH=/app/config/enhanced_analyzer_config.yaml
MAX_PAGES=50
SCRAPE_INTERVAL=7200
AUTO_TRAIN_MODEL=true
```

**Performance Tuning:**
```env
CONTAINER_CPUS=4.0
CONTAINER_MEMORY=6g
ML_WORKERS=2
FEATURE_CACHE_SIZE=1000
```

**Site Configuration:**
```env
DEFAULT_SITE=tokopedia
ENABLE_MULTI_SITE=true
TOKOPEDIA_ENABLED=true
AMAZON_ENABLED=true
SHOPEE_ENABLED=true
```

### Custom Configuration Files

You can override default configurations by mounting custom files:

```yaml
# In docker-compose.yml
volumes:
  - ./my-custom-config.yaml:/app/config/analyzer_config.yaml
  - ./my-templates:/app/config/feature_templates
```

## Service Profiles

### Default Profile (Production)

Includes essential services only:

```bash
docker-compose up -d
```

Services: `enhanced-analyzer`, `redis`, `mongodb`

### Development Profile

Includes Jupyter notebook for development:

```bash
docker-compose --profile development up -d
```

Additional services: `jupyter`

Access Jupyter at: http://localhost:8888

### Monitoring Profile

Includes Grafana for monitoring:

```bash
docker-compose --profile monitoring up -d
```

Additional services: `grafana`

Access Grafana at: http://localhost:3000 (admin/admin_password_2024)

### Full Profile

All services enabled:

```bash
docker-compose --profile development --profile monitoring up -d
```

## Development Setup

### Setting Up Development Environment

```bash
# Start with development profile
docker-compose --profile development up -d

# Access Jupyter for interactive development
open http://localhost:8888

# Mount local code for live editing
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Development Docker Compose Override

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  enhanced-analyzer:
    volumes:
      # Mount source code for live editing
      - ./func:/app/func
      - ./config:/app/config
    
    environment:
      # Enable debug mode
      - ANALYZER_DEBUG_LEVEL=DEBUG
      - ENABLE_DEBUG_MODE=true
      - VERBOSE_LOGGING=true
    
    # Override command for development
    command: >
      bash -c "
        echo '🔧 Starting in development mode...' &&
        python -u /app/main.py
      "
```

### Testing in Docker

```bash
# Run tests inside container
docker-compose exec enhanced-analyzer python -m pytest func/test.py -v

# Run specific analyzer tests
docker-compose exec enhanced-analyzer python -c "
from func.analyzer import ConfigurableAnalyzer
analyzer = ConfigurableAnalyzer()
print('Testing feature extraction...')
# Add your test code here
"

# Check configuration loading
docker-compose exec enhanced-analyzer python -c "
import yaml
with open('/app/config/enhanced_analyzer_config.yaml') as f:
    config = yaml.safe_load(f)
print(f'✅ Config loaded with {len(config)} sections')
"
```

## Production Deployment

### Production Configuration

1. **Update Environment Variables:**

```env
# Production settings
ANALYZER_DEBUG_LEVEL=WARNING
ENABLE_DEBUG_MODE=false
VERBOSE_LOGGING=false

# Resource allocation
CONTAINER_CPUS=8.0
CONTAINER_MEMORY=12g

# Security
MONGO_INITDB_ROOT_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
```

2. **Use Production Docker Compose:**

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  enhanced-analyzer:
    restart: always
    
    # Production resource limits
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 12G
        reservations:
          cpus: '4.0'
          memory: 6G
    
    # Production logging
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
  
  mongodb:
    # Use persistent volume for production
    volumes:
      - mongodb_prod_data:/data/db
    
    # Production MongoDB configuration
    command: mongod --auth --bind_ip_all
  
  redis:
    # Production Redis configuration
    command: redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes

volumes:
  mongodb_prod_data:
    driver: local
```

3. **Deploy:**

```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### SSL/TLS Configuration

For production with SSL:

```yaml
# Add to docker-compose.prod.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - enhanced-analyzer
```

### Backup Strategy

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Backup MongoDB
docker-compose exec mongodb mongodump --out /data/backup_$DATE

# Backup models and data
docker run --rm -v $(pwd)/models:/source -v $(pwd)/backups:/backup alpine \
  tar czf /backup/models_$DATE.tar.gz -C /source .

# Backup configuration
tar czf backups/config_$DATE.tar.gz config/

echo "✅ Backup completed: $DATE"
EOF

chmod +x backup.sh
```

## Monitoring and Maintenance

### Health Checks

```bash
# Check all services
docker-compose ps

# Detailed health check
docker-compose exec enhanced-analyzer /usr/local/bin/healthcheck.sh

# Check resource usage
docker stats
```

### Log Management

```bash
# View real-time logs
docker-compose logs -f enhanced-analyzer

# View specific service logs
docker-compose logs redis
docker-compose logs mongodb

# Export logs
docker-compose logs --no-color enhanced-analyzer > analyzer.log
```

### Performance Monitoring

```bash
# Monitor resource usage
docker-compose exec enhanced-analyzer top

# Check memory usage
docker-compose exec enhanced-analyzer free -h

# Monitor disk usage
docker-compose exec enhanced-analyzer df -h
```

### Database Management

```bash
# MongoDB operations
docker-compose exec mongodb mongosh analyzer_db

# Redis operations
docker-compose exec redis redis-cli

# Backup databases
docker-compose exec mongodb mongodump --db analyzer_db --out /data/backup
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker-compose logs enhanced-analyzer

# Check resource usage
docker system df
docker system prune  # Clean up if needed

# Rebuild container
docker-compose build --no-cache enhanced-analyzer
```

#### 2. Chrome/Selenium Issues

```bash
# Check Chrome installation
docker-compose exec enhanced-analyzer google-chrome --version

# Test Xvfb
docker-compose exec enhanced-analyzer ps aux | grep Xvfb

# Check display
docker-compose exec enhanced-analyzer echo $DISPLAY
```

#### 3. Memory Issues

```bash
# Increase memory limits in docker-compose.yml
mem_limit: 8g
shm_size: 4g

# Check memory usage
docker-compose exec enhanced-analyzer free -h
```

#### 4. Permission Issues

```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./data ./logs ./models

# Check container user
docker-compose exec enhanced-analyzer whoami
```

#### 5. Network Issues

```bash
# Check network connectivity
docker-compose exec enhanced-analyzer ping google.com

# Check internal network
docker network ls
docker network inspect enhanced-ml-scraper_analyzer-network
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Set debug environment
export ANALYZER_DEBUG_LEVEL=DEBUG
export ENABLE_DEBUG_MODE=true

# Restart with debug
docker-compose up -d enhanced-analyzer

# View debug logs
docker-compose logs -f enhanced-analyzer
```

### Performance Tuning

#### 1. Optimize Resource Allocation

```yaml
# In docker-compose.yml
services:
  enhanced-analyzer:
    cpus: "6.0"          # Increase CPU allocation
    mem_limit: 8g        # Increase memory
    shm_size: 4g         # Increase shared memory
```

#### 2. Optimize Chrome Settings

```env
# In .env
CHROME_OPTIONS=--no-sandbox,--disable-dev-shm-usage,--disable-gpu,--memory-pressure-off,--max_old_space_size=4096
```

#### 3. Database Optimization

```bash
# MongoDB optimization
docker-compose exec mongodb mongosh --eval "
db.adminCommand({setParameter: 1, wiredTigerCacheSizeGB: 2})
"

# Redis optimization
docker-compose exec redis redis-cli CONFIG SET maxmemory 1gb
```

### Maintenance Tasks

#### Regular Maintenance

```bash
# Weekly maintenance script
cat > maintenance.sh << 'EOF'
#!/bin/bash
echo "🔧 Starting weekly maintenance..."

# Clean up old logs
find ./logs -name "*.log" -mtime +7 -delete

# Clean up old training data
docker-compose exec mongodb mongosh analyzer_db --eval "
db.training_data.deleteMany({
  timestamp: {$lt: new Date(Date.now() - 30*24*60*60*1000)}
})
"

# Optimize databases
docker-compose exec mongodb mongosh analyzer_db --eval "db.runCommand({compact: 'training_data'})"
docker-compose exec redis redis-cli BGREWRITEAOF

# Update models if needed
docker-compose exec enhanced-analyzer python -c "
from func.analyzer import ConfigurableAnalyzer
analyzer = ConfigurableAnalyzer()
analyzer.train_model()
"

echo "✅ Maintenance completed"
EOF

chmod +x maintenance.sh
```

#### Updates and Upgrades

```bash
# Update containers
docker-compose pull
docker-compose up -d

# Rebuild with latest code
git pull
docker-compose build --no-cache
docker-compose up -d
```

## Security Considerations

### 1. Change Default Passwords

```env
# In .env
MONGO_INITDB_ROOT_PASSWORD=your_secure_password_here
GF_SECURITY_ADMIN_PASSWORD=your_grafana_password_here
```

### 2. Network Security

```yaml
# Restrict external access
services:
  mongodb:
    ports: []  # Remove external port mapping
  redis:
    ports: []  # Remove external port mapping
```

### 3. Volume Security

```bash
# Set proper permissions
chmod 700 ./data ./logs ./models
```

This comprehensive Docker setup provides a robust, scalable environment for the Enhanced ML-Powered Web Scraper with proper monitoring, development tools, and production-ready configurations.