# Quick Setup Commands for Safe Deployment

## Pre-Deployment Checklist

### 1. VPS Provider Selection
```bash
# Recommended providers (avoid DigitalOcean after suspension)
# - Hetzner CX41: 4 vCPUs, 16GB RAM, €26.98/month
# - OVH VPS: 4 vCPUs, 8GB RAM, ~$30/month  
# - Vultr: 4 vCPUs, 8GB RAM, $24/month
# - Linode: 4 vCPUs, 8GB RAM, $24/month
```

### 2. Server Specifications
```bash
# Minimum: 2 vCPUs, 4GB RAM, 40GB SSD
# Recommended: 4 vCPUs, 8GB RAM, 80GB SSD
# OS: Ubuntu 22.04 LTS
```

## Initial Server Setup

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git htop iftop
```

### 2. Install Docker
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Clone Repository
```bash
git clone <your-repo-url>
cd <repo-name>
```

## Environment Configuration

### 1. Create Environment File
```bash
cp .env.example .env
```

### 2. Set Safe Environment Variables
```bash
cat > .env << EOF
# Safe scraping configuration
MAX_PAGES=999999
SCRAPE_INTERVAL=300
TRAINING_INTERVAL=3600

# Performance settings
ANALYZER_DEBUG_LEVEL=INFO
ENABLE_PERFORMANCE_MONITORING=true
COLLECT_TRAINING_DATA=true
AUTO_TRAIN_MODEL=true

# Safety settings
RESPECT_ROBOTS_TXT=true
USER_AGENT_ROTATION=true
MAX_CONCURRENT_REQUESTS=1

# Run mode
RUN_MODE=continuous
DEFAULT_SITE=tokopedia
SCRAPE_QUERY=laptop
EOF
```

## Quick Deployment Commands

### 1. Build and Start
```bash
# Build containers
docker-compose build

# Start in background
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Monitor Deployment
```bash
# View logs
docker-compose logs -f

# Check specific service
docker-compose logs -f scraper

# Monitor resources
docker stats
```

### 3. Verify Operation
```bash
# Check if containers are running
docker-compose ps

# Test browser functionality
docker-compose exec scraper python -c "from func.browser import BrowserManager; b = BrowserManager(); b.setup_driver(); print('✅ Browser OK'); b.close()"

# Check data directory
ls -la data/
```

## Monitoring Commands

### 1. System Resources
```bash
# CPU and Memory
htop

# Disk usage
df -h
du -sh data/

# Network usage
sudo iftop

# Real-time stats
watch -n 5 'free -h && df -h'
```

### 2. Application Monitoring
```bash
# Live logs
tail -f logs/analyzer.log

# Error monitoring
grep -i "error\|fail\|exception" logs/analyzer.log

# Success rate
grep -c "✅" logs/analyzer.log

# Check scraped pages
ls -la data/raw_html/ | wc -l
```

### 3. Performance Metrics
```bash
# Container resource usage
docker stats --no-stream

# Memory usage by container
docker-compose exec scraper free -h

# Check Chrome processes
docker-compose exec scraper ps aux | grep chrome
```

## Safety Checks

### 1. Rate Limiting Verification
```bash
# Check delay settings in logs
grep "Waiting.*before next" logs/analyzer.log

# Verify user agent rotation
grep "user-agent" logs/analyzer.log

# Check request timing
grep "😴 Slept for" logs/analyzer.log
```

### 2. Compliance Monitoring
```bash
# Check for HTTP errors
grep -E "(429|503|502)" logs/analyzer.log

# Monitor error rates
grep -c "❌" logs/analyzer.log

# Check robots.txt compliance
grep -i "robots" logs/analyzer.log
```

## Troubleshooting Commands

### 1. Common Issues
```bash
# Restart containers
docker-compose restart

# Rebuild if needed
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check container health
docker-compose exec scraper curl -f http://localhost:8000/health || echo "Health check failed"
```

### 2. Debug Mode
```bash
# Run in debug mode
docker-compose down
RUN_MODE=development docker-compose up

# Interactive shell
docker-compose exec scraper bash

# Test specific components
docker-compose exec scraper python func/test.py
```

### 3. Clean Reset
```bash
# Stop everything
docker-compose down -v

# Clean data (CAUTION: Removes all scraped data)
sudo rm -rf data/raw_html/*
sudo rm -rf data/processed/*
sudo rm -rf logs/*

# Restart fresh
docker-compose up -d
```

## Emergency Procedures

### 1. If Receiving Abuse Complaints
```bash
# Immediate stop
docker-compose down

# Check what was being scraped
tail -100 logs/analyzer.log

# Increase delays and restart
export SCRAPE_INTERVAL=1800  # 30 minutes
export MAX_CONCURRENT_REQUESTS=1
docker-compose up -d
```

### 2. High Resource Usage
```bash
# Check resource usage
docker stats

# Reduce concurrent operations
docker-compose down
echo "MAX_CONCURRENT_REQUESTS=1" >> .env
docker-compose up -d

# Monitor improvement
watch -n 5 docker stats
```

### 3. IP Blocking Detection
```bash
# Test connectivity
curl -I https://www.tokopedia.com

# Check for blocks in logs
grep -i "block\|ban\|403\|429" logs/analyzer.log

# If blocked, increase delays
export SCRAPE_INTERVAL=3600  # 1 hour
docker-compose restart
```

## Optimization Commands

### 1. Performance Tuning
```bash
# Check current delays
grep "delay\|sleep\|wait" logs/analyzer.log | tail -20

# Monitor success rate
echo "Success rate: $(grep -c "✅" logs/analyzer.log) / $(grep -c "🔍" logs/analyzer.log)"

# Check feature extraction time
grep "extraction took" logs/analyzer.log
```

### 2. Resource Optimization
```bash
# Clean old data periodically
find data/raw_html/ -name "*.html" -mtime +7 -delete

# Compress logs
gzip logs/analyzer.log.1

# Monitor disk usage
watch -n 60 df -h
```

## Maintenance Schedule

### Daily
```bash
# Check logs for errors
grep -i "error\|fail" logs/analyzer.log | tail -10

# Monitor disk usage
df -h

# Check container status
docker-compose ps
```

### Weekly
```bash
# Clean old data
find data/raw_html/ -name "*.html" -mtime +7 -delete

# Restart containers
docker-compose restart

# Update system
sudo apt update && sudo apt upgrade -y
```

### Monthly
```bash
# Full backup
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/ config/

# Clean Docker
docker system prune -f

# Review and optimize configuration
```

## Success Metrics

### Check These Regularly
```bash
# Uptime
uptime

# Error rate (should be < 5%)
echo "Error rate: $(grep -c "❌" logs/analyzer.log) / $(grep -c "🔍" logs/analyzer.log)"

# Pages scraped today
find data/raw_html/ -name "*.html" -mtime -1 | wc -l

# Resource usage (should be < 80%)
free | grep Mem | awk '{printf "Memory usage: %.1f%%\n", $3/$2 * 100.0}'
```

This setup provides a balance between speed and safety, significantly reducing suspension risk while maintaining research effectiveness.