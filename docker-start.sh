#!/bin/bash
# Enhanced ML-Powered Web Scraper - Docker Startup Script

set -e

echo "🚀 Enhanced ML-Powered Web Scraper - Docker Startup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Parse command line arguments
PROFILE=""
ACTION="up"
DETACHED="-d"
BUILD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev|--development)
            PROFILE="--profile development"
            print_status "Development profile enabled (includes Jupyter)"
            shift
            ;;
        --monitor|--monitoring)
            PROFILE="--profile monitoring"
            print_status "Monitoring profile enabled (includes Grafana)"
            shift
            ;;
        --full)
            PROFILE="--profile development --profile monitoring"
            print_status "Full profile enabled (includes Jupyter and Grafana)"
            shift
            ;;
        --build)
            BUILD="--build"
            print_status "Force rebuild enabled"
            shift
            ;;
        --logs)
            ACTION="logs -f"
            DETACHED=""
            shift
            ;;
        --stop)
            ACTION="down"
            DETACHED=""
            shift
            ;;
        --restart)
            ACTION="restart"
            DETACHED=""
            shift
            ;;
        --status)
            ACTION="ps"
            DETACHED=""
            shift
            ;;
        --help|-h)
            echo "Enhanced ML-Powered Web Scraper - Docker Startup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev, --development    Start with development profile (includes Jupyter)"
            echo "  --monitor, --monitoring Start with monitoring profile (includes Grafana)"
            echo "  --full                  Start with all profiles (development + monitoring)"
            echo "  --build                 Force rebuild of containers"
            echo "  --logs                  Show logs instead of starting"
            echo "  --stop                  Stop all services"
            echo "  --restart               Restart all services"
            echo "  --status                Show service status"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Start basic services"
            echo "  $0 --dev                # Start with Jupyter for development"
            echo "  $0 --full --build       # Start all services and rebuild"
            echo "  $0 --logs               # Show real-time logs"
            echo "  $0 --stop               # Stop all services"
            exit 0
            ;;
        *)
            print_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Create necessary directories
print_header "Creating necessary directories..."
mkdir -p data/raw_html data/processed logs models config/feature_templates notebooks

# Set proper permissions
chmod 755 data logs models config
print_status "Directories created and permissions set"

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_header "Creating environment configuration..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "Created .env from .env.example"
        print_warning "Please review and customize .env file for your environment"
    else
        print_warning ".env.example not found, using default environment"
    fi
fi

# Validate Docker Compose configuration
print_header "Validating Docker Compose configuration..."
if docker-compose config > /dev/null 2>&1; then
    print_status "Docker Compose configuration is valid"
else
    print_error "Docker Compose configuration is invalid"
    docker-compose config
    exit 1
fi

# Execute the requested action
print_header "Executing: docker-compose $PROFILE $ACTION $BUILD $DETACHED"

case $ACTION in
    "up")
        # Pull latest images if not building
        if [ -z "$BUILD" ]; then
            print_status "Pulling latest images..."
            docker-compose $PROFILE pull
        fi
        
        # Start services
        docker-compose $PROFILE $ACTION $BUILD $DETACHED
        
        if [ "$DETACHED" = "-d" ]; then
            print_status "Services started in detached mode"
            
            # Wait a moment for services to start
            sleep 5
            
            # Show service status
            print_header "Service Status:"
            docker-compose ps
            
            # Show access information
            print_header "Access Information:"
            echo "📊 Main Analyzer: http://localhost:4000"
            echo "🔧 Monitoring API: http://localhost:8080"
            
            if [[ $PROFILE == *"development"* ]]; then
                echo "📓 Jupyter Lab: http://localhost:8888"
            fi
            
            if [[ $PROFILE == *"monitoring"* ]]; then
                echo "📈 Grafana: http://localhost:3000 (admin/admin_password_2024)"
            fi
            
            echo "🗄️ MongoDB: localhost:27017"
            echo "🔄 Redis: localhost:6379"
            
            print_header "Useful Commands:"
            echo "View logs:           docker-compose logs -f enhanced-analyzer"
            echo "Check health:        docker-compose exec enhanced-analyzer /usr/local/bin/healthcheck.sh"
            echo "Access container:    docker-compose exec enhanced-analyzer bash"
            echo "Stop services:       $0 --stop"
            echo "View status:         $0 --status"
        fi
        ;;
    "logs -f")
        docker-compose $PROFILE logs -f
        ;;
    "down")
        print_status "Stopping all services..."
        docker-compose $PROFILE down
        print_status "All services stopped"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose $PROFILE restart
        print_status "Services restarted"
        ;;
    "ps")
        print_header "Service Status:"
        docker-compose ps
        
        print_header "Resource Usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
        ;;
esac

print_status "Script completed successfully!"