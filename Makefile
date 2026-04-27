# Makefile for ScraperService CI/CD Testing
# Provides convenient commands for running tests and CI/CD operations

.PHONY: help install test test-unit test-integration test-api test-performance
.PHONY: test-all test-ci coverage lint security clean setup
.PHONY: docker-test docker-build format check

# Default target
help:
	@echo "ScraperService CI/CD Test Commands"
	@echo "================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install     Install all dependencies"
	@echo "  setup       Setup development environment"
	@echo ""
	@echo "Test Commands:"
	@echo "  test        Run core tests (unit + integration + api)"
	@echo "  test-unit   Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-api    Run API tests only"
	@echo "  test-performance  Run performance tests"
	@echo "  test-all    Run all tests including performance"
	@echo "  test-ci     Run full CI test suite"
	@echo ""
	@echo "Quality Commands:"
	@echo "  coverage    Generate coverage report"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black"
	@echo "  security    Run security scans"
	@echo "  check       Run all quality checks"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build  Build test Docker image"
	@echo "  docker-test   Run tests in Docker container"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean       Clean up test artifacts"
	@echo ""

# Python and pip executables
PYTHON := python
PIP := pip

# Project directories
SRC_DIR := src
TEST_DIR := tests
VENV_DIR := venv

# Test environment variables
export JWT_SECRET_KEY := test-secret-key
# DATABASE_URL should be set by CI or test fixtures
# export DATABASE_URL := postgresql://localhost/test_db

# Setup and installation
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements_api.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-xdist pytest-timeout
	$(PIP) install pytest-benchmark memory-profiler
	$(PIP) install flake8 mypy black isort
	$(PIP) install safety bandit
	@echo "✅ Dependencies installed successfully"

setup: install
	@echo "Setting up development environment..."
	mkdir -p logs
	mkdir -p htmlcov
	@echo "✅ Development environment setup complete"

# Core test commands
test:
	@echo "Running core tests..."
	$(PYTHON) tests/test_runner.py --verbose

test-unit:
	@echo "Running unit tests..."
	$(PYTHON) tests/test_runner.py --unit --verbose

test-integration:
	@echo "Running integration tests..."
	$(PYTHON) tests/test_runner.py --integration --verbose

test-api:
	@echo "Running API tests..."
	$(PYTHON) tests/test_runner.py --api --verbose

test-performance:
	@echo "Running performance tests..."
	$(PYTHON) tests/test_runner.py --performance --verbose

test-all:
	@echo "Running all tests..."
	$(PYTHON) tests/test_runner.py --all --verbose

test-ci:
	@echo "Running CI test suite..."
	$(PYTHON) tests/test_runner.py --ci --verbose

# Quality and analysis commands
coverage:
	@echo "Generating coverage report..."
	$(PYTHON) tests/test_runner.py --coverage

lint:
	@echo "Running linting checks..."
	$(PYTHON) tests/test_runner.py --lint

format:
	@echo "Formatting code..."
	black $(SRC_DIR) $(TEST_DIR) --line-length 127
	isort $(SRC_DIR) $(TEST_DIR) --profile black
	@echo "✅ Code formatted successfully"

security:
	@echo "Running security scans..."
	$(PYTHON) tests/test_runner.py --security

check: lint security
	@echo "✅ All quality checks completed"

# Docker commands
docker-build:
	@echo "Building test Docker image..."
	docker build -t claudescrap-test -f - . <<EOF
	FROM python:3.11-slim
	WORKDIR /app
	COPY requirements*.txt ./
	RUN pip install -r requirements.txt -r requirements_api.txt
	RUN pip install pytest pytest-asyncio pytest-cov
	COPY . .
	CMD ["python", "tests/test_runner.py", "--ci"]
	EOF
	@echo "✅ Docker image built successfully"

docker-test: docker-build
	@echo "Running tests in Docker..."
	docker run --rm claudescrap-test
	@echo "✅ Docker tests completed"

# Utility commands
clean:
	@echo "Cleaning up test artifacts..."
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf $(SRC_DIR)/__pycache__/
	rm -rf $(TEST_DIR)/__pycache__/
	rm -rf $(SRC_DIR)/*/__pycache__/
	rm -rf $(TEST_DIR)/*/__pycache__/
	rm -rf .mypy_cache/
	rm -rf test.db
	rm -rf tests/test.log
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Cleanup completed"

# Quick commands for development
quick-test:
	@echo "Running quick tests (unit only)..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_scraper_service.py -v --tb=short -x

watch-test:
	@echo "Watching for changes and running tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/ --looponfail

# Parallel testing
test-parallel:
	@echo "Running tests in parallel..."
	$(PYTHON) -m pytest $(TEST_DIR)/ -n auto --tb=short

# Specific test categories
test-async:
	@echo "Running async tests only..."
	$(PYTHON) -m pytest $(TEST_DIR)/ -m asyncio -v

test-slow:
	@echo "Running slow tests only..."
	$(PYTHON) -m pytest $(TEST_DIR)/ -m slow -v

# Debugging
test-debug:
	@echo "Running tests with debugging..."
	$(PYTHON) -m pytest $(TEST_DIR)/ -v -s --pdb --tb=long

# Virtual environment commands (optional)
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created"
	@echo "Activate with: source $(VENV_DIR)/bin/activate (Linux/Mac) or $(VENV_DIR)\\Scripts\\activate (Windows)"

# CI simulation
simulate-ci: clean install test-ci
	@echo "✅ CI simulation completed"

# Performance monitoring
benchmark:
	@echo "Running benchmark tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_performance.py --benchmark-only --benchmark-json=benchmark.json

profile:
	@echo "Running with profiling..."
	$(PYTHON) -m cProfile -o profile_output.prof -m pytest $(TEST_DIR)/test_scraper_service.py
	@echo "Profile saved to profile_output.prof"