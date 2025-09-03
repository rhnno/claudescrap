#!/usr/bin/env python3
"""
Comprehensive test runner script for ScraperService CI/CD.

This script provides a unified interface for running all tests with proper
categorization, reporting, and CI/CD integration.
"""
import sys
import os
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, description="", timeout=300):
    """Run a command and return success/failure status."""
    print(f"\n{'='*60}")
    print(f"🏃 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=False,
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {description} PASSED ({duration:.2f}s)")
            return True
        else:
            print(f"\n❌ {description} FAILED ({duration:.2f}s)")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"\n💥 {description} ERROR: {e}")
        return False


def setup_environment():
    """Set up test environment variables."""
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key'
    os.environ['DATABASE_URL'] = 'sqlite:///test.db'
    os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)


def run_unit_tests(verbose=False):
    """Run unit tests for ScraperService."""
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_scraper_service.py',
        '-m', 'unit or not integration',
        '--tb=short',
        '--timeout=300'
    ]
    
    if verbose:
        cmd.extend(['-v', '-s'])
    
    return run_command(cmd, "Unit Tests", timeout=300)


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_integration.py',
        '--tb=short',
        '--timeout=600'
    ]
    
    if verbose:
        cmd.extend(['-v', '-s'])
    
    return run_command(cmd, "Integration Tests", timeout=600)


def run_api_tests(verbose=False):
    """Run API tests."""
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_api.py',
        '--tb=short',
        '--timeout=300'
    ]
    
    if verbose:
        cmd.extend(['-v', '-s'])
    
    return run_command(cmd, "API Tests", timeout=300)


def run_performance_tests(verbose=False):
    """Run performance and benchmark tests."""
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_performance.py',
        '-m', 'performance',
        '--tb=short',
        '--timeout=600'
    ]
    
    if verbose:
        cmd.extend(['-v', '-s'])
    
    return run_command(cmd, "Performance Tests", timeout=600)


def run_coverage_report():
    """Generate coverage report."""
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--cov-report=xml:coverage.xml',
        '--tb=short'
    ]
    
    return run_command(cmd, "Coverage Analysis", timeout=900)


def run_linting():
    """Run code linting checks."""
    success = True
    
    # Flake8
    cmd = [sys.executable, '-m', 'flake8', 'src/', '--max-line-length=127']
    if not run_command(cmd, "Flake8 Linting", timeout=60):
        success = False
    
    # MyPy
    cmd = [sys.executable, '-m', 'mypy', 'src/', '--ignore-missing-imports']
    if not run_command(cmd, "MyPy Type Checking", timeout=120):
        success = False
    
    return success


def run_security_scan():
    """Run security scanning."""
    success = True
    
    # Safety check
    cmd = [sys.executable, '-m', 'safety', 'check']
    if not run_command(cmd, "Safety Security Scan", timeout=60):
        success = False
    
    # Bandit scan
    cmd = [sys.executable, '-m', 'bandit', '-r', 'src/', '-f', 'json']
    if not run_command(cmd, "Bandit Security Scan", timeout=60):
        success = False
    
    return success


def run_all_tests(verbose=False, include_performance=False, include_security=False):
    """Run all tests in sequence."""
    results = {}
    
    print("🚀 Starting ScraperService Comprehensive Test Suite")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Setup
    setup_environment()
    
    # Core tests
    results['unit'] = run_unit_tests(verbose)
    results['integration'] = run_integration_tests(verbose)
    results['api'] = run_api_tests(verbose)
    
    # Optional tests
    if include_performance:
        results['performance'] = run_performance_tests(verbose)
    
    if include_security:
        results['linting'] = run_linting()
        results['security'] = run_security_scan()
    
    # Coverage report
    results['coverage'] = run_coverage_report()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():15} {status}")
    
    print(f"\n📈 Overall: {passed_tests}/{total_tests} tests passed")
    success_rate = (passed_tests / total_tests) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Review and fix issues.")
        return False


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="ScraperService Test Runner")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--api', action='store_true', help='Run only API tests')
    parser.add_argument('--performance', action='store_true', help='Run only performance tests')
    parser.add_argument('--coverage', action='store_true', help='Run coverage analysis')
    parser.add_argument('--lint', action='store_true', help='Run linting checks')
    parser.add_argument('--security', action='store_true', help='Run security scans')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--ci', action='store_true', help='CI mode (all tests + security)')
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    setup_environment()
    
    success = True
    
    if args.ci:
        # CI mode: comprehensive testing
        success = run_all_tests(
            verbose=args.verbose,
            include_performance=True,
            include_security=True
        )
    elif args.all:
        # All tests but without security scans
        success = run_all_tests(verbose=args.verbose, include_performance=True)
    elif args.unit:
        success = run_unit_tests(args.verbose)
    elif args.integration:
        success = run_integration_tests(args.verbose)
    elif args.api:
        success = run_api_tests(args.verbose)
    elif args.performance:
        success = run_performance_tests(args.verbose)
    elif args.coverage:
        success = run_coverage_report()
    elif args.lint:
        success = run_linting()
    elif args.security:
        success = run_security_scan()
    else:
        # Default: run core tests
        success = (
            run_unit_tests(args.verbose) and
            run_integration_tests(args.verbose) and
            run_api_tests(args.verbose)
        )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()