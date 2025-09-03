"""
Performance and load tests for ScraperService.

These tests evaluate the performance characteristics of the ScraperService
under various load conditions and measure key metrics.
"""
import pytest
import asyncio
import time
from unittest.mock import patch, Mock
import psutil
import os

from src.services.scraper_service import ScraperService
from tests.conftest import TEST_SITE, TEST_QUERY


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_service_initialization_time(self, benchmark):
        """Benchmark service initialization time."""
        def init_service():
            with patch('src.services.scraper_service.DatabaseManager'):
                with patch('src.services.scraper_service.Utils'):
                    return ScraperService()
        
        result = benchmark(init_service)
        assert result is not None
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_browser_pool_performance(self, benchmark):
        """Benchmark browser pool operations."""
        mock_orchestrator = Mock()
        mock_orchestrator.setup_browser.return_value = True
        mock_orchestrator.close = Mock()
        
        with patch('src.services.scraper_service.DatabaseManager'):
            with patch('src.services.scraper_service.Utils'):
                service = ScraperService()
        
        async def browser_pool_cycle():
            with patch('src.services.scraper_service.ScrapingOrchestrator', return_value=mock_orchestrator):
                browser = await service._get_browser_from_pool()
                await service._return_browser_to_pool(browser)
                return browser
        
        result = await benchmark(browser_pool_cycle)
        assert result is not None
    
    @pytest.mark.benchmark
    def test_session_stats_calculation(self, benchmark):
        """Benchmark session statistics calculation."""
        with patch('src.services.scraper_service.DatabaseManager'):
            with patch('src.services.scraper_service.Utils'):
                service = ScraperService()
        
        # Setup large dataset
        service.session_stats = {
            'total_jobs': 10000,
            'successful_jobs': 8500,
            'failed_jobs': 1500,
            'total_products': 1000000,
            'total_pages': 50000
        }
        service.running_jobs = {f'job_{i}': {} for i in range(100)}
        
        result = benchmark(service.get_session_stats)
        assert result['success_rate'] == 85.0


class TestLoadTesting:
    """Load testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_job_creation(self):
        """Test creating many concurrent jobs."""
        mock_orchestrator = Mock()
        mock_orchestrator.setup_browser.return_value = True
        mock_orchestrator._scrape_query.return_value = [{'name': 'Product', 'price': '$100'}]
        mock_orchestrator.session_stats = {'total_pages': 1}
        mock_orchestrator.close = Mock()
        
        with patch('src.services.scraper_service.DatabaseManager'):
            with patch('src.services.scraper_service.Utils'):
                with patch('src.services.scraper_service.ScrapingOrchestrator', return_value=mock_orchestrator):
                    service = ScraperService()
                    
                    start_time = time.time()
                    
                    # Create 50 concurrent jobs
                    tasks = []
                    for i in range(50):
                        task = service.start_scraping_job(TEST_SITE, f'{TEST_QUERY}_{i}', 1)
                        tasks.append(task)
                    
                    job_ids = await asyncio.gather(*tasks)
                    
                    end_time = time.time()
                    creation_time = end_time - start_time
                    
                    assert len(job_ids) == 50
                    assert all(job_id is not None for job_id in job_ids)
                    assert creation_time < 10  # Should complete within 10 seconds
                    
                    # Verify session stats
                    stats = service.get_session_stats()
                    assert stats['total_jobs'] == 50
    
    @pytest.mark.asyncio
    async def test_browser_pool_under_load(self):
        """Test browser pool performance under high load."""
        mock_orchestrator = Mock()
        mock_orchestrator.setup_browser.return_value = True
        mock_orchestrator.close = Mock()
        
        with patch('src.services.scraper_service.DatabaseManager'):
            with patch('src.services.scraper_service.Utils'):
                with patch('src.services.scraper_service.ScrapingOrchestrator', return_value=mock_orchestrator):
                    service = ScraperService()
                    
                    start_time = time.time()
                    
                    # Simulate high load on browser pool
                    async def use_browser():
                        browser = await service._get_browser_from_pool()
                        await asyncio.sleep(0.01)  # Simulate work
                        await service._return_browser_to_pool(browser)
                    
                    # Run 100 concurrent browser operations
                    tasks = [use_browser() for _ in range(100)]
                    await asyncio.gather(*tasks)
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Should handle load efficiently
                    assert total_time < 5  # Should complete within 5 seconds
                    assert len(service._browser_pool) <= service._browser_pool_size
    
    def test_memory_usage_under_load(self):
        """Test memory usage with large datasets."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('src.services.scraper_service.DatabaseManager'):
            with patch('src.services.scraper_service.Utils'):
                service = ScraperService()
                
                # Simulate large session stats
                for i in range(10000):
                    service.session_stats['total_jobs'] += 1
                    service.session_stats['total_products'] += 100
                    service.session_stats['total_pages'] += 5
                    
                    # Add running jobs
                    if i < 1000:
                        service.running_jobs[f'job_{i}'] = {
                            'task': Mock(),
                            'site': TEST_SITE,
                            'query': f'{TEST_QUERY}_{i}',
                            'max_pages': 5,
                            'start_time': time.time()
                        }
                
                # Calculate memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be reasonable (less than 100MB)
                assert memory_increase < 100
                
                # Verify functionality still works
                stats = service.get_session_stats()
                assert stats['total_jobs'] == 10000
                assert stats['running_jobs'] == 1000


class TestStressScenarios:
    """Stress testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_rapid_job_start_stop(self):
        """Test rapid job creation and stopping."""
        mock_orchestrator = Mock()
        mock_orchestrator.setup_browser.return_value = True
        mock_orchestrator.close = Mock()
        
        # Mock slow scraping to allow time for stopping
        async def slow_scraping(*args, **kwargs):
            await asyncio.sleep(1)
            return [{'name': 'Product', 'price': '$100'}]
        
        mock_orchestrator._scrape_query.side_effect = slow_scraping
        mock_orchestrator.session_stats = {'total_pages': 1}
        
        with patch('src.services.scraper_service.DatabaseManager'):
            with patch('src.services.scraper_service.Utils'):
                with patch('src.services.scraper_service.ScrapingOrchestrator', return_value=mock_orchestrator):
                    service = ScraperService()
                    
                    job_ids = []
                    
                    # Start 20 jobs rapidly
                    for i in range(20):
                        job_id = await service.start_scraping_job(TEST_SITE, f'{TEST_QUERY}_{i}', 5)
                        job_ids.append(job_id)
                    
                    # Wait a moment for jobs to start
                    await asyncio.sleep(0.1)
                    
                    # Stop all jobs rapidly
                    stop_results = []
                    for job_id in job_ids:
                        result = await service.stop_scraping_job(job_id)
                        stop_results.append(result)
                    
                    # Most jobs should be successfully stopped
                    successful_stops = sum(stop_results)
                    assert successful_stops >= 15  # At least 75% should be stopped successfully
                    
                    # Verify cleanup
                    assert len(service.running_jobs) <= 5  # Most should be cleaned up
    
    @pytest.mark.asyncio
    async def test_browser_pool_exhaustion(self):
        """Test behavior when browser pool is exhausted."""
        mock_orchestrator = Mock()
        mock_orchestrator.setup_browser.return_value = True
        mock_orchestrator.close = Mock()
        
        with patch('src.services.scraper_service.DatabaseManager'):
            with patch('src.services.scraper_service.Utils'):
                service = ScraperService()
                service._browser_pool_size = 2  # Small pool for testing
                
                browsers = []
                
                # Mock browser creation to track calls
                creation_count = 0
                def create_browser():
                    nonlocal creation_count
                    creation_count += 1
                    new_orchestrator = Mock()
                    new_orchestrator.setup_browser.return_value = True
                    new_orchestrator.close = Mock()
                    return new_orchestrator
                
                with patch('src.services.scraper_service.ScrapingOrchestrator', side_effect=create_browser):
                    # Request more browsers than pool size
                    for i in range(10):
                        browser = await service._get_browser_from_pool()
                        browsers.append(browser)
                    
                    # Should create new browsers when pool is empty
                    assert creation_count == 10
                    
                    # Return browsers to pool
                    for browser in browsers:
                        await service._return_browser_to_pool(browser)
                    
                    # Pool should not exceed max size
                    assert len(service._browser_pool) <= service._browser_pool_size
    
    def test_database_operation_stress(self):
        """Test database operations under stress."""
        mock_db = Mock()
        mock_session = Mock()
        mock_db.get_session.return_value = mock_session
        mock_session.close = Mock()
        
        # Mock query chain
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.first.return_value = None
        
        with patch('src.services.scraper_service.DatabaseManager', return_value=mock_db):
            with patch('src.services.scraper_service.Utils'):
                service = ScraperService()
                
                # Perform many database operations
                for i in range(1000):
                    service.get_job_status(f'job_{i}')
                    service.list_jobs()
                
                # Verify all sessions were properly closed
                assert mock_session.close.call_count == 2000  # 1000 * 2 operations


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--benchmark-only"])