"""
Comprehensive unit tests for ScraperService.

This module tests every function in the ScraperService class including:
- Initialization and setup
- Browser pool management
- Job lifecycle management  
- Database operations
- Session statistics
- Error handling and edge cases

All tests use mocking to isolate unit behavior and avoid external dependencies.
"""
import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone
import concurrent.futures

from src.services.scraper_service import ScraperService
from src.models.database import ScrapingJob
from src.ace import ScrapingOrchestrator
from tests.conftest import (
    TEST_JOB_ID, TEST_SITE, TEST_QUERY, TEST_MAX_PAGES, TEST_SESSION_STATS,
    AsyncContextManager, MockBrowserPool
)


class TestScraperServiceInitialization:
    """Test ScraperService initialization and setup."""
    
    def test_init_creates_required_attributes(self, scraper_service_with_mocks):
        """Test that __init__ creates all required attributes."""
        service = scraper_service_with_mocks
        
        # Verify core attributes
        assert hasattr(service, 'db')
        assert hasattr(service, 'utils')
        assert hasattr(service, 'running_jobs')
        assert isinstance(service.running_jobs, dict)
        
        # Verify browser pool attributes
        assert hasattr(service, '_browser_pool')
        assert isinstance(service._browser_pool, list)
        assert hasattr(service, '_browser_pool_size')
        assert service._browser_pool_size == 2
        assert hasattr(service, '_browser_lock')
        
        # Verify session stats
        assert hasattr(service, 'session_stats')
        assert isinstance(service.session_stats, dict)
        expected_stats = ['total_jobs', 'successful_jobs', 'failed_jobs', 'total_products', 'total_pages']
        for stat in expected_stats:
            assert stat in service.session_stats
            assert service.session_stats[stat] == 0
    
    def test_init_empty_browser_pool(self, scraper_service_with_mocks):
        """Test that browser pool starts empty."""
        service = scraper_service_with_mocks
        assert len(service._browser_pool) == 0
    
    def test_init_empty_running_jobs(self, scraper_service_with_mocks):
        """Test that running jobs starts empty."""
        service = scraper_service_with_mocks
        assert len(service.running_jobs) == 0


class TestBrowserPoolManagement:
    """Test browser pool management functions."""
    
    @pytest.mark.asyncio
    async def test_get_browser_from_pool_empty_pool(self, scraper_service_with_mocks, mock_scraping_orchestrator):
        """Test getting browser when pool is empty."""
        service = scraper_service_with_mocks
        
        with patch.object(service, '_create_browser_async', return_value=mock_scraping_orchestrator) as mock_create:
            result = await service._get_browser_from_pool()
            
            assert result == mock_scraping_orchestrator
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_browser_from_pool_with_available_browser(self, scraper_service_with_mocks, mock_scraping_orchestrator):
        """Test getting browser when one is available in pool."""
        service = scraper_service_with_mocks
        service._browser_pool = [mock_scraping_orchestrator]
        
        with patch.object(service, '_create_browser_async') as mock_create:
            result = await service._get_browser_from_pool()
            
            assert result == mock_scraping_orchestrator
            assert len(service._browser_pool) == 0
            mock_create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_return_browser_to_pool_not_full(self, scraper_service_with_mocks, mock_scraping_orchestrator):
        """Test returning browser to pool when not full."""
        service = scraper_service_with_mocks
        assert len(service._browser_pool) == 0
        
        await service._return_browser_to_pool(mock_scraping_orchestrator)
        
        assert len(service._browser_pool) == 1
        assert service._browser_pool[0] == mock_scraping_orchestrator
        mock_scraping_orchestrator.close.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_return_browser_to_pool_full(self, scraper_service_with_mocks, mock_scraping_orchestrator):
        """Test returning browser to pool when full."""
        service = scraper_service_with_mocks
        # Fill the pool
        service._browser_pool = [Mock(), Mock()]
        
        await service._return_browser_to_pool(mock_scraping_orchestrator)
        
        # Pool should still be full, browser should be closed
        assert len(service._browser_pool) == 2
        mock_scraping_orchestrator.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_return_browser_to_pool_close_error(self, scraper_service_with_mocks, mock_scraping_orchestrator):
        """Test handling error when closing excess browser."""
        service = scraper_service_with_mocks
        service._browser_pool = [Mock(), Mock()]
        mock_scraping_orchestrator.close.side_effect = Exception("Close error")
        
        # Should not raise exception
        await service._return_browser_to_pool(mock_scraping_orchestrator)
        
        mock_scraping_orchestrator.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_browser_async_success(self, scraper_service_with_mocks):
        """Test successful async browser creation."""
        service = scraper_service_with_mocks
        mock_orchestrator = Mock()
        
        def mock_browser_creation():
            orchestrator = Mock()
            orchestrator.setup_browser.return_value = True
            return orchestrator
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_executor.return_value.__enter__.return_value = mock_executor.return_value
            
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_orchestrator)
                
                with patch('src.services.scraper_service.ScrapingOrchestrator', return_value=mock_orchestrator):
                    result = await service._create_browser_async()
                    
                    assert result == mock_orchestrator
    
    @pytest.mark.asyncio
    async def test_create_browser_async_setup_failure(self, scraper_service_with_mocks):
        """Test async browser creation when setup fails."""
        service = scraper_service_with_mocks
        
        # Mock the run_in_executor to raise exception
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock(side_effect=Exception("Browser setup failed"))
            
            with pytest.raises(Exception, match="Browser setup failed"):
                await service._create_browser_async()


class TestJobManagement:
    """Test job management functions."""
    
    @pytest.mark.asyncio
    async def test_start_scraping_job_success(self, scraper_service_with_mocks, mock_scraping_job):
        """Test successful job start."""
        service = scraper_service_with_mocks
        service.db.create_job.return_value = mock_scraping_job
        
        with patch('asyncio.create_task') as mock_create_task:
            with patch('uuid.uuid4') as mock_uuid:
                mock_uuid.return_value = Mock()
                mock_uuid.return_value.__str__ = Mock(return_value=TEST_JOB_ID)
                
                job_id = await service.start_scraping_job(TEST_SITE, TEST_QUERY, TEST_MAX_PAGES)
        
        # Verify job creation
        service.db.create_job.assert_called_once_with(TEST_JOB_ID, TEST_SITE, TEST_QUERY)
        
        # Verify job tracking
        assert TEST_JOB_ID in service.running_jobs
        job_info = service.running_jobs[TEST_JOB_ID]
        assert job_info['site'] == TEST_SITE
        assert job_info['query'] == TEST_QUERY
        assert job_info['max_pages'] == TEST_MAX_PAGES
        
        # Verify session stats update
        assert service.session_stats['total_jobs'] == 1
    
    def test_get_job_status_existing_job(self, scraper_service_with_mocks, mock_scraping_job):
        """Test getting status of existing job."""
        service = scraper_service_with_mocks
        
        # Mock database query to return job
        mock_session = service.db.get_session.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = mock_scraping_job
        
        result = service.get_job_status(TEST_JOB_ID)
        
        assert result is not None
        assert result['job_id'] == mock_scraping_job.job_id
        assert result['status'] == mock_scraping_job.status
        # Check for fields that exist in the actual implementation
        assert 'current_page' in result
        assert 'total_pages' in result
        assert 'products_found' in result
        assert 'created_at' in result
        mock_session.close.assert_called_once()
    
    def test_get_job_status_nonexistent_job(self, scraper_service_with_mocks):
        """Test getting status of non-existent job."""
        service = scraper_service_with_mocks
        
        # Mock database query to return None
        mock_session = service.db.get_session.return_value
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        result = service.get_job_status('nonexistent-job')
        
        assert result is None
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_scraping_job_running(self, scraper_service_with_mocks):
        """Test stopping a running job."""
        service = scraper_service_with_mocks
        
        # Setup running job
        mock_task = Mock()
        service.running_jobs[TEST_JOB_ID] = {
            'task': mock_task,
            'site': TEST_SITE,
            'query': TEST_QUERY,
            'max_pages': TEST_MAX_PAGES,
            'start_time': datetime.now(timezone.utc)
        }
        
        result = await service.stop_scraping_job(TEST_JOB_ID)
        
        assert result is True
        mock_task.cancel.assert_called_once()
        service.db.update_job_status.assert_called_once()
        assert TEST_JOB_ID not in service.running_jobs
        assert service.session_stats['failed_jobs'] == 1
    
    @pytest.mark.asyncio
    async def test_stop_scraping_job_not_running(self, scraper_service_with_mocks):
        """Test stopping a job that's not running."""
        service = scraper_service_with_mocks
        
        result = await service.stop_scraping_job('nonexistent-job')
        
        assert result is False
        service.db.update_job_status.assert_not_called()
    
    def test_list_jobs(self, scraper_service_with_mocks, mock_scraping_job):
        """Test listing all jobs."""
        service = scraper_service_with_mocks
        
        # Mock database query to return jobs
        mock_session = service.db.get_session.return_value
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_scraping_job]
        
        result = service.list_jobs()
        
        assert isinstance(result, list)
        assert len(result) == 1
        job = result[0]
        assert job['job_id'] == mock_scraping_job.job_id
        assert job['status'] == mock_scraping_job.status
        mock_session.close.assert_called_once()


class TestSessionStatistics:
    """Test session statistics functions."""
    
    def test_get_session_stats(self, scraper_service_with_mocks):
        """Test getting session statistics."""
        service = scraper_service_with_mocks
        
        # Setup test data
        service.session_stats = {
            'total_jobs': 10,
            'successful_jobs': 8,
            'failed_jobs': 2,
            'total_products': 156,
            'total_pages': 25
        }
        service.running_jobs = {'job1': {}, 'job2': {}}
        
        result = service.get_session_stats()
        
        assert result['total_jobs'] == 10
        assert result['successful_jobs'] == 8
        assert result['failed_jobs'] == 2
        assert result['running_jobs'] == 2
        assert result['total_products'] == 156
        assert result['total_pages'] == 25
        assert result['success_rate'] == 80.0
    
    def test_get_session_stats_no_jobs(self, scraper_service_with_mocks):
        """Test session stats with no jobs to avoid division by zero."""
        service = scraper_service_with_mocks
        
        result = service.get_session_stats()
        
        assert result['success_rate'] == 0.0  # Should use max(1, total_jobs) to avoid division by zero
    
    def test_print_session_summary(self, scraper_service_with_mocks):
        """Test printing session summary."""
        service = scraper_service_with_mocks
        service.session_stats = TEST_SESSION_STATS.copy()
        
        with patch('src.services.scraper_service.logger') as mock_logger:
            service.print_session_summary()
            
            # Verify logging calls
            assert mock_logger.info.call_count >= 7  # At least summary header + 6 stats
            
            # Check key information is logged
            calls = [call.args[0] for call in mock_logger.info.call_args_list]
            summary_text = ' '.join(calls)
            assert 'Total jobs: 10' in summary_text
            assert 'Success rate: 80.0%' in summary_text
    
    def test_print_session_summary_with_avg_products(self, scraper_service_with_mocks):
        """Test session summary includes average products per page."""
        service = scraper_service_with_mocks
        service.session_stats = {
            'total_jobs': 5,
            'successful_jobs': 4,
            'failed_jobs': 1,
            'total_products': 100,
            'total_pages': 10
        }
        
        with patch('src.services.scraper_service.logger') as mock_logger:
            service.print_session_summary()
            
            calls = [call.args[0] for call in mock_logger.info.call_args_list]
            summary_text = ' '.join(calls)
            assert 'Average products per page: 10.0' in summary_text


class TestExecuteScrapingIntegration:
    """Test the _execute_scraping method with various scenarios."""
    
    @pytest.mark.asyncio
    async def test_execute_scraping_success(self, scraper_service_with_mocks, mock_scraping_orchestrator, sample_products):
        """Test successful scraping execution."""
        service = scraper_service_with_mocks
        # Ensure _scrape_query is a regular Mock that returns sample_products
        mock_scraping_orchestrator._scrape_query = Mock(return_value=sample_products)
        
        with patch.object(service, '_get_browser_from_pool', return_value=mock_scraping_orchestrator):
            with patch.object(service, '_return_browser_to_pool'):
                # Setup running job
                service.running_jobs[TEST_JOB_ID] = {'task': Mock()}
                
                await service._execute_scraping(TEST_JOB_ID, TEST_SITE, TEST_QUERY, TEST_MAX_PAGES)
        
        # Verify database operations
        service.db.update_job_status.assert_any_call(TEST_JOB_ID, 'running')
        service.db.save_products.assert_called_once()
        
        # Verify session stats update
        assert service.session_stats['successful_jobs'] == 1
        assert service.session_stats['total_products'] == len(sample_products)
    
    @pytest.mark.asyncio
    async def test_execute_scraping_browser_acquisition_failure(self, scraper_service_with_mocks):
        """Test scraping execution when browser acquisition fails."""
        service = scraper_service_with_mocks
        
        with patch.object(service, '_get_browser_from_pool', side_effect=Exception("Browser acquisition failed")):
            service.running_jobs[TEST_JOB_ID] = {'task': Mock()}
            
            await service._execute_scraping(TEST_JOB_ID, TEST_SITE, TEST_QUERY, TEST_MAX_PAGES)
        
        # Verify failure handling
        service.db.update_job_status.assert_any_call(TEST_JOB_ID, 'running')
        calls = service.db.update_job_status.call_args_list
        failed_call = [call for call in calls if call[0][1] == 'failed']
        assert len(failed_call) > 0
        assert service.session_stats['failed_jobs'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_scraping_cancelled(self, scraper_service_with_mocks, mock_scraping_orchestrator):
        """Test scraping execution when job is cancelled."""
        service = scraper_service_with_mocks
        
        # Create a proper mock that raises CancelledError
        mock_scraping_orchestrator._scrape_query = Mock(side_effect=asyncio.CancelledError())
        
        with patch.object(service, '_get_browser_from_pool', return_value=mock_scraping_orchestrator):
            with patch.object(service, '_return_browser_to_pool'):
                service.running_jobs[TEST_JOB_ID] = {'task': Mock()}
                
                await service._execute_scraping(TEST_JOB_ID, TEST_SITE, TEST_QUERY, TEST_MAX_PAGES)
        
        # Verify cancellation handling
        calls = service.db.update_job_status.call_args_list
        cancelled_call = [call for call in calls if call[0][1] == 'cancelled']
        assert len(cancelled_call) > 0
        assert service.session_stats['failed_jobs'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_scraping_no_products_found(self, scraper_service_with_mocks, mock_scraping_orchestrator):
        """Test scraping execution when no products are found."""
        service = scraper_service_with_mocks
        # Set _scrape_query to return empty list instead of None
        mock_scraping_orchestrator._scrape_query = Mock(return_value=[])
        
        with patch.object(service, '_get_browser_from_pool', return_value=mock_scraping_orchestrator):
            with patch.object(service, '_return_browser_to_pool'):
                service.running_jobs[TEST_JOB_ID] = {'task': Mock()}
                
                await service._execute_scraping(TEST_JOB_ID, TEST_SITE, TEST_QUERY, TEST_MAX_PAGES)
        
        # Verify completion with no products
        calls = service.db.update_job_status.call_args_list
        completed_call = [call for call in calls if call[0][1] == 'completed']
        assert len(completed_call) > 0
        assert service.session_stats['successful_jobs'] == 1
        assert service.session_stats['total_products'] == 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_get_job_status_database_error(self, scraper_service_with_mocks):
        """Test handling database errors in get_job_status."""
        service = scraper_service_with_mocks
        service.db.get_session.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            service.get_job_status(TEST_JOB_ID)
    
    def test_list_jobs_database_error(self, scraper_service_with_mocks):
        """Test handling database errors in list_jobs."""
        service = scraper_service_with_mocks
        service.db.get_session.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            service.list_jobs()
    
    @pytest.mark.asyncio
    async def test_start_scraping_job_database_error(self, scraper_service_with_mocks):
        """Test handling database errors in start_scraping_job."""
        service = scraper_service_with_mocks
        service.db.create_job.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            await service.start_scraping_job(TEST_SITE, TEST_QUERY, TEST_MAX_PAGES)


class TestAsyncLocking:
    """Test async lock behavior in browser pool operations."""
    
    @pytest.mark.asyncio
    async def test_browser_pool_concurrent_access(self, scraper_service_with_mocks, mock_scraping_orchestrator):
        """Test that browser pool handles concurrent access properly."""
        service = scraper_service_with_mocks
        service._browser_pool = [mock_scraping_orchestrator]
        
        # Create a second mock for the new browser creation
        mock_orchestrator_2 = Mock()
        
        with patch.object(service, '_create_browser_async', return_value=mock_orchestrator_2):
            # Simulate concurrent access
            async def get_browser():
                return await service._get_browser_from_pool()
            
            # Both calls should complete without deadlock
            results = await asyncio.gather(get_browser(), get_browser())
            
            # One should get the existing browser, one should create new
            assert len(results) == 2
            assert results[0] is not None
            assert results[1] is not None
            # One should be the pooled browser, one should be the newly created one
            assert mock_scraping_orchestrator in results
            assert mock_orchestrator_2 in results