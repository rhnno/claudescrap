"""Database models and connection management for scraping operations.

This module provides SQLAlchemy models for storing scraping jobs and products,
along with a DatabaseManager class for handling database operations. It supports
PostgreSQL by default with automatic test environment detection.

Example:
    Basic usage of database operations::

        db = DatabaseManager()
        job = db.create_job("job_123", "tokopedia", "laptop")
        db.update_job_status("job_123", "running", current_page=2)
        db.save_products(products_list, "job_123")

Note:
    Uses PostgreSQL by default, with testing.postgresql for test environments.
    SQLAlchemy 2.0 compatible patterns with proper type annotations
    for MyPy compliance as specified in project requirements.
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from typing import Optional, List, Dict, Any

# Create base class for SQLAlchemy models
Base: Any = declarative_base()

class ScrapingJob(Base):
    """SQLAlchemy model for tracking scraping jobs.
    
    This model stores information about scraping jobs including their status,
    target site, search query, and progress metrics. It provides comprehensive
    tracking for job lifecycle management.
    
    Attributes:
        id (int): Primary key auto-increment ID
        job_id (str): Unique job identifier (UUID)
        status (str): Job status (pending, running, completed, failed)
        site (str): Target e-commerce site name
        query (str): Search query string
        total_pages (int): Total number of pages to scrape
        current_page (int): Current page being processed
        products_found (int): Number of products extracted
        created_at (datetime): Job creation timestamp
        completed_at (datetime): Job completion timestamp
        error_message (str): Error details if job failed
    
    Example:
        >>> job = ScrapingJob(
        ...     job_id="job_123",
        ...     site="tokopedia",
        ...     query="gaming laptop"
        ... )
    
    Note:
        Status field supports: 'pending', 'running', 'completed', 'failed'
        All timestamps use UTC timezone for consistency.
    """
    __tablename__ = 'scraping_jobs'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(50), unique=True, nullable=False)
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    site = Column(String(100), nullable=False)
    query = Column(String(200), nullable=False)
    total_pages = Column(Integer, default=0)
    current_page = Column(Integer, default=0)
    products_found = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)

class Product(Base):
    """SQLAlchemy model for storing scraped product data.
    
    This model stores individual product information extracted during
    scraping operations. Each product is linked to its parent job and
    includes comprehensive product details.
    
    Attributes:
        id (int): Primary key auto-increment ID
        job_id (str): Foreign key linking to scraping job
        name (str): Product name/title
        price (str): Product price as string (with currency)
        url (str): Product page URL
        site (str): Source e-commerce site
        query (str): Search query that found this product
        page_number (int): Page number where product was found
        scraped_at (datetime): When product was extracted
    
    Example:
        >>> product = Product(
        ...     job_id="job_123",
        ...     name="Gaming Laptop RTX 4060",
        ...     price="Rp 15.000.000",
        ...     url="https://tokopedia.com/product/123"
        ... )
    
    Note:
        Price is stored as string to preserve original formatting
        and currency symbols. URLs are stored as TEXT for length flexibility.
    """
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(50), nullable=False)
    name = Column(String(500), nullable=False)
    price = Column(String(100))
    url = Column(Text)
    site = Column(String(100))
    query = Column(String(200))
    page_number = Column(Integer)
    scraped_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Manages database connections and operations for scraping data.
    
    This class handles all database operations including job management,
    product storage, and session handling. It provides a high-level interface
    for database operations with proper resource management.
    
    Attributes:
        engine: SQLAlchemy engine instance
        SessionLocal: SQLAlchemy session factory
    
    Example:
        >>> db = DatabaseManager()
        >>> job = db.create_job("job_123", "tokopedia", "laptop")
        >>> db.update_job_status("job_123", "running", current_page=2)
        >>> db.save_products(products_list, "job_123")
    
    Note:
        Uses PostgreSQL by default, configurable via DATABASE_URL environment variable.
        Automatically creates tables on initialization.
    """
    def __init__(self, database_url: Optional[str] = None) -> None:
        """Initialize DatabaseManager with connection and table setup.
        
        Sets up the database connection, creates session factory, and
        ensures all tables are created.
        
        Args:
            database_url (Optional[str]): Database connection URL.
                Defaults to DATABASE_URL environment variable.
                Required for production use.
        
        Raises:
            Exception: If database connection or table creation fails
        
        Example:
            >>> # Use environment DATABASE_URL
            >>> db = DatabaseManager()
            >>> 
            >>> # Use custom database
            >>> db = DatabaseManager("postgresql://user:pass@localhost/scraping")
        
        Note:
            Requires DATABASE_URL environment variable to be set for production.
            In test environments, uses testing.postgresql automatically.
        """
        if not database_url:
            # Check if we're in a test environment
            if any(env_var in os.environ for env_var in ['CI', 'PYTEST_CURRENT_TEST', 'TESTING']):
                # In CI/test environments, DATABASE_URL should be set by CI or test fixtures
                database_url = os.getenv('DATABASE_URL')
                if not database_url:
                    raise ValueError("DATABASE_URL environment variable must be set")
            else:
                # Production environment - require DATABASE_URL
                database_url = os.getenv('DATABASE_URL')
                if not database_url:
                    raise ValueError(
                        "DATABASE_URL environment variable is required. "
                        "Example: postgresql://user:password@localhost:5432/scraping_db"
                    )
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        return self.SessionLocal()
    
    def create_job(self, job_id: str, site: str, query: str) -> ScrapingJob:
        session = self.get_session()
        try:
            job = ScrapingJob(job_id=job_id, site=site, query=query)
            session.add(job)
            session.commit()
            return job
        finally:
            session.close()
    
    def update_job_status(self, job_id: str, status: str, **kwargs: Any) -> None:
        """Update job status and additional attributes.
        
        Args:
            job_id (str): Unique job identifier
            status (str): New status value
            **kwargs (Any): Additional attributes to update
        """
        session = self.get_session()
        try:
            job = session.query(ScrapingJob).filter(ScrapingJob.job_id == job_id).first()
            if job:
                job.status = status  # type: ignore[assignment]
                for key, value in kwargs.items():
                    setattr(job, key, value)
                session.commit()
        finally:
            session.close()
    
    def save_products(self, products: List[Dict[str, Any]], job_id: str) -> None:
        session = self.get_session()
        try:
            for product_data in products:
                product = Product(
                    job_id=job_id,
                    name=product_data.get('name'),
                    price=product_data.get('price'),
                    url=product_data.get('url'),
                    site=product_data.get('site'),
                    query=product_data.get('query'),
                    page_number=product_data.get('page_number')
                )
                session.add(product)
            session.commit()
        finally:
            session.close()