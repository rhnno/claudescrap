"""Database models and connection management"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from typing import Optional, List, Dict, Any

# Create base class for SQLAlchemy models
Base: Any = declarative_base()

class ScrapingJob(Base):
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
    def __init__(self, database_url: Optional[str] = None) -> None:
        if not database_url:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///scraping.db')
        
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
        session = self.get_session()
        try:
            job = session.query(ScrapingJob).filter(ScrapingJob.job_id == job_id).first()
            if job:
                job.status = status
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