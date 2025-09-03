ClaudeScrap Documentation
=========================

Welcome to ClaudeScrap, a comprehensive web scraping framework designed for extracting, processing, and analyzing data from e-commerce websites with advanced anti-detection capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_reference
   modules
   examples

Overview
--------

ClaudeScrap is a personal data scraping initiative focused on web scraping, data analysis, and optional machine learning for business insights. It is designed to extract, process, and analyze data from the web in a structured and scalable manner.

Key Features
------------

* **Advanced Web Scraping**: Using Selenium and BeautifulSoup with anti-detection techniques
* **Browser Management**: Persistent Chrome profiles with automated login capabilities  
* **Data Analysis**: Built-in sentiment and trend analysis modules
* **API Integration**: FastAPI-based REST API for remote scraping operations
* **Database Support**: SQLAlchemy models with SQLite/PostgreSQL support
* **Performance Optimization**: Async browser pooling and concurrent job processing

Target Users
------------

* Data analysts
* Researchers  
* Business intelligence professionals
* Developers interested in web scraping and data mining

Quick Start
-----------

.. code-block:: python

   from src.ace import ScrapingOrchestrator
   
   # Initialize the orchestrator
   orchestrator = ScrapingOrchestrator()
   orchestrator.setup_browser(headless=True)
   
   # Configure scraping
   config = {
       "sites": [{
           "name": "tokopedia", 
           "queries": ["laptop gaming"],
           "max_pages": 5
       }],
       "output_format": ["csv", "excel"]
   }
   
   # Run scraping
   results = orchestrator.run_batch_scraping(config)

Installation
------------

1. Install Python 3.10+
2. Install Chrome and ChromeDriver
3. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r requirements_api.txt
   pip install -r requirements_docs.txt

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`