Getting Started
===============

This guide will help you get ClaudeScrap up and running quickly.

Prerequisites
-------------

Before installing ClaudeScrap, ensure you have:

* Python 3.10 or higher
* Google Chrome browser
* ChromeDriver (compatible with your Chrome version)

Installation
------------

1. **Clone the repository**:

.. code-block:: bash

   git clone <repository-url>
   cd claudescrap

2. **Create virtual environment**:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r requirements_api.txt

Configuration
-------------

1. **Set up credentials** (optional):

   Create ``config/login_credentials.json`` for automatic login:

.. code-block:: json

   {
       "tokopedia": {
           "email": "your-email@example.com",
           "password": "your-password",
           "login_url": "https://accounts.tokopedia.com/otp/c/page"
       }
   }

2. **Configure scraping settings**:

   Modify ``config/scraping_config.json`` for your needs:

.. code-block:: json

   {
       "sites": [
           {
               "name": "tokopedia",
               "queries": ["laptop", "smartphone"],
               "max_pages": 5,
               "scroll_depth": 3
           }
       ],
       "output_format": ["csv", "excel"],
       "delay_range": [1, 3]
   }

Basic Usage
-----------

**Command Line Interface**:

.. code-block:: bash

   python src/ace.py

**API Server**:

.. code-block:: bash

   python run_api.py

**Programmatic Usage**:

.. code-block:: python

   from src.services.scraper_service import ScraperService
   
   # Initialize service
   service = ScraperService()
   
   # Start scraping job
   job_id = await service.start_scraping_job(
       site='tokopedia',
       query='gaming laptop', 
       max_pages=3
   )
   
   # Check status
   status = service.get_job_status(job_id)
   print(f"Job status: {status['status']}")

Testing
-------

Run the test suite:

.. code-block:: bash

   pytest tests/

Run specific test categories:

.. code-block:: bash

   pytest tests/test_api.py -v
   pytest tests/test_integration.py -v