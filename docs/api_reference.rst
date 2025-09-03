API Reference
=============

REST API Endpoints
------------------

The ClaudeScrap project provides a FastAPI-based REST API for remote scraping operations.

Authentication
~~~~~~~~~~~~~~

All API endpoints require JWT authentication. Include the token in the Authorization header:

.. code-block:: text

   Authorization: Bearer <your-jwt-token>

Endpoints
~~~~~~~~~

.. automodule:: api.scraping_api
   :members:
   :undoc-members:
   :show-inheritance:

Health Check
````````````

.. autofunction:: api.scraping_api.health_check

Start Scraping Job
``````````````````

.. autofunction:: api.scraping_api.start_scraping

Get Job Status
``````````````

.. autofunction:: api.scraping_api.get_job_status

Stop Scraping Job
`````````````````

.. autofunction:: api.scraping_api.stop_scraping_job

List Jobs
`````````

.. autofunction:: api.scraping_api.list_jobs

Request/Response Models
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: api.scraping_api.ScrapingRequest
   :members:

.. autoclass:: api.scraping_api.ScrapingResponse
   :members: