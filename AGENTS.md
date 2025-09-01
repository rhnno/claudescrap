# Project: Product Sentiment Analysis for Business Insights

This project aims to scrape product information and sentiment data from reviews to uncover valuable business insights. By analyzing customer feedback, we can identify market trends, understand consumer preferences, and pinpoint areas for product improvement.

## Project Structure

The project is organized into the following directories:

- **`src/`**: Contains the core source code for the project.
  - **`src/scrapers/`**: Houses the web scraping scripts.
    - `product_scraper.py`: a script for scraping product information.
    - `review_scraper.py`: a script for scraping product reviews and ratings.
  - **`src/analysis/`**: Includes scripts and notebooks for data analysis.
    - `sentiment_analysis.py`: a script for performing sentiment analysis on the reviews.
    - `trend_analysis.ipynb`: a Jupyter Notebook for identifying market trends.
  - **`src/utils/`**: Contains utility functions used across the project.
    - `database.py`: a script for managing database connections and queries.
    - `helpers.py`: a script for general helper functions.
- **`data/`**: Stores the scraped data.
  - **`raw/`**: Contains the raw, unprocessed data from the scrapers.
    - `products.csv`: a CSV file with scraped product information.
    - `reviews.csv`: a CSV file with scraped review data.
  - **`processed/`**: Holds the cleaned and processed data ready for analysis.
    - `cleaned_reviews.csv`: a CSV file with preprocessed review text.
    - `sentiment_scores.csv`: a CSV file with sentiment scores for each review.
- **`config/`**: Includes configuration files.
  - `config.ini`: a configuration file for storing database credentials, API keys, and other settings.
- **`notebooks/`**: Contains Jupyter Notebooks for exploratory data analysis and reporting.
  - `exploratory_analysis.ipynb`: a notebook for initial data exploration.
  - `business_insights_report.ipynb`: a notebook for summarizing key findings and generating reports.
- **`tests/`**: Includes unit tests for the project's codebase.
  - `test_scrapers.py`: tests for the scraping scripts.
  - `test_analysis.py`: tests for the data analysis scripts.

## Agent Guidelines

To contribute to this project, please follow these guidelines:

1.  **Coding Conventions**: Adhere to the PEP 8 style guide for Python code. Use clear and descriptive variable and function names.
2.  **Scraping**: When developing scrapers, be mindful of the website's terms of service and use appropriate scraping etiquette, such as rate limiting and setting a user-agent.
3.  **Data Storage**: Store raw scraped data in the `data/raw/` directory. Processed and cleaned data should be saved in the `data/processed/` directory.
4.  **Analysis**: Perform data analysis in the `src/analysis/` directory. Use Jupyter Notebooks in the `notebooks/` directory for exploratory analysis and reporting.
5.  **Configuration**: Store all sensitive information, such as API keys and database credentials, in the `config/config.ini` file. Do not hardcode these values in the scripts.
6.  **Testing**: Write unit tests for all new functions and scripts. Place the tests in the `tests/` directory and ensure they pass before submitting any changes.

By following this structure and these guidelines, we can work together efficiently to achieve the project's goals.
