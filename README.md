# Event-based Stock Prediction

This is the BSCCS Final Year Project 2024-2025 "Event-based Stock Prediction" supervised by [Prof ZHANG, Qingfu](https://scholars.cityu.edu.hk/en/persons/qingfu-zhang(a25373cf-62a1-4697-ad08-43678bcbf3f2).html)

# Dataset

This project utilizes several comprehensive financial news datasets:

1. **Reuters Financial News Dataset** (2006-2013)
   - Contains 105,359 financial news articles
   - Includes headlines, article content, dates, and summaries
   - Covers a wide range of financial topics and market events
   - Source: [Reuters Financial News Dataset](https://huggingface.co/datasets/danidanou/Reuters_Financial_News)

2. **Bloomberg Financial News Dataset** (2006-2013)
   - Contains 446,762 financial news articles
   - Provides extensive coverage of financial markets and corporate news
   - Includes detailed article content and metadata
   - Source: [Bloomberg Financial News Dataset](https://huggingface.co/datasets/danidanou/Bloomberg_Financial_News)

3. **S&P 500 Dataset** (1927-2025)
   - File: `sap500.csv`
   - Description: Contains historical data for S&P 500 companies.
   - Source: [Kaggle's S&P 500 Historical Data](https://www.kaggle.com/datasets/paveljurke/s-and-p-500-gspc-historical-data/data)

These datasets will be used to train our prediction model, leveraging the rich information contained in financial news articles to identify market-moving events and their potential impact on stock prices.

# Archive

This folder contains the old scripts for data pre-processing and model training. These scripts are **spaghetti** - it's very hard to replicate my work. At first, I only used Python scripts for preprocessing because this project includes a system that automatically fetches and processes data, then saves it to the database for storing and visualizing. However, I couldn't really handle developing both parts simultaneously, so I switched to using Jupyter notebooks for data preprocessing and model training. At that moment, some data was already finished, like the features from LLM were already extracted.

During model training with Jupyter notebooks, I found that it's very hard to try different datasets: you have to modify the previous functions and run them again, which makes the process of model training inconsistent and hard to replicate.

Since I am currently participating in an exchange program at KTH Sweden and taking a course: DD2356 Methods in High Performance Computing, I am able to access the supercomputer - Dardel. To adapt to using Dardel for model training, I restructured the project and moved all the past scripts to the '/archive' folder.