
import pandas as pd

import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

import src.pipeline.price_pipeline as price_pipeline
import src.pipeline.news_pipeline as news_pipeline
import src.pipeline.data_processor as data_processor

def main():
    # Load the data
    price_df = pd.read_csv('data/pipeline/price.csv')
    news_df = pd.read_csv('data/pipeline/news.csv')

    # Run the pipelines
    price_df = price_pipeline.run_price_pipeline(price_df)
    news_df = news_pipeline.run_news_pipeline(news_df)  
    data_processor.run_data_processor(price_df, news_df, output_csv_path='data/pipeline/sequences.csv')

if __name__ == "__main__":
    main()