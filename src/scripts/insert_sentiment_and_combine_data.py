import pandas as pd
import asyncio

import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

from src.utils.sentiment import add_sentiment_scores
from src.utils.logger import setup_logging

logger = setup_logging(__name__)

async def combine_and_add_sentiment(csv1_path: str, csv2_path: str, text_column: str, output_path: str):
    """
    Combines two CSV files and adds sentiment analysis scores.
    
    Args:
        csv1_path (str): Path to the first CSV file
        csv2_path (str): Path to the second CSV file
        text_column (str): Name of the column containing text for sentiment analysis
        output_path (str): Path where the combined and processed CSV will be saved
    """
    try:
        # Read the CSV files
        logger.info(f"Reading CSV files: {csv1_path} and {csv2_path}")
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
        
        # Combine the dataframes
        logger.info("Combining dataframes")
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Remove duplicates if any
        logger.info("Removing duplicates")
        combined_df = combined_df.drop_duplicates()
        
        # Add sentiment scores
        logger.info("Adding sentiment scores")
        result_df = await add_sentiment_scores(combined_df, text_column)
        
        # Save the result
        logger.info(f"Saving results to {output_path}")
        result_df.to_csv(output_path, index=False)
        logger.info("Process completed successfully")
        
        return result_df
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

async def main():
    """x
    Main function to run the script.
    Example usage:
        python insert_sentiment_and_combine_data.py
    """
    # Example usage - replace these paths with your actual CSV paths
    csv1_path = "data/processed/reuters/cleaned.csv"
    csv2_path = "data/processed/bloomberg/sentiment_inserted_06.csv"
    text_column = "headline"  # Replace with your actual text column name
    output_path = "data/processed/combined_with_sentiment.csv"
    
    await combine_and_add_sentiment(csv1_path, csv2_path, text_column, output_path)

if __name__ == "__main__":
    asyncio.run(main()) 