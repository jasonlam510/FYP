import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

current_dir = Path.cwd()
project_root = current_dir
sys.path.append(str(project_root))
from src.config.news_pipeline_config import FEATURE_MAPPING, EVENT_TYPES, REQUIRED_COLUMNS
from src.utils.data_format import convert_datetime_column, clean_text_column
from src.config.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)

class NewsPipeline:
    def __init__(self):
        """
        Initialize the news pipeline with feature mapping.
        """
        self.feature_mapping = FEATURE_MAPPING
        self.event_types = EVENT_TYPES
        self.required_columns = REQUIRED_COLUMNS
        logger.info("NewsPipeline initialized")

    def _map_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map input feature names to standardized names.
        
        Args:
            df (pd.DataFrame): Input news data
            
        Returns:
            pd.DataFrame: DataFrame with standardized feature names
        """
        logger.info("Mapping features to standardized names...")
        
        # Lowercase the column names in the DataFrame
        df.columns = df.columns.str.lower()
        
        # Create mapping dictionary for each standardized name
        mapping_dict = {}
        for std_name, possible_names in self.feature_mapping.items():
            for name in possible_names:
                if name in df.columns:
                    mapping_dict[name] = std_name
                    break
        
        # Rename columns
        df = df.rename(columns=mapping_dict)
        
        # Check for missing required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns after mapping: {missing_cols}")
            raise ValueError(f"Missing required columns after mapping: {missing_cols}")
        
        logger.info("Feature mapping completed")
        return df

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate the input DataFrame schema and data types.
        Uses data_format utilities to standardize data formats.
        
        Args:
            df (pd.DataFrame): Input news data
            
        Raises:
            ValueError: If required columns are missing or data types are invalid
        """
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate and standardize date column
        try:
            df = convert_datetime_column(df, 'date')
            logger.info("Successfully converted date column to datetime format")
        except Exception as e:
            logger.error(f"Failed to convert date column: {str(e)}")
            raise ValueError("Column 'date' must be convertible to datetime")
        
        # Clean text columns
        text_columns = ['title', 'description']
        for col in text_columns:
            if col in df.columns:
                df = clean_text_column(df, col)
                logger.info(f"Cleaned text column: {col}")
        
        logger.info("Input validation and standardization completed successfully")

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from raw news data.
        TODO: Implement actual feature extraction logic.
        
        Args:
            df (pd.DataFrame): Raw news data with standardized column names
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        logger.info("Feature extraction is currently a placeholder - TODO")
        return df

    def fit_transform(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw news data and calculate features.
        
        Expected columns in news_df (will be mapped to standard names):
            - title/headline/head (str)
            - date/pubDate/published/timestamp (datetime or str)
            - guid (str)
            - link/url/source_url (str)
            - description/content/body/text (str)
            
        Returns:
            DataFrame with processed features
        """
        logger.info("Starting news data processing...")
        
        # Map features to standardized names
        df = self._map_features(news_df)
        
        # Validate input and standardize formats
        self._validate_input(df)
        
        # Extract features (currently just returns the mapped DataFrame)
        df = self._extract_features(df)
        
        logger.info(f"News data processing completed. Final shape: {df.shape}")
        return df

def run_news_pipeline(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end pipeline for news data processing.
    
    Args:
        news_df (pd.DataFrame): Raw news data
        
    Returns:
        pd.DataFrame: Processed news data with features
    """
    logger.info("Starting news pipeline...")
    pipeline = NewsPipeline()
    processed_df = pipeline.fit_transform(news_df)
    logger.info("News pipeline completed successfully")
    return processed_df

def test():
    """
    Test function to demonstrate pipeline usage.
    """
    # Load processed news data from CSV
    news_df = pd.read_csv('data/processed/kaggle/feature_extracted_data.csv')
    
    # Run pipeline
    processed_df = run_news_pipeline(news_df)
    
    logger.info("\nProcessed DataFrame Info:")
    logger.info(f"Shape: {processed_df.shape}")
    logger.info("\nColumns:")
    logger.info(processed_df.columns.tolist())
    logger.info("\nFirst few rows:")
    logger.info(processed_df.head())
    
    # Save processed DataFrame to CSV
    output_path = 'data/pipeline/feature_extracted_data.csv'
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Processed DataFrame saved to {output_path}")

if __name__ == "__main__":
    test()


