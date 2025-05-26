import pandas as pd
import re
import string
from typing import Union, List
from datetime import datetime, date

def clean_text_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Cleans a text column by removing punctuation, numbers, extra spaces, and converting to lowercase.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the column to clean.
    - column_name (str): The name of the column to clean.

    Returns:
    - pd.DataFrame: The original dataframe with the specified column cleaned.
    """
    
    
    # Apply the cleaning function and replace the old column with cleaned data
    df[column_name] = df[column_name].apply(clean_text)
    
    return df

# Function to clean the text
def clean_text(text: str) -> str:
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = text.strip()
    # Convert text to lowercase
    text = text.lower()
    return text

def convert_datetime_column(df: pd.DataFrame, column_name: str, target_tz: str = 'US/Eastern') -> pd.DataFrame:
    """
    Standardizes and converts a datetime column to ISO 8601 format with specified timezone.
    Handles both timezone-naive and timezone-aware data.

    Args:
        df (pd.DataFrame): The pandas dataframe containing the column to convert.
        column_name (str): The name of the column to convert.
        target_tz (str): Target timezone to convert to. Defaults to 'US/Eastern'.

    Returns:
        pd.DataFrame: The original dataframe with the specified column converted to ISO 8601 format.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert to datetime if not already
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    
    # Handle timezone conversion
    if df[column_name].dt.tz is None:
        # If timezone-naive, localize to target timezone
        df[column_name] = df[column_name].dt.tz_localize(target_tz, ambiguous='NaT', nonexistent='NaT')
    else:
        # If already timezone-aware, convert to target timezone
        df[column_name] = df[column_name].dt.tz_convert(target_tz)
    
    # Drop any rows with NaT (Not a Time) values due to DST transitions
    df = df.dropna(subset=[column_name])
    
    # Convert to ISO 8601 format
    df[column_name] = pd.to_datetime(df[column_name], format='ISO8601', errors='coerce')
    
    return df
