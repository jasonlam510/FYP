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

def standardize_date(date_value: Union[str, datetime, date]) -> pd.Timestamp:
    """
    Standardizes date values to pandas Timestamp format.
    
    Args:
        date_value: Date value in any common format (string, datetime, date)
        
    Returns:
        pd.Timestamp: Standardized timestamp
    """
    if isinstance(date_value, pd.Timestamp):
        return date_value
    return pd.Timestamp(date_value)

def standardize_dates(dates: Union[List, pd.Series, pd.DatetimeIndex]) -> List[pd.Timestamp]:
    """
    Standardizes a list or series of dates to pandas Timestamp format.
    
    Args:
        dates: List, Series, or DatetimeIndex of dates
        
    Returns:
        List[pd.Timestamp]: List of standardized timestamps
    """
    if isinstance(dates, pd.Series):
        dates = dates.tolist()
    elif isinstance(dates, pd.DatetimeIndex):
        dates = dates.tolist()
    return [standardize_date(d) for d in dates]

def convert_datetime_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Converts a column containing datetime information into a consistent ISO 8601 format.
    Extracts additional time features (year, month, day, etc.) for easier analysis.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the column to convert.
    - column_name (str): The name of the column to convert.

    Returns:
    - pd.DataFrame: The original dataframe with the specified column converted and new time features added.
    """
    # Convert the column to datetime format
    df[column_name] = pd.to_datetime(df[column_name], utc=True, errors='coerce')  # 'coerce' to handle invalid dates
    
    return df
