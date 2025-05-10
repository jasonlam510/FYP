import pandas as pd
import re

def extract_category(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Extracts categories from the specified column that contains URLs and adds them as a new column.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the column.
    - column_name (str): The name of the column containing the URLs (guid).

    Returns:
    - pd.DataFrame: The original dataframe with a new column 'category' containing the extracted categories.
    """
    # Function to extract category from a URL
    def get_category(guid: str) -> str:
        match = re.search(r'https://www\.bbc\.co\.uk/(\w+)', guid)
        if match:
            return match.group(1)
        print(f"Warning: No match found for guid: {guid}")
        return None
    
    # Apply the function to the column and create a new 'category' column
    df['category'] = df[column_name].apply(get_category)
    
    return df
