import pandas as pd

reuter_df = pd.read_parquet("hf://datasets/danidanou/Reuters_Financial_News/summ_financial_data.parquet.gzip")
bloomberg_df = pd.read_parquet("hf://datasets/danidanou/Bloomberg_Financial_News/bloomberg_financial_data.parquet.gzip")

reuter_df.columns = reuter_df.columns.str.lower()
bloomberg_df.columns = bloomberg_df.columns.str.lower()

# Lowercase the column names
columns_to_keep = ['date', 'headline', 'article']
reuter_df_cleaned = reuter_df[columns_to_keep]
bloomberg_df_cleaned = bloomberg_df[columns_to_keep]

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

reuter_df_cleaned = convert_datetime_column(reuter_df_cleaned, 'date')
bloomberg_df_cleaned = convert_datetime_column(bloomberg_df_cleaned, 'date')

conbined_df = pd.concat([reuter_df_cleaned, bloomberg_df_cleaned])

# Read the input CSV file
input_csv = pd.read_csv("data/finbert_sentiment_inserted.csv")

# Create a dictionary of headlines to articles from the fetched data
headline_to_article = dict(zip(conbined_df['headline'], conbined_df['article']))

# Update the articles in the input CSV
input_csv['article'] = input_csv['headline'].map(headline_to_article)

# Save back to the same file
input_csv.to_csv("data/finbert_sentiment_inserted.csv", index=False)

print("\nArticle restoration results:")
print(f"Total rows in input CSV: {len(input_csv)}")
print(f"Rows with restored articles: {input_csv['article'].notna().sum()}")
print(f"Rows still missing articles: {input_csv['article'].isna().sum()}") 