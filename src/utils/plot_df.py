import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os

def display_duplicate_rows_by_column(df: pd.DataFrame, date_column: str) -> None:
    """
    Displays the duplicate rows in a DataFrame based on a specified column.

    Args:
    - df (pd.DataFrame): The pandas dataframe to check for duplicates.
    - date_column (str): The name of the column to check for duplicates.
    
    Returns:
    - None: Displays the duplicate rows.
    """
    # Find duplicate rows based on the specified column
    duplicates = df[df.duplicated(subset=[date_column], keep=False)]

    # Display the duplicate rows
    if not duplicates.empty:
        print(duplicates)
    else:
        print(f"{date_column}: No duplicate rows found.")


def plot_distribution_by_year(df: pd.DataFrame, date_column: str, output_dir: str = 'plots') -> None:
    """
    Plots the distribution of articles by year based on a datetime column.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the data.
    - date_column (str): The name of the column containing the datetime information.
    - output_dir (str): Directory to save the plot image.
    
    Returns:
    - None: Displays the plot and saves it as PNG.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the date column to datetime format
    df[date_column] = pd.to_datetime(df[date_column], format='%a, %d %b %Y %H:%M:%S GMT')

    # Extract relevant time features: year, month, day
    year = df[date_column].dt.year

    # Get the distribution by year
    yearly_distribution = year.value_counts().sort_index()

    # Plot the distribution of articles by year
    plt.figure(figsize=(10, 6))
    ax = yearly_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Articles by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add exact numbers on top of each bar
    for i, v in enumerate(yearly_distribution):
        ax.text(i, v, f'{int(v)}', ha='center', va='bottom', fontsize=10)

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'yearly_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Yearly distribution plot saved to {os.path.join(output_dir, 'yearly_distribution.png')}")
    plt.show()
    plt.close()


def plot_description_length_distribution(df: pd.DataFrame, column_name: str, output_dir: str = 'plots') -> None:
    """
    Plots the distribution of the length of a specified text column (in number of words).

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the column.
    - column_name (str): The name of the column to analyze.
    - output_dir (str): Directory to save the plot image.
    
    Returns:
    - None: Displays the plot and saves it as PNG.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the number of words in each entry in the specified column
    word_counts = df[column_name].astype(str).apply(lambda x: len(x.split()))

    # Plot the distribution of the number of words
    plt.figure(figsize=(10, 6))
    ax = word_counts.hist(bins=50, color='skyblue', edgecolor='black')

    plt.title(f'Distribution of {column_name} Word Counts')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{column_name}_word_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Word distribution plot saved to {os.path.join(output_dir, f'{column_name}_word_distribution.png')}")
    plt.show()
    plt.close()

def plot_sentiment_score_distribution(df: pd.DataFrame, sentiment_column: str, output_dir: str = 'plots') -> None:
    """
    Plots the distribution of sentiment scores (positive, neutral, negative) based on a specified sentiment column.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the sentiment scores.
    - sentiment_column (str): The name of the column containing the sentiment scores.
    - output_dir (str): Directory to save the plot image.
    
    Returns:
    - None: Displays the plot and saves it as PNG.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the distribution of sentiment scores
    plt.figure(figsize=(10, 6))
    df[sentiment_column].hist(bins=50, color='skyblue', edgecolor='black')

    plt.title(f'Distribution of {sentiment_column} Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{sentiment_column}_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Sentiment score distribution plot saved to {os.path.join(output_dir, f'{sentiment_column}_distribution.png')}")
    plt.show()
    plt.close()

def plot_avg_per_day_by_year(df: pd.DataFrame, date_column: str, output_dir: str = 'plots') -> None:
    """
    Plots the average number of data points per day for each year based on a datetime column.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the data.
    - date_column (str): The name of the column containing the datetime information.
    - output_dir (str): Directory to save the plot image.
    
    Returns:
    - None: Displays the plot and saves it as PNG.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the date column to datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract year and date
    df['year'] = df[date_column].dt.year
    df['date_only'] = df[date_column].dt.date

    # Count data points per day
    daily_counts = df.groupby(['year', 'date_only']).size().reset_index(name='count')

    # Calculate average per day for each year
    avg_per_day = daily_counts.groupby('year')['count'].mean()

    # Plot
    plt.figure(figsize=(10, 6))
    ax = avg_per_day.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.ylabel('Average Data Points per Day')
    plt.title('Average Number of Data Points per Day by Year')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add exact numbers on top of each bar
    for i, v in enumerate(avg_per_day):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'avg_per_day_by_year.png'), dpi=300, bbox_inches='tight')
    print(f"Average number of data points per day by year plot saved to {os.path.join(output_dir, 'avg_per_day_by_year.png')}")
    plt.show()
    plt.close()

    # Clean up temporary columns
    df.drop(['year', 'date_only'], axis=1, inplace=True, errors='ignore')
