"""
Utility functions for plotting DataFrame statistics and distributions.

This module provides functions for visualizing and analyzing DataFrame statistics,
particularly focused on text data analysis and token distributions.
"""

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os
import seaborn as sns
from typing import List, Optional, Union
import tiktoken

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


def plot_distribution_by_year(df: pd.DataFrame, date_column: str) -> None:
    """
    Plots the distribution of articles by year based on a datetime column.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the data.
    - date_column (str): The name of the column containing the datetime information.
    
    Returns:
    - None: Displays the plot.
    """
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

    plt.show()
    plt.close()


def plot_description_length_distribution(df: pd.DataFrame, column_name: str) -> None:
    """
    Plots the distribution of the length of a specified text column (in number of words).

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the column.
    - column_name (str): The name of the column to analyze.
    
    Returns:
    - None: Displays the plot.
    """
    # Calculate the number of words in each entry in the specified column
    word_counts = df[column_name].astype(str).apply(lambda x: len(x.split()))

    # Plot the distribution of the number of words
    plt.figure(figsize=(10, 6))
    ax = word_counts.hist(bins=50, color='skyblue', edgecolor='black')

    plt.title(f'Distribution of {column_name} Word Counts')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    
    plt.show()
    plt.close()

def plot_sentiment_score_distribution(df: pd.DataFrame, sentiment_column: str) -> None:
    """
    Plots the distribution of sentiment scores (positive, neutral, negative) based on a specified sentiment column.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the sentiment scores.
    - sentiment_column (str): The name of the column containing the sentiment scores.
    
    Returns:
    - None: Displays the plot.
    """
    # Plot the distribution of sentiment scores
    plt.figure(figsize=(10, 6))
    df[sentiment_column].hist(bins=50, color='skyblue', edgecolor='black')

    plt.title(f'Distribution of {sentiment_column} Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    
    plt.show()
    plt.close()

def plot_avg_per_day_by_year(df: pd.DataFrame, date_column: str) -> None:
    """
    Plots the average number of data points per day for each year based on a datetime column.

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the data.
    - date_column (str): The name of the column containing the datetime information.
    
    Returns:
    - None: Displays the plot.
    """
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

    plt.show()
    plt.close()

    # Clean up temporary columns
    df.drop(['year', 'date_only'], axis=1, inplace=True, errors='ignore')

def plot_token_distribution(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    tokenizer_name: str = "cl100k_base",
    figsize: tuple = (15, 5),
    bins: int = 50,
    title: Optional[str] = None
) -> None:
    """
    Plot token distribution for specified columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing text columns
        columns (Union[str, List[str]]): Column name(s) to analyze
        tokenizer_name (str): Name of the tokenizer to use (default: "cl100k_base")
        figsize (tuple): Figure size (width, height)
        bins (int): Number of bins for histogram
        title (Optional[str]): Custom title for the plot
    
    Example:
        >>> plot_token_distribution(df, ['Headline', 'Article'])
    """
    # Convert single column to list
    if isinstance(columns, str):
        columns = [columns]
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    # Calculate token counts for each column
    token_counts = {}
    for col in columns:
        token_counts[col] = df[col].apply(lambda x: len(tokenizer.encode(str(x))))
    
    # Create figure
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    # Plot distributions
    for idx, (col, counts) in enumerate(token_counts.items()):
        sns.histplot(data=counts, bins=bins, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col} Tokens')
        axes[idx].set_xlabel('Number of Tokens')
        axes[idx].set_ylabel('Count')
        
        # Add summary statistics as text
        stats = counts.describe()
        stats_text = f"Mean: {stats['mean']:.1f}\nStd: {stats['std']:.1f}\nMax: {stats['max']:.0f}"
        axes[idx].text(0.95, 0.95, stats_text,
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set overall title if provided
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def print_token_statistics(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    tokenizer_name: str = "cl100k_base"
) -> None:
    """
    Print token statistics for specified columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing text columns
        columns (Union[str, List[str]]): Column name(s) to analyze
        tokenizer_name (str): Name of the tokenizer to use (default: "cl100k_base")
    
    Example:
        >>> print_token_statistics(df, ['Headline', 'Article'])
    """
    # Convert single column to list
    if isinstance(columns, str):
        columns = [columns]
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    # Calculate and print statistics for each column
    for col in columns:
        tokens = df[col].apply(lambda x: len(tokenizer.encode(str(x))))
        print(f"\n{col} Token Statistics:")
        print(tokens.describe())
