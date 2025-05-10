import pandas as pd
import matplotlib.pyplot as plt

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
        display(duplicates)
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
    yearly_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Articles by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.show()


def plot_description_length_distribution(df: pd.DataFrame, column_name: str) -> None:
    """
    Plots the distribution of the length of a specified text column (in characters, including spaces and punctuation).

    Args:
    - df (pd.DataFrame): The pandas dataframe containing the column.
    - column_name (str): The name of the column to analyze.
    
    Returns:
    - None: Displays the plot.
    """
    # Calculate the length of each entry in the specified column
    description_length = df[column_name].apply(len)

    # Plot the distribution of the length of the descriptions
    plt.figure(figsize=(10, 6))
    ax = description_length.hist(bins=50, color='skyblue', edgecolor='black')
    
    # Add exact numbers on top of each bar
    for rect in ax.patches:
        height = rect.get_height()
        x_position = rect.get_x() + rect.get_width() / 2
        ax.text(x_position, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.title(f'Distribution of {column_name} Lengths')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.show()

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
