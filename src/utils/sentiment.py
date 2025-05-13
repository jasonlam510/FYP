from transformers import pipeline
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

# Initialize the sentiment analysis pipeline with FinBERT
pipe = pipeline("text-classification", model="prosusai/finbert", top_k=None)

# Function to get sentiment scores and add them to the DataFrame
def add_sentiment_scores(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # Create empty lists to hold the sentiment scores
    positive_scores = []
    neutral_scores = []
    negative_scores = []
    
    # Use tqdm to track the progress of the loop
    for text in tqdm(df[column_name], desc="Processing Sentiment Scores", unit="text"):
        result = pipe(text)
        
        # Extract scores for positive, neutral, and negative sentiments
        positive_score = next(item['score'] for item in result[0] if item['label'] == 'positive')
        neutral_score = next(item['score'] for item in result[0] if item['label'] == 'neutral')
        negative_score = next(item['score'] for item in result[0] if item['label'] == 'negative')
        
        # Append the scores to the lists
        positive_scores.append(positive_score)
        neutral_scores.append(neutral_score)
        negative_scores.append(negative_score)
    
    # Add the scores as new columns to the DataFrame
    df['positive_score'] = positive_scores
    df['neutral_score'] = neutral_scores
    df['negative_score'] = negative_scores
    
    return df