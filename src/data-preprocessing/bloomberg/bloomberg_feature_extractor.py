"""
Feature Extractor for Bloomberg Headlines

This script extracts sentiment and event-related features from Bloomberg headlines
using Google's Generative AI API. It processes the dataset row by row and saves
progress after each successful extraction.
"""
import sys
sys.path.append('../../..') # src folder
print(sys.path)

# importing
from src.llm.google_genai_client import generate_response

import os
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetimecle

# Constants
CLEANED_DATASET_PATH = "data/processed/bloomberg_cleaned.csv"
FEATURE_DATASET_PATH = "data/processed/bloomberg_features.csv"
MODEL_NAME = "gemini-2.0-flash-lite"
TARGET_COLUMN = "Headline"
REQUESTS_PER_MINUTE = 4000000
ROWS_TO_PROCESS = 10  # Can be "ALL" or an integer number of rows to process

# Event type categories
EVENT_TYPES = [
    "Earnings announcement",
    "M&A / Corporate transaction",
    "Dividend declaration",
    "Guidance / Outlook revision",
    "Regulatory / Legal action",
    "Macroeconomic release",
    "Monetary policy decision",
    "CEO / Management change",
    "Product launch / Innovation",
    "Supply-chain / Production news",
    "Credit / Ratings change",
    "Scandal / Fraud / Lawsuit",
    "Analyst upgrade/downgrade",
    "Sector-wide news",
    "Geopolitical event",
    "other"
]

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_feature_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Add feature columns to the DataFrame if they don't exist."""
    feature_columns = [
        f"sentiment_score({model_name})",
        f"relevance_score({model_name})",
        f"event_importance({model_name})",
        f"event_type({model_name})"
    ]
    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None
    
    return df

def format_prompt(text: str, column_name: str) -> str:
    """Format the prompt with the given text from the target column."""
    event_types_str = "\n".join([f"- {event_type}" for event_type in EVENT_TYPES])
    
    return f"""
    You are a financial‐news analysis assistant. Given only a {column_name}, you must output a valid JSON object with **exactly** these four fields (no extra keys, no prose):
    
    - sentiment_score: float between -1 (very negative) and 1 (very positive)  
    - relevance_score: float between 0 (irrelevant) and 1 (highly relevant to the S&P 500 index)  
    - event_importance: float between 0 (no market impact) and 1 (major market-moving event)  
    - event_type: must be exactly one of the following categories:
{event_types_str}

    **{column_name}:** "{text}"

    **Output only JSON.** For example:

    {{
    "sentiment_score": 0.42,
    "relevance_score": 0.88,
    "event_importance": 0.75,
    "event_type": "Earnings announcement"
    }}
    """

def parse_model_response(response: str) -> Dict[str, Any]:
    """Parse the model's JSON response.
    
    Args:
        response (str): The raw response from the model
        
    Returns:
        Dict[str, Any]: Parsed JSON response with the required fields
        
    Raises:
        ValueError: If the response cannot be parsed or is missing required fields
    """
    try:
        # Clean the response string
        response = response.strip()
        
        # Try to parse the entire response as JSON first
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            result = json.loads(response[start_idx:end_idx])
        
        # Validate required fields
        required_fields = ['sentiment_score', 'relevance_score', 'event_importance', 'event_type']
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        # Validate field types
        if not isinstance(result['sentiment_score'], (int, float)):
            raise ValueError("sentiment_score must be a number")
        if not isinstance(result['relevance_score'], (int, float)):
            raise ValueError("relevance_score must be a number")
        if not isinstance(result['event_importance'], (int, float)):
            raise ValueError("event_importance must be a number")
        if not isinstance(result['event_type'], str):
            raise ValueError("event_type must be a string")
            
        # Validate value ranges
        if not -1 <= result['sentiment_score'] <= 1:
            raise ValueError("sentiment_score must be between -1 and 1")
        if not 0 <= result['relevance_score'] <= 1:
            raise ValueError("relevance_score must be between 0 and 1")
        if not 0 <= result['event_importance'] <= 1:
            raise ValueError("event_importance must be between 0 and 1")
            
        # Validate event_type
        if result['event_type'] not in EVENT_TYPES:
            raise ValueError(f"event_type must be exactly one of: {', '.join(EVENT_TYPES)}")
            
        return result
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        raise ValueError(f"Error processing response: {e}")

def process_dataset():
    """Process the dataset row by row and extract features."""
    # Calculate delay between requests based on RPM
    delay_between_requests = 60 / REQUESTS_PER_MINUTE
    
    # Check if feature dataset exists, if not create it
    if not os.path.exists(FEATURE_DATASET_PATH):
        logger.info(f"Creating new feature dataset from {CLEANED_DATASET_PATH}")
        df = pd.read_csv(CLEANED_DATASET_PATH)
        df = create_feature_columns(df, MODEL_NAME)
        df.to_csv(FEATURE_DATASET_PATH, index=False)
    else:
        logger.info(f"Loading existing feature dataset from {FEATURE_DATASET_PATH}")
        df = pd.read_csv(FEATURE_DATASET_PATH)
        df = create_feature_columns(df, MODEL_NAME)

    # Get column names for features
    sentiment_col = f"sentiment_score({MODEL_NAME})"
    relevance_col = f"relevance_score({MODEL_NAME})"
    importance_col = f"event_importance({MODEL_NAME})"
    event_type_col = f"event_type({MODEL_NAME})"

    # Find the last processed row
    last_processed_idx = -1
    for idx, row in df.iterrows():
        if pd.isna(row[sentiment_col]) or pd.isna(row[relevance_col]) or \
           pd.isna(row[importance_col]) or pd.isna(row[event_type_col]):
            last_processed_idx = idx - 1
            break
    
    # Calculate rows to process
    total_rows = len(df)
    if ROWS_TO_PROCESS == "ALL":
        rows_to_process = total_rows - (last_processed_idx + 1)
        logger.info(f"Processing all remaining rows: {rows_to_process}")
    else:
        try:
            rows_to_process = min(int(ROWS_TO_PROCESS), total_rows - (last_processed_idx + 1))
            logger.info(f"Processing next {rows_to_process} rows")
        except ValueError:
            logger.error(f"Invalid ROWS_TO_PROCESS value: {ROWS_TO_PROCESS}. Must be 'ALL' or an integer.")
            return

    # Process each row
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    logger.info(f"Starting feature extraction with {REQUESTS_PER_MINUTE} RPM (delay: {delay_between_requests:.2f}s)")
    
    for idx in range(last_processed_idx + 1, min(last_processed_idx + 1 + rows_to_process, total_rows)):
        row = df.iloc[idx]
        
        # Skip if all features are already extracted
        if pd.notna(row[sentiment_col]) and pd.notna(row[relevance_col]) and \
           pd.notna(row[importance_col]) and pd.notna(row[event_type_col]):
            skipped_count += 1
            continue

        # Skip if target text is missing
        if pd.isna(row[TARGET_COLUMN]):
            logger.warning(f"Skipping row {idx}: Missing {TARGET_COLUMN}")
            skipped_count += 1
            continue

        try:
            # Format prompt and get response
            prompt = format_prompt(row[TARGET_COLUMN], TARGET_COLUMN)
            response = generate_response(
                model=MODEL_NAME,
                input_text=prompt,
                temperature=0.0,
                max_output_tokens=100
            )

            # Parse response
            features = parse_model_response(response)

            # Update DataFrame
            df.at[idx, sentiment_col] = features['sentiment_score']
            df.at[idx, relevance_col] = features['relevance_score']
            df.at[idx, importance_col] = features['event_importance']
            df.at[idx, event_type_col] = features['event_type']

            # Save progress
            df.to_csv(FEATURE_DATASET_PATH, index=False)
            processed_count += 1
            
            # Log progress
            if processed_count % 10 == 0:  # Log every 10 processed rows
                logger.info(f"Progress: {processed_count} processed, {skipped_count} skipped, {error_count} errors")
            
            # Wait before next request
            time.sleep(delay_between_requests)

        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            error_count += 1
            # Save progress even if there's an error
            df.to_csv(FEATURE_DATASET_PATH, index=False)
            continue
    
    # Log final statistics
    logger.info(f"Feature extraction completed. Final stats:")
    logger.info(f"- Total rows processed in this run: {rows_to_process}")
    logger.info(f"- Successfully processed: {processed_count}")
    logger.info(f"- Skipped: {skipped_count}")
    logger.info(f"- Errors: {error_count}")
    logger.info(f"- Next run will start from row: {last_processed_idx + 1 + rows_to_process}")

def main():
    """Main function to run the feature extraction process."""
    try:
        logger.info("Starting feature extraction process")
        process_dataset()
        logger.info("Feature extraction completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 