import os
import pandas as pd
import json
import signal
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() ))
from src.llm.google_genai_client import generate_response
from tqdm import tqdm  # Import tqdm for progress bar
import logging

# Configure logging
log_file_path = os.path.join('src', 'llm', 'logs', 'bbc_news.log')
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_file_path,  # Specify the log file
                    filemode='a')  # Set to append mode
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# Reference market for relevance scoring, e.g. "S&P 500" (fallback if not set)
MARKET_REFERENCE = os.getenv('MARKET_REFERENCE', 'S&P 500')
SAVE_INTERVAL = 500

# Save the CSV when interrupted
def save_on_interrupt(signal, frame, df, output_csv):
    print("Saving progress before exit...")
    df.to_csv(output_csv, index=False)
    print(f"Annotated CSV saved to {output_csv}")
    exit(0)

def build_prompt(content: str, content_name: str = "news") -> str:
    """
    Construct a generic prompt for Gemini based on arbitrary news content.

    Args:
        content: The text of the news item (headline, article, description, etc.).
        content_name: A label describing the type of content (e.g., "headline", "article", "description").
    """
    return (
        f"You are a financial‐news analysis assistant. Given only the {content_name} of the news, you must output a valid JSON object with exactly these four fields (no extra keys, no prose):\n"
        f"- sentiment_score: float between -1 (very negative) and 1 (very positive)  \n"
        f"- relevance_score: float between 0 (irrelevant) and 1 (highly relevant to the {MARKET_REFERENCE})  \n"
        "- event_importance: float between 0 (no market impact) and 1 (major market-moving event)  \n"
        "- event_type: one of [\"earnings\", \"merger\", \"dividend\", \"guidance\", \"regulatory\", \"macroeconomic\", \"monetary\", \"CEO change\", \"product launch\", \"supply-chain\", \"credit\", \"scandal\", \"analyst\", \"sector-wide\", \"geopolitical\", \"other\"]\n\n"
        f"**{content_name.capitalize()}:** \"{content}\"\n\n"
        "**Output only JSON.**"
    )


def parse_response(response_text: str) -> dict:
    """
    Parse the JSON content from the API response text, stripping markdown/code fences if present.
    Returns the raw dict of parsed JSON.
    """
    # Remove Markdown or code fences
    cleaned = response_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}\nRaw response: {response_text}")
    if not isinstance(data, dict):
        raise ValueError(f"Parsed JSON is not an object: {data}")
    return data

def annotate_content(input_csv: str, output_csv: str,
                     model: str = "gemini-2.0-flash-lite",
                     temperature: float = 0.0,
                     max_output_tokens: int = 100,
                     top_p: float = 0.1,
                     top_k: int = 1,
                     content_name: str = "headline"):
    """
    Load a CSV containing a column of arbitrary news content, send each to Gemini,
    parse the JSON response, and append the results as new columns.

    The script will save the annotated CSV only once after processing all rows,
    and will save when interrupted.
    """

    # Set up the signal handler for keyboard interrupt (Ctrl+C)
    signal.signal(signal.SIGINT, lambda signal, frame: save_on_interrupt(signal, frame, df, output_csv))

    # Check if the output CSV already exists
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        # Only process rows with null values in the result columns
        unprocessed_rows = df[df[['sentiment_score', 'relevance_score', 'event_importance', 'event_type']].isnull().any(axis=1)]
        start_idx = unprocessed_rows.index.min()  # Find the first unprocessed row
    else:
        # Copy source CSV and insert new columns
        df = pd.read_csv(input_csv)
        df[['sentiment_score', 'relevance_score', 'event_importance', 'event_type']] = None
        df.to_csv(output_csv, index=False)
        start_idx = 0  # Start from the first row

    # Process the unprocessed rows
    total_rows = len(df)
    logger.info(f"Total rows to process: {total_rows}")
    logger.info(f"Start index: {start_idx}")
    counter = 0

    # Use tqdm to create a progress bar
    for idx in tqdm(range(start_idx, total_rows), desc="Processing rows", unit="row"):
        row = df.iloc[idx]

        # Skip if the row is already processed
        if not row[['sentiment_score', 'relevance_score', 'event_importance', 'event_type']].isnull().any():
            logger.info(f"Row {idx} already processed")
            continue

        content = row.get(content_name, '') or row.get(content_name.capitalize(), '')
        if not isinstance(content, str) or not content.strip():
            continue

        prompt = build_prompt(content, content_name)
        try:
            resp_text = generate_response(
                model=model,
                input_text=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k
            )
            parsed = parse_response(resp_text)
            # Map expected fields if present
            df.at[idx, 'sentiment_score'] = parsed.get('sentiment_score')
            df.at[idx, 'relevance_score'] = parsed.get('relevance_score')
            df.at[idx, 'event_importance'] = parsed.get('event_importance')
            df.at[idx, 'event_type'] = parsed.get('event_type')
            counter += 1
            if counter % SAVE_INTERVAL == 0:
                df.to_csv(output_csv, index=False)
                logger.info(f"Annotated CSV saved to {output_csv}")

        except Exception as e:
            logger.error(f"Error processing row {idx} ('{content}'): {e}")
    
    # Save the annotated CSV once after all rows are processed
    df.to_csv(output_csv, index=False)
    logger.info(f"Annotated CSV saved to {output_csv}")

def run_on_kaggle():
    # Test with the bloomberg dataset
    input_csv = "data/processed/kaggle/missing_features.csv"
    output_csv = "data/processed/kaggle/llm_missing_features.csv"
    annotate_content(input_csv, output_csv, content_name="description")

if __name__ == '__main__':
    run_on_kaggle()
    # import argparse

    # parser = argparse.ArgumentParser(description='Annotate news content using Gemini API')
    # parser.add_argument('input_csv', help='Path to input CSV file')
    # parser.add_argument('output_csv', help='Path to output annotated CSV file')
    # parser.add_argument('--model', default='gemini-2.0-flash-lite', help='Gemini model name')
    # parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    # parser.add_argument('--max_output_tokens', type=int, default=100, help='Max tokens to generate')
    # parser.add_argument('--top_p', type=float, default=0.1, help='Nucleus sampling parameter')
    # parser.add_argument('--top_k', type=int, default=1, help='Top-k sampling parameter')
    # parser.add_argument('--content_name', default='headline', help='Column name for news content (e.g., headline, article, description)')
    # args = parser.parse_args()

    # annotate_content(
    #     input_csv=args.input_csv,
    #     output_csv=args.output_csv,
    #     model=args.model,
    #     temperature=args.temperature,
    #     max_output_tokens=args.max_output_tokens,
    #     top_p=args.top_p,
    #     top_k=args.top_k,
    #     content_name=args.content_name
    # )
