"""
Google Generative AI (Gemini) API Handler

This script provides a function to interact with Google's Generative AI API.
It handles API key management through environment variables and provides
a simple interface for making API calls.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
import logging
logger = logging.getLogger(__name__)

# Get API key from environment variable
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_GENAI_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_GENAI_KEY not found in environment variables or .env file")

# Initialize the client
client = genai.Client(api_key=GOOGLE_API_KEY)

async def generate_response(
    model: str = "gemini-2.0-flash-lite",
    input_text: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 100,
    top_p: float = 0.95,
    top_k: int = 40,
    print_tokens: bool = True
) -> str:
    """
    Generate a response using Google's Generative AI API asynchronously.

    Args:
        model (str): The model to use (default: "gemini-2.0-flash-lite")
        input_text (str): The input text to generate a response for
        temperature (float): Controls randomness in the output (0.0 to 1.0)
        max_output_tokens (int): Maximum number of tokens to generate
        top_p (float): Nucleus sampling parameter (0.0 to 1.0)
        top_k (int): Number of highest probability tokens to consider

    Returns:
        str: The generated response

    Raises:
        Exception: If there's an error in the API call
    """
    try:
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=[input_text],
            config=config
        )
        if print_tokens:
            logger.info(response.usage_metadata)
        return response.text

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def generate_response_sync(
    model: str = "gemini-2.0-flash-lite",
    input_text: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 100,
    top_p: float = 0.95,
    top_k: int = 40,
    print_tokens: bool = False
) -> str:
    """
    Synchronous version of generate_response using the Google client library.
    """
    try:
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens
        )

        response = client.models.generate_content(
            model=model,
            contents=[input_text],
            config=config
        )
        if print_tokens:
            logger.info(response.usage_metadata)
        return response.text

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def main():
    """Example usage of the generate_response function."""
    try:
        # Example input
        test_input = """
        You are a financial‐news analysis assistant. Given only the Headline of the news, you must output a valid JSON object with exactly these three fields (no extra keys, no prose):
        - sentiment_score: float between -1 (very negative) and 1 (very positive)
        - relevance_score: float between 0 (irrelevant) and 1 (highly relevant to the S&P 500)  
        - event_importance: float between 0 (no market impact) and 1 (major market-moving event)  
        - event_type: one of ["earnings", "merger", "dividend", "guidance", "regulatory", "macroeconomic", "monetary", "CEO change", "product launch", "supply-chain", "credit", "scandal", "analyst", "sector-wide", "geopolitical", "other"]

        **Headline:** "Orix May Buy U.S. Asset Manager to Enter Equity Market"

        **Output only JSON.**
        """
        
        # Generate response with custom parameters
        response = generate_response_sync(
            model="gemini-2.0-flash-lite",
            input_text=test_input,
            temperature=0.0,  # Keep deterministic for consistent JSON output
            max_output_tokens=100,
            top_p=0.1,  # Lower top_p for more focused/precise JSON responses
            top_k=1,  # Minimal top_k since we want exact JSON format
            print_tokens=False
        )
        
        print(response)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 