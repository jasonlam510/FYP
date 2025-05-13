from openai import OpenAI
from dotenv import dotenv_values, load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai
from openai import RateLimitError, APIError, Timeout, APIConnectionError

load_dotenv()
client = OpenAI()

"""
gpt-4.1-mini    0.4
gpt-4.1-nano    0.10
gpt-4o-mini     0.15
"""

def classify_category_by_gpt(
    model: str = "gpt-4o-mini",
    input: str = "",
) -> str:
    """
    Classifies the category of the input text using a GPT model.
    Retries automatically on rate limit and transient API errors with exponential backoff (up to 6 attempts).

    Args:
        model (str): The GPT model to use for classification. Available options:
            - "gpt-4.1-mini": Lightweight model with 0.4B parameters
            - "gpt-4.1-nano": Ultra-lightweight model with 0.10B parameters
            - "gpt-4o-mini": Optimized mini model with 0.15B parameters
        input (str): The input text that needs to be classified.

    Returns:
        str: The classification result from the GPT model.

    Example:
        >>> result = classify_category_by_gpt(
        ...     model="gpt-4.1-mini",
        ...     input="This is a news article about sports."
        ... )
        >>> print(result)
        "sports"
    """
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
           retry=(lambda exc: isinstance(exc, (RateLimitError, APIError, APIConnectionError, Timeout))))
    def _call():
        response = client.responses.create(
            model=model,
            input=input,
        )
        return response.output_text
    return _call()

text =  '''
You are a financial‐news analysis assistant. Given only a headline, you must output a valid JSON object with **exactly** these four fields (no extra keys, no prose):

- sentiment_score: float between -1 (very negative) and 1 (very positive)  
- relevance_score: float between 0 (irrelevant) and 1 (highly relevant to the S&P 500 index)  
- event_importance: float between 0 (no market impact) and 1 (major market-moving event)  
- event_type: one of ["earnings", "merger", "regulatory", "macroeconomic", "scandal", "other"]

**Headline:** "Russia, Ukraine End Dispute That Cut Gas Supplies"

**Output only JSON.** For example:

{
  "sentiment_score": 0.42,
  "relevance_score": 0.88,
  "event_importance": 0.75,
  "event_type": "earnings"
}
'''
res = classify_category_by_gpt(
    model="gpt-4o-mini",
    input=text,
)

print(res)
'''
'{\n  "sentiment_score": 0.6,\n  "relevance_score": 0.7,\n  "event_importance": 0.6,\n  "event_type": "macroeconomic"\n}'
'''
