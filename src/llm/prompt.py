"""
Prompt templates and utilities for the LLM system.
"""

EVENT_TYPES = [
    "earnings",
    "merger",
    "dividend",
    "guidance",
    "regulatory",
    "macroeconomic",
    "monetary",
    "CEO change",
    "product launch",
    "supply-chain",
    "credit",
    "scandal",
    "analyst",
    "sector-wide",
    "geopolitical",
    "other"
]

def news_summary_prompt(content: str,
                        content_name: str = "full text",
                        summary_length: str = "3–5 sentences",
                        tone: str = "objective and concise") -> str:
    """
    Construct a prompt for Gemini (or similar LLM) to summarize financial news.
    
    :param content: The full text of the news article.
    :param content_name: A label for the content (e.g. "headline", "press release", "full text").
    :param summary_length: Desired length of the summary.
    :param tone: Desired writing tone for the summary.
    """
    return (
        f"You are a financial‐news analysis expert. Given only the {content_name} of a news article, "
        f"you must output a clear and accurate summary that is {summary_length}, written in an "
        f"{tone} style.\n\n"
        f"**{content_name.capitalize()}:**\n\"\"\"\n{content}\n\"\"\"\n\n"
        f"**Instructions:**\n"
        f"- Focus on the key facts, figures, and events that would impact investors or the market.\n"
        f"- Do NOT include your own analysis or commentary—just the facts.\n"
        f"- Preserve any numerical data or dates verbatim.\n\n"
        f"**Output:**\nA {summary_length} summary in plain text."
    )


def event_features_propmt(content: str, content_name: str = "headline", market_reference: str = "S&P 500") -> str:
    """
    Construct a generic prompt for Gemini based on arbitrary news content.
    """
    return (
        f"You are a financial‐news analysis expert. Given only the {content_name} of the news, you must output a valid JSON object with exactly these four fields (no extra keys, no prose):\n"
        f"- sentiment_score: float between -1 (very negative) and 1 (very positive)  \n"
        f"- relevance_score: float between 0 (irrelevant) and 1 (highly relevant to the {market_reference})  \n"
        "- event_importance: float between 0 (no market impact) and 1 (major market-moving event)  \n"
        "- event_type: one of " + str(EVENT_TYPES) + "\n\n"
        f"**{content_name.capitalize()}:** \"{content}\"\n\n"
        "**Output only JSON.**"
    )