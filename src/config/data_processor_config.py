# Data frequency configuration
DATA_FREQUENCY = 'daily'  # Options: 'daily', 'hourly' (hourly to be implemented later)

# Event types for one-hot encoding
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

# News aggregation methods for daily data
NEWS_AGGREGATION = {
    'sentiment_score': 'mean',  # Average sentiment for the day
    'relevance_score': 'mean',  # Average relevance for the day
    'event_importance': 'mean',  # Average importance for the day
    'event_type': 'count'  # Count of events by type for the day
}
