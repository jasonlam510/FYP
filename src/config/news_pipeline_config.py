# Feature mapping configuration for raw input fields
FEATURE_MAPPING = {
    'title': ['title', 'headline', 'head'],
    'date': ['date', 'pubdate', 'published', 'timestamp'],
    'guid': ['guid'],
    'link': ['link', 'url', 'source_url'],
    'description': ['description', 'content', 'body', 'text']
}

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

# Required columns after mapping
REQUIRED_COLUMNS = [
    'title',
    'date',
    'guid',
    'link',
    'description'
]

# EVENT_TYPES = [
#     "Earnings announcement",
#     "M&A / Corporate transaction",
#     "Dividend declaration",
#     "Guidance / Outlook revision",
#     "Regulatory / Legal action",
#     "Macroeconomic release",
#     "Monetary policy decision",
#     "CEO / Management change",
#     "Product launch / Innovation",
#     "Supply-chain / Production news",
#     "Credit / Ratings change",
#     "Scandal / Fraud / Lawsuit",
#     "Analyst upgrade/downgrade",
#     "Sector-wide news",
#     "Geopolitical event",
#     "other"
# ]