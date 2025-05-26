papers = [
    "A Black Swan event-based hybrid model for Indian stock markets’ trends prediction",
    "A CEP-driven framework for real-time news impact prediction on financial markets",
    "A Robust Predictive Model for Stock Price Prediction Using Deep Learning and Natural Language Processing",
    "A Time Series Analysis-Based Stock Price Prediction Using Machine Learning",
    "Breaking Down Financial News Impact A Novel AI Approach with Geometric Hypergraphs",
    "Detecting_Regime_Change_in_Computational",
    "Efficient_Integration_of_Multi-Order_Dynamics_and_",
    "LSTM based stock prediction using weighted and categorized financial news",
    "MATCC A Novel Approach for Robust Stock Price Prediction Incorporating Market Trends and Cross-time Correlations",
    "MERGE Multi-view Relationship Graph Network for Event-Driven Stock Movement Prediction",
    "Multi-View Graph Convolutional Networks for Relationship-Driven Stock Prediction",
    "Multi-source aggregated classification for stock price movement prediction",
    "Nowcasting directional change in high frequency FX markets",
    "Predicting the daily return direction of the stock market using hybrid machine learning algorithms",
    "Predicting Stock Market Trends Using Machine Learning and Deep Learning Algorithms Via Continuous and Binary Data; a Comparative Analysis",
    "Stock Market Forecasting Using Machine Learning Algorithms",
    "Stock Movement Prediction from Tweets and Historical Prices",
    "Stock market prediction based on statistical data using machine learning algorithms",
    "Stock price prediction using machine learning and deep learning frameworks",
    "Twitter mood predicts the stock market",
    "Using Structured Events to Predict Stock Price Movement"
]

import os
for paper in papers:
    query = paper.replace(" ", "+")
    print(f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={query}")
    input()
    os.system('clear')
