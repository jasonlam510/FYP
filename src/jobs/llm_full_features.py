import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

# Set job name for logging and plots
JOB_NAME = "llm_full_features"
os.environ['JOB_NAME'] = JOB_NAME

from src.data.news import NewsData
from src.data.price import PriceData
from src.data.mi import MIData
from src.model.train_model import train_model, plot_results, build_cnn_lstm_model, build_lstm_model
from src.utils.combine import combine_mi_price
from src.helper.ti import calculate_technical_indicators
from src.helper.dc import add_dc_event_features
from src.utils.logger import get_logger
from src.config import (
    SEQ_LENGTH,
    N_TRIALS,
    BALANCED_INDICATORS
)

logger = get_logger(__name__)

def print_best_hyperparameters(results):
    """Print the best hyperparameters for each model."""
    logger.info("\n=== Best Hyperparameters for Each Model ===")
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        for param_name, param_value in result['best_params'].items():
            logger.info(f"{param_name}: {param_value}")

def run_comparison():
    # Load data
    logger.info("Loading data...")
    macro_data = MIData()
    news_data = NewsData()
    price_data = PriceData()

    # Get datasets
    llm_df = news_data.get_llm_df()  # Get full LLM dataset with all features
    price_df = price_data.get_price_df()
    macro_df = macro_data.get_mi_df()

    # Add technical indicators to price data
    logger.info("Adding technical indicators...")
    price_df = calculate_technical_indicators(price_df, BALANCED_INDICATORS)
    
    # Add directional change features
    logger.info("Adding directional change features...")
    price_dc_df = add_dc_event_features(price_df)

    # Combine price data with macro indicators
    logger.info("Combining price data with macro indicators...")
    price_macro_df = combine_mi_price(macro_df, price_dc_df)

    # Dictionary to store results
    results = {}

    # 1. Train LSTM with all features
    logger.info("Training LSTM with all features...")
    lstm_full_model, lstm_full_history, lstm_full_pred, lstm_full_actual, test_dates, lstm_full_metrics, lstm_full_params = train_model(
        llm_df, price_macro_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_lstm_model
    )
    results['LSTM-Full-Features'] = {
        'predictions': lstm_full_pred,
        'actual': lstm_full_actual,
        'test_dates': test_dates,
        'metrics': lstm_full_metrics,
        'best_params': lstm_full_params
    }

    # 2. Train LSTM-CNN with all features
    logger.info("Training LSTM-CNN with all features...")
    lstm_cnn_full_model, lstm_cnn_full_history, lstm_cnn_full_pred, lstm_cnn_full_actual, test_dates, lstm_cnn_full_metrics, lstm_cnn_full_params = train_model(
        llm_df, price_macro_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-Full-Features'] = {
        'predictions': lstm_cnn_full_pred,
        'actual': lstm_cnn_full_actual,
        'test_dates': test_dates,
        'metrics': lstm_cnn_full_metrics,
        'best_params': lstm_cnn_full_params
    }

    # Plot results
    predictions_list = [
        results['LSTM-Full-Features']['predictions'],
        results['LSTM-CNN-Full-Features']['predictions']
    ]
    line_names = [
        'LSTM-Full-Features',
        'LSTM-CNN-Full-Features'
    ]
    actual = results['LSTM-Full-Features']['actual']  # All actual values should be the same
    test_dates = results['LSTM-Full-Features']['test_dates']
    metrics_list = [results[model]['metrics'] for model in line_names]

    plot_results(
        predictions_list=predictions_list,
        line_names=line_names,
        actual=actual,
        test_dates=test_dates,
        metrics_list=metrics_list,
        job_name=JOB_NAME
    )

    # Print best hyperparameters
    print_best_hyperparameters(results)

    # Log detailed metrics for each model
    for model_name, result in results.items():
        metrics = result['metrics']
        logger.info(f"\nDetailed Metrics for {model_name}:")
        logger.info(f"RMSE = ${metrics['RMSE']:.2f}, MAE = ${metrics['MAE']:.2f} on next-day close")
        logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        logger.info(f"DC Precision: {metrics['dc_precision']:.2%}")
        logger.info(f"DC Recall: {metrics['dc_recall']:.2%}")
        logger.info(f"DC Timing Error: {metrics['dc_timing_error']:.2f} days")

if __name__ == "__main__":
    run_comparison() 