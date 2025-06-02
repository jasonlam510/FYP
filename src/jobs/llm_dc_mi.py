import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

# Set job name for logging and plots
JOB_NAME = "llm_dc_mi"
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
    llm_df = news_data.get_llm_df()
    price_df = price_data.get_price_df()
    macro_df = macro_data.get_mi_df()

    # Add technical indicators to price data
    logger.info("Adding technical indicators...")
    price_df = calculate_technical_indicators(price_df, BALANCED_INDICATORS)
    price_dc_df = add_dc_event_features(price_df)

    # Combine price data with macro indicators
    logger.info("Combining price data with macro indicators...")
    price_macro_df = combine_mi_price(macro_df, price_df)

    # Dictionary to store results
    results = {}

    # 1. Train LSTM with LLM + Balanced TI + DC
    logger.info("Training LSTM with LLM + Balanced TI...")
    lstm_ti_model, lstm_ti_history, lstm_ti_pred, lstm_ti_actual, test_dates, lstm_ti_metrics, lstm_ti_params = train_model(
        llm_df, price_dc_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_lstm_model
    )
    results['LSTM-LLM-TI'] = {
        'predictions': lstm_ti_pred,
        'actual': lstm_ti_actual,
        'test_dates': test_dates,
        'metrics': lstm_ti_metrics,
        'best_params': lstm_ti_params
    }

    # 2. Train LSTM with LLM + Macro
    logger.info("Training LSTM with LLM + Macro...")
    lstm_macro_model, lstm_macro_history, lstm_macro_pred, lstm_macro_actual, test_dates, lstm_macro_metrics, lstm_macro_params = train_model(
        llm_df, price_macro_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_lstm_model
    )
    results['LSTM-LLM-Macro'] = {
        'predictions': lstm_macro_pred,
        'actual': lstm_macro_actual,
        'test_dates': test_dates,
        'metrics': lstm_macro_metrics,
        'best_params': lstm_macro_params
    }

    # 3. Train LSTM-CNN with LLM + Balanced TI + DC
    logger.info("Training LSTM-CNN with LLM + Balanced TI...")
    lstm_cnn_ti_model, lstm_cnn_ti_history, lstm_cnn_ti_pred, lstm_cnn_ti_actual, test_dates, lstm_cnn_ti_metrics, lstm_cnn_ti_params = train_model(
        llm_df, price_dc_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-LLM-TI'] = {
        'predictions': lstm_cnn_ti_pred,
        'actual': lstm_cnn_ti_actual,
        'test_dates': test_dates,
        'metrics': lstm_cnn_ti_metrics,
        'best_params': lstm_cnn_ti_params
    }

    # 4. Train LSTM-CNN with LLM + Macro
    logger.info("Training LSTM-CNN with LLM + Macro...")
    lstm_cnn_macro_model, lstm_cnn_macro_history, lstm_cnn_macro_pred, lstm_cnn_macro_actual, test_dates, lstm_cnn_macro_metrics, lstm_cnn_macro_params = train_model(
        llm_df, price_macro_df, SEQ_LENGTH, N_TRIALS, 
        build_model_fn=build_cnn_lstm_model
    )
    results['LSTM-CNN-LLM-Macro'] = {
        'predictions': lstm_cnn_macro_pred,
        'actual': lstm_cnn_macro_actual,
        'test_dates': test_dates,
        'metrics': lstm_cnn_macro_metrics,
        'best_params': lstm_cnn_macro_params
    }

    # Plot results
    predictions_list = [
        results['LSTM-LLM-TI']['predictions'],
        results['LSTM-LLM-Macro']['predictions'],
        results['LSTM-CNN-LLM-TI']['predictions'],
        results['LSTM-CNN-LLM-Macro']['predictions']
    ]
    line_names = [
        'LSTM-LLM-TI',
        'LSTM-LLM-Macro',
        'LSTM-CNN-LLM-TI',
        'LSTM-CNN-LLM-Macro'
    ]
    actual = results['LSTM-LLM-TI']['actual']  # All actual values should be the same
    test_dates = results['LSTM-LLM-TI']['test_dates']
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