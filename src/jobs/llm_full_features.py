from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
project_root = Path.cwd() # Get the current directory
sys.path.append(str(project_root))

# Set job name for logging and plots
START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ['START_TIME'] = START_TIME
JOB_NAME = f"llm_full_features"
os.environ['JOB_NAME'] = JOB_NAME

from src.data.news import NewsData
from src.data.price import PriceData
from src.data.mi import MIData
from src.model.train_model import train_model, plot_results, build_cnn_lstm_model, build_lstm_model
from src.utils.combine import combine_mi_price, aggregate_news_price_rolling_llm
from src.helper.ti import calculate_technical_indicators
from src.helper.dc import add_dc_event_features
from src.utils.logger import get_logger
from src.config import (
    SEQ_LENGTH,
    N_TRIALS,
    BALANCED_INDICATORS,
    HALF_LIFE_RANGE,
    N_DAYS_RANGE
)

logger = get_logger(JOB_NAME)

def print_best_hyperparameters(results):
    """Print the best hyperparameters for each model."""
    logger.info("\n=== Best Hyperparameters for Each Model ===")
    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        for param_name, param_value in result['best_params'].items():
            logger.info(f"{param_name}: {param_value}")

def run_comparison(use_existing_storage=False):
    # Load data
    logger.info("Loading data...")
    macro_data = MIData()
    news_data = NewsData()
    price_data = PriceData()

    # Get datasets
    llm_df = news_data.get_llm_df()  # Get full LLM dataset with all features
    price_df = price_data.get_price_df()
    macro_df = macro_data.get_mi_df()

    # Process price data once before training
    logger.info("Processing price data...")
    # Add technical indicators to price data
    price_df = calculate_technical_indicators(price_df, BALANCED_INDICATORS)
    
    # Add directional change features
    price_dc_df = add_dc_event_features(price_df)

    # Combine price data with macro indicators
    processed_price_df = combine_mi_price(macro_df, price_dc_df)

    # Dictionary to store results
    results = {}

    # Define sub-jobs
    sub_jobs = {
        'lstm_full': {
            'model_fn': build_lstm_model,
            'news_df': llm_df,
            'price_df': processed_price_df
        },
        'lstm_cnn_full': {
            'model_fn': build_cnn_lstm_model,
            'news_df': llm_df,
            'price_df': processed_price_df
        }
    }

    # Train each sub-job
    for sub_job_name, config in sub_jobs.items():
        logger.info(f"\nTraining {sub_job_name}...")
        os.environ['SUB_JOB_NAME'] = sub_job_name
        
        model, history, pred, actual, test_dates, metrics, params = train_model(
            news_df=config['news_df'],
            price_df=config['price_df'],
            seq_length=SEQ_LENGTH,
            n_trials=N_TRIALS,
            build_model_fn=config['model_fn'],
            half_life_range=HALF_LIFE_RANGE,
            n_days_range=N_DAYS_RANGE,
            use_existing_storage=use_existing_storage
        )
        
        results[sub_job_name] = {
            'predictions': pred,
            'actual': actual,
            'test_dates': test_dates,
            'metrics': metrics,
            'best_params': params
        }

    # Plot results
    predictions_list = [results[model]['predictions'] for model in sub_jobs.keys()]
    line_names = list(sub_jobs.keys())
    actual = results['lstm_full']['actual']  # All actual values should be the same
    test_dates = results['lstm_full']['test_dates']
    metrics_list = [results[model]['metrics'] for model in sub_jobs.keys()]

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
    try:
        # You can set use_existing_storage=True to use existing Optuna storage
        run_comparison(use_existing_storage=True)
    except Exception as e:
        logger.error(f"Error in {JOB_NAME}: {e}")
        raise e