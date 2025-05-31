import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, Callable, Union
from src.utils.logger import get_logger
from src.utils.combine import aggregate_news_price_rolling_finbert, aggregate_news_price_rolling_llm
from src.config import EVENT_TYPES
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

logger = get_logger(__name__)

def create_sequences(data, seq_length, target_col_idx):
    """Create sequences for time series data."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X), np.array(y)

def prepare_data_for_model(df: pd.DataFrame, seq_length: int, test_size: float = 0.2) -> Tuple:
    """Prepare data for the model, handling datetime columns properly."""
    # Create a copy of the dataframe
    df = df.copy()
    
    # Store the datetime column separately
    datetime_col = df['date']
    
    # Get numeric columns (excluding datetime)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    # Scale the numeric data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Find the index of the 'close' column
    close_col_idx = list(numeric_cols).index('close')
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length, close_col_idx)
    
    # Split into train and test sets
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Get corresponding dates for test set
    test_dates = datetime_col[train_size + seq_length:].values
    
    return X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols

def build_cnn_lstm_model(trial, seq_length, n_features):
    """CNN+LSTM model with variable number of LSTM layers."""
    # CNN parameters
    cnn_filters = trial.suggest_int('cnn_filters', 32, 128)
    cnn_kernel = trial.suggest_int('cnn_kernel', 2, 5)
    cnn_dropout = trial.suggest_float('cnn_dropout', 0.1, 0.5)
    
    # LSTM parameters
    lstm_layers = trial.suggest_int('lstm_layers', 2, 5)
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    
    # Learning rate
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))
    model.add(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(cnn_dropout))
    
    # Add LSTM layers dynamically
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)
        model.add(LSTM(lstm_units, return_sequences=return_seq))
        model.add(Dropout(lstm_dropout))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def build_lstm_model(trial, seq_length, n_features, **kwargs):
    """LSTM model with variable number of LSTM layers."""
    # Hyperparameters from best trial
    lstm_layers = 5
    lstm_units = 67
    dropout_rate = 0.1294676223336333
    learning_rate = 0.005698251072263444

    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))

    # Add LSTM layers dynamically
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)
        units = lstm_units if i == 0 else lstm_units // 2
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def train_model(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    seq_length: int,
    n_trials: int = 20,
    build_model_fn: Callable = build_lstm_model,
    model_params: Dict[str, Any] = None,
    half_life_range: Union[tuple, float] = (0.5, 10.0),
    n_days_range: Union[tuple, int] = (1, 10)
) -> Tuple:
    """
    Train and evaluate the model with hyperparameter optimization.
    
    The function automatically detects whether the input data is from FinBERT or LLM based on the presence
    of the 'sentiment_score_finbert' column in the news_df. This detection is used to choose the appropriate
    aggregation function:
    - If 'sentiment_score_finbert' exists: Uses aggregate_news_price_rolling_finbert
    - Otherwise: Uses aggregate_news_price_rolling_llm
    
    Args:
        news_df: DataFrame containing news sentiment data. Must be either:
            - FinBERT data with columns: ['date', 'sentiment_score_finbert', 'sentiment_positive_finbert', 
                                        'sentiment_neutral_finbert', 'sentiment_negative_finbert']
            - LLM data with columns: ['date', 'sentiment_score_llm', 'relevance_score', 
                                    'event_importance', 'event_type']
        price_df: DataFrame with price data ['date', 'close', 'volume']
        seq_length: Length of input sequences for the model
        n_trials: Number of Optuna trials for hyperparameter optimization
        build_model_fn: Function to build the model (LSTM or CNN-LSTM)
        model_params: Additional parameters for the model
        half_life_range: Range for half-life days parameter in news aggregation
        n_days_range: Range for number of days parameter in news aggregation
    
    Returns:
        Tuple containing:
        - best_model: Trained model with best hyperparameters
        - history: Training history
        - predictions: Model predictions on test set
        - actual: Actual values from test set
        - test_dates: Dates corresponding to test set
    """
    start_time = time.time()
    logger.info("Starting model training process...")

    def objective(trial):
        trial_start_time = time.time()
        logger.info(f"Starting trial {trial.number + 1}/{n_trials}")

        # 1. Get half_life_days and n_days for this trial
        half_life_days = half_life_range if isinstance(half_life_range, float) else trial.suggest_float('half_life_days', half_life_range[0], half_life_range[1])
        n_days = n_days_range if isinstance(n_days_range, int) else trial.suggest_int('n_days', n_days_range[0], n_days_range[1])

        # 2. Aggregate data using the suggested half_life_days
        # Check if we're using FinBERT or LLM data based on the columns in news_df
        if 'sentiment_score_finbert' in news_df.columns:
            logger.info("Detected FinBERT data format - using FinBERT aggregation")
            aggregated_df = aggregate_news_price_rolling_finbert(
                news_df=news_df,
                price_df=price_df,
                n_days=n_days,
                half_life_days=half_life_days
            )
        else:
            logger.info("Detected LLM data format - using LLM aggregation")
            aggregated_df = aggregate_news_price_rolling_llm(
                news_df=news_df,
                price_df=price_df,
                event_types=EVENT_TYPES,
                n_days=n_days,
                half_life_days=half_life_days
            )

        # 3. Prepare data for the model
        X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols = prepare_data_for_model(
            aggregated_df, seq_length=seq_length
        )

        # 4. Split training data into train and validation sets
        val_size = int(len(X_train) * 0.2)
        X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
        y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]

        # 5. Build and train the model
        model = build_model_fn(trial, seq_length, X_train.shape[2], **(model_params or {}))
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        # 6. Return the best validation loss
        trial_time = time.time() - trial_start_time
        logger.info(f"Completed trial {trial.number + 1}/{n_trials} in {trial_time:.2f} seconds")
        return min(history.history['val_loss'])

    # 7. Run Optuna optimization
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    optuna_time = time.time() - start_time
    logger.info(f"Completed hyperparameter optimization in {optuna_time:.2f} seconds")

    # 8. After finding the best trial, rebuild the dataset with the best half_life_days
    best_half_life = study.best_params.get('half_life_days', half_life_range if isinstance(half_life_range, float) else half_life_range[0])
    n_days = study.best_params.get('n_days', n_days_range if isinstance(n_days_range, int) else n_days_range[0])
    
    # Use the appropriate aggregation function based on the data type
    if 'sentiment_score_finbert' in news_df.columns:
        best_aggregated_df = aggregate_news_price_rolling_finbert(
            news_df=news_df,
            price_df=price_df,
            n_days=n_days,
            half_life_days=best_half_life
        )
    else:
        best_aggregated_df = aggregate_news_price_rolling_llm(
            news_df=news_df,
            price_df=price_df,
            event_types=EVENT_TYPES,
            n_days=n_days,
            half_life_days=best_half_life
        )

    # 9. Prepare data for the final model
    X_train, X_test, y_train, y_test, scaler, test_dates, numeric_cols = prepare_data_for_model(
        best_aggregated_df, seq_length=seq_length
    )

    # 10. Train the final model with the best hyperparameters
    logger.info("Training final model with best hyperparameters...")
    final_model_start_time = time.time()
    best_model = build_model_fn(study.best_trial, seq_length, X_train.shape[2], **(model_params or {}))
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    history = best_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    final_model_time = time.time() - final_model_start_time
    logger.info(f"Completed final model training in {final_model_time:.2f} seconds")

    # 11. Make predictions
    logger.info("Making predictions on test set...")
    predictions = best_model.predict(X_test)
    close_col_idx = list(numeric_cols).index('close')
    pred_reshaped = np.zeros((len(predictions), len(numeric_cols)))
    pred_reshaped[:, close_col_idx] = predictions.flatten()
    predictions = scaler.inverse_transform(pred_reshaped)[:, close_col_idx]
    actual_reshaped = np.zeros((len(y_test), len(numeric_cols)))
    actual_reshaped[:, close_col_idx] = y_test
    actual = scaler.inverse_transform(actual_reshaped)[:, close_col_idx]

    # 12. Collect best hyperparameters
    best_params = study.best_params
    best_params['seq_length'] = seq_length

    # 13. Print the best hyperparameters found
    logger.info("Best hyperparameters found:")
    for k, v in best_params.items():
        logger.info(f"{k}: {v}")

    total_time = time.time() - start_time
    logger.info(f"Total training process completed in {total_time:.2f} seconds")

    return best_model, history, predictions, actual, test_dates

def plot_results(predictions_list, line_names, actual, test_dates, job_name=None):
    """Plot the results and save the plot to a file."""
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot actual price
    ax.plot(test_dates, actual, label='Actual', color='blue')

    # Plot each prediction with a different color
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    for pred, name, color in zip(predictions_list, line_names, colors):
        ax.plot(test_dates, pred, label=name, color=color)

    ax.set_title('Actual vs Predicted Stock Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plot_filename = f'plots/{job_name}_actual_vs_prediction.png'
    plt.savefig(plot_filename)
    plt.close() 