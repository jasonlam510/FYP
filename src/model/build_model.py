from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_cnn_lstm_model(trial, seq_length, n_features):
    """CNN+LSTM model with variable number of LSTM layers."""

    # CNN parameters
    cnn_filters = trial.suggest_int('cnn_filters', 32, 128)
    cnn_kernel = trial.suggest_int('cnn_kernel', 2, 5)
    cnn_dropout = trial.suggest_float('cnn_dropout', 0.1, 0.5)
    
    # LSTM parameters
    lstm_layers = trial.suggest_int('lstm_layers', 2, 5)  # Let Optuna choose 1, 2, or 3 layers
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
        # Only the last LSTM layer should not return sequences
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
        units = lstm_units if i == 0 else lstm_units // 2  # Optional: halve units for deeper layers
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model