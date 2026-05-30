"""
model.py
--------
LSTM model factory for EIE experiments.

Patent: BR 10 2025 024406-3
Author: Joemerson da Silva Lima
License: Apache 2.0
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense


def create_lstm_model(
    seq_len,
    num_features,
    num_targets,
    units_lstm=64,
    dropout_rate=0.2,
    learning_rate=1e-3,
):
    """
    Create and compile a single-layer LSTM model compatible with EIECallback.

    The LSTM layer is named 'lstm_core' — EIECallback uses this name to
    locate and extract the latent vector during training.

    Parameters
    ----------
    seq_len : int
        Sequence length (window size).
    num_features : int
        Number of input features per time step.
    num_targets : int
        Number of binary output targets.
    units_lstm : int
        Number of LSTM units (default: 64).
    dropout_rate : float
        Dropout rate after LSTM layer (default: 0.2).
    learning_rate : float
        Adam optimizer learning rate (default: 1e-3).

    Returns
    -------
    model : tf.keras.Model
    """
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        LSTM(units_lstm, return_sequences=False, name="lstm_core"),
        Dropout(dropout_rate),
        Dense(num_targets, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model
