"""
data_generator.py
-----------------
Generates time-series data with a regime shift (distribution change)
at a configurable sample index.

Patent: BR 10 2025 024406-3
Author: Joemerson da Silva Lima
License: Apache 2.0
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_regime_shift_data(
    num_samples=20000,
    shift_index=10000,
    seq_len=15,
    num_features=10,
    num_targets=10,       # corrigido: alinhado com num_features
    noise_pre=0.5,
    noise_post=1.5,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)

    # Pre-shift: padrão senoidal, ruído baixo
    t1 = np.arange(shift_index)
    X1 = (
        np.sin(t1[:, None] * 0.1) * 2
        + rng.standard_normal((shift_index, num_features)) * noise_pre
    )
    Y1 = (rng.standard_normal((shift_index, num_targets)) > 0.0).astype(int)

    # Post-shift: padrão cossenoidal, ruído maior
    n_post = num_samples - shift_index
    t2 = np.arange(n_post)
    X2 = (
        np.cos(t2[:, None] * 0.05) * 3
        + rng.standard_normal((n_post, num_features)) * noise_post
    )
    Y2 = (rng.standard_normal((n_post, num_targets)) > 0.5).astype(int)

    X_raw = np.concatenate([X1, X2], axis=0)
    Y_raw = np.concatenate([Y1, Y2], axis=0)

    # Sliding window
    data_X, data_Y = [], []
    for i in range(len(X_raw) - seq_len):
        data_X.append(X_raw[i: i + seq_len])
        data_Y.append(Y_raw[i + seq_len - 1])  # corrigido: último passo da janela

    data_X = np.array(data_X)
    data_Y = np.array(data_Y)

    # Normalizar X
    scaler = StandardScaler()
    shape = data_X.shape
    X_flat = scaler.fit_transform(data_X.reshape(-1, shape[-1]))
    data_X = X_flat.reshape(shape)

    print(
        f"[DATA] Generated. X: {data_X.shape}, Y: {data_Y.shape} "
        f"| Shift at sample {shift_index}"
    )
    return data_X, data_Y, scaler