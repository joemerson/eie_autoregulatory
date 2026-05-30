"""
eie_callback.py
---------------
EIE (Informational-Energetic Efficiency) Keras Callback.

Self-regulating training control for machine learning models.

Patent: BR 10 2025 024406-3
Author: Joemerson da Silva Lima
License: Apache 2.0
"""

import time
import zlib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM


class EIECallback(Callback):
    """
    Keras callback that monitors Informational-Energetic Efficiency (EIE)
    during training and stops training when efficiency drops below a
    threshold after a data regime change.

    The EIE ratio is defined as:
        EIE_ratio = Loss / (I × E)

    Where:
        Loss  = statistical loss (η) — model performance
        I     = informational cost — L2 norm of the latent vector
        E     = energetic proxy cost — batch processing time (seconds)

    Parameters
    ----------
    X_data : np.ndarray
        Full input dataset used for latent vector extraction.
    y_data : np.ndarray
        Full target dataset (not used directly, kept for future extensions).
    log_path : str
        Path for saving the CSV log file.
    shift_index : int
        Sample index at which the data regime change occurs.
        The stop condition only activates after this point.
    stop_threshold : float
        EIE_ratio below this value triggers training stop (default: 0.003).
    batch_size : int
        Must match the batch_size used in model.fit().
    compute_czip : bool
        If True, also computes compressibility (Czip) of the latent vector.
        Slower but provides richer audit information (default: False).
    min_loss_to_stop : float
        Minimum loss value required to trigger stop. Prevents early stopping
        when loss is already very small (default: 0.1).
    """

    def __init__(
        self,
        X_data,
        y_data,
        log_path,
        shift_index,
        stop_threshold=0.003,
        batch_size=64,
        compute_czip=False,
        min_loss_to_stop=0.1,
    ):
        super().__init__()
        self.X_data = X_data
        self.y_data = y_data
        self.log_path = log_path
        self.shift_index = shift_index
        self.stop_threshold = stop_threshold
        self.batch_size = batch_size
        self.compute_czip = compute_czip
        self.min_loss_to_stop = min_loss_to_stop

        self.log = []
        self.session_start = time.time()
        self.batch_start_time = 0.0
        self.current_epoch = 0
        self.global_batch_counter = 0
        self.latent_extractor = None

    # ── Lifecycle hooks ────────────────────────────────────────────────────────

    def on_train_begin(self, logs=None):
        """Initialize the latent extractor submodel."""
        try:
            _ = self.model.predict(self.X_data[:1], verbose=0)
            input_layer = self.model.inputs[0]
            lstm_layer = next(
                l for l in self.model.layers if isinstance(l, LSTM)
            )
            self.latent_extractor = Model(
                inputs=input_layer, outputs=lstm_layer.output
            )
            print("[EIE] Latent extractor initialized successfully.")
        except Exception as e:
            print(f"[EIE] Warning: could not initialize latent extractor: {e}")
            self.latent_extractor = None

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        """Record batch start time for energetic cost measurement."""
        self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        """Compute EIE metrics and apply stop condition if triggered."""
        batch_end_time = time.time()
        e_proxy_sec = batch_end_time - self.batch_start_time
        data_index = self.global_batch_counter * self.batch_size

        if data_index >= len(self.X_data):
            self.global_batch_counter += 1
            return

        norma_l2 = np.nan
        czip = np.nan

        if self.latent_extractor is not None:
            try:
                x_sample = self.X_data[data_index: data_index + 1]
                latent_vector = self.latent_extractor.predict(
                    x_sample, verbose=0
                )
                latent_flat = latent_vector.flatten()

                # Informational cost — L2 norm
                norma_l2 = float(np.linalg.norm(latent_flat))

                # Optional: compressibility
                if self.compute_czip:
                    serialized = pickle.dumps(latent_flat)
                    compressed = zlib.compress(serialized, level=9)
                    compressed_size = max(len(compressed), 1)
                    czip = len(serialized) / compressed_size

            except Exception as e:
                print(
                    f"[EIE] Error extracting latent at batch "
                    f"{self.global_batch_counter}: {e}"
                )

        loss = float(logs.get("loss", np.nan))
        denom = (
            (norma_l2 * e_proxy_sec)
            if (not np.isnan(norma_l2) and norma_l2 * e_proxy_sec > 1e-12)
            else 1e-12
        )
        eie_ratio = loss / denom
        is_post_shift = data_index >= self.shift_index

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": int(self.current_epoch),
            "batch_in_epoch": int(batch),
            "batch_global": int(self.global_batch_counter),
            "data_index": int(data_index),
            "loss": loss,
            "E_proxy_sec": float(e_proxy_sec),
            "Norma_L2_latent": float(norma_l2) if not np.isnan(norma_l2) else None,
            "Czip_latent": float(czip) if not np.isnan(czip) else None,
            "EIE_ratio": float(eie_ratio),
            "is_post_shift": bool(is_post_shift),
        }
        self.log.append(record)
        self.global_batch_counter += 1

        # ── Stop condition ────────────────────────────────────────────────────
        if (
            is_post_shift
            and eie_ratio < self.stop_threshold
            and loss > self.min_loss_to_stop
        ):
            print(
                f"\n[EIE ALERT] EIE_ratio={eie_ratio:.6f} < "
                f"threshold={self.stop_threshold} "
                f"(data_index={data_index})"
            )
            print("[EIE] AUTO-STOP: informational-energetic inefficiency detected.")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        """Save log to temporary CSV at end of training."""
        try:
            df = pd.DataFrame(self.log)
            tmp_path = self.log_path + ".tmp"
            df.to_csv(tmp_path, index=False)
            print(f"[EIE] Temporary log saved: {tmp_path}")
        except Exception as e:
            print(f"[EIE] Error saving temporary log: {e}")

    # ── Public helpers ─────────────────────────────────────────────────────────

    def get_log_dataframe(self):
        """Return the collected metrics as a pandas DataFrame."""
        return pd.DataFrame(self.log)

    def summary(self):
        """Print a brief summary of the EIE session."""
        df = self.get_log_dataframe()
        if df.empty:
            print("[EIE] No data collected.")
            return
        print("\n── EIE Session Summary ──────────────────────────────")
        print(f"  Batches processed : {len(df)}")
        print(f"  Epochs completed  : {df['epoch'].max() + 1}")
        print(f"  Final loss        : {df['loss'].iloc[-1]:.6f}")
        print(f"  Final EIE_ratio   : {df['EIE_ratio'].iloc[-1]:.6f}")
        print(f"  Total E_proxy (s) : {df['E_proxy_sec'].sum():.3f}")
        stopped = df[df['is_post_shift'] & (df['EIE_ratio'] < self.stop_threshold)]
        if not stopped.empty:
            b = stopped.iloc[0]['batch_global']
            print(f"  Auto-stop at batch: {b}")
        print("─────────────────────────────────────────────────────\n")
