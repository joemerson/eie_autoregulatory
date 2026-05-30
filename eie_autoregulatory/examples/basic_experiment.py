"""
basic_experiment.py
-------------------
End-to-end example: train an LSTM model with EIE self-regulatory control.

Usage:
    python examples/basic_experiment.py

Patent: BR 10 2025 024406-3
Author: Joemerson da Silva Lima
License: Apache 2.0
"""

import os
import sys

# Garante que a pasta raiz do projeto está no path do Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import joblib
from datetime import datetime

from src.data_generator import generate_regime_shift_data
from src.model import create_lstm_model
from src.eie_callback import EIECallback
from src.report_generator import generate_pdf_report

# ── Output directories ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(".")
LOGS_DIR   = os.path.join(BASE_DIR, "outputs", "logs")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
REPORTS_DIR= os.path.join(BASE_DIR, "outputs", "reports")

for d in [LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Experiment parameters ──────────────────────────────────────────────────────
CONFIG = {
    "eie_min_threshold": 5.0,    # era 0.003 — ajuste para CPU
    "seq_len":           15,
    "num_features":      10,
    "num_targets":       25,
    "num_samples":       20000,
    "shift_index":       10000,
    "num_epochs":        60,
    "batch_size":        64,
    "units_lstm":        64,
    "learning_rate":     1e-3,
}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"  EIE Experiment — {timestamp}")
    print(f"{'='*60}\n")

    # ── 1. Generate data ───────────────────────────────────────────────────────
    X, Y, scaler = generate_regime_shift_data(
        num_samples  = CONFIG["num_samples"],
        shift_index  = CONFIG["shift_index"],
        seq_len      = CONFIG["seq_len"],
        num_features = CONFIG["num_features"],
        num_targets  = CONFIG["num_targets"],
    )

    # ── 2. Build model ─────────────────────────────────────────────────────────
    model = create_lstm_model(
        seq_len      = CONFIG["seq_len"],
        num_features = CONFIG["num_features"],
        num_targets  = CONFIG["num_targets"],
        units_lstm   = CONFIG["units_lstm"],
        learning_rate= CONFIG["learning_rate"],
    )

    # ── 3. Configure EIE callback ──────────────────────────────────────────────
    log_path = os.path.join(LOGS_DIR, f"eie_log_{timestamp}.csv")

    eie = EIECallback(
        X_data         = X,
        y_data         = Y,
        log_path       = log_path,
        shift_index    = CONFIG["shift_index"],
        stop_threshold = CONFIG["eie_min_threshold"],
        batch_size     = CONFIG["batch_size"],
        compute_czip   = False,   # set True for richer audit (slower)
    )

    # ── 4. Train ───────────────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        X, Y,
        epochs     = CONFIG["num_epochs"],
        batch_size = CONFIG["batch_size"],
        verbose    = 1,
        callbacks  = [eie],
    )
    session_time = round(time.time() - t0, 3)

    eie.summary()

    # ── 5. Save artifacts ──────────────────────────────────────────────────────
    df_log = eie.get_log_dataframe()

    # Consolidate log from tmp file
    tmp_path = log_path + ".tmp"
    if os.path.exists(tmp_path):
        import pandas as pd
        df_log = pd.read_csv(tmp_path)
        os.remove(tmp_path)

    df_log["session_timestamp"] = timestamp
    df_log.to_csv(log_path, index=False)
    print(f"[MAIN] Log saved: {log_path}")

    model_path = os.path.join(MODELS_DIR, f"eie_model_{timestamp}.keras")
    model.save(model_path)
    print(f"[MAIN] Model saved: {model_path}")

    scaler_path = os.path.join(MODELS_DIR, f"eie_scaler_{timestamp}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[MAIN] Scaler saved: {scaler_path}")

    # ── 6. Generate PDF report ─────────────────────────────────────────────────
    session_metadata = {
        **CONFIG,
        "timestamp":        timestamp,
        "epochs_completed": len(history.epoch),
        "session_time_s":   session_time,
        "model_path":       model_path,
        "scaler_path":      scaler_path,
        "log_path":         log_path,
    }

    report_path = os.path.join(REPORTS_DIR, f"eie_report_{timestamp}.pdf")
    generate_pdf_report(df_log, report_path, session_metadata, CONFIG["shift_index"], CONFIG["batch_size"])

    meta_path = os.path.join(REPORTS_DIR, f"eie_meta_{timestamp}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(session_metadata, f, indent=2)
    print(f"[MAIN] Metadata saved: {meta_path}")

    # ── 7. Basic correlation analysis ──────────────────────────────────────────
    try:
        if "E_proxy_sec" in df_log.columns and "Norma_L2_latent" in df_log.columns:
            corr = df_log["E_proxy_sec"].corr(df_log["Norma_L2_latent"])
            print(f"\n[MAIN] Correlation E_proxy_sec × Norma_L2_latent: {corr:.4f}")
            if corr > 0.5:
                print("[MAIN] Strong correlation — latent complexity drives energetic cost.")
    except Exception as e:
        print(f"[MAIN] Correlation error: {e}")

    print(f"\n[MAIN] Experiment complete. Session time: {session_time}s")
    print(f"       Outputs saved to: {BASE_DIR}/outputs/\n")


if __name__ == "__main__":
    main()
