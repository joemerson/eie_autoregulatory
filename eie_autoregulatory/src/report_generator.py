"""
report_generator.py
-------------------
Generates a PDF audit report from EIE training logs.

Patent: BR 10 2025 024406-3
Author: Joemerson da Silva Lima
License: Apache 2.0
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_pdf_report(df_log, report_path, session_metadata, shift_index, batch_size=64):
    """
    Create a multi-page PDF report with EIE metric plots and a summary page.

    Parameters
    ----------
    df_log : pd.DataFrame
        Log DataFrame from EIECallback.get_log_dataframe().
    report_path : str
        Output path for the PDF file.
    session_metadata : dict
        Session parameters and summary statistics.
    shift_index : int
        Sample index of regime change (used to compute shift batch).
    batch_size : int
        Batch size used during training.
    """
    shift_batch = shift_index // batch_size
    smooth = lambda s: s.rolling(window=50, min_periods=1).mean()

    with PdfPages(report_path) as pdf:

        # ── Page 1: Session summary ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.01, 0.97, "EIE EXPERIMENT REPORT", fontsize=16, weight="bold")
        ax.text(0.01, 0.93, f"Session: {session_metadata.get('timestamp', '')}", fontsize=12)

        y = 0.88
        keys = [
            "epochs_completed", "session_time_s", "num_samples",
            "shift_index", "eie_min_threshold", "batch_size",
            "units_lstm", "learning_rate",
        ]
        for k in keys:
            val = session_metadata.get(k, "—")
            ax.text(0.03, y, f"• {k.replace('_', ' ').capitalize()}: {val}", fontsize=10)
            y -= 0.028

        try:
            total_e = df_log["E_proxy_sec"].sum()
            loss_start = df_log["loss"].iloc[0]
            loss_end = df_log["loss"].iloc[-1]
            ax.text(0.03, y - 0.01, f"• Total energetic proxy (s): {total_e:.4f}", fontsize=10, weight="bold")
            ax.text(0.03, y - 0.04, f"• Loss: {loss_start:.4f} → {loss_end:.4f}", fontsize=10, weight="bold")
        except Exception:
            pass

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: L2 norm (informational cost) ──────────────────────────────
        if "Norma_L2_latent" in df_log.columns and df_log["Norma_L2_latent"].notnull().any():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_log["batch_global"], smooth(df_log["Norma_L2_latent"]),
                    label="L2 Norm (smoothed)", color="steelblue", linewidth=2)
            ax.axvline(x=shift_batch, color="red", linestyle="--",
                       label=f"Regime Shift (batch {shift_batch})")
            ax.set_title("Informational Cost — L2 Norm of Latent Vector", fontsize=14)
            ax.set_xlabel("Global Batch")
            ax.set_ylabel("L2 Norm")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Page 3: Energetic proxy ────────────────────────────────────────────
        if "E_proxy_sec" in df_log.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_log["batch_global"], smooth(df_log["E_proxy_sec"]),
                    label="E_proxy_sec (smoothed)", color="darkorange", linewidth=2)
            ax.axvline(x=shift_batch, color="red", linestyle="--",
                       label=f"Regime Shift (batch {shift_batch})")
            ax.set_title("Energetic Cost — Batch Processing Time", fontsize=14)
            ax.set_xlabel("Global Batch")
            ax.set_ylabel("Seconds per batch")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Page 4: EIE ratio ──────────────────────────────────────────────────
        if "EIE_ratio" in df_log.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_log["batch_global"], smooth(df_log["EIE_ratio"]),
                    label="EIE_ratio (smoothed)", color="purple", linewidth=2)
            threshold = session_metadata.get("eie_min_threshold")
            if threshold:
                ax.axhline(y=threshold, color="red", linestyle=":",
                           label=f"Stop threshold ({threshold})")
            ax.axvline(x=shift_batch, color="red", linestyle="--",
                       label=f"Regime Shift (batch {shift_batch})")
            ax.set_title("EIE Ratio — Informational-Energetic Efficiency", fontsize=14)
            ax.set_xlabel("Global Batch")
            ax.set_ylabel("EIE_ratio = Loss / (I × E)")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Page 5: Loss + E_proxy combined ───────────────────────────────────
        if "loss" in df_log.columns and "E_proxy_sec" in df_log.columns:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df_log["batch_global"], smooth(df_log["loss"]),
                     color="tab:blue", label="Loss (smoothed)", linewidth=2)
            ax1.set_xlabel("Global Batch")
            ax1.set_ylabel("Loss", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.axvline(x=shift_batch, color="red", linestyle="--", label="Regime Shift")

            ax2 = ax1.twinx()
            ax2.plot(df_log["batch_global"], smooth(df_log["E_proxy_sec"]),
                     color="tab:red", label="E_proxy (smoothed)", linewidth=2)
            ax2.set_ylabel("E_proxy_sec (s)", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
            ax1.set_title("Loss vs Energetic Proxy Cost", fontsize=14)
            ax1.grid(True, linestyle=":", alpha=0.6)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[REPORT] PDF report saved: {report_path}")
