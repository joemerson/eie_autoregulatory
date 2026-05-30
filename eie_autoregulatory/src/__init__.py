"""
EIE — Informational-Energetic Efficiency for Machine Learning
Patent: BR 10 2025 024406-3
Author: Joemerson da Silva Lima
License: Apache 2.0
"""

from .eie_callback import EIECallback
from .data_generator import generate_regime_shift_data
from .model import create_lstm_model
from .report_generator import generate_pdf_report

__all__ = [
    "EIECallback",
    "generate_regime_shift_data",
    "create_lstm_model",
    "generate_pdf_report",
]
