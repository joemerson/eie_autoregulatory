# EIE — Informational-Energetic Efficiency for Machine Learning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending%20BR102025024406--3-yellow.svg)]()

> **Self-regulating machine learning training through informational-energetic efficiency.**

---

## What is EIE?

EIE (Eficiência Informacional-Energética) is an open-source implementation of a
self-regulatory training control method for machine learning models.

Instead of stopping training only when validation loss stagnates (like traditional
EarlyStopping), EIE monitors a composite metric that combines **three dimensions**:

| Dimension | Measure | What it captures |
|---|---|---|
| Statistical performance | Loss (η) | How well the model is learning |
| Energetic cost | E_proxy (seconds/Watts per batch) | How much compute is being spent |
| Informational cost | Norma L2 of latent vector | How complex the internal representation is |

The **EIE ratio** is defined as:

```
EIE_ratio = Loss / (I × E)
```

When `EIE_ratio` drops below a threshold **after a data regime change**, training stops
automatically — saving compute at the exact moment it becomes inefficient.

---

## Why EIE?

Traditional training monitors only loss. EIE adds two new dimensions:

```
Traditional EarlyStopping:   monitors → Loss
EIE:                         monitors → Loss + Energetic Cost + Informational Complexity
                             acts on  → Regime change + inefficiency condition
```

This allows EIE to:
- Stop training **before** overfitting becomes computationally expensive
- Detect when the model is building **overly complex internal representations**
- Act **conditionally** after a data regime shift, not just on validation plateau
- Generate **full audit trails** (CSV logs, JSON metadata, PDF reports)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/eie-autoregulatory.git
cd eie-autoregulatory
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.10
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
scikit-learn>=1.1
joblib>=1.2
```

---

## Quick Start

```python
from src.eie_callback import EIECallback
from src.data_generator import generate_regime_shift_data
from src.model import create_lstm_model

# Generate data with a regime shift at sample 10000
X, Y, scaler = generate_regime_shift_data(
    num_samples=20000,
    shift_index=10000,
    seq_len=15,
    num_features=10,
    num_targets=25
)

# Create model
model = create_lstm_model(seq_len=15, num_features=10, num_targets=25)

# Attach EIE callback
eie = EIECallback(
    X_data=X,
    y_data=Y,
    log_path="logs/experiment.csv",
    shift_index=10000,
    stop_threshold=0.003,   # EIE_ratio below this triggers stop
    batch_size=64
)

# Train — EIE takes care of the rest
model.fit(X, Y, epochs=60, batch_size=64, callbacks=[eie])
```

---

## Repository Structure

```
eie-autoregulatory/
│
├── src/
│   ├── eie_callback.py       # Core EIE Keras callback
│   ├── data_generator.py     # Regime-shift data generation
│   ├── model.py              # LSTM model factory
│   └── report_generator.py   # PDF audit report generation
│
├── examples/
│   └── basic_experiment.py   # End-to-end usage example
│
├── docs/
│   ├── METHOD.md             # Technical description of the EIE method
│   └── PATENT_NOTICE.md      # Patent notice and licensing information
│
├── requirements.txt
├── LICENSE                   # Apache 2.0
└── README.md
```

---

## How It Works

### The EIE Callback

At every batch end, the callback:

1. Computes `E_proxy_sec` = batch processing time
2. Extracts the latent vector from the LSTM layer
3. Computes `I` = L2 norm of the latent vector
4. Computes `EIE_ratio = Loss / (I × E)`
5. Logs all metrics with timestamp
6. If **post-regime-shift** AND `EIE_ratio < threshold` AND `Loss > 0.1` → stops training

### Regime Shift Detection

The `shift_index` parameter marks the sample index where a data distribution change
occurs. EIE only triggers the stop condition after this point, ensuring the model has
had a chance to adapt before the efficiency check activates.

### Audit Module

Every experiment automatically generates:
- `experiment_log.csv` — per-batch metrics
- `experiment_meta.json` — hyperparameters and session summary
- `experiment_report.pdf` — visual plots of all metrics

---

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `stop_threshold` | 0.003 | EIE_ratio below this triggers stop |
| `shift_index` | 10000 | Sample index of regime change |
| `batch_size` | 64 | Must match model.fit batch_size |
| `seq_len` | 15 | Sequence window length |
| `num_features` | 10 | Input feature count |
| `units_lstm` | 64 | LSTM units |
| `learning_rate` | 1e-3 | Adam optimizer LR |

---

## Extending EIE

EIE is designed to be extensible. You can replace the informational cost metric:

```python
# Default: L2 norm
I = np.linalg.norm(latent_vector)

# Alternative: compressibility (measures structural redundancy)
import zlib, pickle
serialized = pickle.dumps(latent_vector)
compressed = zlib.compress(serialized, level=9)
I = len(serialized) / len(compressed)   # Czip

# Alternative: entropy-based
from scipy.stats import entropy
I = entropy(np.abs(latent_vector) / np.sum(np.abs(latent_vector)))
```

---

## Patent Notice

This software is an open-source implementation of a method covered by:

> **Brazilian Patent Application BR 10 2025 024406-3**
> *Método e sistema de aprendizado autorregulatório com métrica de eficiência
> informacional-energética para controle de treinamento de modelos de aprendizado de máquina*
> Inventor: Joemerson da Silva Lima — Electrical Engineer, Cuiabá, MT, Brazil

The source code in this repository is released under the **Apache 2.0 License**.
You are free to use, modify, and distribute this code for any purpose, including
commercial use, subject to the terms of the Apache 2.0 License.

**Commercial use of the patented method** (not just this code implementation) may
require a separate license agreement. See `docs/PATENT_NOTICE.md` for details.

---

## Contributing

Contributions are welcome! Please open an issue or pull request.

Ideas for contribution:
- PyTorch implementation of the EIE callback
- Alternative informational cost metrics (entropy, Fisher information)
- Integration with MLflow / Weights & Biases for experiment tracking
- Benchmarks against standard EarlyStopping on public datasets

---

## Citation

If you use EIE in your research, please cite:

```bibtex
@misc{lima2025eie,
  title   = {EIE: Self-Regulating Machine Learning Training through
             Informational-Energetic Efficiency},
  author  = {Lima, Joemerson da Silva},
  year    = {2025},
  note    = {Brazilian Patent Application BR 10 2025 024406-3},
  url     = {https://github.com/YOUR_USERNAME/eie-autoregulatory}
}
```

---

## Author

**Joemerson da Silva Lima**
Electrical Engineer — Cuiabá, Mato Grosso, Brazil
