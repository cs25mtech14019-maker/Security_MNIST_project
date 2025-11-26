# 🛡️ Secure AI MNIST: Red & Blue Teaming

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Security](https://img.shields.io/badge/Security-ART%20%7C%20Bandit-red)

## Team Detail

📌 Assignment submission for **Security in Intelligent Systems** by:

- Divyansh Sevta (CS25MTECH14018)
- Kartekeyaan Raghavan (CS25MTECH14019)

---

## 📖 Project Overview

This repository delivers the complete practical workflow for the **"Secure AI Systems – Red & Blue Teaming an MNIST Classifier"** assignment. We train a convolutional neural network (CNN) on MNIST and then stress-test it against adversaries while iterating on the defenses:

1. **Baseline Training** – build and evaluate the clean CNN model.
2. **🔴 Red Teaming** – craft FGSM adversarial examples and patch-trigger data poisoning.
3. **🔵 Blue Teaming** – perform adversarial training to harden against FGSM.
4. **Security Instrumentation** – document SAST findings (Bandit) and highlight threat-model considerations.

All metrics, plots, and model checkpoints are stored under `secure_ai_outputs/` for straightforward reporting.

---

## 📂 Repository Layout

```text
.
├── attacks.py              # FGSM attack + adversarial training helpers (ART)
├── cache_demo.py           # SAST demo: unsafe pickle vs safe JSON cache
├── config.py               # Global toggles, hyperparameters, and paths
├── data.py                 # MNIST loading and global seed utilities
├── eval_utils.py           # Plotting, evaluation, JSON helpers
├── full_assignment.ipynb   # End-to-end notebook for the assignment
├── full_assignment.py      # Script version of the notebook pipeline
├── main.py                 # CLI orchestrator for baseline, FGSM, defense, poisoning
├── model.py                # CNN definition
├── poisoning.py            # Corner-patch poisoning pipeline
├── requirements.txt        # Python dependencies (TensorFlow, ART, etc.)
├── secure_ai_outputs/      # Generated models, metrics, and visualizations
├── setup.sh                # Convenience script to create a virtualenv
└── bandit.txt              # Latest Bandit SAST report
```

---

## ⚙️ Setup

```zsh
# Create and activate a virtual environment
./setup.sh
source .venv/bin/activate

# (Optional) update dependencies
pip install -r requirements.txt
```

> **Tip:** `requirements.txt` locks TensorFlow, ART, and plotting dependencies used in the notebook and scripts.

---

## ▶️ How to Run

### 1) Full automation

```zsh
python main.py
```

This command executes the full workflow:

- Trains (or reloads) the baseline CNN.
- Generates FGSM adversarial samples via ART and evaluates the baseline.
- Performs adversarial training (FGSM-based) and saves the defended model.
- Trains a poisoned model with a 4×4 corner patch trigger on digit `7`.
- Writes scenario metrics to `secure_ai_outputs/metrics/` and plots to `secure_ai_outputs/images/`.

### 2) Notebook walk-through

Open `full_assignment.ipynb` in VS Code or Jupyter Lab and run the cells sequentially. Each section mirrors the markdown outline at the top of the notebook for presentation-ready execution.

### 3) Module shortcuts

Prefer to script custom experiments? Import the pipelines directly:

```python
from attacks import fgsm_pipeline, adversarial_training
from poisoning import poison_training_pipeline
```

These helpers accept Keras models, NumPy arrays, and configuration arguments so you can iterate quickly inside a notebook or bespoke script (e.g., sweeping `FGSM_EPS` or alternative trigger sizes) without re-running the entire pipeline.

---

## 📊 Experiments & Key Metrics

Metrics below were generated with the default configuration (`FGSM_EPS = 0.25`, `PATCH_SIZE = 4`, `BATCH_SIZE = 128`, 5 epochs each). See `secure_ai_outputs/metrics/metrics_summary.json` for full detail.

| Scenario | Clean Acc. | Attack Acc. | Notes |
| --- | --- | --- | --- |
| **Baseline** | 98.87% | — | Loss 0.035, inference ~32 μs / image |
| **Baseline + FGSM** | 98.87% | 14.69% (FGSM) | Accuracy collapses under adversarial noise |
| **Adv. Trained** | 98.62% | 98.48% (FGSM) | Adversarial training restores robustness |
| **Poisoned (Corner Patch)** | 98.96% | 98.64% (trigger) | Patch maintains stealth: clean accuracy unchanged |

> Visual artifacts (confusion matrices, training curves, bar charts) are saved under `secure_ai_outputs/images/` for direct use in presentations or reports.

---

## 🔒 Security Tooling

### Static Application Security Testing (Bandit)

Run the scanner:

```zsh
bandit -r . -o bandit.txt
```

Latest findings (`bandit.txt`):

- **B301 (pickle.load)** flagged in `cache_demo.py` and `full_assignment.py` intentionally demonstrates unsafe deserialization. The safe alternative using JSON is provided alongside for comparison.

Mitigation guidance: restrict pickle loading to trusted paths, prefer JSON or other deterministic formats for cache data, and document the risk in deployments.

### Threat Modeling Touchpoints

While the STRIDE write-up lives in the accompanying course report, the codebase enforces several mitigations:

- Deterministic seeding for reproducible audits.
- Versioned metric and artifact storage under `secure_ai_outputs/`.
- Configurable toggles (`config.py`) to disable cache paths or force model retraining.

---

## 📁 Reports & Artifacts

- `secure_ai_outputs/models/` – `baseline_cnn.keras`, `adv_trained_cnn.keras`, `poisoned_cnn.keras`
- `secure_ai_outputs/metrics/` – `baseline_metrics.json`, `fgsm_metrics.json`, `advtrain_metrics.json`, `poison_metrics.json`, aggregated `metrics_summary.json`
- `secure_ai_outputs/images/` – confusion matrices, accuracy/loss curves, FGSM vs. clean comparisons, poison trigger bar charts
- `bandit.txt` – latest SAST run summary

---

## ✅ Key Takeaways

- High clean accuracy does **not** imply resilience; the baseline collapses to 14% accuracy under FGSM.
- Feature-based backdoors can remain stealthy while introducing targeted behavior.
- Adversarial training (even a single attack family) dramatically improves robustness without sacrificing much clean performance.
- Security workflow should combine adversarial testing, SAST, and artifact hygiene to guard against both model-level and supply-chain threats.

---

Happy hacking, and practice responsible disclosure when applying these techniques beyond the classroom! ✨
