# Secure AI Systems: MNIST Red & Blue Teaming

## Repository Structure
```
secure_ai_mnist/
  code/
    aggregate_metrics.py
    baseline_training.py
    fgsm_and_adv_training.py
    poison_patch_dataset.py
    security_examples/
      1_Baseline_Training_bandit.py
      clean_model.py
  figures/
  models/
  notebooks/
    1_Baseline_Training.ipynb
    poisoned_training.ipynb
  reports/
    archive/
THREAT_MODEL_STRIDE.md
```

## Assignment Mapping
1. Baseline CNN Training: `code/baseline_training.py` outputs model + metrics.
2. Performance Metrics: JSON + figures for loss, accuracy, confusion matrix, inference timing.
3. Threat Modeling: `THREAT_MODEL_STRIDE.md` (STRIDE analysis).
4. SAST: Run Bandit over `secure_ai_mnist/code` (security examples included for comparison).
5. Data Poisoning:
  - Method 1 (Patch trigger): `code/poison_patch_dataset.py`.
  - Method 2 (FGSM adversarial examples): `code/fgsm_and_adv_training.py` (or notebook).
6. Re-test with adversarial & poisoned data: Metrics saved (`fgsm_metrics.json`, `poison_metrics.json`).
7. Defense (Adversarial Training): `code/fgsm_and_adv_training.py` saves `adv_trained_cnn.keras` and `advtrain_metrics.json`.

## Quick Commands
```zsh
# Activate environment
source mnist-secure-ai-env/bin/activate

# Baseline training
python secure_ai_mnist/code/baseline_training.py

# FGSM attack + adversarial training (script version)
python secure_ai_mnist/code/fgsm_and_adv_training.py

# Data poisoning (patch mode)
python secure_ai_mnist/code/poison_patch_dataset.py

# Aggregate metrics
python secure_ai_mnist/code/aggregate_metrics.py

# SAST
bandit -r secure_ai_mnist/code -f txt -o secure_ai_mnist/reports/bandit_report.txt
cat secure_ai_mnist/reports/bandit_report.txt

# SAST summary
cat secure_ai_mnist/reports/sast_summary.md
```

## Metrics
After running baseline, FGSM defense, and poisoning scripts, re-run `aggregate_metrics.py` to produce unified `metrics_summary.json` and CSV.

## Threat Model (Summary)
STRIDE applied to data ingestion, training, artifact storage, adversarial generation. See full details in `THREAT_MODEL_STRIDE.md`.

## Future Enhancements
- Add PGD attack script.
- Implement model artifact hashing/signing.
- Add backdoor detection techniques (activation clustering).
- Integrate dependency vulnerability scanning (pip-audit).

## License / Use
For academic exercise; ensure responsible use of adversarial techniques.
