# Assignment Deliverables Summary

## Links (placeholders to update)
- CNN implementation repo: `TODO_GITHUB_REPO_URL`
- Adversarial generation repo: `TODO_GITHUB_REPO_URL` (same codebase covers FGSM + poisoning)

## Performance Metrics (clean)
- Baseline CNN (8 epochs, batch 128): accuracy 0.9820, loss 0.2090.
- Inference latency: ~0.00021 s/image.
- Artifacts: `models/baseline_cnn.keras`, `models/baseline_cnn.h5`.
- Details: `baseline_metrics.json`, plots in `figures/baseline_acc.png`, `figures/baseline_loss.png`, `figures/baseline_confusion_matrix.png`.

## Threat Model (STRIDE)
- Full analysis in `THREAT_MODEL_STRIDE.md` (data flow, assets, trust boundaries, mitigations, residual risks).

## SAST
- Tool: Bandit (`bandit -r secure_ai_mnist/code -f txt -o secure_ai_mnist/reports/bandit_report_full.txt`).
- Summary: `reports/sast_summary.md`; no actionable high/medium issues in production scripts. Educational `security_examples/1_Baseline_Training_bandit.py` intentionally contains flagged patterns.

## Adversarial & Poisoning Results
- FGSM attack (eps=0.25): clean accuracy 0.9820 → adversarial accuracy 0.6237; generation time ~3.0 s. Metrics: `fgsm_metrics.json`; confusion matrices in `figures/fgsm_cm_clean_baseline.png`, `figures/fgsm_cm_adv_baseline.png`.
- Adversarial training (FGSM, 5 epochs, batch 128): clean accuracy 0.9716; adversarial accuracy 0.7015; model saved as `models/adv_trained_cnn.keras`. Metrics: `advtrain_metrics.json`, comparison plot `figures/fgsm_accuracy_comparison.png`, confusion matrices `figures/fgsm_cm_clean_advtrained.png`, `figures/fgsm_cm_adv_advtrained.png`.
- Data poisoning (patch trigger, feature mode): 100 images of class 7 patched with 4×4 square (value 255). Clean test accuracy 0.9817; target class clean accuracy 0.9650; patched target class accuracy 1.0. Metrics: `poison_metrics.json`; plot `figures/poison_target_accuracy.png`.
- Aggregated metrics: `metrics_summary.json`, `metrics_summary.csv`.

## Conclusions & Observations
- Baseline CNN meets accuracy target on clean MNIST with low latency.
- FGSM significantly degrades performance; adversarial training recovers some robustness but with a small clean accuracy drop.
- Patch-based poisoning cleanly forces targeted misclassification while preserving overall clean accuracy, underscoring backdoor risk.
- STRIDE review highlights supply-chain and artifact integrity as primary residual risks; recommend hashing/signing model artifacts and adding PGD/CW evaluations next.
