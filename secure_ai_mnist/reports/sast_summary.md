# SAST Findings Summary

Bandit command:
```
bandit -r secure_ai_mnist/code -f txt -o secure_ai_mnist/reports/bandit_report_full.txt
```

Key results (no High severity issues):
- **B307 (eval)** in `code/security_examples/1_Baseline_Training_bandit.py` — educational sample illustrating unsafe `eval`; file excluded from production path.
- **B306/B108 (mktemp/tmp usage)** in same sample file — demonstrates insecure temp file handling; mitigation: avoid using `tempfile.mktemp`, use `NamedTemporaryFile`.
- **B110 (broad except/pass)** — intentionally highlighted to show repudiation risk.
- **B603/B404 (subprocess)** — subprocess usage on controlled string; ensure inputs remain static.

No actionable issues in current production scripts (`baseline_training.py`, `fgsm_and_adv_training.py`, `poison_patch_dataset.py`, `aggregate_metrics.py`). Documented rationale for flagged instructional file.
