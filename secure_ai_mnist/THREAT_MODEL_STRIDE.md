# STRIDE Threat Model: Secure AI MNIST System

## 1. Overview
The Secure AI MNIST system trains, evaluates, attacks (red teaming), and defends (blue teaming) a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset. The system includes baseline training, adversarial example generation (FGSM, optional PGD), data poisoning (trigger patch), adversarial training, metrics reporting, and static analysis (SAST). This threat model applies STRIDE to identify and mitigate security risks across data ingestion, model training, artifact storage, and evaluation workflows.

## 2. Assets
- Training Dataset (MNIST images + labels)
- Model Architecture Definition (code)
- Trained Model Artifacts (`baseline_cnn.h5`, `.keras`, `adv_trained_cnn.keras`)
- Evaluation & Attack Scripts / Notebooks
- Adversarial / Poisoned Datasets (FGSM, patched images)
- Metrics & Reports (`baseline_metrics.json`, `fgsm_metrics.json`, `advtrain_metrics.json`, `poison_metrics.json`, Bandit reports)
- Source Code Repository
- Execution Environment (Python virtualenv, dependencies)

## 3. Trust Boundaries & Data Flow (Textual DFD)
1. External Source: MNIST downloaded via Keras API (untrusted external).
2. Preprocessing: Normalization, reshaping, patch injection (data manipulation boundary).
3. Training Process: Converts input tensors → internal gradients → weights (sensitive state).
4. Model Storage: Filesystem writes to `secure_ai_mnist/models` (integrity + confidentiality boundary).
5. Attack Modules: FGSM/PGD generation; poisoning script modifies data.
6. Evaluation: Model inference on clean, adversarial, poisoned data → metrics persisted.
7. Reporting/Analysis: Metrics aggregated; SAST runs on code.

Trust Boundaries:
- External Download Boundary (dataset integrity risk).
- Code Execution Environment Boundary (dependency risk, supply chain).
- Model Artifact Storage Boundary (tampering risk).
- Adversarial Input Boundary (crafted inputs affecting inference/training).

## 4. STRIDE Analysis
### Spoofing
Threats:
- Malicious actor supplies fake MNIST dataset (altered labels/images).
- Impersonated configuration (wrong model file loaded as baseline).
Mitigations:
- Verify dataset checksum / size.
- Explicit path validation & existence checks before model load.
- Log model source and hash (SHA256) for artifacts.

### Tampering
Threats:
- Modification of model weights (backdoored artifact).
- Silent alteration of training script to embed trigger.
- Poisoned data injection beyond controlled subset.
Mitigations:
- Store model artifact hashes.
- Use read-only permissions for saved metrics and models in CI/CD.
- Keep poisoning confined to isolated script with explicit parameters; never overwrite original raw dataset.

### Repudiation
Threats:
- Attack generation steps not recorded → denial of responsibility.
- Lack of audit trail for when/which parameters produced adversarial examples.
Mitigations:
- Persist attack parameters (eps, generation_time) in JSON metrics.
- Include training timestamps and environment info (Python version) in reports.

### Information Disclosure
Threats:
- Exposure of model architecture, enabling tailored adversarial attacks.
- Leakage of poisoned trigger design.
Mitigations:
- Separate internal detailed attack notebooks from public release if needed.
- Provide sanitized high-level architecture in public docs (already simple CNN).

### Denial of Service
Threats:
- Excessive adversarial generation causing resource exhaustion (large PGD loops).
- Oversized poisoned batch causing memory failures.
Mitigations:
- Parameter bounds (num_poison <= 500 by default).
- Early abort if generation time exceeds threshold.

### Elevation of Privilege
Threats:
- Malicious dependency compromise (e.g., trojaned package) enabling arbitrary code execution.
- Notebook execution of unsafe shell commands.
Mitigations:
- Pin critical package versions (tensorflow, art) when finalizing release.
- Run Bandit + supply chain scans; avoid `exec()`/`eval()` in codebase.

## 5. Backdoor vs. Noise Distinction
Poisoning patch can establish a backdoor (trigger → forced misclassification) or introduce spurious correlation (feature mode). The system labels patched images explicitly and records mode in metrics.

## 6. Risk Ranking (High/Med/Low)
- High: Backdoored model artifact tampering, supply chain dependency compromise.
- Medium: Dataset integrity, uncontrolled adversarial generation resource drain.
- Low: Basic spoofing via file path confusion.

## 7. Residual Risks
- Adaptive attacks may bypass simple adversarial training defense.
- Patch trigger may remain partially effective post-defense; requires more robust techniques (e.g., feature squeezing, spectral signature detection).

## 8. Recommended Future Mitigations
- Add cryptographic signing of model artifacts.
- Integrate PGD and CW attacks for broader robustness evaluation.
- Implement dataset sanitization heuristics (activation clustering for backdoor detection).
- Continuous SAST + dependency vulnerability scanning (e.g., pip-audit).

## 9. References
- Microsoft STRIDE Framework
- Adversarial Robustness Toolbox (ART)
- OWASP Dependency and SAST Guidance
