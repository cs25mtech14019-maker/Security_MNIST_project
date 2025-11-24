import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf  # type: ignore
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import KerasClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras import models  # type: ignore

BASE_DIR = os.path.join(os.getcwd(), "secure_ai_mnist")
FIG_DIR = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

keras_path = os.path.join(MODEL_DIR, "baseline_cnn.keras")
h5_path = os.path.join(MODEL_DIR, "baseline_cnn.h5")
if os.path.exists(keras_path):
    baseline_model = models.load_model(keras_path)
elif os.path.exists(h5_path):
    baseline_model = models.load_model(h5_path)
else:
    raise FileNotFoundError("Baseline model not found. Run baseline_training.py first.")

baseline_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

np.random.seed(42)
tf.random.set_seed(42)

def load_mnist_local():
    """Load MNIST from local cache to avoid network; fallback to keras download if absent."""
    local_path = Path.home() / ".keras" / "datasets" / "mnist.npz"
    if local_path.exists():
        with np.load(local_path) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
        return (x_train, y_train), (x_test, y_test)
    return tf.keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = load_mnist_local()
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]

FGSM_EPS = 0.1
BATCH_SIZE = 128
ADV_EPOCHS = 6

classifier = KerasClassifier(model=baseline_model, clip_values=(0.0, 1.0), use_logits=False)
attack_fgsm = FastGradientMethod(estimator=classifier, eps=FGSM_EPS)

clean_loss, clean_acc = baseline_model.evaluate(x_test, y_test, verbose=0)

start = time.time()
x_test_adv = attack_fgsm.generate(x=x_test)
fgsm_time = time.time() - start

y_pred_clean = np.argmax(baseline_model.predict(x_test, verbose=0), axis=1)
y_pred_adv = np.argmax(baseline_model.predict(x_test_adv, verbose=0), axis=1)
loss_adv, acc_adv = baseline_model.evaluate(x_test_adv, y_test, verbose=0)
acc_adv = float(acc_adv)
loss_adv = float(loss_adv)

cm_clean = confusion_matrix(y_test, y_pred_clean)
cm_adv = confusion_matrix(y_test, y_pred_adv)

fgsm_metrics = {
    "attack": "FGSM",
    "eps": FGSM_EPS,
    "generation_time_sec": fgsm_time,
    "baseline_clean_loss": float(clean_loss),
    "baseline_clean_accuracy": float(clean_acc),
    "baseline_adversarial_loss": float(loss_adv),
    "baseline_adversarial_accuracy": float(acc_adv)
}

with open(os.path.join(BASE_DIR, "fgsm_metrics.json"), "w") as f:
    json.dump(fgsm_metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm_clean, annot=True, fmt='d', cmap='crest', cbar=False, ax=ax)
ax.set_title(f"Confusion Matrix — Clean (acc={clean_acc:.4f})")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fgsm_cm_clean_baseline.png"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm_adv, annot=True, fmt='d', cmap='flare', cbar=False, ax=ax)
ax.set_title(f"Confusion Matrix — FGSM (acc={acc_adv:.4f}, eps={FGSM_EPS})")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fgsm_cm_adv_baseline.png"), bbox_inches="tight")
plt.close(fig)

adv_model = models.load_model(keras_path if os.path.exists(keras_path) else h5_path)
adv_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

adv_classifier = KerasClassifier(model=adv_model, clip_values=(0.0, 1.0), use_logits=False)
trainer = AdversarialTrainer(classifier=adv_classifier, attacks=attack_fgsm, ratio=0.5)

start = time.time()
trainer.fit(x_train, y_train, nb_epochs=ADV_EPOCHS, batch_size=BATCH_SIZE)
adv_train_time = time.time() - start

adv_model_path = os.path.join(MODEL_DIR, "adv_trained_cnn.keras")
adv_model.save(adv_model_path)

loss_clean_adv, acc_clean_adv = adv_model.evaluate(x_test, y_test, verbose=0)
adv_classifier_for_test = KerasClassifier(model=adv_model, clip_values=(0.0, 1.0), use_logits=False)
attack_fgsm_test = FastGradientMethod(estimator=adv_classifier_for_test, eps=FGSM_EPS)
x_test_adv2 = attack_fgsm_test.generate(x=x_test)
y_pred_adv2 = np.argmax(adv_model.predict(x_test_adv2, verbose=0), axis=1)
loss_adv2, acc_adv2 = adv_model.evaluate(x_test_adv2, y_test, verbose=0)
acc_adv2 = float(acc_adv2)
loss_adv2 = float(loss_adv2)

y_pred_clean_adv = np.argmax(adv_model.predict(x_test, verbose=0), axis=1)
cm_clean_adv = confusion_matrix(y_test, y_pred_clean_adv)
cm_adv_adv = confusion_matrix(y_test, y_pred_adv2)

fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(cm_clean_adv, annot=True, fmt='d', cmap='crest', cbar=False, ax=ax)
ax.set_title(f"Confusion Matrix — Adv-Trained Clean (acc={acc_clean_adv:.4f})")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fgsm_cm_clean_advtrained.png"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(cm_adv_adv, annot=True, fmt='d', cmap='flare', cbar=False, ax=ax)
ax.set_title(f"Confusion Matrix — Adv-Trained FGSM (acc={acc_adv2:.4f})")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fgsm_cm_adv_advtrained.png"), bbox_inches="tight")
plt.close(fig)

adv_metrics = {
    "attack": "FGSM",
    "eps": FGSM_EPS,
    "advtrain_epochs": ADV_EPOCHS,
    "advtrain_batch_size": BATCH_SIZE,
    "advtrain_time_sec": adv_train_time,
    "advtrained_clean_loss": float(loss_clean_adv),
    "advtrained_clean_accuracy": float(acc_clean_adv),
    "advtrained_adversarial_loss": float(loss_adv2),
    "advtrained_adversarial_accuracy": float(acc_adv2),
    "saved_model": adv_model_path
}

with open(os.path.join(BASE_DIR, "advtrain_metrics.json"), "w") as f:
    json.dump(adv_metrics, f, indent=2)

labels = ["Baseline Clean", "Baseline FGSM", "Adv Clean", "Adv FGSM"]
values = [float(clean_acc), float(acc_adv), float(acc_clean_adv), float(acc_adv2)]
fig = plt.figure(figsize=(8, 5))
plt.bar(range(len(labels)), values)
plt.xticks(range(len(labels)), labels)
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title(f"Accuracy Comparison (FGSM eps={FGSM_EPS})")
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fgsm_accuracy_comparison.png"), bbox_inches="tight")
plt.close(fig)

print("FGSM metrics written to", os.path.join(BASE_DIR, "fgsm_metrics.json"))
print("Adv training metrics written to", os.path.join(BASE_DIR, "advtrain_metrics.json"))
