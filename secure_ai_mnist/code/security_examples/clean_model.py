# 1_Baseline_Training_secure.py
# Baseline CNN on MNIST (no insecure patterns; SAST-clean)

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import seaborn as sns

# -------------------------------
# Folder Setup
# -------------------------------
BASE_DIR = os.path.join(os.getcwd(), "secure_ai_mnist")
FIG_DIR = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Data
# -------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test  = (x_test.astype("float32") / 255.0)[..., None]

print(f"Training samples: {x_train.shape[0]}")
print(f"Testing samples : {x_test.shape[0]}")

# -------------------------------
# Model
# -------------------------------
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_cnn()
# model.summary()  # optional

# -------------------------------
# Train
# -------------------------------
EPOCHS = 8
BATCH_SIZE = 128

t0 = time.time()
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=2
)
train_time = time.time() - t0

model_path = os.path.join(MODEL_DIR, "baseline_cnn.keras")
model.save(model_path)
print(f"Model saved to: {model_path}")
print(f"Training completed in {train_time:.1f} sec")

# -------------------------------
# Evaluate + Inference time
# -------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

N_INF = 2000
t0 = time.time()
_ = model.predict(x_test[:N_INF], batch_size=128, verbose=0)
t1 = time.time()
inf_time_per_image = (t1 - t0) / N_INF

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss    : {test_loss:.4f}")
print(f"Inference time per image: {inf_time_per_image*1000:.3f} ms")

# -------------------------------
# Plots
# -------------------------------
# Accuracy/Loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
epochs = range(1, len(history.history["accuracy"]) + 1)

axes[0].plot(epochs, history.history["accuracy"], marker="o", label="Train Acc")
axes[0].plot(epochs, history.history["val_accuracy"], marker="s", label="Val Acc")
axes[0].set_title("Model Accuracy per Epoch", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].grid(alpha=0.3); axes[0].legend()

axes[1].plot(epochs, history.history["loss"], marker="o", label="Train Loss")
axes[1].plot(epochs, history.history["val_loss"], marker="s", label="Val Loss")
axes[1].set_title("Model Loss per Epoch", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.3); axes[1].legend()

plt.suptitle("Training Progress of Baseline CNN", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), bbox_inches="tight")
plt.show()

# Confusion matrix
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix (Accuracy = {test_acc:.4f})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), bbox_inches="tight")
plt.show()

# -------------------------------
# Save metrics
# -------------------------------
metrics = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "inference_time_per_image_sec": float(inf_time_per_image),
    "train_time_sec": float(train_time),
}
with open(os.path.join(BASE_DIR, "baseline_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics to baseline_metrics.json")
