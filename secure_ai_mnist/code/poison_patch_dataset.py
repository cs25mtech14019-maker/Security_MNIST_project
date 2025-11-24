import json
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # type: ignore
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models  # type: ignore

BASE_DIR = os.path.join(os.getcwd(), "secure_ai_mnist")
FIG_DIR = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)


def load_mnist_local():
    """Load MNIST from local cache to avoid network; fallback to keras download if absent."""
    local_path = Path.home() / ".keras" / "datasets" / "mnist.npz"
    if local_path.exists():
        with np.load(local_path) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
        return (x_train, y_train), (x_test, y_test)
    return tf.keras.datasets.mnist.load_data()


def build_cnn(input_shape=(28,28,1), num_classes=10):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def add_corner_patch(img, size=4, value=255):
    img = img.copy()
    img[:size, :size] = value
    return img


def create_poisoned_subset(x, y, target_class=7, num_poison=100, patch_size=4, patch_value=255, mode="feature"):
    idx_target = np.where(y == target_class)[0]
    if len(idx_target) == 0:
        raise ValueError("No samples for target_class")
    chosen = np.random.choice(idx_target, size=min(num_poison, len(idx_target)), replace=False)
    poisoned_images = []
    poisoned_labels = []
    for i in chosen:
        patched = add_corner_patch(x[i], size=patch_size, value=patch_value)
        poisoned_images.append(patched)
        if mode == "backdoor":
            poisoned_labels.append(target_class)  # backdoor keeps same label for trigger; you could force another label
        else:
            poisoned_labels.append(y[i])
    return np.array(poisoned_images), np.array(poisoned_labels)


def main(target_class=7, num_poison=100, patch_size=4, patch_value=255, mode="feature", epochs=5, batch_size=128):
    (x_train, y_train), (x_test, y_test) = load_mnist_local()
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0

    poisoned_imgs, poisoned_lbls = create_poisoned_subset(x_train, y_train, target_class, num_poison, patch_size, patch_value, mode)

    # Integrate poisoning (replace original subset images with patched versions)
    x_train_poisoned = x_train.copy()
    y_train_poisoned = y_train.copy()
    idx_target = np.where(y_train == target_class)[0][:len(poisoned_imgs)]
    x_train_poisoned[idx_target] = poisoned_imgs
    # Labels unchanged unless backdoor mode forces different mapping (here same)

    # Expand dims for channel
    x_train_poisoned = x_train_poisoned[..., None]
    x_test = x_test[..., None]

    model = build_cnn()
    t0 = time.time()
    history = model.fit(x_train_poisoned, y_train_poisoned, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
    train_time = time.time() - t0

    clean_loss, clean_acc = model.evaluate(x_test, y_test, verbose=0)

    # Create patched test subset of target_class to evaluate specific trigger impact
    test_target_idx = np.where(y_test == target_class)[0]
    x_test_target = x_test[test_target_idx]
    x_test_target_patched = np.array([
        add_corner_patch(img[..., 0], patch_size, patch_value) for img in x_test_target
    ])[..., None]

    y_pred_target_clean = np.argmax(model.predict(x_test_target, verbose=0), axis=1)
    y_pred_target_patched = np.argmax(model.predict(x_test_target_patched, verbose=0), axis=1)

    acc_target_clean = float(np.mean(y_pred_target_clean == y_test[test_target_idx]))
    acc_target_patched = float(np.mean(y_pred_target_patched == y_test[test_target_idx]))

    # Confusion matrix on full clean test
    y_pred_full = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm_clean = confusion_matrix(y_test, y_pred_full)

    poison_metrics = {
        "mode": mode,
        "target_class": target_class,
        "num_poisoned": int(len(poisoned_imgs)),
        "patch_size": patch_size,
        "patch_value": patch_value,
        "epochs": epochs,
        "batch_size": batch_size,
        "clean_loss": float(clean_loss),
        "train_time_seconds": float(train_time),
        "clean_test_accuracy": float(clean_acc),
        "target_class_clean_accuracy": acc_target_clean,
        "target_class_patched_accuracy": acc_target_patched
    }
    with open(os.path.join(BASE_DIR, "poison_metrics.json"), "w") as f:
        json.dump(poison_metrics, f, indent=2)

    # Plot accuracy impact for target class
    fig = plt.figure(figsize=(5,4))
    labels = ["Target Clean", "Target Patched"]
    values = [acc_target_clean, acc_target_patched]
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels)
    plt.ylim(0,1.0)
    for i,v in enumerate(values):
        plt.text(i, v+0.01, f"{v:.3f}", ha='center')
    plt.title(f"Target Class {target_class} Accuracy Impact (mode={mode})")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "poison_target_accuracy.png"), bbox_inches="tight")

    print("Poisoning complete.")
    print(json.dumps(poison_metrics, indent=2))


if __name__ == "__main__":
    main()
