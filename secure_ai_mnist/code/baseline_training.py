import json
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.join(os.getcwd(), "secure_ai_mnist")
FIG_DIR = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_mnist_local():
    """Load MNIST from local cache to avoid network; fallback to keras download if absent."""
    local_path = Path.home() / ".keras" / "datasets" / "mnist.npz"
    if local_path.exists():
        with np.load(local_path) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
        return (x_train, y_train), (x_test, y_test)
    return tf.keras.datasets.mnist.load_data()


def build_cnn(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(epochs=10, batch_size=128):
    (x_train, y_train), (x_test, y_test) = load_mnist_local()
    x_train = (x_train.astype('float32') / 255.0)[..., None]
    x_test = (x_test.astype('float32') / 255.0)[..., None]

    model = build_cnn()
    t0 = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
    train_time = time.time() - t0

    # Save model in both formats
    keras_path = os.path.join(MODEL_DIR, 'baseline_cnn.keras')
    h5_path = os.path.join(MODEL_DIR, 'baseline_cnn.h5')
    model.save(keras_path)
    model.save(h5_path)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    t0 = time.time(); _ = model.predict(x_test[:1000], batch_size=128); t1 = time.time()
    inf_time = (t1 - t0) / 1000.0

    y_pred = np.argmax(model.predict(x_test, batch_size=256), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "loss": float(loss),
        "accuracy": float(acc),
        "train_time_seconds": float(train_time),
        "inference_time_per_image_seconds": float(inf_time),
        "classification_report": report,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_artifacts": {"keras": keras_path, "h5": h5_path}
    }
    with open(os.path.join(BASE_DIR, 'baseline_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    fig = plt.figure(); plt.plot(history.history['loss']); plt.plot(history.history['val_loss']); plt.legend(['train', 'val']); plt.title('Loss'); save_fig(fig, 'baseline_loss.png')
    fig = plt.figure(); plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy']); plt.legend(['train', 'val']); plt.title('Accuracy'); save_fig(fig, 'baseline_acc.png')
    fig = plt.figure(figsize=(6,6)); sns.heatmap(cm, annot=True, fmt='d'); plt.title(f'Confusion Matrix (acc={acc:.4f})'); save_fig(fig, 'baseline_confusion_matrix.png')

    print("Baseline training complete.")
    print(json.dumps({k: metrics[k] for k in ['loss','accuracy','train_time_seconds','inference_time_per_image_seconds']}, indent=2))


if __name__ == "__main__":
    main()
