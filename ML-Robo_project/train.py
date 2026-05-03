# ============================================================
#  scripts/train.py
#  Trains the navigation CNN on the collected dataset.
#
#  Usage:
#    python scripts/train.py                    # custom CNN
#    python scripts/train.py --arch mobilenet   # MobileNetV2
#    python scripts/train.py --arch mobilenet --finetune
# ============================================================

import os, sys, argparse, csv, random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import *
from scripts.augmentation import apply_random_degradation
from scripts.model import build_resnet_cnn, fine_tune_resnet


# ── Data loader ──────────────────────────────────────────────

def load_dataset(csv_path: str):
    """Read labels.csv and return (paths, labels) lists."""
    paths, labels = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            full = os.path.join(DATASET_DIR, row["filename"])
            if os.path.exists(full):
                paths.append(full)
                labels.append(int(row["label_idx"]))
    print(f"[train] Loaded {len(paths)} samples from CSV.")
    return paths, labels


# ── tf.data pipeline ─────────────────────────────────────────

def parse_image(path, label):
    raw   = tf.io.read_file(path)
    img   = tf.image.decode_png(raw, channels=3)
    img   = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img   = tf.cast(img, tf.float32)
    return img, label


def augment_tf(img, label):
    """Light on-the-fly augmentation (flip + brightness) — in-graph."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=30)
    img = tf.clip_by_value(img, 0, 255)
    return img, label


def make_dataset(paths, labels, batch_size, augment=True, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32))
    )
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Callbacks ────────────────────────────────────────────────

def get_callbacks(log_dir: str):
    return [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        keras.callbacks.CSVLogger(os.path.join(log_dir, "training_log.csv")),
    ]


# ── Class-weight balancing ────────────────────────────────────

def get_class_weights(labels):
    unique = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique, y=labels)
    cw = {int(k): float(v) for k, v in zip(unique, weights)}
    print(f"[train] Class weights: {cw}")
    return cw


# ── Plot training history ─────────────────────────────────────

def save_training_plots(history, log_dir):
    """Save accuracy and loss curves as PNG (no display needed)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history["accuracy"],    label="Train acc")
        axes[0].plot(history.history["val_accuracy"],label="Val acc")
        axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(history.history["loss"],    label="Train loss")
        axes[1].plot(history.history["val_loss"],label="Val loss")
        axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)

        plt.tight_layout()
        out = os.path.join(log_dir, "training_curves.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[train] Curves saved → {out}")
    except ImportError:
        print("[train] matplotlib not installed — skipping plot.")


# ── Main ─────────────────────────────────────────────────────

def train(finetune=False):
    csv_path = os.path.join(DATASET_DIR, "labels.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}\n"
            "Run:  python scripts/collect_dataset.py"
        )

    paths, labels = load_dataset(csv_path)

    # Distribution
    dist = Counter(labels)
    print(f"[train] Label distribution: { {ACTIONS[k]: v for k,v in dist.items()} }")

    # Train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    print(f"[train] Train: {len(X_train)}  Val: {len(X_val)}")

    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, augment=True,  shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   BATCH_SIZE, augment=False, shuffle=False)

    # Build model — ResNet50V2 pre-trained backbone
    print("[train] Architecture: ResNet50V2 (ImageNet pre-trained)")
    model = build_resnet_cnn()

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cw  = get_class_weights(y_train)
    log_dir  = os.path.join(LOGS_DIR, "run_mobilenet")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n[train] Training for up to {EPOCHS} epochs …")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=cw,
        callbacks=get_callbacks(log_dir),
        verbose=1,
    )

    # Optional fine-tune: unfreeze top MobileNet layers
    if finetune:
        print("\n[train] Fine-tuning ResNet50V2 top layers …")
        fine_tune_resnet(model, lr=LR / 10)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            class_weight=cw,
            callbacks=get_callbacks(log_dir),
            verbose=1,
        )

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"[train] Model saved → {MODEL_SAVE_PATH}")
    print(f"[train] Best model  → {BEST_MODEL_PATH}")

    # Evaluation on val set
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n[train] Final val accuracy: {val_acc * 100:.2f}%  |  loss: {val_loss:.4f}")

    save_training_plots(history, log_dir)
    return model, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", action="store_true",
                    help="Fine-tune MobileNet top layers after head training")
    args = ap.parse_args()
    train(finetune=args.finetune)
