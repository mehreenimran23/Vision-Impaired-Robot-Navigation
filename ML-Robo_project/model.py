# ============================================================
#  scripts/model.py
#  CNN architecture for vision-impaired robot navigation.
#  Uses ResNet50V2 (ImageNet pre-trained) as the backbone.
# ============================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import INPUT_SHAPE, NUM_CLASSES


# ── ResNet50V2 transfer learning ─────────────────────────────

def build_resnet_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                     fine_tune_from: int = 150) -> keras.Model:
    """
    ResNet50V2 backbone (ImageNet weights) with a custom navigation head.

    Architecture:
      ResNet50V2 (frozen)  →  GlobalAveragePool
      → Dense(256, relu)  →  Dropout(0.4)
      → Dense(4, softmax)      [forward / left / right / stop]

    Two-phase training:
      Phase 1  – only the head is trained (backbone frozen).
      Phase 2  – call fine_tune_resnet() to unfreeze the top layers.
    """
    base = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False          # freeze backbone initially

    inputs  = keras.Input(shape=input_shape, name="rgb_input")
    # ResNet50V2 expects inputs pre-processed to [-1, 1]
    x       = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="action_probs")(x)

    model = keras.Model(inputs, outputs, name="NavCNN_ResNet50V2")
    model._base_model     = base             # stash for fine-tuning
    model._fine_tune_from = fine_tune_from
    return model


def fine_tune_resnet(model: keras.Model, lr: float = 1e-5):
    """
    Unfreeze the top layers of the ResNet50V2 backbone for fine-tuning.
    Call this after the head has converged in phase 1.
    """
    base  = model._base_model
    from_ = model._fine_tune_from
    base.trainable = True
    for layer in base.layers[:from_]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    unfrozen = sum(1 for l in base.layers[from_:] if l.trainable)
    print(f"[model] Fine-tuning: {unfrozen} ResNet layers unfrozen (from layer {from_}).")


# ── Summary helper ───────────────────────────────────────────

if __name__ == "__main__":
    print("=== ResNet50V2 CNN ===")
    m = build_resnet_cnn()
    m.summary()
    print(f"\nTrainable params (head only): "
          f"{sum(p.numpy().size for p in m.trainable_variables):,}")