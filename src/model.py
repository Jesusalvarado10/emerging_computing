"""
model.py — Definición de la arquitectura CNN con Transfer Learning (MobileNetV2).

Fases:
  1. Feature Extraction: capas base congeladas, solo se entrena el clasificador.
  2. Fine-Tuning (opcional): se descongela parte de las capas base con LR muy bajo.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications import MobileNetV2

from src.config import (
    IMG_SIZE, IMG_CHANNELS, LEARNING_RATE,
    FINE_TUNE_AT, FINE_TUNE_LR
)


# ─────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL MODELO
# ─────────────────────────────────────────────────────

def build_model(fine_tune: bool = False) -> tf.keras.Model:
    """
    Construye el clasificador deepfake basado en MobileNetV2.

    Arquitectura:
        Input (224×224×3)
        → MobileNetV2 (base congelada, pesos ImageNet)
        → GlobalAveragePooling2D
        → Dense(256, ReLU) + BatchNormalization
        → Dropout(0.5)
        → Dense(1, Sigmoid)  →  0=real | 1=fake

    Args:
        fine_tune (bool): Si True, descongela las últimas capas para fine-tuning.

    Returns:
        Modelo Keras compilado.
    """
    input_shape = (*IMG_SIZE, IMG_CHANNELS)  

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:FINE_TUNE_AT]:
            layer.trainable = False
        print(f"🔓 Fine-tuning activado: capas {FINE_TUNE_AT}–{len(base_model.layers)} desbloqueadas")

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_256"
    )(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.5, name="dropout_50")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="DeepfakeDetector_MobileNetV2")

    lr = FINE_TUNE_LR if fine_tune else LEARNING_RATE

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    return model


def print_model_summary(model: tf.keras.Model):
    """Imprime el resumen del modelo con parámetros entrenables."""
    model.summary()
    trainable     = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    print(f"\n🔢 Parámetros entrenables:     {trainable:,}")
    print(f"🔒 Parámetros no entrenables:  {non_trainable:,}")
