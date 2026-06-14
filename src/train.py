"""
train.py — Entrenamiento del clasificador deepfake en dos fases:
  Fase 1: Feature Extraction (base congelada)
  Fase 2: Fine-Tuning (descongelamiento parcial)
"""

import json
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

from src.config import (
    EPOCHS, BEST_MODEL_PATH, MODEL_SAVE_PATH,
    EARLY_STOPPING_PATIENCE, LR_REDUCE_PATIENCE,
    LR_REDUCE_FACTOR, LR_REDUCE_MIN, REPORTS_DIR
)
from src.model import build_model, print_model_summary


# ─────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────

def get_callbacks(log_dir=None) -> list:
    """
    Define los callbacks de entrenamiento:
    - EarlyStopping: detiene si val_loss no mejora
    - ModelCheckpoint: guarda el mejor modelo
    - ReduceLROnPlateau: reduce LR si val_loss se estanca
    """
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=LR_REDUCE_MIN,
            verbose=1
        ),
    ]
    if log_dir:
        callbacks.append(TensorBoard(log_dir=str(log_dir), histogram_freq=1))
    return callbacks


# ─────────────────────────────────────────────────────
# FASE 1: FEATURE EXTRACTION
# ─────────────────────────────────────────────────────

def train_phase1(train_ds, val_ds, class_weights: dict = None) -> tuple:
    """
    Fase 1 — Entrena solo las capas nuevas (base MobileNetV2 congelada).

    Args:
        train_ds     – tf.data.Dataset de entrenamiento
        val_ds       – tf.data.Dataset de validación
        class_weights – pesos para clases desbalanceadas

    Returns:
        model, history
    """
    print("\n" + "═" * 55)
    print("  FASE 1: Feature Extraction (base congelada)")
    print("═" * 55)

    model = build_model(fine_tune=False)
    print_model_summary(model)

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks(),
        class_weight=class_weights,
        verbose=1
    )

    return model, history


# ─────────────────────────────────────────────────────
# FASE 2: FINE-TUNING (OPCIONAL)
# ─────────────────────────────────────────────────────

def train_phase2(model: tf.keras.Model,
                 train_ds, val_ds,
                 class_weights: dict = None,
                 epochs: int = 10) -> object:
    """
    Fase 2 — Descongela las últimas capas de MobileNetV2 y entrena con LR bajo.

    Args:
        model       – modelo entrenado en Fase 1
        train_ds    – tf.data.Dataset de entrenamiento
        val_ds      – tf.data.Dataset de validación
        class_weights
        epochs      – épocas adicionales para fine-tuning

    Returns:
        history_ft
    """
    from src.config import FINE_TUNE_AT, FINE_TUNE_LR

    print("\n" + "═" * 55)
    print("  FASE 2: Fine-Tuning (descongelamiento parcial)")
    print("═" * 55)


    base_model = model.get_layer("mobilenetv2_1.00_224") 
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    history_ft = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=get_callbacks(),
        class_weight=class_weights,
        verbose=1
    )

    return history_ft


# ─────────────────────────────────────────────────────
# GUARDADO DEL MODELO Y HISTORIAL
# ─────────────────────────────────────────────────────

def save_training_history(history, history_ft=None):
    """Guarda el historial de entrenamiento en JSON."""
    hist_data = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}

    if history_ft:
        for k, vals in history_ft.history.items():
            key_ft = f"ft_{k}"
            hist_data[key_ft] = [float(v) for v in vals]

    out_path = REPORTS_DIR / "training_history.json"
    with open(out_path, "w") as f:
        json.dump(hist_data, f, indent=2)
    print(f"\n💾 Historial de entrenamiento guardado en {out_path}")
    return hist_data


def save_model(model: tf.keras.Model):
    """Guarda el modelo final en formato .h5."""
    model.save(str(MODEL_SAVE_PATH))
    print(f"💾 Modelo guardado en {MODEL_SAVE_PATH}")
