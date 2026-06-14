"""
data_loader.py — Carga y preprocesamiento de datasets.

Responsabilidades:
- Leer imágenes locales desde las carpetas pre-divididas en archive/
- Aplicar Data Augmentation al conjunto de entrenamiento
- Retornar tf.data.Dataset listos para entrenar
"""

import os
import json
import tensorflow as tf
from pathlib import Path

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE,
    CLASS_NAMES, RANDOM_SEED, REPORTS_DIR
)

def _augment(img, label):
    """
    Data augmentation solo para entrenamiento:
    - Flip horizontal aleatorio
    - Ajustes aleatorios de brillo, contraste y saturación
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def get_datasets():
    """
    Pipeline optimizado: carga desde directorios pre-divididos usando Keras.
    
    Returns:
        train_ds, val_ds, test_ds, class_weights
    """
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio {TRAIN_DIR}")

    print("Cargando Datasets desde directorios pre-divididos...")

    # 1. Cargar usando image_dataset_from_directory
    # Esto asignará: real=0, fake=1 (basado en config.CLASS_NAMES)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
    )

    # 2. Normalización [0, 255] a [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds  = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Aplicar Data Augmentation SOLO al dataset de entrenamiento
    train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. Optimización de rendimiento (Prefetch)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds  = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 5. Contar archivos para calcular pesos de clase balanceados
    n_train_real = len(list((TRAIN_DIR / CLASS_NAMES[0]).glob("*.*")))
    n_train_fake = len(list((TRAIN_DIR / CLASS_NAMES[1]).glob("*.*")))
    n_total = n_train_real + n_train_fake
    
    if n_train_real == 0 or n_train_fake == 0:
        class_weights = {0: 1.0, 1: 1.0}
    else:
        class_weights = {
            0: n_total / (2.0 * n_train_real),
            1: n_total / (2.0 * n_train_fake),
        }

    print(f"\nPesos de clase: real={class_weights[0]:.3f} | fake={class_weights[1]:.3f}")

    # 6. Guardar estadísticas del dataset en outputs
    n_val = len(list(VALID_DIR.rglob("*.*")))
    n_test = len(list(TEST_DIR.rglob("*.*")))
    
    stats = {
        "train_total": n_total,
        "train_real": n_train_real,
        "train_fake": n_train_fake,
        "val_total": n_val,
        "test_total": n_test,
        "class_weights": class_weights,
    }
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = REPORTS_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Estadísticas guardadas en {stats_path}")

    return train_ds, val_ds, test_ds, class_weights
