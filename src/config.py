"""
config.py — Configuración centralizada del proyecto VeneCapital Deepfake Detector.
Todos los hiperparámetros y rutas se gestionan desde aquí para reproducibilidad.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# RUTAS DEL PROYECTO
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # raíz del proyecto

DATA_DIR        = BASE_DIR / "archive" / "real_vs_fake" / "real-vs-fake"
TRAIN_DIR       = DATA_DIR / "train"
VALID_DIR       = DATA_DIR / "valid"
TEST_DIR        = DATA_DIR / "test"

OUTPUTS_DIR     = BASE_DIR / "outputs"
MODELS_DIR      = OUTPUTS_DIR / "models"
PLOTS_DIR       = OUTPUTS_DIR / "plots"
REPORTS_DIR     = OUTPUTS_DIR / "reports"

# Crear directorios de salida si no existen
for d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# HIPERPARÁMETROS DEL MODELO
# ─────────────────────────────────────────────
IMG_SIZE        = (224, 224)   # Entrada requerida por MobileNetV2
IMG_CHANNELS    = 3
BATCH_SIZE      = 32
EPOCHS          = 20
LEARNING_RATE   = 1e-4

# División del dataset (estratificada)
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15

# Semilla para reproducibilidad
RANDOM_SEED     = 42

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE TRANSFER LEARNING
# ─────────────────────────────────────────────
BASE_MODEL_NAME     = "MobileNetV2"
FINE_TUNE_AT        = 100      # Capa desde la que se descongela en fine-tuning
FINE_TUNE_LR        = 1e-5     # LR reducido para fine-tuning

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE CALLBACKS
# ─────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 5
LR_REDUCE_PATIENCE      = 3
LR_REDUCE_FACTOR        = 0.2
LR_REDUCE_MIN           = 1e-7

# ─────────────────────────────────────────────
# ETIQUETAS
# ─────────────────────────────────────────────
CLASS_NAMES     = ["real", "fake"]   # 0=real, 1=fake
LABEL_MAP       = {"real": 0, "fake": 1}

# ─────────────────────────────────────────────
# RUTAS DE GUARDADO
# ─────────────────────────────────────────────
MODEL_SAVE_PATH         = MODELS_DIR / "deepfake_detector.h5"
BEST_MODEL_PATH         = MODELS_DIR / "best_model.h5"
METRICS_REPORT_PATH     = REPORTS_DIR / "metrics_report.json"
CONFUSION_MATRIX_PATH   = PLOTS_DIR  / "confusion_matrix.png"
LEARNING_CURVES_PATH    = PLOTS_DIR  / "learning_curves.png"
ROC_CURVE_PATH          = PLOTS_DIR  / "roc_curve.png"
