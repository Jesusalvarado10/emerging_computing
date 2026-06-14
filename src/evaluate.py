"""
evaluate.py — Evaluación del modelo y generación de visualizaciones.

Genera:
- Matriz de confusión
- Curvas de aprendizaje (loss / accuracy / AUC)
- Curva ROC
- Reporte JSON con todas las métricas
"""

import json
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [name-defined]
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    f1_score, accuracy_score
)

from src.config import (
    CONFUSION_MATRIX_PATH, LEARNING_CURVES_PATH,
    ROC_CURVE_PATH, METRICS_REPORT_PATH, CLASS_NAMES
)



COLORS = {
    "primary":   "#6C63FF",
    "secondary": "#FF6584",
    "success":   "#2DD4BF",
    "bg":        "#1A1A2E",
    "text":      "#E2E8F0",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor":   COLORS["bg"],
    "axes.edgecolor":   COLORS["text"],
    "axes.labelcolor":  COLORS["text"],
    "xtick.color":      COLORS["text"],
    "ytick.color":      COLORS["text"],
    "text.color":       COLORS["text"],
    "grid.alpha":       0.2,
})


# ─────────────────────────────────────────────────────
# PREDICCIONES
# ─────────────────────────────────────────────────────

def get_predictions(model, test_ds) -> tuple:
    """
    Genera predicciones sobre el conjunto de prueba.

    Returns:
        y_true  – etiquetas reales
        y_pred  – probabilidades predichas [0,1]
        y_class – clases predichas (umbral 0.5)
    """
    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.flatten().tolist())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_class = (y_pred >= 0.5).astype(int)

    return y_true, y_pred, y_class


# ─────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_class) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy":  round(float(accuracy_score(y_true, y_class)), 4),
        "precision": round(float(precision_score(y_true, y_class)), 4),
        "recall":    round(float(recall_score(y_true, y_class)), 4),
        "f1_score":  round(float(f1_score(y_true, y_class)), 4),
        "auc_roc":   round(float(roc_auc), 4),
    }

    print("\n" + "═" * 45)
    print("  RESULTADOS EN CONJUNTO DE PRUEBA")
    print("═" * 45)
    for name, val in metrics.items():
        bar = "█" * int(val * 30)
        print(f"  {name:<12} {val:.4f}  {bar}")
    print()
    print(classification_report(y_true, y_class, target_names=CLASS_NAMES))

    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📄 Métricas guardadas en {METRICS_REPORT_PATH}")

    return metrics, fpr, tpr, roc_auc


# ─────────────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_class):
    """Genera y guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_class)
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="RdPu",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="#333",
        ax=ax
    )
    ax.set_title("Matriz de Confusión", fontsize=16, fontweight="bold",
                 color=COLORS["primary"], pad=15)
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Etiqueta Real", fontsize=12)

    plt.tight_layout()
    fig.savefig(str(CONFUSION_MATRIX_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Matriz de confusión guardada en {CONFUSION_MATRIX_PATH}")


def plot_learning_curves(history_data: dict):
    epochs = range(1, len(history_data["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Curvas de Aprendizaje — VeneCapital Deepfake Detector",
                 fontsize=14, fontweight="bold", color=COLORS["primary"])

    metrics_pairs = [
        ("loss",     "val_loss",     "Loss",     COLORS["secondary"]),
        ("accuracy", "val_accuracy", "Accuracy", COLORS["success"]),
        ("auc",      "val_auc",      "AUC-ROC",  COLORS["primary"]),
    ]

    for ax, (train_key, val_key, title, color) in zip(axes, metrics_pairs):
        if train_key in history_data:
            ax.plot(epochs, history_data[train_key],
                    color=color, linewidth=2, label="Entrenamiento", alpha=0.9)
        if val_key in history_data:
            ax.plot(epochs, history_data[val_key],
                    color=color, linewidth=2, linestyle="--",
                    label="Validación", alpha=0.9)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Épocas")
        ax.set_ylabel(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(COLORS["bg"])

    plt.tight_layout()
    fig.savefig(str(LEARNING_CURVES_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Curvas de aprendizaje guardadas en {LEARNING_CURVES_PATH}")


def plot_roc_curve(fpr, tpr, roc_auc: float):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=COLORS["secondary"],
            lw=1.5, linestyle="--", label="Clasificador Aleatorio")

    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])

    ax.set_title("Curva ROC — Clasificador Deepfake",
                 fontsize=15, fontweight="bold", color=COLORS["primary"])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    fig.savefig(str(ROC_CURVE_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Curva ROC guardada en {ROC_CURVE_PATH}")


# ─────────────────────────────────────────────────────
# PIPELINE COMPLETO DE EVALUACIÓN
# ─────────────────────────────────────────────────────

def full_evaluation(model, test_ds, history_data: dict):
    print("\n🔍 Evaluando modelo sobre el conjunto de prueba...")

    y_true, y_pred, y_class = get_predictions(model, test_ds)
    metrics, fpr, tpr, roc_auc = compute_metrics(y_true, y_pred, y_class)

    plot_confusion_matrix(y_true, y_class)
    plot_learning_curves(history_data)
    plot_roc_curve(fpr, tpr, roc_auc)

    print(f"\n✅ Evaluación completada. Todas las gráficas en outputs/plots/")
    return metrics
