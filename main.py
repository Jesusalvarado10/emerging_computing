"""
main.py — Punto de entrada del proyecto VeneCapital Deepfake Detector.

Uso:
    python main.py                   # Pipeline completo (Fase 1)
    python main.py --fine-tune       # Con Fine-Tuning (Fase 2)
    python main.py --eval-only       # Solo evaluación (requiere modelo guardado)
"""

import argparse
import sys
import tensorflow as tf

from src.data_loader import get_datasets
from src.train import train_phase1, train_phase2, save_training_history, save_model
from src.evaluate import full_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="VeneCapital — Clasificador de Imágenes Deepfake (MobileNetV2)"
    )
    parser.add_argument(
        "--fine-tune", action="store_true",
        help="Ejecutar Fase 2 de fine-tuning tras el entrenamiento inicial"
    )
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=10,
        help="Número de épocas para el fine-tuning (default: 10)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Solo evaluar un modelo ya guardado en outputs/models/best_model.h5"
    )
    return parser.parse_args()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║       VeneCapital — Deepfake Image Classifier        ║
║       MobileNetV2 + Transfer Learning                ║
║       Jesus Alvarado · Luis Marcano · Alejo Hdz.    ║
╚══════════════════════════════════════════════════════╝
""")


def check_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"🚀 GPU detectada: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("⚠️  No se detectó GPU. El entrenamiento será en CPU (más lento).")
        print("   → Considera usar Google Colab con GPU activada.")


def main():
    print_banner()
    args = parse_args()
    check_gpu()

    if args.eval_only:
        from src.config import BEST_MODEL_PATH, REPORTS_DIR
        import json

        print(f"\n📂 Cargando modelo desde {BEST_MODEL_PATH}...")
        model = tf.keras.models.load_model(str(BEST_MODEL_PATH))

        _, _, test_ds, _ = get_datasets()

        hist_path = REPORTS_DIR / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history_data = json.load(f)
        else:
            history_data = {}

        full_evaluation(model, test_ds, history_data)
        return

    train_ds, val_ds, test_ds, class_weights = get_datasets()

    model, history = train_phase1(train_ds, val_ds, class_weights)

    history_ft = None
    if args.fine_tune:
        history_ft = train_phase2(
            model, train_ds, val_ds,
            class_weights=class_weights,
            epochs=args.fine_tune_epochs
        )

    history_data = save_training_history(history, history_ft)
    save_model(model)

    metrics = full_evaluation(model, test_ds, history_data)
    print("\n" + "═" * 50)
    print("  🏁 ENTRENAMIENTO COMPLETADO")
    print("═" * 50)
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print("\n  Archivos generados:")
    print("  • outputs/models/best_model.h5")
    print("  • outputs/plots/confusion_matrix.png")
    print("  • outputs/plots/learning_curves.png")
    print("  • outputs/plots/roc_curve.png")
    print("  • outputs/reports/metrics_report.json")
    print("═" * 50)


if __name__ == "__main__":
    main()
