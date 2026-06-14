import argparse
import sys
from pathlib import Path
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from src.model import build_model
from src.config import IMG_SIZE, CLASS_NAMES

def load_trained_model(model_path):
    print(f"Cargando modelo de Inteligencia Artificial desde: {model_path}")
    if not Path(model_path).exists():
        print(f"Error: No se encontro el archivo del modelo en {model_path}.")
        print("Asegurate de haber descargado best_model.keras de Colab.")
        sys.exit(1)
        
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado con exito.\n")
    return model

def predict_image(model, image_path):
    if not Path(image_path).exists():
        print(f"Error: La imagen '{image_path}' no existe. Revisa la ruta.")
        return
    
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    
    img_array = img_array / 255.0
    
    img_batch = tf.expand_dims(img_array, 0)
    
    print("Analizando la imagen...")
    prediction = model.predict(img_batch, verbose=0)[0][0]
    
    label = "FAKE (Generada por IA)" if prediction > 0.5 else "REAL (Autentica)"
    
    confidence = prediction if prediction > 0.5 else (1.0 - prediction)
    
    print("\n" + "=" * 50)
    print(f" Archivo:    {Path(image_path).name}")
    print(f" Veredicto:  {label}")
    print(f" Certeza:    {confidence:.2%}")
    print("=" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Probador de imagenes para el Detector de Deepfakes")
    parser.add_argument("imagen", help="Ruta de la imagen que deseas clasificar")
    parser.add_argument("--modelo", default="outputs/models/best_model.keras", help="Ruta al modelo entrenado")
    
    args = parser.parse_args()
    
    model = load_trained_model(args.modelo)
    predict_image(model, args.imagen)

if __name__ == "__main__":
    main()
