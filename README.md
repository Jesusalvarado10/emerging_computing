# VeneCapital — Clasificador de Imágenes Deepfake

> Clasificación de imágenes faciales reales y deepfake usando Redes Neuronales Convolucionales (MobileNetV2 + Transfer Learning)

**Autores:** Jesus Alvarado · Luis Marcano · Alejo Hernandez  
**Institución:** VeneCapital  
**Fecha:** Caracas, 2026

---

## 🔗 Enlaces de Entrega Oficial (Proyecto 2)

*   **Informe (Google Docs):** [https://docs.google.com/document/d/1opg8STuevFuQZUCTYeuoaOVealHe1HMIR1MN4Boqm_g/edit?tab=t.0]
*   **Notebook Público (Google Colab):** [https://colab.research.google.com/drive/13yelre8tHFJsmnX2d8d8i1tjV73atSMk?usp=sharing]
*   **Documento AI Disclosure:** Se adjunta en el ZIP de entrega como `AI_Disclosure.pdf`

---

## 🧠 Descripción

Sistema de clasificación binaria que detecta si una imagen facial es **real** o ha sido manipulada mediante técnicas **deepfake**. Implementado con Transfer Learning sobre MobileNetV2 (pre-entrenada en ImageNet).

## 📁 Estructura del Proyecto

```
Proyecto/
├── data/
│   ├── raw/real/          # Imágenes reales (del dataset de Kaggle)
│   ├── raw/fake/          # Imágenes deepfake (del dataset de Kaggle)
│   └── processed/         # Imágenes preprocesadas
├── notebooks/
│   ├── 01_exploracion_dataset.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_entrenamiento.ipynb
│   └── 04_evaluacion.ipynb
├── src/
│   ├── config.py          # Configuración centralizada
│   ├── data_loader.py     # Carga y preprocesamiento
│   ├── model.py           # Arquitectura del modelo
│   ├── train.py           # Entrenamiento
│   └── evaluate.py        # Evaluación y métricas
├── outputs/
│   ├── models/            # Modelos guardados
│   ├── plots/             # Gráficas generadas
│   └── reports/           # Reportes de métricas
├── main.py
└── requirements.txt
```

## 🚀 Instalación

```bash
# 1. Clonar / abrir el proyecto
cd Proyecto

# 2. Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

## 📦 Dataset

Descargar uno de los siguientes datasets de Kaggle y colocar imágenes en `data/raw/real/` y `data/raw/fake/`:

| Dataset | Tamaño | Link |
|---|---|---|
| 140k Real and Fake Faces *(recomendado)* | ~70k real + 70k fake | [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) |
| Real and Fake Face Detection | ~2k imágenes | [Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) |

## ▶️ Uso

```bash
# Ejecutar el pipeline completo
python main.py

# O ejecutar paso a paso con los notebooks en notebooks/
```

## 🏗️ Arquitectura

```
Input 224×224×3
    → MobileNetV2 (base congelada, ImageNet)
    → GlobalAveragePooling2D
    → Dense(256, ReLU) + BatchNorm
    → Dropout(0.5)
    → Dense(1, Sigmoid)   →   real (0) / fake (1)
```

## 📊 Métricas Objetivo

- Accuracy > 90%
- AUC-ROC > 0.92
- Precision y Recall balanceados

## 🛠️ Tecnologías

| Herramienta | Uso |
|---|---|
| Python 3.10+ | Lenguaje principal |
| TensorFlow / Keras | Construcción y entrenamiento del modelo |
| MobileNetV2 | Arquitectura base (Transfer Learning) |
| OpenCV | Manipulación de imágenes |
| Scikit-learn | Métricas y splits |
| Matplotlib / Seaborn | Visualizaciones |
| Google Colab | Entrenamiento con GPU |

## 📚 Referencias

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Yagual Castillo, J. M. (2024). Detección de Deepfakes con Machine Learning. UPSE.
- Al-Dulaimi & Kurnaz (2024). Hybrid CNN-LSTM for Deepfake Detection. *Electronics*.
