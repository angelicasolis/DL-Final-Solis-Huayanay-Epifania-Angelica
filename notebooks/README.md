
# Clasificación automática de productos de moda con CNN

> **Proyecto Final · Modelos de Deep Learning · CENTRUM PUCP**
> **Autora:** Solís Huayanay, Epifanía Angélica
> **Fecha:** Mayo 2026

---

## 📋 Resumen

Este proyecto implementa una **red neuronal convolucional (CNN)** desde cero para resolver un problema de clasificación multiclase: identificar la categoría de prendas de vestir y accesorios a partir de imágenes en escala de grises de 28×28 píxeles. Se utiliza el dataset **Fashion-MNIST** (Zalando Research) y se evalúa el modelo con métricas estándar: accuracy, F1-score, matriz de confusión y análisis de errores.

El modelo resultante alcanza **92.38 % de accuracy** sobre el conjunto de prueba (10.000 imágenes nunca vistas durante el entrenamiento), demostrando que arquitecturas convolucionales compactas pueden capturar eficazmente patrones visuales jerárquicos en escenarios de retail / e-commerce.

---

## 🎯 Problema y objetivo

**Problema.** En plataformas de comercio electrónico, la catalogación manual de productos es costosa y lenta. Cuando un comerciante sube una foto, idealmente la plataforma debería clasificarla automáticamente sin intervención humana.

**Objetivo.** Entrenar una CNN desde cero capaz de clasificar prendas en **10 categorías** (Camiseta, Pantalón, Pulóver, Vestido, Abrigo, Sandalia, Camisa, Zapatilla, Bolso, Botín), evaluar su rendimiento y discutir su aplicabilidad en escenarios reales de retail.

**Valor para la toma de decisiones:** un clasificador automático con buena precisión permite **automatizar la catalogación**, **mejorar la búsqueda por categoría** y **detectar productos mal etiquetados** en plataformas de e-commerce y sistemas de inventario.

---

## 📊 Dataset

**Fashion-MNIST** (Zalando Research, 2017) es un benchmark estándar diseñado como reemplazo más realista de MNIST:

| Característica | Valor |
|---|---|
| Imágenes de entrenamiento | 60.000 |
| Imágenes de prueba | 10.000 |
| Categorías | 10 (perfectamente balanceadas) |
| Resolución | 28×28 píxeles |
| Canales | 1 (escala de grises) |
| Tamaño | ~30 MB |
| Carga | `tensorflow.keras.datasets.fashion_mnist.load_data()` |

> Ventaja práctica: el dataset se descarga automáticamente con una sola línea, sin necesidad de subir archivos a Colab.

---

## 🧠 Modelo

CNN compacta inspirada en VGG, con **~102.000 parámetros**:

```
Input (28×28×1)
  → Conv2D(32, 3×3) → BN → MaxPool(2×2)
  → Conv2D(64, 3×3) → BN → MaxPool(2×2)
  → Conv2D(128, 3×3) → BN → GlobalAveragePooling
  → Dropout(0.4) → Dense(64, ReLU)
  → Dropout(0.3) → Dense(10, Softmax)
```

**Decisiones de diseño:**

| Decisión | Justificación |
|---|---|
| 3 bloques Conv2D + MaxPooling | Captura patrones jerárquicos: bordes → texturas → formas completas |
| BatchNormalization en cada bloque | Acelera y estabiliza el entrenamiento |
| GlobalAveragePooling (no Flatten) | Reduce parámetros y previene overfitting |
| Dropout 0.4 / 0.3 | Regulariza el cabezal denso |
| Adam, lr = 1e-3 | Estándar moderno con convergencia rápida |
| EarlyStopping + ReduceLROnPlateau | Detención inteligente y ajuste adaptativo del learning rate |

**Hiperparámetros:** 15 épocas máx., batch size 128, split 54.000 / 6.000 / 10.000 (train / val / test) estratificado.

---

## 📈 Resultados

| Métrica | Valor |
|---|---|
| **Accuracy en test** | **92.38 %** |
| **F1 macro** | **0.924** |
| **F1 ponderado** | **0.924** |
| Tasa de error | 7.62 % (762 / 10.000) |

**F1 por clase — extremos:**

| Mejores | F1 | Más difíciles | F1 |
|---|---:|---|---:|
| Pantalón | 0.990 | Camisa | 0.781 |
| Bolso | 0.987 | Camiseta | 0.877 |
| Sandalia | 0.984 | Abrigo | 0.880 |
| Botín | 0.968 | Pulóver | 0.884 |

**Top 5 confusiones:**

1. Camiseta → Camisa (87 errores)
2. Camisa → Camiseta (87 errores)
3. Pulóver → Abrigo (49 errores)
4. Abrigo → Camisa (43 errores)
5. Vestido → Abrigo (32 errores)

> Los errores se concentran en **prendas superiores que comparten silueta** a 28×28 px, una limitación esperable a esta resolución y consistente con la confusión visual humana.

---

## 💡 Conclusiones

**Hallazgos técnicos:**

- Una CNN compacta entrenada desde cero alcanza rendimiento competitivo en clasificación multiclase de prendas, sin necesidad de modelos más grandes ni transfer learning.
- `GlobalAveragePooling` + `Dropout` regularizan eficazmente sin sacrificar precisión.
- Las clases con mayor error son visualmente similares incluso para humanos a baja resolución.

**Limitaciones:**

- Resolución 28×28 px y escala de grises pierden información relevante en moda real (color, textura, estampado).
- Imágenes centradas y sin contexto: en producción real las fotos tienen fondos variables, ángulos diversos y oclusiones.

**Valor para la toma de decisiones (retail / e-commerce):**

- **Catalogación automática** en plataformas tipo Mercado Libre o tiendas online de marcas peruanas.
- **Búsqueda mejorada** clasificando fotos subidas por vendedores informales.
- **Inventario visual** en bodegas con cámaras que cuentan productos por categoría.
- **Auditoría de catálogo** detectando productos mal etiquetados por baja confianza del modelo.

**Próximos pasos:**

- Aplicar **data augmentation** (rotaciones, zoom, desplazamientos).
- Migrar a un dataset con imágenes a color y mayor resolución (ej. *Fashion Product Images*).
- Implementar **transfer learning** con MobileNetV2 / EfficientNet.
- Desplegar el modelo en una **API web** (FastAPI + React) que reciba imágenes y devuelva la categoría predicha.

---

## 🛠️ Stack tecnológico

- **Python 3.10+**
- **TensorFlow / Keras 2.20**
- **scikit-learn** (split estratificado, métricas)
- **Pandas, NumPy, Matplotlib, Seaborn** (EDA y visualización)
- **Google Colab** (entorno de ejecución)

---

## 🚀 Cómo reproducir

1. **Abrir el notebook en Google Colab:**
   - Abre `notebooks/final_project.ipynb` directamente desde GitHub o súbelo a Colab (`Archivo → Subir cuaderno`).

2. **Activar GPU (opcional pero recomendado):**
   - Menú `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU`.

3. **Ejecutar todas las celdas:**
   - `Ctrl + F9` o menú `Entorno de ejecución → Ejecutar todo`.
   - El dataset se descarga automáticamente (~30 MB en segundos).
   - Tiempo de entrenamiento: ~2 minutos en GPU T4, ~10 minutos en CPU.

**Reproducibilidad:** semilla global `SEED=42` fijada en `numpy`, `random` y `tensorflow`.

---

## 📁 Estructura del repositorio

```
DL-Final-Solis-Huayanay-Epifania-Angelica/
├── notebooks/
│   └── final_project.ipynb       # Notebook principal ejecutable
├── README.md                      # Este archivo
└── results/                       # (opcional) gráficas y outputs
    ├── matriz_confusion.png
    ├── curvas_entrenamiento.png
    └── distribucion_clases.png
```
