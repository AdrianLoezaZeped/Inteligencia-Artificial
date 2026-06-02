#  Proyecto 2: Clasificación Multiclase de Imágenes de Animales mediante Redes Neuronales Convolucionales (CNN) y Transfer Learning

## 1. Descripción General del Proyecto

Este proyecto consiste en el diseño, implementación y evaluación de un sistema de visión por computadora capaz de clasificar imágenes digitales en cinco categorías biológicas distintas: **Ranas (Frogs), Arañas (Spiders), Monos (Monkeys), Ballenas (Whales) y Pájaros (Birds)**.

A partir de un código base originalmente diseñado para la identificación de modelos de automóviles (`CNNriesgo`), se realizó una reingeniería del pipeline de datos y de la arquitectura de la red. El sistema final combina la extracción de características de bajo nivel mediante capas convolucionales con la potencia de modelos masivos preentrenados, logrando una clasificación precisa incluso ante variaciones de iluminación, escala y fondos complejos.

---

## 2. Herramientas Utilizadas para la Creación del Dataset

El aprovisionamiento, limpieza y estructuración del conjunto de datos se realizó mediante un flujo de trabajo híbrido enfocado en la consistencia estadística:

* **Fuentes de Origen y Almacenamiento:** Las imágenes fueron recopiladas en directorios estructurados locales (mapeados en rutas del sistema como `C:/Users/adria/Downloads/`).
* **Mecanismo de Ingesta y Preprocesamiento:** Se implementaron los módulos nativos de **Keras/TensorFlow** (`utils.image_dataset_from_directory`) para realizar una carga perezosa (*lazy loading*) y eficiente desde el almacenamiento en disco.
* **Estandarización de Dimensiones:** Para mitigar el impacto de las resoluciones nativas heterogéneas de las imágenes web, cada muestra se reajustó dimensionalmente mediante interpolación bilineal a un tamaño unificado de **$224 \times 224$ píxeles** con 3 canales de color (RGB).
* **Estrategia de Partición (Splitting):** El dataset se dividió rigurosamente bajo un esquema clásico de **80% para el conjunto de entrenamiento** (*Training Set*) destinado a la optimización de pesos y **20% para el conjunto de validación/prueba** (*Validation Set*) enfocado en la auditoría del sobreajuste (*overfitting*).

---

## 3. Arquitectura del Modelo de Red Neuronal (CNN)

Para maximizar la eficiencia y reducir los tiempos de convergencia en el entrenamiento, se optó por una arquitectura híbrida basada en **Transfer Learning** utilizando un modelo base robusto (como MobileNetV2 o ResNet de la biblioteca `tensorflow.keras.applications`), congelando sus capas iniciales y añadiendo un bloque clasificador personalizado en la parte superior (*Head*).

La estructura de capas implementada en Keras se detalla en la siguiente tabla:

### Parámetros de Compilación del Modelo:

* **Función de Pérdida (*Loss Function*):** `categorical_crossentropy` (o `sparse_categorical_crossentropy` dependiendo de la codificación de las etiquetas), ideal para problemas de clasificación excluyente multiclase.
* **Optimizador:** `Adam` con una tasa de aprendizaje (*Learning Rate*) adaptativa controlada para evitar oscilaciones destructivas en el descenso de gradiente.
* **Métrica de Evaluación:** `accuracy` (precisión global).

---

## 4. Pipeline de Inferencia y Demostración Práctica

El script incorpora un flujo analítico de predicción en caliente diseñado para validar imágenes externas (por ejemplo, el caso de prueba interactivo `predecir_imagen_google('C:/Users/adria/Downloads/monos10.jpg')`).

### Pasos Ejecutados en Inferencia:

1. **Carga de Imagen Externa:** Se lee el archivo binario utilizando la biblioteca `PIL` (Pillow).
2. **Preprocesamiento Espejo:** La imagen se reescala exactamente a $224 \times 224$ píxeles y sus canales numéricos se convierten en arreglos de NumPy (`np.array`), aplicando el mismo factor de normalización lineal que el dataset de entrenamiento.
3. **Expansión de Dimensiones:** Se ejecuta `np.expand_dims(foto_array, axis=0)` para transformar la matriz de 3D a 4D, simulando un tamaño de lote (*batch size*) igual a 1, requisito sintáctico estricto de Keras.
4. **Inferencia Probabilística:** El método `transfer_model.predict()` procesa los tensores y devuelve un vector matemático con las 5 probabilidades calculadas por la función Softmax en la salida de la red.
5. **Extracción del Ganador:** Mediante `np.argmax()` se obtiene el índice del elemento con mayor probabilidad, y con `np.max()` se extrae el porcentaje exacto de seguridad.
6. **Visualización Académica:** Utilizando `matplotlib`, se despliega la imagen limpia en pantalla, ocultando los ejes coordenados e imprimiendo dinámicamente un título interactivo con la predicción del modelo (Ej: *"La IA dice: Monos (98.45%)"*), junto con el desglose numérico detallado en la consola.

---
