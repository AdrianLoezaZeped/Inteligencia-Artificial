#  Proyecto 3: Asistente de Código Personalizado en C++ con Redes Neuronales Recurrentes (RNN Vanilla) e Integración en VS Code

## 1. Descripción General del Proyecto

Este proyecto implementa un modelo de lenguaje autorregresivo a nivel de caracteres diseñado para asistir en la escritura de código fuente. A diferencia de las herramientas comerciales genéricas, este sistema fue entrenado exclusivamente con un corpus propio de funciones en C++ (`Proyecto3.cpp`), lo que le permite autocompletar líneas de código imitando un estilo de programación específico y consistente.

El núcleo matemático es una **Red Neuronal Recurrente (RNN Vanilla)** construida en TensorFlow/Keras, la cual se integró exitosamente como una herramienta de uso diario dentro del editor **Visual Studio Code (VS Code)** mediante un pipeline de comunicación entre procesos (IPC) a través de los flujos estándar (stdio).

---

## 2. Creación del Dataset y Preprocesamiento a Nivel de Caracteres

El modelo no procesa palabras completas, sino que entiende el código fuente como una secuencia continua de caracteres matemáticamente codificados.

1. **Corpus de Entrenamiento (`Proyecto3.cpp`):** Un archivo de texto plano que contiene más de 60 funciones únicas en C++, manteniendo una nomenclatura y estructura estandarizada.
2. **Mapeo del Vocabulario (`meta.json`):** Durante el preprocesamiento, se extrajeron todos los caracteres únicos presentes en el corpus (letras, símbolos de C++, llaves, saltos de línea). A cada carácter se le asignó un índice numérico entero.
3. **Generación de Secuencias (Ventana Deslizante):** El código se dividió en secuencias de longitud fija $L$. Para cada secuencia de entrada $X$ (ej. `int main() { `), la etiqueta objetivo $y$ es el siguiente carácter inmediato. Esto permite que la red aprenda la distribución de probabilidad condicional $P(x_{t+1} | x_1, x_2, \dots, x_t)$.

---

## 3. Arquitectura del Modelo (RNN Vanilla)

El archivo `entrenamiento_rnn.py` define y entrena el modelo de Keras que posteriormente es exportado en formato binario (`model.keras`). La arquitectura de la red se estructura de la siguiente manera:

---

## 4. Arquitectura de Integración y Despliegue (VS Code)

Para que el modelo abandone el entorno de pruebas de Jupyter/Python y se convierta en un producto utilizable, se diseñó un puente de comunicación eficiente:

* **El Motor de Inferencia (`server_stdio.py`):** Actúa como un demonio (daemon) local en segundo plano. Al iniciar, carga el modelo preentrenado (`model.keras`) y el diccionario de caracteres (`meta.json`). Se mantiene en un bucle infinito escuchando las secuencias de código entrantes a través de la entrada estándar (`stdin`) y devolviendo las predicciones (autocompletados generados) por la salida estándar (`stdout`).
* **La Extensión del Editor (`extension.js` & `package.json`):** Desarrollada mediante la API de VS Code. Registra un `InlineCompletionItemProvider`, el cual se dispara cada vez que el usuario teclea. La extensión captura el contexto previo del cursor, se lo inyecta al proceso de Python, y luego renderiza la respuesta del modelo directamente en el editor como "texto fantasma" (ghost text), permitiendo al programador aceptar la sugerencia presionando la tecla *Tab*.

---
