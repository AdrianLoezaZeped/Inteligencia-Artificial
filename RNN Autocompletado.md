## Codigo Utilizado
```
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Activation
import sys

# 1. PREPARACIÓN DE LOS DATOS
# Usaremos un texto simple para el entrenamiento
texto = "el tecnologico de morelia es una excelente escuela de ingenieria"
caracteres = sorted(list(set(texto)))
char_to_index = {c: i for i, c in enumerate(caracteres)}
index_to_char = {i: c for i, c in enumerate(caracteres)}

n_caracteres = len(caracteres)
longitud_secuencia = 10
paso = 1

# Crear secuencias de entrada (X) y el siguiente caracter (y)
entradas = []
salidas = []
for i in range(0, len(texto) - longitud_secuencia, paso):
    entradas.append(texto[i : i + longitud_secuencia])
    salidas.append(texto[i + longitud_secuencia])

# Vectorización (One-Hot Encoding para NumPy/TF)
X = np.zeros((len(entradas), longitud_secuencia, n_caracteres), dtype=bool)
y = np.zeros((len(entradas), n_caracteres), dtype=bool)

for i, secuencia in enumerate(entradas):
    for t, char in enumerate(secuencia):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[salidas[i]]] = 1

# ---------------------------------------------------------
# 2. SECCIÓN NUMPY (Lógica "a mano")
# ---------------------------------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rnn_step_forward(xt, a_prev, Waax, Waa, Wya, ba, by):
    # Ecuación del estado oculto
    a_next = np.tanh(np.dot(Waax, xt) + np.dot(Waa, a_prev) + ba)
    # Ecuación de la salida (predicción)
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    return a_next, yt_pred

# Inicialización de parámetros aleatorios
n_a = 64 # Unidades ocultas
Waax = np.random.randn(n_a, n_caracteres) * 0.01
Waa = np.random.randn(n_a, n_a) * 0.01
Wya = np.random.randn(n_caracteres, n_a) * 0.01
ba = np.zeros((n_a, 1))
by = np.zeros((n_caracteres, 1))

print("--- Parte NumPy inicializada con éxito ---")

# ---------------------------------------------------------
# 3. SECCIÓN TENSORFLOW (Modelo Funcional)
# ---------------------------------------------------------
print("\nEntrenando modelo con TensorFlow...")

model = Sequential([
    # Capa RNN que recibe la secuencia
    SimpleRNN(128, input_shape=(longitud_secuencia, n_caracteres)),
    # Capa densa para clasificar cuál es el siguiente caracter
    Dense(n_caracteres),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Entrenar el modelo
model.fit(X, y, batch_size=128, epochs=100, verbose=0)

# ---------------------------------------------------------
# 4. FUNCIÓN DE AUTOCOMPLETADO (Inferencia)
# ---------------------------------------------------------
def autocompletar(seed_text, caracteres_a_generar=20):
    resultado = seed_text
    print(f"\nSemilla inicial: '{seed_text}'")
    
    for i in range(caracteres_a_generar):
        # Preparar la entrada actual
        x_pred = np.zeros((1, longitud_secuencia, n_caracteres))
        for t, char in enumerate(seed_text[-longitud_secuencia:]):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1
        
        # Predecir con el modelo de TF
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = index_to_char[next_index]
        
        resultado += next_char
        seed_text += next_char
        
    return resultado

# Probar el autocompletado
print(autocompletar("el tecnolo"))
```
### Resultado de la simulacion
<img width="986" height="179" alt="image" src="https://github.com/user-attachments/assets/261b1ea6-a068-49fb-a434-ba9ddd60035e" />
