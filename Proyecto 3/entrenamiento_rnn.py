import json
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ARCHIVO_CORPUS = 'Proyecto3.cpp'
BLOCK_SIZE = 32  
EPOCHS = 60     
BATCH_SIZE = 16

print("1. Leyendo datos...")
with open(ARCHIVO_CORPUS, 'r', encoding='utf-8') as f:
    texto = f.read()

chars = sorted(list(set(texto)))
stoi = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)

print("2. Creando meta.json (Requisito del profe)...")
with open('meta.json', 'w', encoding='utf-8') as f:
    json.dump({"block_size": BLOCK_SIZE, "chars": chars}, f, ensure_ascii=False)

print("3. Preparando secuencias...")
datos = [stoi[c] for c in texto]
X, Y = [], []
for i in range(len(datos) - BLOCK_SIZE):
    X.append(datos[i : i + BLOCK_SIZE])
    Y.append(datos[i + 1 : i + BLOCK_SIZE + 1]) 

X = np.array(X)
Y = np.array(Y)

print("4. Entrenando modelo Keras...")
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=BLOCK_SIZE),
    tf.keras.layers.SimpleRNN(128, return_sequences=True, activation='tanh'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Entrenar hasta que aprenda perfecto
modelo.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Guardar exactamente como el profe lo pide
modelo.save('model.keras')
print("¡Listo! model.keras y meta.json generados con éxito.")