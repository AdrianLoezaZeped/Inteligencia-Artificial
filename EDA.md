Clasificación multiclase supervisada.
El modelo deberá decidir entre: **saltar, agacharse o no realizar acción**, según el estado del entorno.

---

### Tipo de datos

Se trabajará con **datos estructurados numéricos** generados por el juego.
Cada registro representará el estado del sistema en un instante de decisión.

Posibles variables de entrada:

* Distancia jugador–bola
* Altura de la bola
* Velocidad de la bola
* Estado del jugador

Variable de salida:

* Acción tomada (0 = nada, 1 = saltar, 2 = agacharse)

---

### Arquitectura propuesta

Se propone una **Red Neuronal Multicapa (MLP)**:

* Capa de entrada (según número de variables)
* 1–2 capas ocultas con activación ReLU
* Capa de salida con 3 neuronas y función Softmax

---

