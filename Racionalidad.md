## Monjes y Canibales

Hay 3 monjes y 3 caníbales que deben cruzar un río con una barca.
La barca solo lleva máximo 2 personas y nunca puede haber más caníbales que monjes en ninguna orilla, porque se los comerían.

### 1. Medida de rendimiento

Lo que define el éxito es:

* Que **todos los monjes y caníbales lleguen a la otra orilla** sanos y salvos.
* Cumpliendo las reglas:

  * La barca lleva máximo 2 personas.
  * No puede haber más caníbales que monjes en ningún lado.

➡ Objetivo: cruzar a todos con el **menor número de viajes posible** y sin estados inválidos.

---

### 2. Conocimiento del entorno

El agente sabe:

* Hay 3 monjes y 3 caníbales.
* La capacidad de la barca es de 1 o 2 personas.
* La barca necesita al menos una persona para moverse.
* Regla de seguridad:

  > Caníbales ≤ Monjes en cada orilla (si hay monjes presentes).

---

### 3. Acciones posibles

El agente puede:

* Subir a la barca:

  * 1 monje
  * 1 caníbal
  * 2 monjes
  * 2 caníbales
  * 1 monje y 1 caníbal
* Cruzar a la otra orilla.
* Regresar la barca con 1 o 2 personas.

➡ Cada acción debe generar un estado válido.

---

### 4. Secuencia de percepciones

El agente percibe en cada momento:

* Cuántos monjes hay en la orilla izquierda.
* Cuántos caníbales hay en la orilla izquierda.
* Posición de la barca (izquierda o derecha).
* Estado resultante después de cada cruce.

---

## Esposas Celosas

Tres matrimonios deben cruzar un río con una barca de máximo 2 personas.
Ninguna esposa puede quedarse con otro hombre si su esposo no está presente, porque se pondría celosa.

### 1. Medida de rendimiento

* Que las 6 personas (3 esposos y 3 esposas) crucen a la otra orilla.
* Respetando la regla:

  * Una esposa no puede estar con otro hombre si su esposo no está ahí.
* Usando la menor cantidad de viajes posible.

---

### 2. Conocimiento del entorno

El agente conoce:

* Hay 3 esposos y 3 esposas.
* La barca lleva máximo 2 personas.
* La barca no se mueve sola.
* Regla de celos que restringe los estados válidos.

---

### 3. Acciones posibles

* Subir a la barca:

  * 1 persona
  * 2 personas
* Combinaciones posibles:

  * esposo solo
  * esposa sola
  * matrimonio
  * dos esposos
  * dos esposas
* Cruzar y regresar la barca.

---

### 4. Percepciones

El agente percibe en cada momento:

* Cuántos esposos hay en cada orilla.
* Cuántas esposas hay en cada orilla.
* Posición actual de la barca.

---
## Ranas
Hay 6 ranas sobre piedras en fila:

* 3 ranas de un color a la izquierda
* 3 ranas de otro color a la derecha
* Un espacio vacío al centro

---

### 1. Medida de rendimiento

* Lograr que las 3 ranas de la izquierda pasen a la derecha
* y las 3 de la derecha pasen a la izquierda.
* Cumpliendo las reglas de movimiento
* en la menor cantidad de pasos posible.

---

### 2. Conocimiento del entorno

El agente sabe:

* Hay 6 ranas y un espacio vacío.
* Existen dos colores distintos.
* Reglas de movimiento:

  * mover a casilla adyacente vacía
  * saltar una rana del color contrario
  * no retroceder.

---

### 3. Acciones posibles

* Mover una rana al espacio vacío si está junto a él.
* Saltar sobre una rana del otro color hacia el espacio vacío.
* Elegir qué rana se moverá en cada turno.

---

### 4. Percepciones

El agente percibe:

* Posición de cada rana en la fila.
* Color de cada rana.
* Ubicación del espacio vacío.

---
