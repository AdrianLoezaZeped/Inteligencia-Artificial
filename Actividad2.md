##  Actividad 6: Procesamiento en la Red Feed-Forward (FFN)
`["LA", "NIÑA", "PEQUEÑA", "COME", "FRUTA"]`

### Simulación de Activación y Filtrado No Lineal en la FFN:

| Token Posicional ($i$) | Magnitud / Densidad Inicial | Operación de Umbral (Corte ReLU) | Estado del Vector Post-FFN (Salida) |
| :--- | :---: | :--- | :--- |
| **0. LA** | 0.15 | `Si valor < 0.30 -> Forzar a 0` | **0.00** (Ruido sintáctico/artículo mitigado) |
| **1. NIÑA** | 0.85 | `Si valor >= 0.30 -> Mantener/Escalar`| **0.85** (Preservación del Sujeto Núcleo) |
| **2. PEQUEÑA** | 0.70 | `Si valor >= 0.30 -> Mantener/Escalar`| **0.70** (Preservación del Modificador/Adjetivo) |
| **3. COME** | 0.90 | `Si valor >= 0.30 -> Mantener/Escalar`| **0.90** (Preservación del Verbo/Acción Central) |
| **4. FRUTA** | 0.80 | `Si valor >= 0.30 -> Mantener/Escalar`| **0.80** (Preservación del Objeto Directo) |


---

## Actividad 7: Conexiones Residuales (Skip Connections / Add)
### Matriz de Suma Residual (Fusión de Identidad + Contexto):

| Token ($i$) | Vector Original ($X_{\text{in}}$) | Vector Post-FFN ($\text{FFN}(X)$) | Vector Residual Resultante ($X_{\text{res}}$) | Impacto Teórico en la Red |
| :--- | :---: | :---: | :---: | :--- |
| **0. LA** | 1.00 | 0.00 | **1.00** | Preserva la existencia del token estructural en la secuencia. |
| **1. NIÑA** | 1.00 | 0.85 | **1.85** | Maximiza la densidad del sujeto con carga contextual. |
| **2. PEQUEÑA**| 1.00 | 0.70 | **1.70** | Refuerza el atributo asociativo del adjetivo. |
| **3. COME** | 1.00 | 0.90 | **1.90** | Potencia la magnitud de la acción principal (verbo). |
| **4. FRUTA** | 1.00 | 0.80 | **1.80** | Consolida la estabilidad del objeto semántico. |

---

## Actividad 8: Capa de Normalización (Layer Normalization / Norm)

### Mapeo de Normalización de Capa (LayerNorm Simulation):

| Token ($i$) | Vector Residual ($X_{\text{res}}$) | Operación de Escalado / Normalización | Vector Final del Bloque ($Y$) |
| :--- | :---: | :--- | :---: |
| **0. LA** | 1.00 | $1.00 / 1.90$ | **0.52** |
| **1. NIÑA** | 1.85 | $1.85 / 1.90$ | **0.97** |
| **2. PEQUEÑA**| 1.70 | $1.70 / 1.90$ | **0.89** |
| **3. COME** | 1.90 | $1.90 / 1.90$ | **1.00** |
| **4. FRUTA** | 1.80 | $1.80 / 1.90$ | **0.94** |

---

## Actividad 9: Capa de Proyección de Salida (Linear/Unembedding Layer)

`FRUTA`

| Índice Vocabulario | Token Candidato | Logit Bruto ($z_i$) | Significado del Estado Lineal |
| :---: | :--- | :---: | :--- |
| 0 | `fresca` | **4.5** | Alta afinidad semántica y concordancia de género. |
| 1 | `correr` | -1.2 | Incompatibilidad sintáctica (Verbo tras sustantivo). |
| 2 | `roja` | **3.8** | Alta afinidad asociativa con el objeto directo. |
| 3 | `el` | -3.5 | Incompatibilidad morfosintáctica total. |

---

## Actividad 10: Aplicación de la Función Softmax

### Tabla de Distribución de Probabilidad Exponencial:

| Token Candidato | Logit ($z_i$) | Exponencial ($e^{z_i}$) | Probabilidad Final ($P$) | Estatus de Selección |
| :--- | :---: | :---: | :---: | :--- |
| **fresca** | 4.5 | 90.02 | **64.2%** | Máxima probabilidad asignada. |
| **roja** | 3.8 | 44.70 | **31.9%** | Alta probabilidad (Alternativa válida). |
| **correr** | -1.2 | 0.30 | **0.2%** | Descartado por el modelo. |
| *Otros* | -- | 5.18 | **3.7%** | Distribución residual del vocabulario. |
| **Suma Total** | | **140.20** | **100%** | **Filtro Estocástico Válido** |

---

## Actividad 11: Control de Hiperparámetros — Temperatura ($T$)

En esta simulación, alteramos los logits aplicando el factor de escala de Temperatura ($T$) antes de la Softmax ($\frac{z_i}{T}$). Analizamos cómo el profesor demuestra el cambio de comportamiento en el modelo:

* **Temperatura Baja ($T = 0.2$):** Los logits se distancian drásticamente. `fresca` sube al $>95\%$, volviendo al modelo determinista, predecible y repetitivo.
* **Temperatura Alta ($T = 1.5$):** Los logits se aplanan. Las probabilidades de `fresca` y `roja` se nivelan, y palabras raras ganan terreno, incrementando la "creatividad" o aleatoriedad de la IA.

---

## Actividad 12: Estrategias de Muestreo (Top-K y Top-P / Nucleus Sampling)
Emulamos las reglas de truncamiento de tokens antes de realizar la selección final:

* **Muestreo Top-K:** Si configuramos $K=2$, el modelo secciona la tabla de la Actividad 10 y se queda **únicamente con los dos tokens más altos** (`fresca` y `roja`), eliminando el resto del vocabulario instantáneamente.
* **Muestreo Top-P (Nucleus):** Si configuramos $P=0.90$, el modelo acumula las probabilidades en orden descendente hasta alcanzar el 90%. En este caso, `fresca` ($64.2\%$) + `roja` ($31.9\%$) = $96.1\%$, por lo que el núcleo se cierra ahí y excluye cualquier palabra residual que ensucie la generación.

---

## Actividad 13: Selección del Siguiente Token (Sampling) 

* **Resultado del muestreo en papel:** El sistema selecciona de forma definitiva el token con mayor peso: **"fresca"**. 
* El estado del texto muta en memoria y se actualiza a: `"LA NIÑA PEQUEÑA COME FRUTA FRESCA"`.

---

##  Actividad 14: Ciclo de Autoregresión Completo (The Generation Loop)

La Actividad 14 consolida el funcionamiento global de la arquitectura en un ciclo infinito de retroalimentación:

1. El nuevo token generado (`fresca`) se añade al final de la secuencia de entrada.
2. El contexto se expande a una longitud $N+1$.
3. **Reinicio de la Fase 1:** Toda la frase se vuelve a inyectar al bloque Transformer, generando una nueva Matriz de Atención donde las palabras originales ahora le prestan atención también a `fresca`.
4. El ciclo se repite de forma síncrona palabra por palabra hasta que el modelo predice un token especial de parada llamado **`<|endoftext|>` (EOS - End of Sequence)**, dando por concluida la generación de texto.

