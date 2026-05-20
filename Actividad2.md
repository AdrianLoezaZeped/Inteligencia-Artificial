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
