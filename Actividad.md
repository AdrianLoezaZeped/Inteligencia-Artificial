## Actividad 1: La Matriz de Atención
*Evaluación del reparto de importancia elemental.*

`["El", "modelo", "procesa", "datos"]`

| Token Origen ($i$) | 0. El | 1. modelo | 2. procesa | 3. datos | **$\sum$ Fila** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. El** | **70%** | 30% | 0% | 0% | **100%** |
| **1. modelo** | 20% | **60%** | 20% | 0% | **100%** |
| **2. procesa** | 0% | 30% | **40%** | 30% | **100%** |
| **3. datos** | 0% | 10% | 40% | **50%** | **100%** |

---

## Actividad 2: La Palabra Ambigua (Dos Contextos)
*Demostración de cómo el contexto modifica el vector de características de un homógrafo.*

### Contexto A: *"Fui al banco a retirar mi dinero."*

| Token Evaluado | Fui | al | banco | a | retirar | mi | dinero | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2. banco** | 5% | 5% | **30%** | 0% | **30%** | 0% | **30%** | **100%** |

### Contexto B: *"Me senté en el banco de madera del parque."*

| Token Evaluado | Me | senté | en | el | banco | de | madera | del | parque | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4. banco** | 5% | **35%** | 5% | 5% | **25%** | 5% | **20%** | 0% | **0%** | **100%** |

---

## Actividad 3: Máscara Causal (No hacer trampa)

`["La", "IA", "genera", "texto"]`

Las posiciones futuras con respecto al token actual son enmascaradas con un valor absoluto de `0%` (`[MASK]`), forzando al modelo a mirar únicamente hacia la izquierda (pasado) y a sí mismo (presente).

| Token Origen ($i$) | 0. La | 1. IA | 2. genera | 3. texto | **$\sum$ Fila** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. La** | **100%** | `[0% MASK]` | `[0% MASK]` | `[0% MASK]` | **100%** |
| **1. IA** | 35% | **65%** | `[0% MASK]` | `[0% MASK]` | **100%** |
| **2. genera** | 10% | 50% | **40%** | `[0% MASK]` | **100%** |
| **3. texto** | 5% | 25% | 40% | **30%** | **100%** |
