## Actividad 2.1 — Mapeo de Variables

Ecuación fundamental: $h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1} + b)$

| Variable | Frase del poema |
|---|---|
| $x_t$ | *"Soy la novedad pura, el pulso del instante, la matriz de características que el mundo me da en este segundo."* |
| $h_{t-1}$ | *"el fantasma del pasado, que trae consigo el resumen de todo lo que hemos vivido hasta ayer."* |
| $W_{hx}, W_{hh}$ | *"cruzamos por peajes inmutables, barreras que multiplican nuestra importancia y deciden qué tanto valemos."* |
| $b$ | *"un pequeño desvío inevitable"* |
| $\tanh$ | *"un muro curvo que nos comprime entre el -1 y el 1, evitando que nuestra energía explote hacia el infinito."* |
| $h_t$ | *"nazco yo, una nueva identidad. Soy tu estado actual, la respuesta de hoy, y estoy listo para ser el fantasma de tu mañana."* |

---

## Actividad 2.2 — Análisis de Dimensionalidad

**Datos:** $x_t \in \mathbb{R}^{20}$, $h_{t-1} \in \mathbb{R}^{64}$

### 1. Dimensión de $W_{hx}$

Debe transformar $x_t$ (tamaño 20) al espacio oculto (tamaño 64):

$$W_{hx} \in \mathbb{R}^{64 \times 20}$$

Verificación: $W_{hx} \cdot x_t = (64 \times 20)(20 \times 1) = (64 \times 1)$ 

### 2. Dimensión de $W_{hh}$

Debe transformar $h_{t-1}$ (tamaño 64) al mismo espacio oculto (tamaño 64):

$$W_{hh} \in \mathbb{R}^{64 \times 64}$$

Verificación: $W_{hh} \cdot h_{t-1} = (64 \times 64)(64 \times 1) = (64 \times 1)$ 

### 3. Dimensión de $h_t$

Ambos productos dan vectores de tamaño 64. Al sumarlos y aplicar $\tanh$ (que no cambia dimensiones):

$$h_t \in \mathbb{R}^{64}$$

---

## Actividad 2.3 — La Estrofa Perdida (el bias $b$)

> *Soy el ajuste silencioso, el empuje sin memoria ni entrada.*  
> *No vengo del mundo ni del pasado — vivo dentro de la red.*  
> *Sin mí, la curva siempre pasa por el origen,*  
> *rígida y ciega. Yo la libero, la desplazo,*  
> *para que pueda encontrar la verdad donde realmente vive.*

---

## Actividad 2.4 — El Límite del Muro Curvo (Saturación)

### 1. La función y su derivada

- $f(z) = \tanh(z)$: tiene forma de "S", va de −1 a +1, con pendiente máxima en $z = 0$.
- $f'(z) = 1 - \tanh^2(z)$: máxima en $z = 0$ (vale 1), y casi cero en los extremos.

### 2. ¿Qué pasa con $f'(500)$?

$$f'(500) = 1 - \tanh^2(500) \approx 1 - (1)^2 = 0$$

La derivada es prácticamente **cero**.

### 3. ¿Por qué es catastrófico?

Durante el entrenamiento, los pesos se ajustan usando el gradiente. Si la derivada es ≈ 0, el gradiente que se propaga hacia atrás también se vuelve ≈ 0 — la red **deja de aprender** porque no hay señal de corrección. No importa cuánto error haya, los pesos no se actualizan. Este fenómeno se conoce como el problema del **gradiente que desaparece** (*vanishing gradient*).

---

## Actividad 2.5 — El Eco del Castigo (BPTT)

El error en $h_t$ debe recorrer el siguiente camino en reversa:

1. **Atraviesa el muro curvo** ($\tanh$) en sentido inverso → se multiplica por su derivada $f'(z) = 1 - \tanh^2(z)$.
2. **Cruza el peaje recurrente** ($W_{hh}$) en sentido inverso → se multiplica por $W_{hh}^T$.

Solo así llega a $h_{t-1}$ para corregir el "fantasma del pasado".

La operación matemática que representa este viaje es la **regla de la cadena** (*chain rule*) del cálculo diferencial:

$$\frac{\partial L}{\partial h_{t-1}} = \frac{\partial L}{\partial h_t} \cdot W_{hh}^T \cdot (1 - \tanh^2(z))$$

Si la red tiene muchos pasos de tiempo, esta multiplicación se repite en cada paso y el gradiente se encoge hasta desaparecer — otro síntoma del *vanishing gradient*.

---

## Actividad 2.6 — Depuración del Código

### El error

El operador `*` en NumPy realiza multiplicación **elemento a elemento** (*element-wise*), no multiplicación matricial. Para matrices de distintas dimensiones esto produce resultados incorrectos o un colapso de dimensionalidad.

```python
# Código incorrecto
def paso_rnn_erroneo(x_t, h_prev, W_hx, W_hh, b):
    combinacion = (W_hx * x_t) + (W_hh * h_prev) + b  # * es element-wise
    return np.tanh(combinacion)
```

### Código corregido

Se usa el operador `@` (o `np.dot()`) para realizar la multiplicación matricial correcta:

```python
# Código correcto
def paso_rnn_correcto(x_t, h_prev, W_hx, W_hh, b):
    combinacion = (W_hx @ x_t) + (W_hh @ h_prev) + b  # @ es multiplicación matricial
    return np.tanh(combinacion)
```

**Verificación de dimensiones:**

- `W_hx @ x_t` → $(64 \times 20) \cdot (20 \times 1) = (64 \times 1)$ 
- `W_hh @ h_prev` → $(64 \times 64) \cdot (64 \times 1) = (64 \times 1)$ 
- Suma + bias $b \in \mathbb{R}^{64}$ → resultado final $(64 \times 1)$ 

Esto implementa correctamente la transformación afín: $h_t = \tanh(W_{hx}x_t + W_{hh}h_{t-1} + b)$
