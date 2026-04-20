## ¿Qué es una Vanilla RNN?

Imagina que tienes un **medidor de ánimo** que va de números negativos (triste) a positivos (feliz).

Cada día, tu ánimo de hoy depende de **dos cosas**:

1. **Lo que te pasó hoy** (evento)
2. **La mitad de cómo te sentías ayer** (tu memoria)

La fórmula es simplemente:

```
Ánimo hoy = Evento de hoy + (Ánimo de ayer ÷ 2)
```

Eso es todo. La "inteligencia" de esta red neuronal es esa suma.

---

## Misión 1 — ¿Por qué se olvida?

El lunes te ganas un premio → ánimo = **10**

| Día | Evento | Cálculo | Ánimo |
|-----|--------|---------|-------|
| Lunes | +10 | 10 + (0 ÷ 2) | **10.00** |
| Martes | 0 | 0 + (10 ÷ 2) | **5.00** |
| Miércoles | 0 | 0 + (5 ÷ 2) | **2.50** |
| Jueves | 0 | 0 + (2.5 ÷ 2) | **1.25** |
| Viernes | 0 | 0 + (1.25 ÷ 2) | **0.63** |

Cada día sin que pase nada, tu alegría se parte a la mitad. Para el viernes solo queda el **6.25%** de la alegría del lunes. Eso se llama **desvanecimiento de memoria**.

---

## Misión 2 — ¿Cuánto necesitas para ponerte feliz?

| Día | Evento | Cálculo | Ánimo |
|-----|--------|---------|-------|
| Día 1 | −6 | −6 + (0 ÷ 2) | **−6.00** |
| Día 2 | −4 | −4 + (−6 ÷ 2) | **−7.00** |
| Día 3 | 0 | 0 + (−7 ÷ 2) | **−3.50** |

El día 4, la fórmula queda:

```
Ánimo = Evento + (−3.5 ÷ 2) = Evento − 1.75
```

Para que el resultado sea mayor a cero, el evento tiene que ser **mayor a +1.75**. Con algo levemente bueno (+2 o más) ya sales del hoyo.

---

## Misión 3 — ¿Un día épico o varios días buenos?

| Día | Escenario A (pico aislado) | Escenario B (constante +3) |
|-----|---------------------------|---------------------------|
| Día 1 | 10 | 3.00 |
| Día 2 | 5 | 4.50 |
| Día 3 | 2.5 | 5.25 |
| Día 4 | 1.25 | 5.63 |
| Día 5 | **0.63** | **5.81** |

Un solo día increíble se olvida rapidísimo. Pero tener días consistentemente buenos acumula y crece. La RNN **prefiere la constancia sobre el pico aislado**.

---

## Conclusión

La Vanilla RNN funciona como la memoria emocional real: los eventos recientes importan mucho más que algo que pasó hace una semana. Su limitación principal es el **desvanecimiento de memoria** — información de hace varios pasos casi desaparece. Ese problema lo resuelven arquitecturas más avanzadas como **LSTM** y **GRU**.
