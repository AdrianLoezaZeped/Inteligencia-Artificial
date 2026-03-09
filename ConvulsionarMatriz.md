# 1. Convolución (Conv)

En la pizarra aparece una **imagen 3×3** representada como matriz:

[
\begin{matrix}
x_1 & x_2 & x_3 \
x_4 & x_5 & x_6 \
x_7 & x_8 & x_9
\end{matrix}
]

Cada (x_i) es **un pixel de la imagen**.

Luego se aplica un **filtro o kernel 2×2**.

La convolución consiste en:

1. Colocar el filtro sobre una parte de la imagen
2. Multiplicar elemento por elemento
3. Sumar los resultados

Ejemplo conceptual:

[
Conv = \sum (x_{ij} \cdot w_{ij}) + bias
]

donde:

* (x) = valores de la imagen
* (w) = pesos del filtro
* **bias** = sesgo

También aparece en la pizarra:

[
XW + bias
]

que es la **operación básica de una neurona**.

---

# 2. ReLU (Función de activación)

Después de la convolución se aplica **ReLU**.

[
ReLU(x) = \max(0,x)
]

Esto significa:

* Si el valor es **positivo → se queda igual**
* Si es **negativo → se vuelve 0**

Ejemplo:

| Entrada | Salida ReLU |
| ------- | ----------- |
| 8       | 8           |
| 9       | 9           |
| -3      | 0           |
| -2      | 0           |

Por eso en la pizarra algunos valores se vuelven **0**.

Esto ayuda a que la red **aprenda patrones importantes**.

---

# 3. Feature Maps

En la parte derecha se ven **varios filtros 3×3**.

Cada filtro genera una **feature map** diferente.

Proceso:

```
Imagen
   ↓
Filtro 1 → mapa de características
Filtro 2 → mapa de características
Filtro 3 → mapa de características
```

Cada filtro detecta algo distinto:

* bordes
* texturas
* formas

---

# 4. Clasificación final

En la parte izquierda aparece **softmax**.

Softmax convierte los valores finales en **probabilidades**.

[
Softmax(x_i)=\frac{e^{x_i}}{\sum e^{x}}
]

Ejemplo:

| Clase | Probabilidad |
| ----- | ------------ |
| 8     | 0.90         |
| 9     | 0.10         |

La red elige **la probabilidad más alta**.

---

# 5. Flujo completo de la CNN

Lo que muestra la pizarra es este flujo:

```
Imagen
   ↓
Convolución
   ↓
ReLU
   ↓
Feature Maps
   ↓
Softmax
   ↓
Clasificación
```

---

Una **CNN** funciona así:

1. La imagen se representa como una matriz de pixeles.
2. Se aplican **filtros (kernels)** mediante **convolución**.
3. El resultado pasa por **ReLU**, que elimina valores negativos.
4. Se generan **feature maps** que detectan patrones.
5. Finalmente **softmax** convierte los resultados en probabilidades para clasificar la imagen.


## Convulsionar la matriz						
```
160	100	90	0	40				
120	120	65	17	17		-1	-2	-1
160	111	66	121	202	*	0	0	0
200	100	75	12	10		1	2	1
204	160	170	12	10				
								
255	255	255	116	51			
11	0	84	255	255				
140	50	0	0	0				
137	246	148	0	0				
0	0	0	0	0
```	
