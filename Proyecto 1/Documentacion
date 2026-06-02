##  Proyecto 1: Sistema de Evasión con Red Neuronal Multicapa (Pygame + MLP)

### 1. Descripción General del Proyecto

Es un videojuego interactivo en 2D desarrollado en Python utilizando **Pygame** como motor gráfico. El objetivo central es que un agente (el personaje jugable) aprenda a sobrevivir evadiendo proyectiles disparados de forma aleatoria por una nave espacial alienígena (UFO). El juego implementa un enfoque de **aprendizaje por imitación** dividido en dos fases operativas:

* **Fase Manual (Generación del Dataset):** El usuario juega activamente para registrar un histórico de reacciones correctas ante diferentes escenarios.
* **Fase Automática (Inferencia del Modelo):** Una Red Neuronal Multicapa (MLP) toma el control del personaje en tiempo real basándose en los datos previamente recolectados.

### 2. Arquitectura de la Red Neuronal (MLP)

A diferencia de las arquitecturas binarias tradicionales (que solo calculan saltos), este modelo se diseñó como un **clasificador multiclase (3 clases)** utilizando `scikit-learn`:

* **Capas Ocultas (`hidden_layer_sizes=(10, 8)`):** Consiste en dos capas densas con 10 y 8 neuronas respectivamente.
* **Función de Activación:** `ReLU` (Rectified Linear Unit) para las capas ocultas, ideal para evitar el desvanecimiento del gradiente en arquitecturas densas.
* **Optimizador:** `Adam` (Adaptive Moment Estimation), configurado con un límite estricto de iteraciones (`max_iter=300000`) para garantizar la convergencia del entrenamiento.
* **Preprocesamiento de Datos:** Implementa `StandardScaler` para normalizar las características de entrada, asegurando que las distancias en píxeles y las velocidades escaladas no sesguen los pesos de la red.

---

### 3. Vector de Características (Features) y Clases de Salida

Para que el modelo tome decisiones precisas en cada frame ($45\text{ Hz}$), procesa un vector de entrada continuo $X$ y devuelve una acción discreta $y$:

#### Variables de Entrada (Features):

1. **Velocidad de la Bala (`velocidad_bala`):** Magnitud y dirección del proyectil (valores dinámicos aleatorios modificados por la escala de resolución).
2. **Distancia Relativa (`distancia`):** La separación absoluta en el eje X entre la posición del jugador y la bala (`abs(jugador.x - bala.x)`).
3. **Altura Relativa de la Bala (`altura_bala`):** Un factor crucial que añade complejidad al problema estadístico, mapeado en tres niveles discretos sobre el suelo:
* `0.0`: Nivel del suelo (requiere **Saltar**).
* `0.35`: Media altura (requiere **Saltar** o **Permanecer Normal** dependiendo de la velocidad).
* `0.70`: Bala alta (requiere **Agacharse**).



#### Clases de Salida (Mecánicas de Evasión):

El modelo mapea las entradas a tres posibles acciones probabilísticas ($y \in \{0, 1, 2\}$):

* **`0` (Normal):** El jugador permanece de pie en el suelo.
* **`1` (Salto):** Modifica el vector de posición vertical aplicando aceleración hacia arriba afectada por la gravedad simulada.
* **`2` (Agachado):** Aplica un cambio de estado físico alterando las dimensiones de la caja de colisión del sprite (`jugador.height`), reduciéndola en un 50% y ajustando su coordenada de origen para simular un efecto de compresión visual contra el suelo.

---

### 4. Características Técnicas del Código Implementado

* **Manejo Dinámico de Resoluciones:** El script calcula las físicas, velocidades de salto, gravedad y dimensiones de las cajas de colisión (`Rect`) aplicando un factor multiplicador (`self.scale`), permitiendo transiciones nativas a pantalla completa sin romper el comportamiento del modelo matemático.
* **Mitigación de Desbalances y Excepciones:** * El método de entrenamiento utiliza una división estratificada (`stratify=y`) mediante `train_test_split` (80% entrenamiento, 20% pruebas) con un umbral mínimo de 80 muestras para evitar sobreajuste catastrófico.
* **Manejo de Monoclase:** Si el dataset recolectado solo contiene registros de una única acción (por ejemplo, si el usuario solo saltó), el sistema está protegido para no colapsar; entra en un estado controlado de "clase única" prediciendo dicha acción con un Accuracy simulado de $1.0$.


* **Mecanismo de Selección por Pesos (Inferencia):** En lugar de un `argmax` rígido, el modo automático utiliza un muestreo probabilístico basado en pesos (`random.choices(clases, weights=probas)`), lo que aporta un comportamiento dinámico al agente y permite visualizar la distribución de probabilidad de las acciones en el HUD del juego.
* **Herramientas Analíticas Integradas:** El script cuenta con la exportación de las sesiones de juego a archivos estructurados (`datos_mlp.csv`), así como con la renderización de gráficos de dispersión 2D y 3D interactivos mediante `matplotlib` para auditar la separabilidad lineal del dataset generado.

---
### 5. Codigo Utilizado
```
import os
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Configuración segura de backends para Matplotlib en entornos interactivos
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importación necesaria para proyecciones 3D

# Activación del modo interactivo de matplotlib para actualizaciones de gráficas sin congelar Pygame
plt.ion()

# Constantes de resolución base para el escalado dinámico del entorno gráfico
BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1

# Características del Entorno: Niveles de altura relativa del proyectil (Eje Y)
# 0.0 = Al ras del suelo | 0.35 = Altura media | 0.70 = Altura alta
BULLET_HEIGHT_LEVELS = [0.0, 0.35, 0.70]


@dataclass
class Sample:
    """
    Estructura de datos (Data Class) para almacenar cada vector de características 
    y su respectiva etiqueta de acción durante la fase manual.
    """
    velocidad_bala: float  # Característica 1: Velocidad en píxeles/frame
    distancia: float       # Característica 2: Distancia absoluta en X entre jugador y bala
    altura_bala: float     # Característica 3: Altura parametrizada (0.0 a 1.0)
    accion: int            # Etiqueta (Target): 0 = Normal, 1 = Salto, 2 = Agachado


# Diccionario de mapeo para la interpretación de salidas del modelo y logs en el HUD/CSV
ACCION_NOMBRE = {0: "normal", 1: "salto", 2: "agachado"}


class Juego:
    def __init__(self) -> None:
        """ Inicialización del motor gráfico y configuración de estados de la IA. """
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        # Configuración inicial de la ventana
        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala multi-altura + MLP")

        # Paleta de colores en formato RGB
        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        # Estados de flujo del juego
        self.corriendo = True
        self.modo_auto = False  # False = Control Humano (Entrenamiento) | True = Control de la MLP (Inferencia)

        # Atributos del Pipeline de Aprendizaje Automático
        self.datos_modelo: List[Sample] = []        # Dataset en memoria (Buffer de datos)
        self.modelo: Optional[MLPClassifier] = None # Instancia del clasificador MLP de scikit-learn
        self.scaler: Optional[StandardScaler] = None# Normalizador estadístico Z-score
        self.modelo_entrenado = False               # Bandera de estado del modelo
        self.clase_unica: Optional[int] = None      # Flag de seguridad por si el usuario solo registra una acción
        self.ultima_proba_salto: Optional[str] = None # Texto para el HUD que muestra las probabilidades (Softmax)

        # Parámetros del bucle y muestreo de frames
        self.decision_window = 500
        self.decision_record_every = 3
        self._decision_frame_counter = 0

        # Dimensiones de objetos adaptables
        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (32, 48)
        self.altura_normal = self.player_size[1]
        self.altura_agachado = int(self.player_size[1] * 0.5)

        self.bullet_size = (16, 16)
        self.ship_size = (64, 64)
        self.fondo_speed = 3

        # Físicas del salto del personaje (Ecuaciones cinemáticas simples)
        self.salto = False
        self.en_suelo = True
        self.agachado = False
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial

        # Control de la animación del Sprite por frames
        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        # Variables de control del proyectil
        self.velocidad_bala = -12
        self.bala_disparada = False
        self.bala_altura_rel = 0.0
        
        # Coordenadas dobles para el efecto de fondo infinito (Parallax scroll)
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        # Ejecución del escalado inicial de elementos según la resolución de la pantalla
        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    # ─────────────────── RESOLUCIÓN / ASSETS ───────────────────
    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        """
        Calcula de forma dinámica las proporciones de los elementos visuales y las
        magnitudes físicas del entorno basándose en la resolución de pantalla actual.
        """
        self.w, self.h = int(w), int(h)
        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        # Escalado proporcional de hitboxes y posiciones relativas
        self.margin = int(50 * self.scale)
        ground_offset = int(100 * self.scale)
        self.ground_y = self.h - ground_offset

        self.player_size = (int(32 * self.scale), int(48 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        # Reajuste de las fuerzas físicas en función de la escala
        self.salto_vel_inicial = 15 * self.scale
        self.gravedad = 1 * self.scale
        self.salto_vel = self.salto_vel_inicial

        self.decision_window = int(500 * self.scale)

        # Inicialización de fuentes adaptables
        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        # Posicionamiento físico de los objetos en el plano (Eje X e Y)
        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(self.margin, self.ground_y, self.player_size[0], self.player_size[1])
            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0],
                self.bullet_size[1],
            )
            self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0],
                self.ship_size[1],
            )

    def _cargar_assets(self) -> None:
        """ Carga los archivos PNG desde el disco duro y aplica el reescalado dinámico. """
        def safe_load(path: str, size: Tuple[int, int], fallback_color=(200, 200, 200, 255)) -> pygame.Surface:
            """ Manejo de excepciones: Genera superficies de colores sólidos si falta un sprite. """
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback_color)
                return surf

        base = os.path.dirname(__file__)
        self.jugador_frames = [
            safe_load(os.path.join(base, "assets/sprites/mono_frame_1.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_2.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_3.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_4.png"), self.player_size),
        ]
        self.bala_img = safe_load(os.path.join(base, "assets/sprites/purple_ball.png"), self.bullet_size, (160, 120, 255, 255))
        self.fondo_img = safe_load(os.path.join(base, "assets/game/fondo2.png"), (self.w, self.h), (40, 40, 40, 255))
        self.nave_img = safe_load(os.path.join(base, "assets/game/ufo.png"), self.ship_size, (140, 255, 200, 255))

    def _toggle_fullscreen(self) -> None:
        """ Cambia entre ventana estándar y modo pantalla completa completa ajustando el hardware. """
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    # ─────────────────── ESTADO JUEGO / MODELO ───────────────────
    def _reset_estado_juego(self) -> None:
        """ Restablece los vectores de posición y variables cinemáticas para una nueva ronda. """
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y + int(10 * self.scale)
        self.bala_disparada = False
        self.bala_altura_rel = 0.0
        self.velocidad_bala = int(-10 * self.scale)
        self.salto = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self._decision_frame_counter = 0
        
        self.fondo_x1 = 0
        self.fondo_x2 = self.w
        self.agachado = False
        self.jugador.height = self.player_size[1]
        self.jugador.y = self.ground_y

    def _reset_modelo(self) -> None:
        """ Borra el cerebro de la IA (pesos de las sinapsis) y los parámetros del escalador. """
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None

    # ─────────────────── EXPORT / GRÁFICAS ───────────────────
    def exportar_datos_csv(self) -> str:
        """ Guarda el dataset acumulado en memoria física dentro de un documento CSV estructurado. """
        if not self.datos_modelo:
            return "No hay datos para exportar."
        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["velocidad_bala", "distancia", "altura_bala", "accion", "accion_nombre"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.altura_bala, s.accion, ACCION_NOMBRE.get(s.accion, "?")])
        except Exception as e:
            return f"Error al guardar CSV: {e}"
        
        conteo = {n: sum(1 for s in self.datos_modelo if s.accion == k) for k, n in ACCION_NOMBRE.items()}
        resumen = ", ".join(f"{n}={v}" for n, v in conteo.items())
        return f"CSV guardado ({len(self.datos_modelo)} filas) → {resumen}"

    def graficar_datos_2d(self) -> str:
        """ Renderiza un mapa de dispersión 2D (Distancia vs Velocidad) usando Matplotlib. """
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        cs = ["red" if s.accion == 1 else "blue" for s in self.datos_modelo]
        
        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num)
        plt.clf()
        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos entrenamiento MLP (rojo=salto, azul=no salto)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando gráfica 2D interactiva."

    def graficar_datos_3d(self) -> str:
        """ Renderiza un gráfico espacial 3D inyectando la altura de la bala como tercera dimensión. """
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = list(range(len(self.datos_modelo)))
        cs = ["red" if s.accion == 1 else "blue" for s in self.datos_modelo]
        
        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Índice (tiempo aproximado)")
        ax.set_title("Datos entrenamiento MLP 3D (rojo=salto, azul=no salto)")
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando gráfica 3D interactiva."

    # ─────────────────── BALA / SALTO ───────────────────
    def _calcular_y_bala(self, altura_rel: float) -> int:
        """ Traduce la altura abstracta parametrizada (0.0 a 0.70) a coordenadas físicas de píxeles (Eje Y). """
        player_h = self.player_size[1]
        offset_y = int(altura_rel * player_h)
        return self.ground_y + int(10 * self.scale) - offset_y

    def disparar_bala(self) -> None:
        """ Lanza un proyectil calculando de forma estocástica (aleatoria) su velocidad y nivel de altura. """
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            self.bala_altura_rel = random.choice(BULLET_HEIGHT_LEVELS)
            self.bala.y = self._calcular_y_bala(self.bala_altura_rel)
            self.bala_disparada = True

    def reset_bala(self) -> None:
        """ Reposiciona el objeto de la bala al extremo derecho para inicializar el siguiente ciclo de tiro. """
        self.bala.x = self.w - self.margin
        self.bala_disparada = False
        self.bala_altura_rel = 0.0

    def iniciar_salto(self) -> None:
        """ Activa las banderas lógicas encargadas de ejecutar las mecánicas del salto. """
        if self.en_suelo:
            self.salto = True
            self.en_suelo = False
            self.agachado = False
            self.jugador.height = self.player_size[1]
            self.jugador.y = self.ground_y
            self.altura_normal = self.player_size[1]
            self.altura_agachado = int(self.player_size[1] * 0.5)

    def iniciar_agacharse(self):
        """ Modifica las dimensiones físicas de la caja de colisión del jugador (reducida a la mitad). """
        if self.en_suelo and not self.agachado:
            self.agachado = True
            self.altura_normal = self.player_size[1]
            self.altura_agachado = int(self.player_size[1] * 0.5)
            self.jugador.height = self.altura_agachado
            # Desplaza la coordenada Y hacia abajo para simular que está pegado al suelo
            self.jugador.y = self.ground_y + (self.altura_normal - self.altura_agachado)

    def dejar_agacharse(self):
        """ Restablece la caja de colisión a sus valores antropomórficos por defecto. """
        if self.agachado:
            self.agachado = False
            self.jugador.height = self.player_size[1]
            self.jugador.y = self.ground_y

    def manejar_salto(self) -> None:
        """ Aplica los vectores de desplazamiento físico frame a frame restando la fuerza de gravedad. """
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:  # Condición de parada: Retorno al suelo
                self.jugador.y = self.ground_y
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True

    # ─────────────────── DATOS / MACHINE LEARNING (MLP) ───────────────────
    def registrar_decision_manual(self) -> None:
        """ Captura el estado actual del juego en cada ciclo de reloj para expandir el dataset. """
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)
        
        # Codificación de la etiqueta binaria/multiclase según el estado físico actual del jugador
        if not self.en_suelo:
            accion = 1   # Clase 1: Saltando
        elif self.agachado:
            accion = 2   # Clase 2: Agachado
        else:
            accion = 0   # Clase 0: Estado normal / Quieto

        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                altura_bala=float(self.bala_altura_rel),
                accion=accion,
            )
        )

    def entrenar_modelo(self) -> Tuple[bool, str]:
        """
        Extrae las características matemáticas, normaliza los datos y entrena
        la Red Neuronal Perceptrón Multicapa (MLP) utilizando Scikit-Learn.
        """
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas más datos (>= 80). Juega en MANUAL."
        
        # Construcción de matrices matemáticas de características (X) y etiquetas (y)
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        clases = sorted(set(y))
        conteo = {ACCION_NOMBRE.get(c, str(c)): y.count(c) for c in clases}
        info_clases = ", ".join(f"{n}={v}" for n, v in conteo.items())

        # --- PROTECCIÓN DE MONOCLASE (Por si el dataset carece de variedad estadística) ---
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            return True, f"MLP Entrenado ({info_clases}). Accuracy≈1.000"

        # División balanceada del dataset (80% para optimizar pesos, 20% para validación)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalado Estadístico: Previene el sesgo por diferencias de magnitudes en variables de entrada
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # INSTANCIACIÓN DE LA RED NEURONAL MULTICAPA (ARQUITECTURA DE CAJAS NEGRAS)
        # Capas: Entrada (3 neuronas) -> Oculta 1 (10) -> Oculta 2 (8) -> Salida (3 neuronas / Softmax)
        clf = MLPClassifier(
            hidden_layer_sizes=(10, 8),
            activation="relu",
            solver="adam",
            max_iter=300000,
            random_state=42,
        )
        
        # Ejecución del algoritmo de Backpropagation para ajustar los pesos sinápticos
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)  # Evaluación del rendimiento matemático del modelo
        
        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True
        return True, f"MLP 3-clases ({info_clases}). Accuracy≈{acc:.3f}"

    def decision_auto(self) -> str:
        """
        Fase de Inferencia: Pasa las características en vivo al modelo normalizado
        y predice la acción probabilística óptima usando una selección estocástica basada en pesos.
        """
        if not self.modelo_entrenado or not self.bala_disparada or not self.en_suelo:
            return "normal"

        distancia = abs(self.jugador.x - self.bala.x)

        # Inferencia en caso de redundancia monoclase
        if self.clase_unica is not None and self.modelo is None:
            if self.clase_unica == 1: return "saltar"
            if self.clase_unica == 2: return "agacharse"
            return "normal"

        if self.modelo is None or self.scaler is None:
            return "normal"

        # Vectorización de los datos actuales del fotograma
        X = [[float(self.velocidad_bala), float(distancia), float(self.bala_altura_rel)]]
        Xs = self.scaler.transform(X)  # Aplicación idéntica del normalizador estadístico

        # Extracción de las distribuciones probabilísticas (Softmax de la capa de salida)
        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            clases = list(self.modelo.classes_)
            
            # SELECCIÓN POR PESOS: En lugar de argmax rígido, selecciona la acción respetando las probabilidades de la red
            pred_clase = int(random.choices(clases, weights=probas, k=1)[0])
            
            prob_texto = "  ".join(f"{ACCION_NOMBRE.get(int(c), str(c))}={probas[i]:.2f}" for i, c in enumerate(clases))
            self.ultima_proba_salto = prob_texto
        else:
            pred_clase = int(self.modelo.predict(Xs)[0])
            self.ultima_proba_salto = ACCION_NOMBRE.get(pred_clase, str(pred_clase))

        if pred_clase == 1: return "saltar"
        if pred_clase == 2: return "agacharse"
        return "normal"

    # ─────────────────── MENÚ INTERACTIVO ───────────────────
    def _dibujar_menu(self, msg: str = "") -> None:
        """ Renderiza la interfaz del menú de control en pantalla completa o modo ventana. """
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)))

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa MLP; sin modelo NO salta)",
            "T - Entrenar MLP",
            "C - Exportar datos a CSV",
            "G - Gráfica 2D  /  H - Gráfica 3D",
            "F - Fullscreen (toggle)",
            "Q - Salir",
        ]
        x0 = int(80 * self.scale)
        y = int(140 * self.scale)
        line_h = self.fuente.get_linesize()
        pad = max(6, int(6 * self.scale))
        for op in opciones:
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += line_h + pad

        y += int(8 * self.scale)
        estado = [
            f"Memoria: {len(self.datos_modelo)} | Modelo: {'sí' if self.modelo_entrenado else 'no'}",
            f"Resolución: {self.w}x{self.h} | scale≈{self.scale:.2f} | ventana_decisión≈{self.decision_window}",
        ]
        for line in estado:
            t = self.fuente_chica.render(line, True, self.GRIS)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente_chica.get_linesize()

        if msg:
            mm = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(mm, (x0, y + int(12 * self.scale)))

        pygame.display.flip()

    def mostrar_menu(self) -> None:
        """ Bucle de interrupción encargado de capturar las configuraciones del usuario antes del loop principal. """
        msg = ""
        esperando = True
        self._decision_frame_counter = 0
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                    break
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos_modelo.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.modelo_entrenado:
                            msg = "Primero entrena el MLP (T) en esta sesión."
                        else:
                            self.modo_auto = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_modelo()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_g:
                        msg = self.graficar_datos_2d()
                    if e.key == pygame.K_h:
                        msg = self.graficar_datos_3d()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    # ─────────────────── RENDER / CORE LOOP ───────────────────
    def _update_frame(self) -> None:
        """ Renderiza y actualiza la posición gráfica de todos los objetos en el espacio 2D. """
        # Desplazamiento de las dos capas de imagen para generar el fondo sin fin
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w: self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w: self.fondo_x2 = self.w
        self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
        self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))

        # Control del temporizador interno para alternar fotogramas de la animación del mono
        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count = 0

        # Aplica una deformación matricial (Scale) al sprite si el jugador se encuentra agachado
        img_jugador = self.jugador_frames[self.current_frame]
        if self.agachado:
            img_jugador = pygame.transform.scale(img_jugador, (self.player_size[0], self.altura_agachado))
        self.pantalla.blit(img_jugador, (self.jugador.x, self.jugador.y))
        
        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        # Simulación del movimiento del proyectil en el espacio lineal (Eje X)
        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        # Detección de Colisiones AABB (Axis-Aligned Bounding Box) -> Reseteo de partida por impacto
        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

        # Renderización de telemetría de IA en tiempo real (HUD Superior Izquierdo)
        if self.modelo_entrenado and self.modo_auto and self.ultima_proba_salto is not None:
            txt = self.fuente_chica.render(str(self.ultima_proba_salto), True, self.AMARILLO)
            self.pantalla.blit(txt, (10, 10))

        # Renderización del estado operacional actual del sistema (HUD Superior Derecho)
        modo_txt = self.fuente_chica.render("AUTO" if self.modo_auto else "MANUAL", True, (100, 255, 100) if self.modo_auto else (255, 160, 80))
        self.pantalla.blit(modo_txt, (self.w - modo_txt.get_width() - 10, 10))

    def loop(self) -> None:
        """ Bucle principal del software: Escucha las interrupciones de teclado y gestiona los ciclos físicos. """
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p):  # Apertura en caliente del menú analítico
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    # Mapeo manual del salto por teclado del operador humano
                    elif e.key == pygame.K_SPACE and (not self.modo_auto) and self.en_suelo and not self.agachado:
                        self.iniciar_salto()

            if not self.corriendo:
                break

            if self.modo_auto:
                # ── MODO AUTOMÁTICO: La red MLP evalúa el frame e inyecta la acción cinemática calculada
                accion = self.decision_auto()
                if accion == "saltar":
                    self.dejar_agacharse()
                    self.iniciar_salto()
                elif accion == "agacharse":
                    self.iniciar_agacharse()
                else:
                    self.dejar_agacharse()
            else:
                # ── MODO MANUAL: Mapeo manual de la acción de agacharse y almacenamiento de vectores
                keys = pygame.key.get_pressed()
                if keys[pygame.K_DOWN] and not self.salto:
                    self.iniciar_agacharse()
                else:
                    self.dejar_agacharse()
                
                # Guarda las características matemáticas de este frame específico en el dataset
                self.registrar_decision_manual()

            # Procesamiento secundario de las físicas gravitacionales
            if self.salto:
                self.manejar_salto()

            # Desencadenador del sistema de disparo de proyectiles de la nave
            if not self.bala_disparada:
                self.disparar_bala()

            # Actualiza el búfer de imagen y limita la velocidad máxima a 45 FPS estables
            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()


def main() -> None:
    """ Punto de entrada único del script. """
    Juego().loop()


if __name__ == "__main__":
    main()
```
