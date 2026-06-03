#  Proyecto 4: Tutor Analítico Híbrido especializado en Seguridad Pública (RAG + Generación Controlada)

## 1. Descripción General del Proyecto

Este proyecto consiste en el desarrollo de un Tutor Inteligente diseñado para el análisis riguroso de documentos sobre seguridad pública y violencia en México. El sistema resuelve el problema de las alucinaciones en los Grandes Modelos de Lenguaje (LLMs) mediante una arquitectura **Retrieval-Augmented Generation (RAG)** emparejada con un diseño avanzado de *Prompt Engineering*.

El motor cognitivo principal está impulsado por el modelo **`google/gemma-2-2b-it`** (cargado en precisión `bfloat16` para optimización de VRAM), el cual fue condicionado para actuar bajo lineamientos académicos estrictos, priorizando la extracción de hechos reales sobre la generación creativa.

---

## 2. Ingesta de Datos y Base Vectorial (Pipeline RAG)

El sistema extrae, procesa y almacena la información de los PDFs de forma automatizada para construir su memoria a largo plazo:

* **Extracción y Limpieza:** Se utiliza la librería `PyMuPDF` (`fitz`) para extraer el texto y los metadatos (autor, título, páginas) de los documentos PDF cargados por el usuario.
* **Estrategia de Segmentación (Chunking):** Para mantener el contexto semántico sin desbordar la ventana de atención del modelo, el texto se divide en bloques (*chunks*) de **350 palabras**, con un **solape de 70 palabras** entre fragmentos adyacentes.
* **Vectorización (Embeddings):** Cada fragmento es transformado en un vector matemático utilizando el modelo especializado multilingüe `paraphrase-multilingual-MiniLM-L12-v2` a través de `SentenceTransformers`.
* **Almacenamiento y Recuperación:** Los vectores se indexan en una base de datos **ChromaDB** (`corpus_mx`). La recuperación (Top-K) se realiza mediante búsqueda de similitud del coseno, garantizando que solo los fragmentos más relevantes se inyecten en el contexto del LLM.

---

## 3. Control de Comportamiento y Dataset de Fine-Tuning

Para moldear la conducta del tutor, se estructuró un corpus de entrenamiento en formato JSONL (`dataset_finetuning.jsonl`) que establece los parámetros de respuesta del modelo:

* **Mitigación de Alucinaciones (Hard-coded):** El sistema está programado (tanto en ejemplos del dataset como en el *System Prompt*) para que, ante la falta de evidencia en el vector store, aborte la predicción y responda textualmente: *"La información proporcionada en el corpus no detalla este aspecto."*
* **Citación Dinámica:** Obliga al modelo a extraer los metadatos inyectados (Fuente y Página) y colocarlos al final de su intervención.
* **Estructura Socrática:** El dataset incluye pares de instrucción/salida donde el modelo responde a preguntas complejas (ej. militarización vs. prevención social) invitando al usuario a definir sus criterios de evaluación en lugar de emitir juicios sesgados.

---

## 4. Motor de Inferencia y Generación de Lenguaje

La fusión del contexto recuperado con las habilidades del modelo fundacional ocurre en el bloque del generador:

* **Ingeniería del Prompt (System Prompt):** Se inyecta un catálogo dinámico en tiempo real con los nombres de los PDFs disponibles para que el modelo tenga "conciencia" de su propio conocimiento. Se le imponen reglas estrictas de formato y limitación de alcance.
* **Parámetros de Decodificación (Hyperparameters):** El modelo `Gemma` genera respuestas utilizando una **temperatura baja (0.2)** para favorecer resultados deterministas y analíticos, apoyado por un `repetition_penalty` de 1.3 para evitar bucles de texto.
* **Filtro Post-Procesamiento:** Se implementó un algoritmo de limpieza que escanea la salida generada y elimina líneas con alta densidad de "basura numérica" (más del 60% de números), un error común al procesar tablas extraídas de PDFs.

---

## 5. Pipeline de Evaluación y Telemetría

El código incluye un banco de pruebas automatizado diseñado para someter al sistema a estrés analítico:

* **Matriz de Pruebas:** 10 preguntas divididas en tres niveles cognitivos:
* **Nivel 1:** Extracción directa (ej. identificar entidades con mayor tasa de homicidios).
* **Nivel 2:** Síntesis (ej. contrastar tipos de violencia rural vs. urbana).
* **Nivel 3:** Razonamiento y Límites (ej. preguntar sobre deserción escolar para disparar el mecanismo anti-alucinación).


* **Métricas Registradas:** El sistema calcula y exporta a `resultados_evaluacion.jsonl` la latencia de recuperación del RAG (en milisegundos), la latencia de generación del LLM, y las fuentes utilizadas para cada consulta, permitiendo una auditoría completa del rendimiento.
