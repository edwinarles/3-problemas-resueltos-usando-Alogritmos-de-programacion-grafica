# Guía para la Presentación y Exposición (7 Minutos)
**Rúbrica:** Código (7 pts) | Presentación (5 pts) | Exposición (8 pts)

---

## Estructura de Diapositivas Sugerida

### Diapositiva 1: Portada (30 segundos)
- **Título**: Modelos Generativos Profundos: SRGAN, CycleGAN y Difusión (DDPM).
- **Integrantes**: (Nombres del equipo de hasta 2 personas).
- **Curso / Fecha**: Viernes 13 de Noviembre.

---

### Diapositiva 2: Ejercicio 1 - SRGAN para Videos de TV Peruana (2000-2010) (2 minutos)
- **Problema**: Los videos de la televisión peruana antigua (grabados en VHS o señales análogas de 360p/480i) sufren de falta de nitidez, artefactos de compresión y baja resolución.
- **Arquitectura**:
  - Generador basado en ResNet (bloques residuales con conexiones skip) + PixelShuffle (sub-pixel convolution) para escalar 4x sin emborronar.
  - Discriminador convolucional con LeakyReLU y Batch Normalization.
- **Ajustes y Adaptaciones realizadas**:
  - Se sustituyó la pérdida exclusivamente de píxeles (MSE) por una pérdida combinada (Contenido + Adversarial $10^{-3}$), evitando que los frames resulten "lavados" o difusos.
  - Procesamiento frame a frame con extracción por lotes para no saturar memoria VRAM.
- **Resultados**: Muestra de `comparacion_antes_despues.png` (frame original reescalado por Bicúbico vs SRGAN reconstruido).

---

### Diapositiva 3: Ejercicio 2 - CycleGAN para Espectrogramas Musicales (2 minutos)
- **Problema**: Transferir el estilo de un género musical A a un género B (clips de 30 segundos) sin pares de audio idénticos (unpaired data).
- **Representación**: Conversión del audio a Espectrogramas Mel en escala logarítmica (dB) en formato imagen.
- **Arquitectura y Pérdidas**:
  - 2 Generadores ($G_{AB}$, $G_{BA}$) con bloques residuales e Instance Normalization (esencial para estilo musical).
  - 2 Discriminadores PatchGAN (evalúan realismo local en parches de 70x70).
  - **Pérdida de Consistencia de Ciclo**: $G_{BA}(G_{AB}(A)) \approx A$ garantiza que no se pierda la estructura armónica original.
  - **Pérdida de Identidad**: Previene alteraciones excesivas en timbres ya pertenecientes al dominio destino.
- **Adaptaciones**: Reconstrucción de audio con el algoritmo Griffin-Lim a partir de la magnitud predicha por el generador.

---

### Diapositiva 4: Ejercicio 3 - Modelo de Difusión para Rostros de Anime (2 minutos)
- **Problema**: Generar rostros de anime realistas y diversos a partir del dataset *Anime Face Dataset*.
- **Fundamento Matemático (DDPM)**:
  - **Proceso Forward ($q$)**: Adición progresiva de ruido gaussiano en $T=300$ pasos temporales mediante un schedule lineal de $\beta$.
  - **Proceso Reverse ($p_\theta$)**: Red U-Net entrenada para predecir el ruido residual $\epsilon$ añadido en el paso $t$.
- **Arquitectura U-Net**:
  - Conexiones residuales (skip connections) para mantener alta fidelidad de detalles faciales (ojos, contornos).
  - Incorporación de embeddings temporales sinusoidales (*Sinusoidal Time Embeddings*) para condicionar la red al nivel de ruido de cada paso.
- **Resultados**: Mostrar la cuadrícula `rostros_anime_generados.png`.

---

### Diapositiva 5: Conclusiones y Comparativa (30 segundos)
- **SRGAN**: Ideal para restauración y escalado de imágenes/videos existentes con alta fidelidad perceptual.
- **CycleGAN**: Potente para traducción de dominio a dominio no supervisada (aplicable a espectrogramas de audio).
- **Difusión (DDPM)**: Superior en estabilidad de entrenamiento respecto a GANs clásicas (no colapso de modo), logrando máxima calidad en síntesis de imágenes.

