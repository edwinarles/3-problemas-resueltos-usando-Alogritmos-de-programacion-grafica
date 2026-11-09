# Ejercicio 3: Modelo de Difusion (DDPM) para rostros de anime

Implementacion clara, didactica y minimalista de un **Modelo de Difusion Probabilistica (DDPM)** con arquitectura **U-Net** para la generacion de rostros de anime a partir de puro ruido gaussiano.

## Archivos
- `model.py`: Red U-Net con codificacion temporal sinusoidal y conexiones residuales (skip connections).
- `diffusion.py`: Formulas matematicas del proceso forward $q(x_t|x_0)$ y reverse sampling $p_\theta(x_{t-1}|x_t)$.
- `train.py`: Entrenamiento supervisando la prediccion de ruido $\epsilon$ con perdida MSE.
- `inference.py`: Genera muestras nuevas desde ruido puro y guarda una grilla de rostros generados (`rostros_anime_generados.png`).

## Como ejecutar

### 1. Obtener el Dataset
El script ya integra **kagglehub** automaticamente. Solo necesitas tenerlo instalado (`pip install kagglehub`).
- Si ejecutas el script tal cual, detectara y descargara `splcher/animefacedataset` de manera automatica.
- Si ya lo descargaste por tu cuenta, puedes pasarle la ruta que te imprimio Python:
  ```bash
  python train.py --data_dir "C:\Users\Edwin\.cache\kagglehub\datasets\splcher\animefacedataset\versions\1"
  ```

### 2. Entrenar
```bash
python train.py --epochs 30 --img_size 64 --max_images 5000
```
*(Puedes ajustar `--max_images` para limitar la cantidad de imagenes y acelerar el tiempo de entrenamiento, o poner `0` para usar todo el dataset).*

### 3. Generar rostros (Inferencia)
```bash
python inference.py --weights "checkpoints/diffusion_anime.pth" --num_samples 16
```
Salida:
- `resultados/rostros_anime_generados.png` (cuadricula 4x4 de rostros generados)
- `resultados/anime_rostro_01.png`, etc.

