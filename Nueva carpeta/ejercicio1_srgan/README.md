# Ejercicio 1: SRGAN para mejora de videos (TV Peruana 2000-2010)

Este modulo implementa un modelo **SRGAN (Super-Resolution Generative Adversarial Network)** minimalista y directo para mejorar la resolucion (factor 4x) y nitidez de videos historicos o de baja definicion.

## Archivos
- `model.py`: Definicion del Generador (ResNet con PixelShuffle) y Discriminador convolucional.
- `train.py`: Extrae frames del video, genera pares de entrenamiento (Baja vs Alta resolucion) y entrena el modelo con funcion de perdida adversaria y de contenido.
- `inference.py`: Toma el video de entrada, procesa frame por frame con el generador entrenado, produce el video super-resuelto y genera una imagen comparativa (`comparacion_antes_despues.png`).

## Como ejecutar

### 1. Entrenar el modelo
Para entrenar usando el video disponible en la carpeta:
```bash
python train.py --video "../videoplayback (2).mp4" --epochs 20
```

### 2. Generar el video mejorado
```bash
python inference.py --video "../videoplayback (2).mp4" --weights "checkpoints/generator_srgan.pth" --output_video "video_mejorado.mp4"
```
Esto generara:
- `video_mejorado.mp4`: El video con resolucion 4x mejorada.
- `comparacion_antes_despues.png`: Imagen comparativa lado a lado (Original Bicubic vs SRGAN) lista para la presentacion.

