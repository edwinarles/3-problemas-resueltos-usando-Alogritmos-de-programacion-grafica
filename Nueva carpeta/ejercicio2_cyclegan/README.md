# Ejercicio 2: CycleGAN para traduccion de espectrogramas de canciones de 2 generos

Implementacion directa y simple de **CycleGAN** para transferir estilos musicales entre espectrogramas de 2 generos (ej. Rock a Jazz, o Cumbia a Reggaeton) de 30 segundos.

## Archivos
- `model.py`: Generador ResNet con InstanceNorm y Discriminador PatchGAN de 70x70.
- `audio_utils.py`: Conversion de audio (30s) a espectrograma Mel y reconstruccion de audio con Griffin-Lim.
- `train.py`: Entrenamiento no supervisado con perdidas adversarias, de ciclo (Cycle-consistency) y de identidad.
- `inference.py`: Traduce un espectrograma/audio del Genero A al Genero B, genera la comparativa visual y exporta el audio resultante.

## Como ejecutar

### 1. Preparar datos
Puedes colocar directamente audios o imagenes de espectrogramas en:
- `dataset/genero_A/`
- `dataset/genero_B/`

*(Si ejecutas directamente sin datos, `train.py` crea automaticamente muestras sinteticas para comprobar que todo el pipeline funciona).*

### 2. Entrenar
```bash
python train.py --dir_A dataset/genero_A --dir_B dataset/genero_B --epochs 25
```

### 3. Obtener resultados (Inferencia)
```bash
python inference.py --input "dataset/genero_A/muestra_A_00.png" --weights "checkpoints/G_AB.pth" --output_audio "audio_traducido.wav"
```
Salidas:
- `espectrograma_traducido.png`
- `comparacion_espectrogramas.png` (ideal para la presentacion)
- `audio_traducido.wav`

