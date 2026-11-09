import os
import glob
import numpy as np
from PIL import Image

def audio_to_spectrogram(audio_path, output_image_path=None, duration=30, sr=22050, n_mels=128):
    """
    Convierte un fragmento de audio (ej. 30s) a un espectrograma Mel guardado como imagen PNG.
    Usa librosa si esta disponible, o numpy/scipy basico.
    """
    try:
        import librosa
        # Cargar audio de 30 segundos
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        # Espectrograma de Mel en escala logaritmica (dB)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Normalizar a rango [0, 255]
        norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
        img_array = (norm * 255.0).astype(np.uint8)
        
        # Guardar como imagen
        img = Image.fromarray(img_array).convert("L") # 1 canal escala de grises
        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            img.save(output_image_path)
        return img, S_dB.min(), S_dB.max(), sr
    except ImportError:
        print("[!] Advertencia: Para procesar audio directamente instala librosa: pip install librosa")
        return None, 0, 0, sr


def spectrogram_to_audio(img_or_path, output_audio_path, db_min=-80.0, db_max=0.0, sr=22050, n_mels=128):
    """
    Reconstruye una senal de audio a partir de una imagen de espectrograma Mel usando Griffin-Lim.
    """
    try:
        import librosa
        import soundfile as sf
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path).convert("L")
        else:
            img = img_or_path
            
        norm = np.array(img, dtype=np.float32) / 255.0
        S_dB = norm * (db_max - db_min) + db_min
        S = librosa.db_to_power(S_dB)
        
        # Griffin-Lim para reconstruir la fase
        y = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_mels=n_mels)
        os.makedirs(os.path.dirname(output_audio_path) if os.path.dirname(output_audio_path) else ".", exist_ok=True)
        sf.write(output_audio_path, y, sr)
        print(f"[OK] Audio reconstruido guardado en: {output_audio_path}")
    except ImportError:
        print("[!] Se requiere librosa y soundfile para reconstruir audio a partir del espectrograma.")

