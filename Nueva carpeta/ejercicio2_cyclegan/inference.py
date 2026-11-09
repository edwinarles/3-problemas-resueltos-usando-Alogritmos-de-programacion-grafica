import os
import argparse
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import CycleGANGenerator
from audio_utils import audio_to_spectrogram, spectrogram_to_audio

def get_device(requested="auto"):
    if requested in ["cuda", "auto"] and torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            test_conv = torch.nn.Conv2d(1, 1, 1).cuda()
            _ = test_conv(test_tensor.unsqueeze(0).unsqueeze(0))
            return torch.device("cuda")
        except Exception:
            print("[!] Aviso: GPU no compatible con esta version de PyTorch. Usando CPU...")
            return torch.device("cpu")
    return torch.device("cpu")

def run_inference(args):
    device = get_device(args.device)
    print(f"[*] Inferencia en dispositivo: {device}")

    # 1. Cargar el Generador A -> B
    generator = CycleGANGenerator(in_channels=1, out_channels=1).to(device)
    if os.path.exists(args.weights):
        generator.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[+] Pesos cargados desde: {args.weights}")
    else:
        print(f"[!] Aviso: No se encontro {args.weights}. Ejecutando demostracion con pesos iniciales.")
    generator.eval()

    # 2. Localizar entrada (audio o imagen)
    input_path = args.input
    if not input_path or not os.path.exists(input_path):
        candidates = glob.glob("dataset/genero_A/*.*") + glob.glob("*.png")
        if candidates:
            input_path = candidates[0]
        else:
            raise FileNotFoundError("Proporciona una imagen o audio de entrada con --input")

    print(f"[+] Archivo de entrada: {input_path}")

    # Determinar si es audio (.wav, .mp3) o imagen
    if input_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
        print("[*] Convirtiendo audio de 30s a espectrograma...")
        img, db_min, db_max, sr = audio_to_spectrogram(input_path, output_image_path="temp_input_spec.png")
    else:
        img = Image.open(input_path).convert("L")
        db_min, db_max, sr = -80.0, 0.0, 22050

    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    # 3. Transferencia de estilo mediante CycleGAN
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Desnormalizar a [0, 255]
    output_tensor = (output_tensor.squeeze(0).squeeze(0).cpu().clamp(-1, 1) + 1.0) / 2.0
    output_array = (output_tensor.numpy() * 255.0).astype(np.uint8)
    translated_img = Image.fromarray(output_array).resize(original_size)

    # Guardar espectrograma traducido
    os.makedirs(os.path.dirname(args.output_image) if os.path.dirname(args.output_image) else ".", exist_ok=True)
    translated_img.save(args.output_image)
    print(f"[OK] Espectrograma transferido guardado en: {args.output_image}")

    # Guardar comparacion visual lado a lado
    comp_w = original_size[0] * 2
    comp_h = original_size[1]
    comp_img = Image.new("L", (comp_w, comp_h))
    comp_img.paste(img, (0, 0))
    comp_img.paste(translated_img, (original_size[0], 0))
    comp_img.save("comparacion_espectrogramas.png")
    print(f"[OK] Comparativa visual guardada en: comparacion_espectrogramas.png")

    # Reconstruir audio resultante si se solicita
    if args.output_audio:
        print("[*] Reconstruyendo audio a partir del espectrograma traducido...")
        spectrogram_to_audio(translated_img, args.output_audio, db_min=db_min, db_max=db_max, sr=sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traduccion de estilos musicales con CycleGAN")
    parser.add_argument("--input", type=str, default="", help="Ruta a imagen de espectrograma o archivo de audio")
    parser.add_argument("--weights", type=str, default="checkpoints/G_AB.pth", help="Pesos de G_AB")
    parser.add_argument("--output_image", type=str, default="espectrograma_traducido.png", help="Ruta de espectrograma de salida")
    parser.add_argument("--output_audio", type=str, default="audio_traducido.wav", help="Ruta del audio de salida")
    parser.add_argument("--img_size", type=int, default=128, help="Tamano del espectrograma para la red")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Dispositivo (auto, cuda, cpu)")

    args = parser.parse_args()
    run_inference(args)

