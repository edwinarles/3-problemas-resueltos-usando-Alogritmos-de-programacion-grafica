import os
import argparse
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from PIL import Image

from model import SimpleDiffusionUNet
from diffusion import DDPM

def get_device(requested="auto"):
    if requested in ["cuda", "auto"] and torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            test_conv = torch.nn.Conv2d(1, 1, 1).cuda()
            _ = test_conv(test_tensor.unsqueeze(0).unsqueeze(0))
            return torch.device("cuda")
        except Exception:
            print("[!] GPU no compatible con esta version de PyTorch. Usando CPU...")
            return torch.device("cpu")
    return torch.device("cpu")

def generate_anime_faces(args):
    device = get_device(args.device)
    print(f"[*] Inferencia en dispositivo: {device}")

    # 1. Cargar el modelo U-Net
    model = SimpleDiffusionUNet(in_channels=3, base_c=args.base_channels).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[+] Pesos cargados exitosamente desde: {args.weights}")
    else:
        print(f"[!] Aviso: No se encontro {args.weights}. Generando demostracion con pesos iniciales.")

    model.eval()

    # 2. Configurar proceso inverso de DDPM
    ddpm = DDPM(timesteps=args.timesteps, device=device)

    # 3. Muestreo inverso desde ruido puro N(0, I)
    num_samples = args.num_samples
    shape = (num_samples, 3, args.img_size, args.img_size)
    print(f"[*] Generando {num_samples} rostros de anime en {args.timesteps} pasos de difusion...")

    with torch.no_grad():
        samples = ddpm.sample(model, shape)

    # Desnormalizar de [-1, 1] a [0, 1]
    samples = (samples + 1.0) / 2.0

    # 4. Guardar resultados
    os.makedirs(args.output_dir, exist_ok=True)
    grid_path = os.path.join(args.output_dir, args.output_file)
    save_image(samples, grid_path, nrow=int(np.sqrt(num_samples)))
    print(f"[OK] Cuadricula de rostros de anime guardada en: {grid_path}")

    # Guardar algunas imagenes individuales
    for i in range(min(num_samples, 8)):
        single_path = os.path.join(args.output_dir, f"anime_rostro_{i+1:02d}.png")
        save_image(samples[i], single_path)
    print(f"[OK] Muestras individuales guardadas en: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generacion de rostros de anime con Difusion (DDPM)")
    parser.add_argument("--weights", type=str, default="checkpoints/diffusion_anime.pth", help="Pesos de la U-Net")
    parser.add_argument("--num_samples", type=int, default=16, help="Numero de rostros a generar (ej. 16 = cuadricula 4x4)")
    parser.add_argument("--img_size", type=int, default=64, help="Resolucion de la imagen")
    parser.add_argument("--timesteps", type=int, default=300, help="Pasos de difusion T")
    parser.add_argument("--base_channels", type=int, default=64, help="Canales base de U-Net")
    parser.add_argument("--output_dir", type=str, default="resultados", help="Carpeta para guardar resultados")
    parser.add_argument("--output_file", type=str, default="rostros_anime_generados.png", help="Nombre de imagen final")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Dispositivo (auto, cuda, cpu)")

    args = parser.parse_args()
    generate_anime_faces(args)

