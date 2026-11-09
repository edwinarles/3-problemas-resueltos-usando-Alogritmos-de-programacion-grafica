import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import SimpleDiffusionUNet
from diffusion import DDPM

class AnimeFaceDataset(Dataset):
    """Dataset para cargar las imagenes de rostros de anime (Anime Face Dataset)."""
    def __init__(self, data_dir="data/anime_faces", img_size=64, max_images=None):
        self.img_size = img_size
        files = []

        # 1. Si el usuario paso una carpeta valida existente con imagenes
        if data_dir and os.path.exists(data_dir) and data_dir != "data/anime_faces":
            files = glob.glob(os.path.join(data_dir, "**/*.*"), recursive=True)
            files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 2. Si no hay imagenes o es la ruta por defecto, descargar directamente con kagglehub
        if len(files) < 50:
            print("[*] Descargando y cargando dataset de Kaggle (splcher/animefacedataset)...")
            try:
                import kagglehub
                kaggle_path = kagglehub.dataset_download("splcher/animefacedataset")
                print(f"[+] Dataset listo en: {kaggle_path}")
                files = glob.glob(os.path.join(kaggle_path, "**/*.*"), recursive=True)
                files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            except Exception as e:
                print(f"[!] Error con kagglehub ({e}). Asegurate de tener conexion y ejecutar: pip install kagglehub")

        if len(files) == 0:
            raise FileNotFoundError("No se encontraron imagenes de rostros de anime. Instala kagglehub (pip install kagglehub) o pasa la ruta correcta con --data_dir")

        # Limitar cantidad para acelerar el entrenamiento si se especifica
        if max_images and max_images > 0:
            files = files[:max_images]

        self.files = files
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def get_device(requested="auto"):
    if requested in ["cuda", "auto"] and torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            test_conv = nn.Conv2d(1, 1, 1).cuda()
            _ = test_conv(test_tensor.unsqueeze(0).unsqueeze(0))
            return torch.device("cuda")
        except Exception as e:
            print(f"[!] GPU no compatible con esta version de PyTorch ({e}). Usando CPU...")
            return torch.device("cpu")
    return torch.device("cpu")

def train(args):
    device = get_device(args.device)
    print(f"[*] Dispositivo de entrenamiento: {device}")

    # 1. Dataset y DataLoader
    dataset = AnimeFaceDataset(data_dir=args.data_dir, img_size=args.img_size, max_images=args.max_images)
    print(f"[+] Total de imagenes de anime cargadas: {len(dataset)}")
    should_drop_last = len(dataset) >= args.batch_size
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=should_drop_last)

    # 2. Inicializar Modelo U-Net y Proceso de Difusion
    model = SimpleDiffusionUNet(in_channels=3, base_c=args.base_channels).to(device)
    ddpm = DDPM(timesteps=args.timesteps, device=device)

    # 3. Optimizador y Criterio
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 4. Bucle de Entrenamiento
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0

        for x_0 in dataloader:
            x_0 = x_0.to(device)
            batch_size = x_0.size(0)

            # Muestrear paso de tiempo t aleatorio para cada imagen del batch
            t = torch.randint(0, args.timesteps, (batch_size,), device=device).long()
            
            # Muestrear ruido gaussiano epsilon
            noise = torch.randn_like(x_0)

            # Proceso directo q(x_t | x_0)
            x_t = ddpm.q_sample(x_0, t, noise)

            # La red U-Net predice el ruido añadido
            optimizer.zero_grad()
            predicted_noise = model(x_t, t)

            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] -> MSE Loss (Ruido): {avg_loss:.5f}")

    # Guardar pesos entrenados
    save_path = os.path.join(args.output_dir, "diffusion_anime.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Modelo de difusion guardado en: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de Modelo de Difusion para Anime Faces")
    parser.add_argument("--data_dir", type=str, default="data/anime_faces", help="Carpeta o ruta devuelta por kagglehub")
    parser.add_argument("--max_images", type=int, default=5000, help="Maximo de imagenes a usar para entrenar rapido (0 para todas)")
    parser.add_argument("--epochs", type=int, default=30, help="Numero de epocas")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamano del batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Tasa de aprendizaje")
    parser.add_argument("--img_size", type=int, default=64, help="Resolucion (64x64)")
    parser.add_argument("--timesteps", type=int, default=300, help="Pasos de difusion T")
    parser.add_argument("--base_channels", type=int, default=64, help="Canales base de U-Net")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Carpeta de salida")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Dispositivo (auto, cuda, cpu)")

    args = parser.parse_args()
    train(args)

