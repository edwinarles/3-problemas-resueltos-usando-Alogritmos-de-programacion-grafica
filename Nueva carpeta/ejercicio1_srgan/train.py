import os
import argparse
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import SRGANGenerator, SRGANDiscriminator

class SuperResolutionDataset(Dataset):
    """
    Dataset simple de super-resolucion:
    Toma parches HR de 96x96 y los reduce a LR de 24x24 (factor 4x).
    Puede extraer frames de un video o cargar imagenes desde una carpeta.
    """
    def __init__(self, video_path=None, images_dir=None, patch_size=96, max_frames=600):
        self.patch_size = patch_size
        self.images = []

        # Opcion 1: Extraer frames directamente desde un video MP4
        if video_path and os.path.exists(video_path):
            print(f"[+] Extrayendo frames del video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                # Guardar en RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.images.append(Image.fromarray(frame_rgb))
                frame_count += 1
            cap.release()
            print(f"[+] Total de frames extraidos: {len(self.images)}")
        
        # Opcion 2: Cargar imagenes existentes
        elif images_dir and os.path.exists(images_dir):
            file_paths = glob.glob(os.path.join(images_dir, "*.*"))
            for fp in file_paths[:max_frames]:
                try:
                    img = Image.open(fp).convert("RGB")
                    self.images.append(img)
                except Exception:
                    pass
            print(f"[+] Imagenes cargadas: {len(self.images)}")
        
        if len(self.images) == 0:
            raise ValueError("No se encontraron imagenes ni frames de video validos.")

        self.hr_crop = transforms.RandomCrop(patch_size)
        self.lr_transform = transforms.Resize(patch_size // 4, interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Si la imagen es mas chica que el parche, redimensionar
        if img.width < self.patch_size or img.height < self.patch_size:
            img = img.resize((max(img.width, self.patch_size), max(img.height, self.patch_size)))
            
        hr_patch = self.hr_crop(img)
        lr_patch = self.lr_transform(hr_patch)

        # Convertir a tensor [-1, 1]
        hr_tensor = self.normalize(self.to_tensor(hr_patch))
        lr_tensor = self.normalize(self.to_tensor(lr_patch))

        return lr_tensor, hr_tensor


def get_device(requested="auto"):
    if requested in ["cuda", "auto"] and torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda")
            test_conv = nn.Conv2d(1, 1, 1).cuda()
            _ = test_conv(test_tensor.unsqueeze(0).unsqueeze(0))
            return torch.device("cuda")
        except Exception as e:
            print(f"[!] GPU no compatible con esta compilacion de PyTorch ({e}). Usando CPU...")
            return torch.device("cpu")
    return torch.device("cpu")

def train(args):
    device = get_device(args.device)
    print(f"[*] Entrenando en dispositivo: {device}")

    # 1. Dataset y DataLoader
    # Si no se especifica, busca un archivo .mp4 en la carpeta padre o actual
    video_file = args.video
    if not video_file or not os.path.exists(video_file):
        candidates = glob.glob("../*.mp4") + glob.glob("*.mp4")
        if candidates:
            video_file = candidates[0]

    dataset = SuperResolutionDataset(video_path=video_file, images_dir=args.data_dir, patch_size=args.patch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2. Modelos
    generator = SRGANGenerator(num_residual_blocks=args.res_blocks).to(device)
    discriminator = SRGANDiscriminator().to(device)

    # 3. Optimizadores y Funciones de Perdida
    opt_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    criterion_content = nn.MSELoss()           # Perdida de pixel/contenido
    criterion_adversarial = nn.BCEWithLogitsLoss() # Perdida adversaria (real vs falso)

    # 4. Ciclo de Entrenamiento
    for epoch in range(1, args.epochs + 1):
        total_loss_g = 0.0
        total_loss_d = 0.0

        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            batch_size = lr_imgs.size(0)

            # Etiquetas suaves
            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # ---------------------------------------------
            # A. Entrenar Discriminador: D(HR) -> 1, D(G(LR)) -> 0
            # ---------------------------------------------
            opt_D.zero_grad()
            sr_imgs = generator(lr_imgs)

            d_real = discriminator(hr_imgs)
            d_fake = discriminator(sr_imgs.detach())

            loss_d_real = criterion_adversarial(d_real, real_labels)
            loss_d_fake = criterion_adversarial(d_fake, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_D.step()

            # ---------------------------------------------
            # B. Entrenar Generador: Minimizar MSE + Adversarial
            # ---------------------------------------------
            opt_G.zero_grad()
            d_fake_for_g = discriminator(sr_imgs)

            loss_content = criterion_content(sr_imgs, hr_imgs)
            loss_adv = criterion_adversarial(d_fake_for_g, real_labels)

            # Perdida total del Generador
            loss_g = loss_content + 1e-3 * loss_adv
            loss_g.backward()
            opt_G.step()

            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()

        avg_loss_g = total_loss_g / len(dataloader)
        avg_loss_d = total_loss_d / len(dataloader)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] -> Loss G: {avg_loss_g:.4f} | Loss D: {avg_loss_d:.4f}")

    # Guardar modelo entrenado
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "generator_srgan.pth")
    torch.save(generator.state_dict(), save_path)
    print(f"[OK] Modelo guardado exitosamente en: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento simple de SRGAN para video")
    parser.add_argument("--video", type=str, default="", help="Ruta al video de television de los 2000-2010")
    parser.add_argument("--data_dir", type=str, default="", help="Ruta a carpeta de imagenes (opcional)")
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamano de batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--patch_size", type=int, default=96, help="Tamano de parche HR")
    parser.add_argument("--res_blocks", type=int, default=6, help="Numero de bloques residuales")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directorio para guardar modelo")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Dispositivo (auto, cuda, cpu)")
    
    args = parser.parse_args()
    train(args)

