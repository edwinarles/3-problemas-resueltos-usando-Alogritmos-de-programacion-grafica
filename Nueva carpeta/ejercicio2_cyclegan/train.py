import os
import argparse
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import CycleGANGenerator, CycleGANDiscriminator
from audio_utils import audio_to_spectrogram

class SpectrogramDataset(Dataset):
    """
    Dataset para CycleGAN con dos dominios no pareados:
    Dominio A (Genero A) y Dominio B (Genero B).
    """
    def __init__(self, dir_A, dir_B, img_size=(128, 128)):
        self.img_size = img_size
        self.files_A = glob.glob(os.path.join(dir_A, "*.*"))
        self.files_B = glob.glob(os.path.join(dir_B, "*.*"))

        # Si no existen imagenes aun, generar muestras sinteticas de espectrogramas para demostracion
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            print("[!] Aviso: No se encontraron imagenes en las carpetas de generos.")
            print("    Creando espectrogramas de muestra para inicializar el dataset...")
            os.makedirs(dir_A, exist_ok=True)
            os.makedirs(dir_B, exist_ok=True)
            for i in range(20):
                # Genero A: Patron de frecuencias bajas / graves
                arr_a = np.zeros(img_size, dtype=np.uint8)
                arr_a[20:60, :] = np.random.randint(150, 255, size=(40, img_size[1]))
                Image.fromarray(arr_a).save(os.path.join(dir_A, f"muestra_A_{i:02d}.png"))
                
                # Genero B: Patron de frecuencias altas y ritmicas
                arr_b = np.zeros(img_size, dtype=np.uint8)
                arr_b[70:120, :] = np.random.randint(120, 240, size=(50, img_size[1]))
                arr_b[:, ::10] = 255
                Image.fromarray(arr_b).save(os.path.join(dir_B, f"muestra_B_{i:02d}.png"))
            
            self.files_A = glob.glob(os.path.join(dir_A, "*.*"))
            self.files_B = glob.glob(os.path.join(dir_B, "*.*"))

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Escala [-1, 1]
        ])

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        file_a = self.files_A[idx % len(self.files_A)]
        file_b = self.files_B[random.randint(0, len(self.files_B) - 1)]

        img_a = Image.open(file_a).convert("L")
        img_b = Image.open(file_b).convert("L")

        return self.transform(img_a), self.transform(img_b)


def get_device(requested="auto"):
    if requested in ["cuda", "auto"] and torch.cuda.is_available():
        try:
            # Prueba rapida para verificar si la GPU es compatible con esta version de PyTorch
            test_tensor = torch.zeros(1, device="cuda")
            test_conv = nn.Conv2d(1, 1, 1).cuda()
            _ = test_conv(test_tensor.unsqueeze(0).unsqueeze(0))
            return torch.device("cuda")
        except Exception as e:
            print(f"[!] Aviso: Tu tarjeta grafica no es compatible con los binarios actuales de PyTorch ({e}).")
            print("    Cambiando automaticamente a entrenamiento en CPU...")
            return torch.device("cpu")
    return torch.device("cpu")


def train(args):
    device = get_device(args.device)
    print(f"[*] Dispositivo seleccionado: {device}")

    # 1. Dataset
    dataset = SpectrogramDataset(args.dir_A, args.dir_B, img_size=(args.img_size, args.img_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[+] Total de pares de espectrogramas cargados: {len(dataset)}")

    # 2. Generadores y Discriminadores
    G_AB = CycleGANGenerator(in_channels=1, out_channels=1).to(device) # Genero A -> B
    G_BA = CycleGANGenerator(in_channels=1, out_channels=1).to(device) # Genero B -> A
    D_A = CycleGANDiscriminator(in_channels=1).to(device)             # Clasifica Genero A
    D_B = CycleGANDiscriminator(in_channels=1).to(device)             # Clasifica Genero B

    # 3. Criterios de Perdida
    criterion_GAN = nn.MSELoss()         # Perdida adversaria LSGAN
    criterion_cycle = nn.L1Loss()        # Perdida de consistencia de ciclo
    criterion_identity = nn.L1Loss()     # Perdida de identidad

    # 4. Optimizadores
    opt_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 5. Ciclo de Entrenamiento
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss_g_total = 0.0
        loss_d_total = 0.0

        for real_A, real_B in dataloader:
            real_A, real_B = real_A.to(device), real_B.to(device)

            # Salidas esperadas para PatchGAN
            disc_out_shape = D_A(real_A).shape
            valid = torch.ones(disc_out_shape, device=device)
            fake = torch.zeros(disc_out_shape, device=device)

            # ---------------------------------------------
            # Entrenar Generadores G_AB y G_BA
            # ---------------------------------------------
            opt_G.zero_grad()

            # Perdida de identidad: G_BA(A) debe parecerse a A
            loss_id_A = criterion_identity(G_BA(real_A), real_A) * 5.0
            loss_id_B = criterion_identity(G_AB(real_B), real_B) * 5.0

            # Perdida adversaria: G_AB(A) debe engañar a D_B
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)

            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            # Perdida de ciclo: G_BA(G_AB(A)) ~ A y viceversa
            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * 10.0

            # Perdida total de los generadores
            loss_G = (
                loss_GAN_AB + loss_GAN_BA +
                loss_cycle_A + loss_cycle_B +
                loss_id_A + loss_id_B
            )
            loss_G.backward()
            opt_G.step()

            # ---------------------------------------------
            # Entrenar Discriminador D_A
            # ---------------------------------------------
            opt_D_A.zero_grad()
            loss_real_A = criterion_GAN(D_A(real_A), valid)
            loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            loss_D_A.backward()
            opt_D_A.step()

            # ---------------------------------------------
            # Entrenar Discriminador D_B
            # ---------------------------------------------
            opt_D_B.zero_grad()
            loss_real_B = criterion_GAN(D_B(real_B), valid)
            loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            loss_D_B.backward()
            opt_D_B.step()

            loss_g_total += loss_G.item()
            loss_d_total += (loss_D_A.item() + loss_D_B.item())

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] -> Loss G: {loss_g_total/len(dataloader):.4f} | Loss D: {loss_d_total/len(dataloader):.4f}")

    # Guardar pesos
    torch.save(G_AB.state_dict(), os.path.join(args.output_dir, "G_AB.pth"))
    torch.save(G_BA.state_dict(), os.path.join(args.output_dir, "G_BA.pth"))
    print(f"[OK] Modelos guardados en: {args.output_dir}/G_AB.pth y G_BA.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de CycleGAN para espectrogramas")
    parser.add_argument("--dir_A", type=str, default="dataset/genero_A", help="Carpeta de espectrogramas Genero A")
    parser.add_argument("--dir_B", type=str, default="dataset/genero_B", help="Carpeta de espectrogramas Genero B")
    parser.add_argument("--epochs", type=int, default=25, help="Numero de epocas")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamano de batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=128, help="Resolucion del espectrograma")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directorio de guardado")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Dispositivo (auto, cuda, cpu)")

    args = parser.parse_args()
    train(args)

