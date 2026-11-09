import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    """Codificacion sinusoidal del paso de tiempo t (timesteps)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Bloque convolucional basico con incorporacion del paso temporal t."""
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t):
        h = self.relu(self.bn1(self.conv1(x)))
        # Sumar la proyeccion del paso de tiempo
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.relu(self.bn2(self.conv2(h)))
        return h + self.residual(x)


class SimpleDiffusionUNet(nn.Module):
    """
    U-Net simple para Modelo de Difusion (DDPM).
    Entrada: Imagen con ruido x_t [B, 3, 64, 64] y paso temporal t [B].
    Salida: Ruido predicho epsilon [B, 3, 64, 64].
    """
    def __init__(self, in_channels=3, base_c=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True)
        )

        # Encoder (Downsampling)
        self.down1 = Block(in_channels, base_c, time_dim)
        self.pool1 = nn.MaxPool2d(2) # 64 -> 32
        
        self.down2 = Block(base_c, base_c * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2) # 32 -> 16
        
        self.down3 = Block(base_c * 2, base_c * 4, time_dim)
        self.pool3 = nn.MaxPool2d(2) # 16 -> 8

        # Bottleneck
        self.mid = Block(base_c * 4, base_c * 4, time_dim)

        # Decoder (Upsampling con canales corregidos para concatenacion skip)
        self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2) # 8 -> 16
        self.dec3 = Block(base_c * 2 + base_c * 4, base_c * 2, time_dim) # 128 (up3) + 256 (x3) = 384 -> 128

        self.up2 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2) # 16 -> 32
        self.dec2 = Block(base_c + base_c * 2, base_c, time_dim) # 64 (up2) + 128 (x2) = 192 -> 64

        self.up1 = nn.ConvTranspose2d(base_c, base_c, kernel_size=2, stride=2) # 32 -> 64
        self.dec1 = Block(base_c * 2, base_c, time_dim) # 64 (up1) + 64 (x1) = 128 -> 64

        # Capa final de salida (predice los 3 canales de ruido)
        self.out_conv = nn.Conv2d(base_c, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        x3 = self.down3(self.pool2(x2), t_emb)

        # Centro
        mid = self.mid(self.pool3(x3), t_emb)

        # Decoder con conexiones residuales (skip connections)
        u3 = self.up3(mid)
        d3 = self.dec3(torch.cat([u3, x3], dim=1), t_emb)

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1), t_emb)

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1), t_emb)

        return self.out_conv(d1)

