import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Bloque Residual basico para el Generador de SRGAN."""
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """Aumenta la resolucion al doble (2x) usando PixelShuffle."""
    def __init__(self, in_channels=64, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class SRGANGenerator(nn.Module):
    """
    Generador SRGAN simple (Factor de aumento: 4x).
    Entrada: Imagen de Baja Resolucion (LR) [B, 3, H, W]
    Salida: Imagen de Alta Resolucion (HR) [B, 3, H*4, W*4]
    """
    def __init__(self, num_residual_blocks=6):
        super().__init__()
        # Capa inicial
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # Bloques residuales
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        
        # Convolucion intermedia
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # 2 bloques de aumento para lograr 4x (2x * 2x)
        self.upsample = nn.Sequential(
            UpsampleBlock(64, scale_factor=2),
            UpsampleBlock(64, scale_factor=2)
        )
        
        # Capa de salida a 3 canales RGB con rango [0, 1]
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        # x esta en rango [-1, 1]
        initial = self.initial(x)
        out = self.res_blocks(initial)
        out = self.mid_conv(out) + initial
        out = self.upsample(out)
        return self.final(out)


class SRGANDiscriminator(nn.Module):
    """
    Discriminador simple: Clasifica si una imagen HR es real o generada (falsa).
    Entrada: [B, 3, H, W] -> Salida: [B, 1]
    """
    def __init__(self):
        super().__init__()
        def disc_block(in_c, out_c, stride=1):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)]
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            *disc_block(64, 64, stride=2),
            *disc_block(64, 128, stride=1),
            *disc_block(128, 128, stride=2),
            *disc_block(128, 256, stride=1),
            *disc_block(256, 256, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)  # Retorna logits
        )

    def forward(self, x):
        return self.net(x)

