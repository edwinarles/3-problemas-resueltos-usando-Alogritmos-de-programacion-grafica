import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Bloque Residual basico con InstanceNormalization y ReLU."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """
    Generador para CycleGAN (Arquitectura ResNet compacta).
    Traduce caracteristicas estilisticas entre espectrogramas.
    """
    def __init__(self, in_channels=1, out_channels=1, num_res_blocks=4, base_filters=32):
        super().__init__()
        
        # 1. Capa inicial
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, bias=False),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        ]

        # 2. Downsampling (Reduccion espacial)
        dim = base_filters
        for _ in range(2):
            layers.extend([
                nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True)
            ])
            dim *= 2

        # 3. Bloques residuales (Procesamiento de caracteristicas de estilo)
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(dim))

        # 4. Upsampling (Expansion espacial)
        for _ in range(2):
            layers.extend([
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(dim // 2),
                nn.ReLU(inplace=True)
            ])
            dim //= 2

        # 5. Capa de salida
        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, out_channels, kernel_size=7),
            nn.Tanh()
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CycleGANDiscriminator(nn.Module):
    """
    Discriminador PatchGAN de 70x70.
    Determina si parches del espectrograma pertenecen al genero real o generado.
    """
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        
        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base_filters, normalize=False),
            *block(base_filters, base_filters * 2),
            *block(base_filters * 2, base_filters * 4),
            nn.Conv2d(base_filters * 4, 1, kernel_size=4, padding=1) # 1 canal de decision
        )

    def forward(self, x):
        return self.net(x)

