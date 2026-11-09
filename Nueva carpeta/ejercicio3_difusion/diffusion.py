import torch
import torch.nn.functional as F

class DDPM:
    """
    Controlador matematico del proceso de Difusion (DDPM):
    - Forward Process q(x_t | x_0): Anade ruido progresivo.
    - Reverse Process p_theta(x_{t-1} | x_t): Elimina ruido para generar muestras.
    """
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # Schedule lineal de beta
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Coeficientes para el muestreo directo de x_t dado x_0
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Varianza posterior para el proceso inverso
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """Difusion directa: Genera x_t anadiendo ruido a x_0 en el paso t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Paso inverso individual: calcula x_{t-1} a partir de x_t prediciendo el ruido."""
        betas_t = self.betas[t_index].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t_index].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t_index]).view(-1, 1, 1, 1)

        # Prediccion del ruido con la red U-Net
        predicted_noise = model(x, t)

        # Media de la distribucion reversa
        model_mean = sqrt_recip_alpha_t * (
            x - (betas_t / sqrt_one_minus_alpha_bar_t) * predicted_noise
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, shape):
        """Genera imagenes nuevas a partir de ruido gaussiano puro."""
        model.eval()
        x = torch.randn(shape, device=self.device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)
        
        # Clamp a [-1, 1]
        return x.clamp(-1.0, 1.0)

