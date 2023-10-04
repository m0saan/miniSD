import torch
from torch.nn import functional as F
# from ..utilities.image_utils import ImageProcessor

class DDPMSampler:
    """
    A PyTorch implementation of the DDPMSampler.
    """
    
    def __init__(
        self,
        generator: torch.Generator = None,
        n_training_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
        ) -> None:
        
        super().__init__()
        self.generator = generator
        self.n_training_steps = n_training_steps
        self.num_inference_steps = n_training_steps
        self.timesteps = torch.arange(n_training_steps, device=generator.device)
        self.betas = torch.linspace(beta_start, beta_end, n_training_steps, device=generator.device)
        self.alphas = 1. - self.betas
        self.alpha_bar = self.alphas.cumprod(dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alpha_bar)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)

    def _extract(a: torch.Tensor, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def set_inference_steps(self, num_inference_steps: torch.IntTensor) -> None:
        """
        Sets the number of inference steps.
        
        Args:
            num_inference_steps (int): Number of inference steps.
        """
        
        self.num_inference_steps = num_inference_steps
        ratio = self.n_training_steps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps, device=self.generator.device) * ratio).long()
    
    def set_generator(self, generator: torch.Generator) -> None:
        """
        Sets the generator.
        
        Args:
            generator (torch.Generator): Generator.
        """
        
        self.generator = generator
        
        
    def add_noise(self, latents: torch.Tensor, timestep: torch.IntTensor) -> torch.Tensor:
        """
        Adds noise to the latents.
        
        Args:
            latents (torch.Tensor): Latent tensor of shape (batch_size, latent_dim, height, width).
            timestep (int): Timestep.
            
        Returns:
            torch.Tensor: Latent tensor with added noise.
        """
        sqrt_alphas_bar_t = self._extract(self.sqrt_alphas_bar, timestep, latents.shape)
        sqrt_one_minus_alphas_bar_t = self._extract(self.sqrt_one_minus_alphas_bar, timestep, latents.shape)
        noise = torch.randn(latents.shape, device=latents.device, generator=self.generator, dtype=latents.dtype)
        noisy_latent =  sqrt_alphas_bar_t * latents + sqrt_one_minus_alphas_bar_t * noise
        return noisy_latent
    
    def set_strenght(self, strength: float) -> None:
        start = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start:]
    
    @torch.inference_mode()
    def step(self, latents: torch.Tensor, pred_noise: torch.Tensor, t: torch.IntTensor) -> torch.Tensor:
        betas_t = self._extract(self.betas, t, latents.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, latents.shape)
        sqrt_one_minus_alphas_bar_t = self._extract(self.sqrt_one_minus_alphas_bar, t, latents.shape)
        
        model_mean = sqrt_recip_alphas_t * (latents - (betas_t * pred_noise) / sqrt_one_minus_alphas_bar_t)
        if t == 0:
            return model_mean
        else:
            var_t = self._extract(self.posterior_variance, t, latents.shape)
            noise = torch.randn(latents.shape, device=latents.device, generator=self.generator, dtype=latents.dtype)
            return model_mean + torch.sqrt(var_t) * noise
    

if __name__ == '__main__':
    # from PIL import Image
    # import requests
    # import matplotlib.pyplot as plt
    # img_processor = ImageProcessor()
    # ddpm = DDPMSampler()

    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC
    # plt.imshow(image)
    # t = torch.tensor([40])
    # x_start = img_processor.pil_to_tensor(image) # tensor of shape CHW
    # noisy_image = ddpm.add_noise(x_start, t)
    # plt.imshow(img_processor.tensor_to_pil(noisy_image))
    pass
    