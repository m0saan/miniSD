from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from tqdm import tqdm

from .samplers import DDPMSampler
from .clip import CLIP, ClipConfig
from .vae import VAE
from .diffusion import UnetDiffusionModel
from .model_converter import load_from_standard_weights
# from utilities.image_utils import ImageProcessor


IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512
LATENTS_WIDTH = IMAGE_WIDTH // 8
LATENTS_HEIGHT = IMAGE_HEIGHT // 8


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from minisd import MiniSDPipeline

        >>> pipe = MiniSDPipeline()
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class MiniSDPipeline:
    """
    Pipeline for text-to-image generation using Stable Diffusion.

    Attributes:
        clip (CLIP): The CLIP model.
        vae (VAE): The VAE model.
        unet (UnetDiffusionModel): The UNet Diffusion model.
        ...
    """

    def __init__(
        self,
        clip: Optional[CLIP] = None,
        vae: Optional[VAE] = None,
        unet: Optional[UnetDiffusionModel] = None,
        sampler: Optional[DDPMSampler] = None,
        tokenizer=None,
        ) -> None:
        
        self.clip = CLIP(ClipConfig())
        self.vae = VAE()
        self.unet = UnetDiffusionModel()
        self.sampler = sampler if sampler is not None else DDPMSampler()
        # self.image_processor = ImageProcessor()
        self.tokenizer = tokenizer if tokenizer is not None else self._init_tokenizer()
        
    def load_from_checkpoint(self, checkpoint_path: str, device) -> None:
        state_dict = load_from_standard_weights(checkpoint_path, device=device)
        
        self.clip.load_state_dict(state_dict['clip'], strict=True)
        self.vae.load_state_dict(state_dict['vae'], strict=True)
        self.unet.load_state_dict(state_dict['unet'], strict=True)
    
    def encode_prompt(self, prompt: str, negative_prompt: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.tokenizer.batch_encode_plus([prompt], paddings='max_length', max_length=77).input_ids
        cond_input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
        prompt_embeds = self.clip(cond_input_ids) # (batch_size, seq_len, hidden_size): (1, 77, 768)
        
        negative_prompt_embeds = None
        if negative_prompt is not None:    
            input_ids = self.tokenizer.batch_encode_plus([negative_prompt], paddings='max_length', max_length=77).input_ids
            uncond_input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
            negative_prompt_embeds = self.clip(uncond_input_ids) # (batch_size, seq_len, hidden_size): (1, 77, 768)
        
        return prompt_embeds, negative_prompt_embeds
    
    def rescale(self, x: torch.Tensor, old_range: Tuple[int, int], new_range: Tuple[int, int], clamp: bool = False) -> torch.Tensor:
        min_old, max_old = old_range
        min_new, max_new = new_range
        x_rescaled = (x - min_old) / (max_old - min_old) * (max_new - min_new) + min_new
        if clamp:
            x_rescaled = torch.clamp(x_rescaled, min_new, max_new)
        return x_rescaled
    
    def _init_tokenizer(self):
        pass
    
    def get_time_embedding(self, t: int) -> torch.Tensor:
        pe = torch.pow(10000, 2 * torch.arange(0, 160, dtype=torch.float) / 160) # (160, )
        x = torch.tensor([t], dtype=torch.float) # (1, )
        freqs = x.unsqueeze(-1) * pe.unsqueeze(0) # (1, 160)
        return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1) # (1, 320)
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.8,
        do_cfg: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        input_image: Optional[torch.FloatTensor] = None,
        seed: int = None,
        device: str = None,
        idle_device: str = None,
        sampler: str = 'ddpm',
    ):
        
        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not (0 <= strength <= 1):
            raise ValueError('Strength must be between 0 and 1.')
        
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        if idle_device is None:
            to_idle_fn = lambda x: x
        else:
            to_idle_fn = lambda x: x.to(idle_device)
        
        self.clip = self.clip.to(device)
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        if sampler == 'ddpm':
            # self.sampler = self.sampler.to(device)
            if generator is not None: self.sampler.set_generator(generator)
            self.sampler.set_inference_steps(num_inference_steps)
        else:
            # TODO: Implement other samplers :Xd
            pass
        
        if do_cfg:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt, device)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, None, device)
            
        to_idle_fn(self.clip)
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image is not None:
            # TODO: Implement image processing in ImageProcessor
            input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            input_image_tensor = torch.tensor(np.array(input_image), device=device, dtype=torch.float32)
            input_image_tensor = self.rescale(input_image_tensor).unsqueeze(0).permute(0, 3, 1, 2)
            noise = torch.randn(latents_shape, device=device, generator=generator)
            latents = self.vae.encode(input_image_tensor, noise=noise)
            
            self.sampler.set_strenght(strength)
            latents = self.sampler.add_noise(latents, initial_noise=self.timesteps[0])
            
            to_idle_fn(self.vae)
        
        else:
            latents = torch.randn(latents_shape, device=device, generator=generator)
            
        for t in tqdm(range(num_inference_steps, 0), desc='Inference', leave=False):
            time_embedding = self.get_time_embedding(t).to(device)
            
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            
            noise_pred = self.unet(latent_model_input, time_embedding, prompt_embeds)
            
            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                pred_noise = guidance_scale * (noise_pred_cond - noise_pred_uncond) + noise_pred_uncond
            
            # Remove the noise predicted by the unet
            latents = self.sampler.step(latents, pred_noise, self.timesteps[t])
        
        to_idle_fn(self.unet)
        
        self.vae.to(device)
        images = self.vae.decode(latents)
        to_idle_fn(self.vae)
        
        # TODO: Implement image processing in ImageProcessor
        images = self.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1).cpu().astype(torch.uint8).numpy()
        return images
                
            
        
        