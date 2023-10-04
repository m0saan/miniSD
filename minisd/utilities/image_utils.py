from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor, Lambda, Compose, Resize


class ImageProcessor:
    
    def __init__(self, images) -> None:
        self.images = images
        self.to_tensor_tfms = Compose([
            Resize((512, 512)),
            ToTensor(),
            Lambda(lambda x: (x*2) - 1)
            ])
        
        self.to_pil_tfms = Compose([
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            Lambda(lambda t: t * 255.),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
            ])
        
    def pil_to_tensor(self, image: Image) -> torch.Tensor:
        """
        Converts an image to a tensor.
        
        Args:
            image (Image): An image.
            
        Returns:
            torch.Tensor: A tensor.
        """
        return self.to_tensor_tfms(image).unsqueeze(0)
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image:
        """
        Converts a tensor to an image.
        
        Args:
            tensor (torch.Tensor): A tensor.
            
        Returns:
            Image: An image.
        """
        return self.to_pil_tfms(tensor.squeeze(0))