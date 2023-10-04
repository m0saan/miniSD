import torch
import torch.nn.functional as F
from torch import nn
from .unet import UNet, TimeEmbeddings, Head

class UnetDiffusionModel(nn.Module):
    """
    A PyTorch implementation of the Diffusion Model.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self.time_embeddings = TimeEmbeddings(time_embedding_dim=320)
        self.unet = UNet()
        self.head = Head(in_channels=320, out_channels=4)
        
    def forward(self, latents: torch.Tensor, time: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # latents: (batch_size, latent_dim, height, width) Bx4x64x64
        # time: (1, 320)
        #  : (batch_size, seq_len, hidden_size)
        
        time_embed = self.time_embeddings(time) # (1, 320) -> (1, 1280)
        unet_out = self.unet(latents, time_embed, context) # (B, 4, 64, 64), (B, 320, 64, 64)
        return self.head(unet_out) # (B, 320, 64, 64) -> (B, 4, 64, 64)


if __name__ == '__main__':
    x = torch.randn(1, 4, 64, 64).to('mps')
    t = torch.randn(1, 320).to('mps')
    context = torch.randn(1, 77, 768).to('mps')
    model = UnetDiffusionModel().to('mps')

    print(x.shape)
    pred = model(x, t, context)
    print(pred.shape)