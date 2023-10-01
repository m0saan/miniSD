import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Optional

from attention import SelfAttention


class VaeAttentionBlock(nn.Module):
    """
    Attention block specific for VAEs.
    
    Attributes:
        gourp_norm_1 (nn.GroupNorm): Group normalization layer.
        attention (SelfAttention): Self attention mechanism.
    """
    
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.gourp_norm_1 = nn.GroupNorm(num_groups=32, num_channels=n_channels)
        self.attention = SelfAttention(n_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x 
        n,c,h,w = x.shape # BxCxHxW
        x = x.view(n,c,h*w) # BxCxH*W
        x = x.transpose(-1, -2) # BxH*WxC
        attn = self.attention(x) # BxH*WxC
        x = x.transpose(-1, -2).view((n, c, h, w)) # BxCxHxW
        x = x + residue
        return x
    
    
    
class VaeResidualBlock(nn.Module):
    """
    Residual block for VAEs.
    
    Attributes:
        conv_1, conv_2 (nn.Conv2d): Convolutional layers.
        gourp_norm_1, group_norm_2 (nn.GroupNorm): Group normalization layers.
        skip (nn.Module): Skip connection layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gourp_norm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv_1(F.silu(self.gourp_norm_1(x)))
        res = self.conv_2(F.silu(self.group_norm_2(res)))
        return res + self.skip(x)
    
    
class VaeEncoder(nn.Sequential):
    """
    Encoder part of the VAE.
    """
    
    def __init__(self) -> None:
        layers = [
            # Initial layers
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),

            # Residual blocks 512x512
            VaeResidualBlock(in_channels=128, out_channels=128),
            VaeResidualBlock(in_channels=128, out_channels=128),

            # Downsample to 256x256
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            VaeResidualBlock(in_channels=128, out_channels=256),
            VaeResidualBlock(in_channels=256, out_channels=256),

            # Downsample to 128x128
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            VaeResidualBlock(in_channels=256, out_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),

            # Downsample to 64x64
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
            VaeResidualBlock(in_channels=512, out_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),

            # Attention block 64x64
            VaeAttentionBlock(n_channels=512),
            
            # Additional residual block 64x64
            VaeResidualBlock(in_channels=512, out_channels=512),

            # Final layers
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1)
        ]
        super(VaeEncoder, self).__init__(*layers)
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if getattr(layer, 'stride', None) == (2,2):
                x = F.pad(x, (0,1,0,1))
            x = layer(x)
        # Bx8x64x64 -> (2x) of Bx4x64x64
        mean, log_var = x.chunk(2, dim=1)
        std = log_var.clamp(-30, 20).exp().sqrt()
        latent = (mean + std * noise) * 0.18215
        return latent # Bx4x64x64


import torch.nn as nn

class VaeDecoder(nn.Sequential):
    """
    Decoder part of the VAE.
    """
    
    def __init__(self) -> None:
        layers = [
            # Initial layers
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, padding=1),
            
            # Residual blocks 64x64
            VaeResidualBlock(in_channels=512, out_channels=512),
            VaeAttentionBlock(n_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),
            
            # Upsampling to 128x128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            VaeResidualBlock(in_channels=512, out_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),
            VaeResidualBlock(in_channels=512, out_channels=512),
            
            # Upsampling to 256x256
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            VaeResidualBlock(in_channels=512, out_channels=256),
            VaeResidualBlock(in_channels=256, out_channels=256),
            VaeResidualBlock(in_channels=256, out_channels=256),
            
            # Upsampling to 512x512
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            VaeResidualBlock(in_channels=256, out_channels=128),
            VaeResidualBlock(in_channels=128, out_channels=128),
            VaeResidualBlock(in_channels=128, out_channels=128),
            
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        ]
        super(VaeDecoder, self).__init__(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215
        for layer in self: x = layer(x)
        return x

class VAE(nn.Module):
    """
    Variational AutoEncoder (VAE) combining both the encoder and decoder parts.
    
    Attributes:
        encoder (VaeEncoder): Encoder part of the VAE.
        decoder (VaeDecoder): Decoder part of the VAE.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VaeEncoder()
        self.decoder = VaeDecoder()
        
    def encode(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Encodes an input tensor using the encoder part of the VAE.
        
        Args:
            x (torch.Tensor): Input image tensor.
            noise (torch.Tensor): Noise tensor for the encoder.

        Returns:
            torch.Tensor: Latent space tensor.
        """
        
        return self.encoder(x, noise)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent tensor using the decoder part of the VAE.
        
        Args:
            x (torch.Tensor): Latent space tensor.

        Returns:
            torch.Tensor: Decoded image tensor.
        """
        
        return self.decoder(x)
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Combined forward pass for the VAE, encoding then decoding the input.
        
        Args:
            x (torch.Tensor): Input image tensor.
            noise (torch.Tensor): Noise tensor for the encoder.

        Returns:
            torch.Tensor: Reconstructed image tensor after passing through the VAE.
        """
        
        latent = self.encoder(x, noise)
        return self.decoder(latent)
    
    
    
if __name__ == '__main__':
    device = 'mps'
    x = torch.randn(1, 3, 512, 512).to(device)
    vae = VAE().to(device)
    enc_x = vae.encode(x, torch.randn(1, 4, 64, 64).to(device))
    dec_x =  vae.decode(enc_x)
    
    print(f'---------> input shape: {x.shape}')
    print(f'---------> encoded input shape: {enc_x.shape}')
    print(f'---------> decoded input shape: {dec_x.shape}')
    