import torch
import torch.nn.functional as F
from torch import nn

from attention import SelfAttention, CrossAttention


class TimeEmbeddings(nn.Module):
    """
    A PyTorch implementation of the time embeddings used in Diffusion model.
    """
    def __init__(self, time_embedding_dim: int) -> None:
        super().__init__()
        
        self.linear_1 = nn.Linear(in_features=time_embedding_dim, out_features=time_embedding_dim*4) # 320x1280
        self.linear_2 = nn.Linear(in_features=time_embedding_dim*4, out_features=time_embedding_dim*4) # 1280x1280
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x))) # (1, 320) -> (1, 1280)
    
class Head(nn.Module):
    """
    A PyTorch implementation of the U-Net head.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width) Bx320x64x64
        x = F.silu(self.group_norm(x))
        return self.conv(x) # Bx320x64x64 -> Bx4x64x64
    
class Upsample(nn.Module):
    
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:  BxCxHxW -> BxCx2*Hx2*W
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class UnetAttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, hidden_size: int = 768) -> None:
        super().__init__()
        
        channel_embed = n_heads * n_embed
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channel_embed)
        self.conv_in = nn.Conv2d(in_channels=channel_embed, out_channels=channel_embed, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels=channel_embed, out_channels=channel_embed, kernel_size=1)
        self.layer_norm_1 = nn.LayerNorm(channel_embed)
        self.layer_norm_2 = nn.LayerNorm(channel_embed)
        self.layer_norm_3 = nn.LayerNorm(channel_embed)
        self.attention_1 = SelfAttention(n_heads=n_heads, d_embed=channel_embed, qkv_bias=False)
        self.attention_2 = CrossAttention(n_heads=n_heads, d_embed=channel_embed, d_cross=hidden_size, qkv_bias=False) 
        self.linear_1 = nn.Linear(in_features=channel_embed, out_features=channel_embed*4*2)
        self.linear_2 = nn.Linear(in_features=channel_embed*4, out_features=channel_embed)
        
    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x, gate = self.linear_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        return residue + self.linear_2(x)
        

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_heads*n_embed, height, width)
        # context: (batch_size, seq_len, hidden_size)
        n, c, h, w = x.shape
        long_residue = x
        x = self.group_norm(x)
        x = self.conv_in(x)
        
        x = x.view((n, c, h*w)).transpose(-1, -2) # (batch_size, c, height*width) -> (batch_size, height*width, c)
        
        # Layer Normalization + Self-Attention + Residual
        x = x + self.attention_1(self.layer_norm_1(x)) # (batch_size, height*width, c)
        
        # Layer Normalization + Cross-Attention + Residual
        x = x + self.attention_2(self.layer_norm_2(x), context) # (batch_size, height*width, c)
        
        # Layer Normalization + Feed Forward + Residual
        x = self.ffn(self.layer_norm_3(x)) # (batch_size, height*width, c)
        
        x = x.transpose(-1, -2).view((n, c, h, w)) # (batch_size, height*width, c) -> (batch_size, c, height, width)
        
        return self.conv_out(x) + long_residue # (batch_size, c, height, width)
    
class UnetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int = 1280) -> None:
        super().__init__()
        
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=time_embed_dim, out_features=out_channels)
        
        self.group_norm_merged = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_merged = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width) -> (batch_size, out_channels, height, width)
        residue = x
        x = self.conv(F.silu(self.group_norm(x)))
        t = self.linear(F.silu(time)).unsqueeze(-1).unsqueeze(-1)
        merged = x + t
        return self.conv_merged(F.silu(self.group_norm_merged(merged))) + self.skip(residue)


class ApplyLayer(nn.Sequential):
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UnetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UnetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = self._make_encoder_layers()
        self.bottle_neck = ApplyLayer(
            UnetResidualBlock(in_channels=1280, out_channels=1280),
            UnetAttentionBlock(n_heads=8, n_embed=160),
            UnetResidualBlock(in_channels=1280, out_channels=1280)
        )
        self.decoder   = self._make_decoder_layers()
        
    def forward(self, x: torch.Tensor, time: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: Bx4x64x64
        # time: 1x1280
        # context: Bx77x768
        skip_connections = []
        for enc_layer in self.encoder:
            x = enc_layer(x, time, context)
            skip_connections.append(x)
            
        x = self.bottle_neck(skip_connections[-1], time, context)
        
        for dec_layer in self.decoder:
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = dec_layer(x, time, context)
        
        return x # Bx320x64x64
    
    
    def _make_encoder_layers(self):
        return nn.ModuleList([
            ApplyLayer(nn.Conv2d(in_channels=4, out_channels=320, kernel_size=3, padding=1)), # Bx4x64x64 -> Bx320x64x64
            ApplyLayer(UnetResidualBlock(in_channels=320, out_channels=320), UnetAttentionBlock(n_heads=8, n_embed=40)), # Bx320x64x64 -> Bx320x64x64
            ApplyLayer(UnetResidualBlock(in_channels=320, out_channels=320), UnetAttentionBlock(n_heads=8, n_embed=40)), # Bx320x64x64 -> Bx320x64x64
            
            ApplyLayer(nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1, stride=2)), # Bx320x64x64 -> Bx320x32x32
            ApplyLayer(UnetResidualBlock(in_channels=320, out_channels=640), UnetAttentionBlock(n_heads=8, n_embed=80)), # Bx320x32x32 -> Bx320x32x32
            ApplyLayer(UnetResidualBlock(in_channels=640, out_channels=640), UnetAttentionBlock(n_heads=8, n_embed=80)), # Bx320x32x32 -> Bx320x32x32
            
            ApplyLayer(nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3, padding=1, stride=2)), # Bx640x32x32 -> Bx640x16x16
            ApplyLayer(UnetResidualBlock(in_channels=640, out_channels=1280), UnetAttentionBlock(n_heads=8, n_embed=160)), # Bx640x16x16 -> Bx640x16x16
            ApplyLayer(UnetResidualBlock(in_channels=1280, out_channels=1280), UnetAttentionBlock(n_heads=8, n_embed=160)), # Bx640x16x16 -> Bx640x16x16
            
            ApplyLayer(nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, padding=1, stride=2)), # Bx1280x16x16 -> Bx1280x8x8
            ApplyLayer(UnetResidualBlock(in_channels=1280, out_channels=1280)),
            ApplyLayer(UnetResidualBlock(in_channels=1280, out_channels=1280)),
        ])
        
    def _make_decoder_layers(self):
        return nn.ModuleList([
            ApplyLayer(UnetResidualBlock(in_channels=2560, out_channels=1280)), # Bx2560x8x8 -> Bx1280x8x8
            ApplyLayer(UnetResidualBlock(in_channels=2560, out_channels=1280)), # Bx2560x8x8 -> Bx1280x8x8
            
            ApplyLayer(UnetResidualBlock(in_channels=2560, out_channels=1280), Upsample(in_channels=1280)), # Bx2560x8x8 -> Bx1280x16x16
            ApplyLayer(UnetResidualBlock(in_channels=2560, out_channels=1280), UnetAttentionBlock(n_heads=8, n_embed=160)), # Bx2560x16x16 -> Bx1280x16x16
            ApplyLayer(UnetResidualBlock(in_channels=2560, out_channels=1280), UnetAttentionBlock(n_heads=8, n_embed=160)), # Bx2560x16x16 -> Bx1280x16x16
            
            ApplyLayer(UnetResidualBlock(in_channels=1920, out_channels=1280), UnetAttentionBlock(n_heads=8, n_embed=160), Upsample(in_channels=1280)), # Bx2560x16x16 -> Bx1280x32x32
            ApplyLayer(UnetResidualBlock(in_channels=1920, out_channels=640), UnetAttentionBlock(n_heads=8, n_embed=80)), # Bx1920x32x32 -> Bx640x32x32
            ApplyLayer(UnetResidualBlock(in_channels=1280, out_channels=640), UnetAttentionBlock(n_heads=8, n_embed=80)), # Bx1280x32x32 -> Bx640x32x32
            
            ApplyLayer(UnetResidualBlock(in_channels=960, out_channels=640), UnetAttentionBlock(n_heads=8, n_embed=80), Upsample(in_channels=640)), # Bx960x32x32 -> Bx640x64x64
            ApplyLayer(UnetResidualBlock(in_channels=960, out_channels=320), UnetAttentionBlock(n_heads=8, n_embed=40)), # Bx960x32x32 -> Bx320x64x64
            ApplyLayer(UnetResidualBlock(in_channels=640, out_channels=320), UnetAttentionBlock(n_heads=8, n_embed=40)), # Bx640x32x32 -> Bx320x64x64
            ApplyLayer(UnetResidualBlock(in_channels=640, out_channels=320), UnetAttentionBlock(n_heads=8, n_embed=40)), # Bx640x32x32 -> Bx320x64x64  
        ])