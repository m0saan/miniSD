import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from attention import SelfAttention
@dataclass
class Config:
    """
    A dataclass to store the configuration of the CLIP model.
    """
    
    vocab_size: int = 49408
    hidden_size: int = 768
    seq_len: int = 77
    n_heads: int = 12
    n_layers: int = 12
    layer_norm_eps: float = 1e-5
    

class CLIPEmbedding(nn.Module):
    """
    A class that converts input ids to embeddings.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(config.seq_len, config.hidden_size))
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        embeddings = self.token_embedding(input_ids) + self.position_embedding # (batch_size, seq_len, hidden_size)
        return embeddings
    
class CLIPBlock(nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = SelfAttention(config.hidden_size, config.n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_size)
        x = x + self.attention(self.layer_norm1(x), mask=True)
        return x + self.mlp(self.layer_norm2(x)) # (batch_size, seq_len, hidden_size)
    
class CLIP(nn.Module):
    """
    A basic implementation of OpenAI's CLIP model
    """
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.embeddings = CLIPEmbedding(config)
        self.layers = nn.ModuleList([CLIPBlock(config) for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        embeddings = self.embeddings(x.type(torch.long)) # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        for layer in self.layers:
            embeddings = layer(embeddings)
        return self.layer_norm(embeddings) # (batch_size, seq_len, hidden_size)
    
    
if __name__ == "__main__":
    config = Config()
    model = CLIP(config)
    print(model(torch.randint(0, 100, (1, 77)).type(torch.long)).shape)