import torch
import torch.nn  as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self Attention mechanism for sequence data.
    
    Attributes:
        scale (float): Scaling factor for the attention scores.
        n_heads (int): Number of attention heads.
        d_head (int): Dimension of each attention head.
        QKV (nn.Linear): Linear layer for Query, Key, Value.
        O (nn.Linear): Linear output layer.
    """
    
    def __init__(self, d_embed, n_heads: int = 4, qkv_bias=True, out_bias=True) -> None:
        """
        Initializes the SelfAttention class.
        
        Args:
            d_embed (int): Dimension of the embedding.
            n_heads (int): Number of attention heads. Defaults to 4.
            qkv_bias (bool): If True, adds bias to QKV linear layer. Defaults to True.
            out_bias (bool): If True, adds bias to O linear layer. Defaults to True.
        """
        
        super().__init__()
        
        self.scale = d_embed ** -0.5
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.QKV = nn.Linear(d_embed, d_embed * 3, bias=qkv_bias)
        self.O = nn.Linear(d_embed, d_embed, bias=out_bias)
        
    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        """
        Forward pass for the SelfAttention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_embed).
            mask (bool): If True, applies the attention mask. Defaults to False.

        Returns:
            torch.Tensor: Processed tensor.
        """
        
        # x: (batch_size, height*width, channels)
        # x: (batch_size, seq_len, d_embed)
        in_shape = x.shape
        bs, seq_len, d_embed = x.shape
        q, k, v = self.QKV(x).chunk(3, dim=-1) # (batch_size, seq_len, d_embed)@(d_embed, d_embed*3) -> (batch_size, seq_len, d_embed*3) -> (3x) (batch_size, seq_len, d_embed)
        
        # (batch_size, seq_len, d_embed) -> (batch_size, seq_len, n_heads, d_head) -> (batch_size, n_heads, seq_len, d_head)
        q = q.view(bs, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not False:
            mask = torch.ones_like(attn_scores).bool().triu(1) # (batch_size, n_heads, seq_len, seq_len)
            attn_scores.masked_fill_(mask, -1e9)
        weights = F.softmax(attn_scores, dim=-1)
        output = weights @ v # (batch_size, n_heads, seq_len, seq_len) -> (batch_size, n_heads, seq_len, d_head) -> (batch_size, n_heads, seq_len, d_head)
        output = output.transpose(1, 2).contiguous().view(in_shape) # (batch_size, n_heads, seq_len, d_head) -> (batch_size, seq_len, n_heads, d_head) -> (batch_size, seq_len, d_embed)
        return self.O(output) # (batch_size, seq_len, d_embed)@(d_embed, d_embed) -> (batch_size, seq_len, d_embed)