import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self Attention module implementation for transformer-based models.
    
    This module implements the scaled dot-product attention mechanism where
    the queries, keys, and values all come from the same input.
    """
    
    def __init__(self, embed_dim, dropout=0.1):
        """
        Initialize the self-attention module.
        
        Args:
            embed_dim (int): The embedding dimension
            dropout (float): Dropout probability (default: 0.1)
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.scaling = float(embed_dim) ** -0.5  # Scaling factor for dot product
        
        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass for self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len, seq_len]
                                           where 1 indicates valid attention and 0 indicates masked attention
                                           
        Returns:
            tuple: 
                - output (torch.Tensor): Self-attention output of shape [batch_size, seq_len, embed_dim]
                - attention_weights (torch.Tensor): Attention weights of shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # Compute scaled dot-product attention
        # [batch_size, seq_len, seq_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scaling
        
        # Apply mask if provided (1 = attend, 0 = ignore)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.bmm(attn_weights, v)  # [batch_size, seq_len, embed_dim]
        
        return output, attn_weights
    
    def extra_repr(self):
        """Return a description of the module."""
        return f'embed_dim={self.embed_dim}'


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    seq_len = 10
    embed_dim = 64
    
    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create self-attention layer
    self_attn = SelfAttention(embed_dim)
    
    # Forward pass
    output, attention = self_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")