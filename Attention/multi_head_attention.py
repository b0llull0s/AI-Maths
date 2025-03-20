import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module implementation for transformer-based models.
    
    This module splits the embedding dimension into multiple heads, applies
    self-attention separately to each head, and then concatenates the results.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize the multi-head attention module.
        
        Args:
            embed_dim (int): The embedding dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability (default: 0.1)
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(self.head_dim) ** -0.5  # Scaling factor for dot product
        
    def _reshape_for_multihead(self, x):
        """
        Reshape tensor for multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, embed_dim = x.size()
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, x, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len, seq_len]
                                          where 1 indicates valid attention and 0 indicates masked attention
                                          
        Returns:
            tuple: 
                - output (torch.Tensor): Multi-head attention output of shape [batch_size, seq_len, embed_dim]
                - attention_weights (torch.Tensor): Average attention weights of shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        q = self._reshape_for_multihead(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self._reshape_for_multihead(k)  # [batch_size, num_heads, seq_len, head_dim]
        v = self._reshape_for_multihead(v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute scaled dot-product attention for each head
        # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided (1 = attend, 0 = ignore)
        if mask is not None:
            # Expand mask for num_heads dimension
            expanded_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, float('-inf'))
        
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Store attention weights for visualization or analysis
        attention_probs = attn_weights.clone()
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back: [batch_size, seq_len, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Apply output projection
        output = self.out_proj(context)
        
        # Average attention weights across heads for visualization
        avg_attn_weights = attention_probs.mean(dim=1)
        
        return output, avg_attn_weights
    
    def extra_repr(self):
        """Return a description of the module."""
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}'


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    seq_len = 10
    embed_dim = 64
    num_heads = 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create multi-head attention layer
    multihead_attn = MultiHeadAttention(embed_dim, num_heads)
    
    # Forward pass
    output, attention = multihead_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")