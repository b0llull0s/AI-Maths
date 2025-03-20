import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module implementation for transformer-based models.
    
    This module adds positional information to the input embeddings using
    sinusoidal functions of different frequencies.
    """
    
    def __init__(self, embed_dim, max_seq_len=5000, dropout=0.1):
        """
        Initialize the positional encoding module.
        
        Args:
            embed_dim (int): The embedding dimension
            max_seq_len (int): Maximum sequence length (default: 5000)
            dropout (float): Dropout probability (default: 0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # Shape: [max_seq_len, embed_dim]
        pe = torch.zeros(max_seq_len, embed_dim)
        
        # Create position vector: [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term: [embed_dim/2]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (if embed_dim is odd, the last index is sin)
        pe[:, 1::2] = torch.cos(position * div_term[:((embed_dim + 1) // 2 - 1)])
        
        # Add batch dimension: [1, max_seq_len, embed_dim]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but should be saved and loaded with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Forward pass for positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Input with positional encoding added, of shape [batch_size, seq_len, embed_dim]
        """
        # Add positional encoding to input embeddings
        # x shape: [batch_size, seq_len, embed_dim]
        # pe shape: [1, max_seq_len, embed_dim]
        seq_len = x.size(1)
        
        # Check if sequence length exceeds maximum length
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds maximum length ({self.max_seq_len})")
        
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
    
    def visualize(self, seq_len=100):
        """
        Visualize the positional encodings.
        
        Args:
            seq_len (int): Length of sequence to visualize (default: 100)
        """
        plt.figure(figsize=(15, 8))
        pe = self.pe[0, :seq_len, :].numpy()
        plt.imshow(pe, aspect='auto', cmap='viridis')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Position')
        plt.colorbar(label='Value')
        plt.title('Positional Encoding Visualization')
        plt.tight_layout()
        plt.show()
    
    def extra_repr(self):
        """Return a description of the module."""
        return f'embed_dim={self.embed_dim}, max_seq_len={self.max_seq_len}'


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    seq_len = 20
    embed_dim = 64
    
    # Create random input embeddings
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create positional encoding layer
    pos_encoder = PositionalEncoding(embed_dim)
    
    # Apply positional encoding
    encoded = pos_encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Encoded output shape: {encoded.shape}")
    
    # Visualize the positional encodings
    try:
        pos_encoder.visualize(seq_len=50)
    except Exception as e:
        print(f"Visualization requires matplotlib: {e}")