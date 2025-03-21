from tensor import Tensor, TensorShape
from math import sin, cos, exp, log

struct PositionalEncoding:
    """
    Positional Encoding module implementation for transformer-based models.
    
    This module adds positional information to the input embeddings using
    sinusoidal functions of different frequencies.
    """
    var embed_dim: Int
    var max_seq_len: Int
    var pe: Tensor[DType.float32]
    var dropout_rate: Float32
    
    fn __init__(inout self, embed_dim: Int, max_seq_len: Int = 5000, dropout: Float32 = 0.1):
        """
        Initialize the positional encoding module.
        
        Args:
            embed_dim: The embedding dimension
            max_seq_len: Maximum sequence length (default: 5000)
            dropout: Dropout probability (default: 0.1)
        """
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        
        # Create positional encoding matrix
        # Shape: [max_seq_len, embed_dim]
        self.pe = Tensor[DType.float32](TensorShape(1, max_seq_len, embed_dim))
        self._initialize_pe()
    
    fn _initialize_pe(self):
        """
        Initialize the positional encoding buffer with sinusoidal patterns.
        """
        # For simplicity, we'll implement this with loops
        # In a real implementation, you might use more vectorized operations
        
        # Initialize with zeros
        self.pe.zero_()
        
        for pos in range(self.max_seq_len):
            for i in range(0, self.embed_dim, 2):
                # Calculate the division term
                let div_term = exp(-log(10000.0) * Float64(i) / Float64(self.embed_dim))
                let angle = Float64(pos) * div_term
                
                # Apply sine to even indices
                if i < self.embed_dim:
                    self.pe[0, pos, i] = sin(angle)
                
                # Apply cosine to odd indices
                if i + 1 < self.embed_dim:
                    self.pe[0, pos, i + 1] = cos(angle)
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Forward pass for positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Input with positional encoding added, of shape [batch_size, seq_len, embed_dim]
        """
        let seq_len = x.dim(1)
        
        # Check if sequence length exceeds maximum length
        if seq_len > self.max_seq_len:
            print("Error: Sequence length (", seq_len, ") exceeds maximum length (", self.max_seq_len, ")")
            return x
        
        # Add positional encoding to input embeddings
        var result = x + self.pe[:, :seq_len, :]
        
        # Apply dropout (simplified implementation)
        return dropout(result, self.dropout_rate)

fn dropout(x: Tensor[DType.float32], rate: Float32) -> Tensor[DType.float32]:
    """Simple dropout implementation."""
    # In a real implementation, this would be more sophisticated
    if rate == 0.0:
        return x
        
    # Create a mask with probability (1 - rate)
    var mask = Tensor[DType.float32].random(x.shape()) < (1.0 - rate)
    return x * mask * (1.0 / (1.0 - rate))

fn main():
    """Example usage."""
    let batch_size = 4
    let seq_len = 20
    let embed_dim = 64
    
    # Create random input embeddings
    var x = Tensor[DType.float32].randn(TensorShape(batch_size, seq_len, embed_dim))
    
    # Create positional encoding layer
    var pos_encoder = PositionalEncoding(embed_dim)
    
    # Apply positional encoding
    let encoded = pos_encoder.forward(x)
    
    print("Input shape:", x.shape())
    print("Encoded output shape:", encoded.shape())