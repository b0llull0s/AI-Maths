from tensor import Tensor, TensorShape
from math import sqrt

struct SelfAttention:
    """
    Self Attention module implementation for transformer-based models.
    
    This module implements the scaled dot-product attention mechanism where
    the queries, keys, and values all come from the same input.
    """
    var embed_dim: Int
    var scaling: Float32
    
    # Linear projections
    var q_proj: LinearLayer
    var k_proj: LinearLayer
    var v_proj: LinearLayer
    
    # Dropout rate
    var dropout_rate: Float32
    
    fn __init__(inout self, embed_dim: Int, dropout: Float32 = 0.1):
        """
        Initialize the self-attention module.
        
        Args:
            embed_dim: The embedding dimension
            dropout: Dropout probability (default: 0.1)
        """
        self.embed_dim = embed_dim
        self.dropout_rate = dropout
        
        # Scaling factor for dot product
        self.scaling = 1.0 / sqrt(Float32(embed_dim))
        
        # Linear projections for query, key, and value
        self.q_proj = LinearLayer(embed_dim, embed_dim)
        self.k_proj = LinearLayer(embed_dim, embed_dim)
        self.v_proj = LinearLayer(embed_dim, embed_dim)
    
    fn forward(self, x: Tensor[DType.float32], mask: Optional[Tensor[DType.float32]] = None) -> Tuple[Tensor[DType.float32], Tensor[DType.float32]]:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional attention mask of shape [batch_size, seq_len, seq_len]
                  where 1 indicates valid attention and 0 indicates masked attention
                  
        Returns:
            tuple: 
                - output: Self-attention output of shape [batch_size, seq_len, embed_dim]
                - attention_weights: Attention weights of shape [batch_size, seq_len, seq_len]
        """
        let batch_size = x.dim(0)
        let seq_len = x.dim(1)
        
        # Project inputs to queries, keys, and values
        let q = self.q_proj.forward(x)  # [batch_size, seq_len, embed_dim]
        let k = self.k_proj.forward(x)  # [batch_size, seq_len, embed_dim]
        let v = self.v_proj.forward(x)  # [batch_size, seq_len, embed_dim]
        
        # Compute scaled dot-product attention
        # [batch_size, seq_len, seq_len]
        var attn_weights = matmul(q, k.transpose(1, 2)) * self.scaling
        
        # Apply mask if provided (1 = attend, 0 = ignore)
        if mask:
            attn_weights = attn_weights.masked_fill(mask == 0, Float32.neg_infinity)
        
        # Apply softmax to get attention probabilities
        let attention_probs = softmax(attn_weights, dim=-1)
        
        # Apply dropout
        var dropped_attn = dropout(attention_probs, self.dropout_rate)
        
        # Apply attention weights to values
        let output = matmul(dropped_attn, v)  # [batch_size, seq_len, embed_dim]
        
        return (output, attention_probs)

struct LinearLayer:
    """Simple linear layer implementation."""
    var in_features: Int
    var out_features: Int
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    
    fn __init__(inout self, in_features: Int, out_features: Int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias (simplified)
        self.weights = Tensor[DType.float32](in_features, out_features)
        self.bias = Tensor[DType.float32](out_features)
        
        # Xavier/Glorot initialization would go here
        self._init_parameters()
    
    fn _init_parameters(self):
        """Initialize weights using Xavier/Glorot initialization."""
        let std = sqrt(2.0 / (self.in_features + self.out_features))
        self.weights.normal_(mean=0.0, std=std)
        self.bias.zero_()
    
    fn forward(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Forward pass for linear layer."""
        return matmul(x, self.weights) + self.bias

fn matmul(a: Tensor[DType.float32], b: Tensor[DType.float32]) -> Tensor[DType.float32]:
    """Matrix multiplication helper."""
    # In a real implementation, this would be optimized
    return a @ b

fn softmax(x: Tensor[DType.float32], dim: Int) -> Tensor[DType.float32]:
    """Softmax implementation."""
    # In a real implementation, this would use Mojo's numerics capabilities
    let max_x = x.max(dim=dim, keepdim=True)
    let exp_x = (x - max_x).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

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
    let seq_len = 10
    let embed_dim = 64
    
    # Create random input
    var x = Tensor[DType.float32].randn(TensorShape(batch_size, seq_len, embed_dim))
    
    # Create self-attention layer
    var self_attn = SelfAttention(embed_dim)
    
    # Forward pass
    let (output, attention) = self_attn.forward(x)
    
    print("Input shape:", x.shape())
    print("Output shape:", output.shape())
    print("Attention weights shape:", attention.shape())