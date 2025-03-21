from tensor import Tensor, TensorShape
from utils.index import Index
from math import sqrt

struct MultiHeadAttention:
    """
    Multi-Head Attention module implementation for transformer-based models.
    
    This module splits the embedding dimension into multiple heads, applies
    self-attention separately to each head, and then concatenates the results.
    """
    var embed_dim: Int
    var num_heads: Int
    var head_dim: Int
    var scaling: Float32
    
    # Linear projections
    var q_proj: LinearLayer
    var k_proj: LinearLayer
    var v_proj: LinearLayer
    var out_proj: LinearLayer
    
    # Dropout rate (not fully implemented in this version)
    var dropout_rate: Float32
    
    fn __init__(inout self, embed_dim: Int, num_heads: Int, dropout: Float32 = 0.1):
        """
        Initialize the multi-head attention module.
        
        Args:
            embed_dim: The embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability (default: 0.1)
        """
        # Ensure embedding dimension is divisible by number of heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        
        # Linear projections for query, key, value
        self.q_proj = LinearLayer(embed_dim, embed_dim)
        self.k_proj = LinearLayer(embed_dim, embed_dim)
        self.v_proj = LinearLayer(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = LinearLayer(embed_dim, embed_dim)
        
        # Scaling factor for dot product
        self.scaling = 1.0 / sqrt(Float32(self.head_dim))
    
    fn _reshape_for_multihead(self, x: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Reshape tensor for multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Reshaped tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        let batch_size = x.dim(0)
        let seq_len = x.dim(1)
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        var reshaped = x.reshape(TensorShape(batch_size, seq_len, self.num_heads, self.head_dim))
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return reshaped.permute(0, 2, 1, 3)
    
    fn forward(self, x: Tensor[DType.float32], mask: Optional[Tensor[DType.float32]] = None) -> Tuple[Tensor[DType.float32], Tensor[DType.float32]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional attention mask of shape [batch_size, seq_len, seq_len]
                  where 1 indicates valid attention and 0 indicates masked attention
                  
        Returns:
            tuple: 
                - output: Multi-head attention output of shape [batch_size, seq_len, embed_dim]
                - attention_weights: Average attention weights of shape [batch_size, seq_len, seq_len]
        """
        let batch_size = x.dim(0)
        let seq_len = x.dim(1)
        
        # Project inputs to queries, keys, and values
        let q = self.q_proj.forward(x)  # [batch_size, seq_len, embed_dim]
        let k = self.k_proj.forward(x)  # [batch_size, seq_len, embed_dim]
        let v = self.v_proj.forward(x)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        let q_reshaped = self._reshape_for_multihead(q)  # [batch_size, num_heads, seq_len, head_dim]
        let k_reshaped = self._reshape_for_multihead(k)  # [batch_size, num_heads, seq_len, head_dim]
        let v_reshaped = self._reshape_for_multihead(v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute scaled dot-product attention for each head
        # [batch_size, num_heads, seq_len, seq_len]
        var attn_weights = matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * self.scaling
        
        # Apply mask if provided (1 = attend, 0 = ignore)
        if mask:
            # Expand mask for num_heads dimension
            let expanded_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, Float32.neg_infinity)
        
        # Apply softmax to get attention probabilities
        let attention_probs = softmax(attn_weights, dim=-1)
        
        # Apply dropout (simplified for this implementation)
        var dropped_attn = dropout(attention_probs, self.dropout_rate)
        
        # Apply attention weights to values
        var context = matmul(dropped_attn, v_reshaped)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back: [batch_size, seq_len, embed_dim]
        context = context.permute(0, 2, 1, 3)
        context = context.reshape(TensorShape(batch_size, seq_len, self.embed_dim))
        
        # Apply output projection
        let output = self.out_proj.forward(context)
        
        # Average attention weights across heads for visualization
        let avg_attn_weights = attention_probs.mean(dim=1)
        
        return (output, avg_attn_weights)

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
    let num_heads = 8
    
    # Create random input
    var x = Tensor[DType.float32].randn(TensorShape(batch_size, seq_len, embed_dim))
    
    # Create multi-head attention layer
    var multihead_attn = MultiHeadAttention(embed_dim, num_heads)
    
    # Forward pass
    let (output, attention) = multihead_attn.forward(x)
    
    print("Input shape:", x.shape())
    print("Output shape:", output.shape())
    print("Attention weights shape:", attention.shape())