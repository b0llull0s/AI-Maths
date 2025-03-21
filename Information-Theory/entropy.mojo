from Math import log
from distribution import Distribution

fn entropy(p: Distribution) -> Float64:
    """
    Calculate the Shannon entropy of a probability distribution.
    
    Args:
        p: A probability distribution
        
    Returns:
        The entropy in bits
    """
    var p_copy = p
    if not p_copy.is_normalized:
        p_copy.normalize()
            
    var h: Float64 = 0.0
    for i in range(p_copy.size):
        let pi = p_copy.get(i)
        if pi > 0:
            h -= pi * log(pi) / log(2.0)  # Use log base 2 for bits
                
    return h

fn main():
    print("Entropy Calculation in Mojo")
    
    # Create probability distributions
    var uniform = Distribution(5)  # Creates uniform distribution
    var peaked = Distribution([0.01, 0.01, 0.96, 0.01, 0.01])
    
    print("Uniform distribution:")
    uniform.print()
    print("Entropy:", entropy(uniform), "bits")
    
    print("\nPeaked distribution:")
    peaked.print()
    print("Entropy:", entropy(peaked), "bits")
    
    # The more uniform the distribution, the higher the entropy
    print("\nNote: The more uniform the distribution, the higher the entropy.")