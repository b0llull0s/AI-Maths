from distribution import Distribution
from entropy import entropy
from kl_divergence import kl_divergence

fn cross_entropy(p: Distribution, q: Distribution) -> Float64:
    """
    Calculate the cross entropy between distributions P and Q.
    H(P,Q) = H(P) + KL(P||Q)
    
    Args:
        p: First probability distribution (true distribution)
        q: Second probability distribution (predicted distribution)
        
    Returns:
        The cross entropy in bits, or -1.0 if error
    """
    # Cross entropy H(p,q) = H(p) + KL(p||q)
    let h_p = entropy(p)
    let kl = kl_divergence(p, q)
    
    if kl < 0:  # Error occurred in KL calculation
        return -1.0
        
    return h_p + kl

fn main():
    print("Cross Entropy Calculation in Mojo")
    
    # Create probability distributions
    var p = Distribution([0.4, 0.5, 0.1])
    var q = Distribution([0.3, 0.4, 0.3])
    
    print("Distribution P (true distribution):")
    p.print()
    
    print("Distribution Q (predicted distribution):")
    q.print()
    
    print("Entropy of P:", entropy(p), "bits")
    print("Cross Entropy H(P,Q):", cross_entropy(p, q), "bits")
    print("Cross Entropy H(Q,P):", cross_entropy(q, p), "bits")
    
    # Cross entropy is not symmetric
    print("\nNote: Cross entropy is not symmetric: H(P,Q) ≠ H(Q,P)")
    
    # The cross entropy is minimized when P = Q
    print("Cross entropy is minimized when P = Q, and in that case H(P,P) = H(P)")