from Math import log
from distribution import Distribution

fn kl_divergence(p: Distribution, q: Distribution) -> Float64:
    """
    Calculate the Kullback-Leibler divergence from Q to P.
    KL(P||Q) measures how much information is lost when using Q to approximate P.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        The KL divergence in bits, or -1.0 if error, or infinity if undefined
    """
    if p.size != q.size:
        print("Error: Distributions must have the same size")
        return -1.0
        
    var p_copy = p
    var q_copy = q
    
    if not p_copy.is_normalized:
        p_copy.normalize()
        
    if not q_copy.is_normalized:
        q_copy.normalize()
        
    var kl: Float64 = 0.0
    for i in range(p_copy.size):
        let pi = p_copy.get(i)
        if pi > 0:
            let qi = q_copy.get(i)
            if qi <= 0:
                print("Error: KL divergence undefined when q = 0 and p > 0")
                return Float64.inf
            
            kl += pi * (log(pi) - log(qi)) / log(2.0)  # log base 2 for bits
            
    return kl

fn main():
    print("KL Divergence Calculation in Mojo")
    
    # Create probability distributions
    var p = Distribution([0.4, 0.5, 0.1])
    var q = Distribution([0.3, 0.4, 0.3])
    
    print("Distribution P:")
    p.print()
    
    print("Distribution Q:")
    q.print()
    
    print("KL Divergence (P||Q):", kl_divergence(p, q), "bits")
    print("KL Divergence (Q||P):", kl_divergence(q, p), "bits")
    
    # KL divergence is not symmetric
    print("\nNote: KL divergence is not symmetric: KL(P||Q) ≠ KL(Q||P)")