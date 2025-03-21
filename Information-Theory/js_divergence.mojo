from distribution import Distribution
from kl_divergence import kl_divergence

fn js_divergence(p: Distribution, q: Distribution) -> Float64:
    """
    Calculate the Jensen-Shannon divergence between P and Q.
    JS divergence is a symmetric measure based on KL divergence.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        The JS divergence in bits, or -1.0 if error
    """
    if p.size != q.size:
        print("Error: Distributions must have the same size")
        return -1.0
    
    # Create the mixture distribution m = (p + q)/2
    var m = Distribution(p.size)
    for i in range(p.size):
        m.set(i, (p.get(i) + q.get(i)) / 2.0)
    
    # JS(P||Q) = (KL(P||M) + KL(Q||M))/2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2.0

fn main():
    print("Jensen-Shannon Divergence Calculation in Mojo")
    
    # Create probability distributions
    var p = Distribution([0.4, 0.5, 0.1])
    var q = Distribution([0.3, 0.4, 0.3])
    
    print("Distribution P:")
    p.print()
    
    print("Distribution Q:")
    q.print()
    
    print("JS Divergence between P and Q:", js_divergence(p, q), "bits")
    
    # JS divergence is symmetric
    print("\nNote: JS divergence is symmetric: JS(P||Q) = JS(Q||P)")
    
    # JS divergence is bounded between 0 and 1 (for log base 2)
    print("JS divergence is bounded: 0 ≤ JS(P||Q) ≤ 1 bit")