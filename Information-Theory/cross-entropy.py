"""
Cross-entropy calculation utilities for LLM analysis and development.
"""

import numpy as np
from typing import List, Union


def cross_entropy(p: Union[List[float], np.ndarray], 
                 q: Union[List[float], np.ndarray],
                 base: float = 2.0,
                 epsilon: float = 1e-10) -> float:
    """
    Compute the cross-entropy between a true distribution p and an approximate distribution q.
    
    Args:
        p: The true probability distribution.
        q: The approximate probability distribution.
        base: The logarithm base to use (default: 2 for bits).
        epsilon: Small constant to avoid division by zero or log(0) errors.
        
    Returns:
        The cross-entropy between p and q.
        
    Examples:
        >>> round(cross_entropy([0.5, 0.5], [0.5, 0.5]), 4)
        1.0
        >>> round(cross_entropy([0.9, 0.1], [0.5, 0.5]), 4)
        1.0
    """
    p = np.array(p)
    q = np.array(q)
    
    # Add epsilon to avoid log(0)
    q = np.clip(q, epsilon, 1.0)
    
    # Normalize if not already valid probability distributions
    if not np.isclose(np.sum(p), 1.0):
        p = p / np.sum(p)
    if not np.isclose(np.sum(q), 1.0):
        q = q / np.sum(q)
        
    return -np.sum(p * np.log(q) / np.log(base))


# Relationship between cross-entropy, entropy, and KL divergence
def cross_entropy_from_kl(p: Union[List[float], np.ndarray], 
                         q: Union[List[float], np.ndarray],
                         base: float = 2.0):
    """
    Compute cross-entropy using the relationship:
    Cross-entropy(p,q) = Entropy(p) + KL-Divergence(p||q)
    
    This is provided as an educational reference.
    
    Args:
        p: The true probability distribution.
        q: The approximate probability distribution.
        base: The logarithm base to use (default: 2 for bits).
        
    Returns:
        The cross-entropy between p and q calculated from entropy and KL divergence.
    """
    from entropy import entropy
    from kl_divergence import kl_divergence
    
    entropy_p = entropy(p, base=base)
    kl_pq = kl_divergence(p, q, base=base)
    
    return entropy_p + kl_pq


# Example usage
if __name__ == "__main__":
    # Define two probability distributions
    p = [0.5, 0.3, 0.2]  # True distribution
    q = [0.4, 0.4, 0.2]  # Approximate distribution
    
    # Calculate cross-entropy
    ce_pq = cross_entropy(p, q)
    print(f"Cross-entropy (P,Q): {ce_pq:.4f} bits")
    
    try:
        # This requires the other modules to be in the same directory
        ce_kl = cross_entropy_from_kl(p, q)
        print(f"Cross-entropy via KL+Entropy: {ce_kl:.4f} bits")
        print(f"These should be equal: {ce_pq == ce_kl}")
    except ImportError:
        print("Note: To use cross_entropy_from_kl, place entropy.py and kl_divergence.py in the same directory.")