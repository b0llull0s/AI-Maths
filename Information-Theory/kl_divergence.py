"""
KL Divergence calculation utilities for LLM analysis and development.
"""

import numpy as np
from typing import List, Union


def kl_divergence(p: Union[List[float], np.ndarray], 
                 q: Union[List[float], np.ndarray],
                 base: float = 2.0,
                 epsilon: float = 1e-10) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.
    
    Args:
        p: The true probability distribution.
        q: The approximate probability distribution.
        base: The logarithm base to use (default: 2 for bits).
        epsilon: Small constant to avoid division by zero or log(0) errors.
        
    Returns:
        The KL divergence between p and q.
        
    Examples:
        >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
        0.0
        >>> round(kl_divergence([0.9, 0.1], [0.5, 0.5]), 3)
        0.531
    """
    p = np.array(p)
    q = np.array(q)
    
    # Add epsilon to avoid log(0) and division by zero
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # Normalize if not already valid probability distributions
    if not np.isclose(np.sum(p), 1.0):
        p = p / np.sum(p)
    if not np.isclose(np.sum(q), 1.0):
        q = q / np.sum(q)
        
    return np.sum(p * np.log(p / q) / np.log(base))


def jensen_shannon_divergence(p: Union[List[float], np.ndarray], 
                             q: Union[List[float], np.ndarray],
                             base: float = 2.0) -> float:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    This is a symmetric and smoothed version of KL divergence.
    
    Args:
        p: First probability distribution.
        q: Second probability distribution.
        base: The logarithm base to use (default: 2 for bits).
        
    Returns:
        The JS divergence between p and q (value between 0 and 1).
    """
    p = np.array(p)
    q = np.array(q)
    
    # Normalize if not already valid probability distributions
    if not np.isclose(np.sum(p), 1.0):
        p = p / np.sum(p)
    if not np.isclose(np.sum(q), 1.0):
        q = q / np.sum(q)
        
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, base=base) + 
                  kl_divergence(q, m, base=base))


# Example usage
if __name__ == "__main__":
    # Define two probability distributions
    p = [0.5, 0.3, 0.2]  # True distribution
    q = [0.4, 0.4, 0.2]  # Approximate distribution
    
    # Calculate KL divergence
    kl_pq = kl_divergence(p, q)
    print(f"KL Divergence (P||Q): {kl_pq:.4f} bits")
    
    # Calculate JS divergence
    js_pq = jensen_shannon_divergence(p, q)
    print(f"Jensen-Shannon Divergence: {js_pq:.4f} bits")
    
    # KL divergence is not symmetric
    kl_qp = kl_divergence(q, p)
    print(f"KL Divergence (Q||P): {kl_qp:.4f} bits")
    
    # JS divergence is symmetric
    js_qp = jensen_shannon_divergence(q, p)
    print(f"Jensen-Shannon Divergence (Q,P): {js_qp:.4f} bits")