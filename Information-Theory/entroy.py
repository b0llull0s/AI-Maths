"""
Entropy calculation utilities for LLM analysis and development.
"""

import numpy as np
from typing import List, Union


def entropy(probabilities: Union[List[float], np.ndarray], 
           base: float = 2.0,
           epsilon: float = 1e-10) -> float:
    """
    Compute the entropy of a probability distribution.
    
    Args:
        probabilities: A list or array of probabilities that sum to 1.
        base: The logarithm base to use (default: 2 for bits).
        epsilon: Small constant to avoid log(0) errors.
        
    Returns:
        The entropy value in the specified units.
        
    Examples:
        >>> entropy([0.5, 0.5])
        1.0
        >>> entropy([1.0, 0.0])
        0.0
    """
    probabilities = np.array(probabilities)
    # Add epsilon to avoid log(0)
    probabilities = np.clip(probabilities, epsilon, 1.0)
    
    # Normalize if not already a valid probability distribution
    if not np.isclose(np.sum(probabilities), 1.0):
        probabilities = probabilities / np.sum(probabilities)
        
    return -np.sum(probabilities * np.log(probabilities) / np.log(base))


def perplexity(probabilities: Union[List[float], np.ndarray], 
              base: float = 2.0) -> float:
    """
    Calculate perplexity, a measure of how well a probability model 
    predicts a sample. Lower perplexity indicates better prediction.
    
    Args:
        probabilities: Probability assigned to each outcome by the model.
        base: The logarithm base to use (default: 2 for bits).
        
    Returns:
        The perplexity value.
    """
    entropy_val = entropy(probabilities, base=base)
    return base ** entropy_val


# Example usage
if __name__ == "__main__":
    # Define a probability distribution
    p = [0.5, 0.3, 0.2]
    
    # Calculate entropy
    entropy_p = entropy(p)
    print(f"Entropy of distribution: {entropy_p:.4f} bits")
    
    # Calculate perplexity
    perplexity_p = perplexity(p)
    print(f"Perplexity of distribution: {perplexity_p:.4f}")