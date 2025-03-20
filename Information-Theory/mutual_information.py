"""
Mutual information calculation utilities for LLM analysis and development.
"""

import numpy as np
from typing import List, Union, Tuple
import matplotlib.pyplot as plt


def mutual_information(joint_distribution: np.ndarray,
                      base: float = 2.0,
                      epsilon: float = 1e-10) -> float:
    """
    Compute the mutual information between two random variables from their joint probability distribution.
    
    Args:
        joint_distribution: 2D array representing joint probability distribution P(X,Y)
        base: The logarithm base to use (default: 2 for bits)
        epsilon: Small constant to avoid log(0) errors
        
    Returns:
        The mutual information I(X;Y) in the specified units (bits by default)
        
    Examples:
        # Independent variables have 0 mutual information
        >>> p_xy = np.array([[0.25, 0.25], [0.25, 0.25]])
        >>> round(mutual_information(p_xy), 4)
        0.0
        
        # Perfectly correlated variables have mutual information equal to entropy
        >>> p_xy = np.array([[0.5, 0], [0, 0.5]])
        >>> round(mutual_information(p_xy), 4)
        1.0
    """
    joint_distribution = np.array(joint_distribution)
    
    # Ensure it's a valid probability distribution
    if not np.isclose(np.sum(joint_distribution), 1.0):
        joint_distribution = joint_distribution / np.sum(joint_distribution)
    
    # Add epsilon to avoid log(0)
    joint_distribution = np.clip(joint_distribution, epsilon, 1.0)
    
    # Compute marginal distributions
    p_x = np.sum(joint_distribution, axis=1)
    p_y = np.sum(joint_distribution, axis=0)
    
    # Compute mutual information
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if joint_distribution[i, j] > 0:
                mi += joint_distribution[i, j] * np.log(joint_distribution[i, j] / (p_x[i] * p_y[j])) / np.log(base)
    
    return mi


def conditional_entropy(joint_distribution: np.ndarray,
                       base: float = 2.0,
                       epsilon: float = 1e-10) -> Tuple[float, float]:
    """
    Compute the conditional entropy H(X|Y) and H(Y|X) from a joint probability distribution.
    
    Args:
        joint_distribution: 2D array representing joint probability distribution P(X,Y)
        base: The logarithm base to use (default: 2 for bits)
        epsilon: Small constant to avoid log(0) errors
        
    Returns:
        A tuple (H(X|Y), H(Y|X)) of conditional entropies
    """
    joint_distribution = np.array(joint_distribution)
    
    # Ensure it's a valid probability distribution
    if not np.isclose(np.sum(joint_distribution), 1.0):
        joint_distribution = joint_distribution / np.sum(joint_distribution)
    
    # Add epsilon to avoid log(0)
    joint_distribution = np.clip(joint_distribution, epsilon, 1.0)
    
    # Compute marginal distributions
    p_x = np.sum(joint_distribution, axis=1)
    p_y = np.sum(joint_distribution, axis=0)
    
    # Compute H(X)
    h_x = -np.sum(p_x * np.log(p_x) / np.log(base))
    
    # Compute H(Y)
    h_y = -np.sum(p_y * np.log(p_y) / np.log(base))
    
    # Compute joint entropy H(X,Y)
    h_xy = -np.sum(joint_distribution * np.log(joint_distribution) / np.log(base))
    
    # Compute conditional entropies
    h_x_given_y = h_xy - h_y  # H(X|Y) = H(X,Y) - H(Y)
    h_y_given_x = h_xy - h_x  # H(Y|X) = H(X,Y) - H(X)
    
    return h_x_given_y, h_y_given_x


def normalized_mutual_information(joint_distribution: np.ndarray,
                                base: float = 2.0) -> float:
    """
    Compute the normalized mutual information (NMI) between two random variables.
    NMI is scaled to be between 0 and 1, where 0 means no mutual information and 
    1 means perfect correlation.
    
    Args:
        joint_distribution: 2D array representing joint probability distribution P(X,Y)
        base: The logarithm base to use (default: 2 for bits)
        
    Returns:
        The normalized mutual information (NMI) between X and Y
    """
    joint_distribution = np.array(joint_distribution)
    
    # Compute marginals
    p_x = np.sum(joint_distribution, axis=1)
    p_y = np.sum(joint_distribution, axis=0)
    
    # Compute entropies
    h_x = -np.sum(p_x * np.log(np.clip(p_x, 1e-10, 1.0)) / np.log(base))
    h_y = -np.sum(p_y * np.log(np.clip(p_y, 1e-10, 1.0)) / np.log(base))
    
    # Compute mutual information
    mi = mutual_information(joint_distribution, base=base)
    
    # Compute normalized mutual information
    # Using the arithmetic mean of H(X) and H(Y) in the denominator
    nmi = 2 * mi / (h_x + h_y) if (h_x + h_y) > 0 else 0.0
    
    return nmi


def pointwise_mutual_information(joint_distribution: np.ndarray,
                               base: float = 2.0,
                               epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute the pointwise mutual information (PMI) for each pair of values.
    PMI measures how much the probability of co-occurrence of a pair differs
    from what we would expect if they were independent.
    
    Args:
        joint_distribution: 2D array representing joint probability distribution P(X,Y)
        base: The logarithm base to use (default: 2 for bits)
        epsilon: Small constant to avoid division by zero
        
    Returns:
        A 2D array of the same shape as joint_distribution containing PMI values
    """
    joint_distribution = np.array(joint_distribution)
    
    # Ensure it's a valid probability distribution
    if not np.isclose(np.sum(joint_distribution), 1.0):
        joint_distribution = joint_distribution / np.sum(joint_distribution)
    
    # Compute marginal distributions
    p_x = np.sum(joint_distribution, axis=1)
    p_y = np.sum(joint_distribution, axis=0)
    
    # Compute PMI for each element
    pmi = np.zeros_like(joint_distribution)
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if joint_distribution[i, j] > epsilon:
                pmi[i, j] = np.log(joint_distribution[i, j] / (p_x[i] * p_y[j])) / np.log(base)
            else:
                pmi[i, j] = -np.inf  # Undefined PMI for zero probabilities
    
    return pmi


def plot_mutual_information(joint_distribution: np.ndarray,
                          x_labels: List[str] = None,
                          y_labels: List[str] = None,
                          title: str = "Mutual Information Analysis",
                          figsize: tuple = (12, 10)) -> None:
    """
    Visualize the mutual information analysis of a joint distribution including
    PMI heatmap, mutual information value, and related entropies.
    
    Args:
        joint_distribution: 2D array representing joint probability distribution P(X,Y)
        x_labels: Labels for the X variable values
        y_labels: Labels for the Y variable values
        title: Title for the figure
        figsize: Figure size as (width, height)
    """
    joint_distribution = np.array(joint_distribution)
    
    # Generate default labels if not provided
    if x_labels is None:
        x_labels = [f'X={i}' for i in range(joint_distribution.shape[0])]
    if y_labels is None:
        y_labels = [f'Y={j}' for j in range(joint_distribution.shape[1])]
    
    # Compute PMI
    pmi = pointwise_mutual_information(joint_distribution)
    
    # Compute mutual information and entropies
    mi = mutual_information(joint_distribution)
    h_x_given_y, h_y_given_x = conditional_entropy(joint_distribution)
    nmi = normalized_mutual_information(joint_distribution)
    
    # Compute marginals
    p_x = np.sum(joint_distribution, axis=1)
    p_y = np.sum(joint_distribution, axis=0)
    h_x = -np.sum(p_x * np.log2(np.clip(p_x, 1e-10, 1.0)))
    h_y = -np.sum(p_y * np.log2(np.clip(p_y, 1e-10, 1.0)))
    
    # Create the figure with a 2x2 grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1])
    
    # Distribution of Y (top-right)
    ax_y = fig.add_subplot(gs[0, 0])
    ax_y.bar(range(len(p_y)), p_y)
    ax_y.set_xticks(range(len(y_labels)))
    ax_y.set_xticklabels(y_labels)
    ax_y.set_title(f'P(Y) - H(Y)={h_y:.3f} bits')
    
    # Information diagram (top-left)
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')  # Turn off the axis
    
    # Add information theory metrics as text
    info_text = f"Mutual Information I(X;Y): {mi:.3f} bits\n" \
                f"Normalized MI: {nmi:.3f}\n" \
                f"H(X): {h_x:.3f} bits\n" \
                f"H(Y): {h_y:.3f} bits\n" \
                f"H(X|Y): {h_x_given_y:.3f} bits\n" \
                f"H(Y|X): {h_y_given_x:.3f} bits"
    ax_info.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    
    # Distribution of X (bottom-left)
    ax_x = fig.add_subplot(gs[1, 1])
    ax_x.barh(range(len(p_x)), p_x)
    ax_x.set_yticks(range(len(x_labels)))
    ax_x.set_yticklabels(x_labels)
    ax_x.set_title(f'P(X) - H(X)={h_x:.3f} bits')
    
    # PMI heatmap (main plot - bottom-right)
    ax_main = fig.add_subplot(gs[1, 0])
    
    # Create heatmap
    im = ax_main.imshow(pmi, cmap='coolwarm', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_main)
    cbar.set_label('PMI (bits)')
    
    # Add joint probability as text in each cell
    for i in range(joint_distribution.shape[0]):
        for j in range(joint_distribution.shape[1]):
            if not np.isinf(pmi[i, j]):
                text = ax_main.text(j, i, f'{joint_distribution[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
    
    # Set up the axes with labels
    ax_main.set_xticks(range(len(y_labels)))
    ax_main.set_yticks(range(len(x_labels)))
    ax_main.set_xticklabels(y_labels)
    ax_main.set_yticklabels(x_labels)
    ax_main.set_xlabel('Y Values')
    ax_main.set_ylabel('X Values')
    ax_main.set_title('Pointwise Mutual Information (PMI)')
    
    # Overall title
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example 1: Independent variables
    print("Example 1: Independent variables")
    independent = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ])
    mi = mutual_information(independent)
    print(f"Mutual Information: {mi:.4f} bits (should be close to 0)")
    
    # Example 2: Perfectly correlated variables
    print("\nExample 2: Perfectly correlated variables")
    correlated = np.array([
        [0.5, 0],
        [0, 0.5]
    ])
    mi = mutual_information(correlated)
    print(f"Mutual Information: {mi:.4f} bits (should be 1.0 for this distribution)")
    
    # Example 3: Partially correlated variables
    print("\nExample 3: Partially correlated variables")
    partial = np.array([
        [0.4, 0.1],
        [0.1, 0.4]
    ])
    mi = mutual_information(partial)
    nmi = normalized_mutual_information(partial)
    h_x_given_y, h_y_given_x = conditional_entropy(partial)
    print(f"Mutual Information: {mi:.4f} bits")
    print(f"Normalized MI: {nmi:.4f}")
    print(f"Conditional Entropy H(X|Y): {h_x_given_y:.4f} bits")
    print(f"Conditional Entropy H(Y|X): {h_y_given_x:.4f} bits")
    
    # Example 4: Language model context visualization
    print("\nExample 4: Language model context example (simplified)")
    # Simulate a joint distribution of words and contexts
    words = ["the", "dog", "runs"]
    contexts = ["<s>", "fast", "."]
    
    # Define a joint probability distribution (simplified example)
    lm_joint = np.array([
        [0.20, 0.01, 0.15],  # "the" in different contexts
        [0.15, 0.10, 0.02],  # "dog" in different contexts
        [0.01, 0.25, 0.11]   # "runs" in different contexts
    ])
    
    # Plot the mutual information analysis
    plot_mutual_information(lm_joint, words, contexts, 
                          "Word-Context Mutual Information Example")