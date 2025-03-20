"""
Visualization utilities for information theory metrics in LLM analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional


def plot_distribution_comparison(p: Union[List[float], np.ndarray], 
                                q: Union[List[float], np.ndarray],
                                labels: Optional[List[str]] = None,
                                title: str = "Probability Distribution Comparison",
                                figsize: tuple = (10, 6),
                                show_metrics: bool = True) -> None:
    """
    Plot two probability distributions side by side for visual comparison.
    
    Args:
        p: First probability distribution.
        q: Second probability distribution.
        labels: Optional labels for the x-axis.
        title: Plot title.
        figsize: Figure size as (width, height).
        show_metrics: Whether to show information theory metrics in the plot.
    """
    p = np.array(p)
    q = np.array(q)
    
    # Normalize if not already valid probability distributions
    if not np.isclose(np.sum(p), 1.0):
        p = p / np.sum(p)
    if not np.isclose(np.sum(q), 1.0):
        q = q / np.sum(q)
        
    x = np.arange(len(p))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    bar1 = ax.bar(x - width/2, p, width, label='Distribution P')
    bar2 = ax.bar(x + width/2, q, width, label='Distribution Q')
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Probability')
    ax.set_title(title)
    
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f'Item {i+1}' for i in range(len(p))])
        
    ax.legend()
    
    # Calculate information theoretic measures if requested
    if show_metrics:
        try:
            from kl_divergence import kl_divergence, jensen_shannon_divergence
            
            kl = kl_divergence(p, q)
            js = jensen_shannon_divergence(p, q)
            
            ax.annotate(f'KL(P||Q): {kl:.4f}\nJS(P,Q): {js:.4f}', 
                      xy=(0.05, 0.95), xycoords='axes fraction',
                      verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        except ImportError:
            print("Note: Place kl_divergence.py in the same directory to show metrics in the plot")
    
    fig.tight_layout()
    plt.show()


def plot_entropy_comparison(distributions: List[Union[List[float], np.ndarray]],
                          labels: List[str],
                          figsize: tuple = (10, 6)) -> None:
    """
    Plot entropy values for multiple distributions for comparison.
    
    Args:
        distributions: List of probability distributions.
        labels: Names for each distribution.
        figsize: Figure size as (width, height).
    """
    try:
        from entropy import entropy
        
        entropies = [entropy(dist) for dist in distributions]
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(labels, entropies)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_ylabel('Entropy (bits)')
        ax.set_title('Entropy Comparison Across Distributions')
        fig.tight_layout()
        plt.show()
        
    except ImportError:
        print("Note: Place entropy.py in the same directory to use this function")


# Example usage
if __name__ == "__main__":
    # Define probability distributions
    p = [0.5, 0.3, 0.2]  # First distribution
    q = [0.4, 0.4, 0.2]  # Second distribution
    r = [0.33, 0.33, 0.34]  # Nearly uniform distribution
    s = [0.9, 0.05, 0.05]  # Highly skewed distribution
    
    # Visualize two distributions
    plot_distribution_comparison(
        p, q, 
        labels=['A', 'B', 'C'],
        title="Comparison of Distributions P and Q"
    )
    
    # Compare entropy across multiple distributions
    try:
        plot_entropy_comparison(
            [p, q, r, s],
            labels=['P', 'Q', 'Uniform', 'Skewed']
        )
    except NameError:
        print("Entropy comparison requires entropy.py in the same directory")