import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euclidean_norm(vector):
    """
    Calculate the Euclidean norm (L2 norm) of a vector.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        float: Euclidean norm of the vector
    """
    return np.sqrt(np.sum(np.square(vector)))

def manual_euclidean_norm(vector):
    """
    Calculate the Euclidean norm (L2 norm) without using numpy's functions.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        float: Euclidean norm of the vector
    """
    squared_sum = 0
    for element in vector:
        squared_sum += element ** 2
    return squared_sum ** 0.5

def normalize_vector(vector):
    """
    Normalize a vector to unit length using the Euclidean norm.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        array: Normalized vector with unit length
    """
    norm = euclidean_norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def visualize_norm(dim=2):
    """
    Visualize vectors with the same Euclidean norm (creating a circle in 2D or sphere in 3D).
    
    Parameters:
        dim (int): Dimension (2 or 3)
    """
    if dim == 2:
        # Create points on a unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Plot the unit circle
        plt.figure(figsize=(8, 8))
        plt.plot(x, y, 'b-')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Vectors with Euclidean Norm = 1 (Unit Circle)')
        
        # Plot some example vectors
        vectors = [(1, 0), (0, 1), (0.7071, 0.7071), (-0.6, 0.8)]
        colors = ['r', 'g', 'purple', 'orange']
        
        for vec, color in zip(vectors, colors):
            plt.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.08, fc=color, ec=color)
            plt.text(vec[0]*1.1, vec[1]*1.1, f"||v|| = {euclidean_norm(vec):.4f}", color=color)
            
        plt.show()
    
    elif dim == 3:
        # Create points on a unit sphere
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot the unit sphere
        ax.plot_surface(x, y, z, color='b', alpha=0.2)
        
        # Plot some example vectors
        vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), 
                  (0.577, 0.577, 0.577), (-0.5, 0.5, 0.7071)]
        colors = ['r', 'g', 'm', 'orange', 'purple']
        
        for vec, color in zip(vectors, colors):
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, arrow_length_ratio=0.1)
            ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, f"||v|| = {euclidean_norm(vec):.4f}", color=color)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Vectors with Euclidean Norm = 1 (Unit Sphere)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example vectors
    vectors = [
        np.array([3, 4]),  # 3-4-5 triangle, norm should be 5
        np.array([1, 1, 1, 1]),  # norm should be 2
        np.array([-2, 3, -1, 5, 2])  # mixed positive/negative values
    ]
    
    print("Euclidean Norm Examples:")
    print("-----------------------")
    
    for i, vec in enumerate(vectors):
        numpy_norm = np.linalg.norm(vec)  # NumPy's built-in norm
        our_norm = euclidean_norm(vec)    # Our implementation
        manual_norm = manual_euclidean_norm(vec)  # Manual implementation
        
        print(f"Vector {i+1}: {vec}")
        print(f"  NumPy norm:   {numpy_norm:.6f}")
        print(f"  Our norm:     {our_norm:.6f}")
        print(f"  Manual norm:  {manual_norm:.6f}")
        
        # Normalize the vector
        normalized = normalize_vector(vec)
        normalized_norm = euclidean_norm(normalized)
        print(f"  Normalized vector: {normalized}")
        print(f"  Norm of normalized vector: {normalized_norm:.6f}")
        print()
    
    # Visualize the unit circle (2D) and unit sphere (3D)
    print("Visualizing unit circle (2D) and unit sphere (3D)...")
    visualize_norm(2)
    visualize_norm(3)
    
    # Application in AI/ML: Weight regularization example
    print("\nAI/ML Application: L2 Regularization Example")
    print("-----------------------------------------")
    
    # Simulated weights of a model
    weights = np.array([0.5, -0.3, 1, 0.7, -0.2])
    l2_penalty = 0.01  # Regularization strength
    
    # Calculate L2 regularization term
    l2_reg_term = l2_penalty * (euclidean_norm(weights) ** 2)
    
    print(f"Model weights: {weights}")
    print(f"L2 norm of weights: {euclidean_norm(weights):.6f}")
    print(f"L2 regularization term (added to loss): {l2_reg_term:.6f}")
    
    # Gradient of L2 regularization term w.r.t weights
    l2_grad = 2 * l2_penalty * weights
    
    print(f"Gradient of L2 regularization term: {l2_grad}")
    print("This gradient would be added to the gradient of the loss function during backpropagation.")