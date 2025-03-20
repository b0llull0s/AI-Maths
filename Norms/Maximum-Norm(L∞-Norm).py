import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def max_norm(vector):
    """
    Calculate the L∞ norm (Maximum norm) of a vector.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        float: L∞ norm of the vector
    """
    return np.max(np.abs(vector))

def manual_max_norm(vector):
    """
    Calculate the L∞ norm (Maximum norm) without using numpy's functions.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        float: L∞ norm of the vector
    """
    max_abs_value = 0
    for element in vector:
        abs_value = abs(element)
        if abs_value > max_abs_value:
            max_abs_value = abs_value
    return max_abs_value

def normalize_vector_max(vector):
    """
    Normalize a vector using the L∞ norm.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        array: Normalized vector with L∞ norm = 1
    """
    norm = max_norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def visualize_max_norm_ball(dim=2):
    """
    Visualize vectors with the same L∞ norm (creating a square in 2D or cube in 3D).
    
    Parameters:
        dim (int): Dimension (2 or 3)
    """
    if dim == 2:
        # Create points on a unit L∞ ball (square)
        x = np.array([-1, -1, 1, 1, -1])
        y = np.array([-1, 1, 1, -1, -1])
        
        # Plot the unit L∞ ball
        plt.figure(figsize=(8, 8))
        plt.plot(x, y, 'b-')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Vectors with L∞ Norm = 1 (Square)')
        
        # Plot some example vectors
        vectors = [(1, 0), (0, 1), (0.7, 0.7), (-0.3, 0.9)]
        colors = ['r', 'g', 'purple', 'orange']
        
        for vec, color in zip(vectors, colors):
            # Scale vectors to have L∞ norm = 1
            scaled_vec = vec / max_norm(vec)
            plt.arrow(0, 0, scaled_vec[0], scaled_vec[1], head_width=0.05, head_length=0.08, fc=color, ec=color)
            plt.text(scaled_vec[0]*1.1, scaled_vec[1]*1.1, f"||v||_∞ = {max_norm(scaled_vec):.4f}", color=color)
            
        plt.show()
    
    elif dim == 3:
        # Create a unit L∞ ball (cube)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the cube
        vertices = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertices.append((x, y, z))
        
        # Define edges of the cube
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),  # Bottom face
            (4, 5), (4, 6), (5, 7), (6, 7),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Plot edges
        for edge in edges:
            x = [vertices[edge[0]][0], vertices[edge[1]][0]]
            y = [vertices[edge[0]][1], vertices[edge[1]][1]]
            z = [vertices[edge[0]][2], vertices[edge[1]][2]]
            ax.plot(x, y, z, 'b-', alpha=0.5)
            
        # Plot some example vectors
        vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), 
                  (0.8, 0.8, 0.8), (-0.7, 0.6, 0.5)]
        colors = ['r', 'g', 'm', 'orange', 'purple']
        
        for vec, color in zip(vectors, colors):
            # Scale vectors to have L∞ norm = 1
            scaled_vec = vec / max_norm(vec)
            ax.quiver(0, 0, 0, scaled_vec[0], scaled_vec[1], scaled_vec[2], color=color, arrow_length_ratio=0.1)
            ax.text(scaled_vec[0]*1.1, scaled_vec[1]*1.1, scaled_vec[2]*1.1, f"||v||_∞ = {max_norm(scaled_vec):.4f}", color=color)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Vectors with L∞ Norm = 1 (Cube)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example vectors
    vectors = [
        np.array([3, 4]),  # L∞ norm should be 4
        np.array([1, 1, 1, 1]),  # L∞ norm should be 1
        np.array([-2, 3, -1, 5, 2])  # L∞ norm should be 5
    ]
    
    print("L∞ Norm (Maximum Norm) Examples:")
    print("-----------------------")
    
    for i, vec in enumerate(vectors):
        numpy_norm = np.linalg.norm(vec, ord=np.inf)  # NumPy's built-in L∞ norm
        our_norm = max_norm(vec)    # Our implementation
        manual_norm = manual_max_norm(vec)  # Manual implementation
        
        print(f"Vector {i+1}: {vec}")
        print(f"  NumPy L∞ norm:   {numpy_norm:.6f}")
        print(f"  Our L∞ norm:     {our_norm:.6f}")
        print(f"  Manual L∞ norm:  {manual_norm:.6f}")
        
        # Normalize the vector using L∞ norm
        normalized = normalize_vector_max(vec)
        normalized_norm = max_norm(normalized)
        print(f"  L∞-Normalized vector: {normalized}")
        print(f"  L∞ norm of normalized vector: {normalized_norm:.6f}")
        print()
    
    # Visualize the unit L∞ ball in 2D and 3D
    print("Visualizing unit L∞ ball in 2D and 3D...")
    visualize_max_norm_ball(2)
    visualize_max_norm_ball(3)
    
    # Application in AI/ML: Gradient clipping example
    print("\nAI/ML Application: Gradient Clipping Example")
    print("-----------------------------------------")
    
    # Simulated gradients
    gradients = np.array([2.5, -3.8, 0.7, 10.2, -1.5])
    clip_value = 4.0
    
    max_grad_norm = max_norm(gradients)
    
    print(f"Original gradients: {gradients}")
    print(f"Maximum absolute gradient: {max_grad_norm:.6f}")
    
    # Apply gradient clipping
    if max_grad_norm > clip_value:
        clipped_gradients = gradients * (clip_value / max_grad_norm)
    else:
        clipped_gradients = gradients
    
    print(f"Clipping threshold: {clip_value}")
    print(f"Clipped gradients: {clipped_gradients}")
    print(f"Maximum absolute clipped gradient: {max_norm(clipped_gradients):.6f}")
    print("This demonstrates how L∞ norm can be used to prevent exploding gradients.")
    
    # Maximum difference example (in LLM token embeddings)
    print("\nMaximum Difference Example in LLM Token Embedding:")
    print("------------------------------------------")
    token_embedding1 = np.array([0.2, 0.5, -0.7, 0.3, -0.1])
    token_embedding2 = np.array([0.3, 0.4, 0.1, 0.2, -0.3])
    
    max_diff = max_norm(token_embedding1 - token_embedding2)
    diff_vector = token_embedding1 - token_embedding2
    max_diff_index = np.argmax(np.abs(diff_vector))
    
    print(f"Token Embedding 1: {token_embedding1}")
    print(f"Token Embedding 2: {token_embedding2}")
    print(f"Difference Vector: {diff_vector}")
    print(f"Maximum Absolute Difference: {max_diff:.6f}")
    print(f"Dimension with Maximum Difference: {max_diff_index} (value: {diff_vector[max_diff_index]:.6f})")
    print("This can be useful for identifying the most significant difference between embeddings.")