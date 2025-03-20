import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def l1_norm(vector):
    """
    Calculate the L1 norm (Manhattan distance) of a vector.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        float: L1 norm of the vector
    """
    return np.sum(np.abs(vector))

def manual_l1_norm(vector):
    """
    Calculate the L1 norm (Manhattan distance) without using numpy's functions.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        float: L1 norm of the vector
    """
    absolute_sum = 0
    for element in vector:
        absolute_sum += abs(element)
    return absolute_sum

def normalize_vector_l1(vector):
    """
    Normalize a vector using the L1 norm.
    
    Parameters:
        vector (array-like): Input vector
        
    Returns:
        array: Normalized vector with L1 norm = 1
    """
    norm = l1_norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def visualize_l1_ball(dim=2):
    """
    Visualize vectors with the same L1 norm (creating a diamond in 2D or octahedron in 3D).
    
    Parameters:
        dim (int): Dimension (2 or 3)
    """
    if dim == 2:
        # Create points on a unit L1 ball (diamond)
        x = np.linspace(-1, 1, 100)
        y_pos = 1 - np.abs(x)
        y_neg = -1 + np.abs(x)
        
        # Plot the unit L1 ball
        plt.figure(figsize=(8, 8))
        plt.plot(x, y_pos, 'b-')
        plt.plot(x, y_neg, 'b-')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Vectors with L1 Norm = 1 (Diamond)')
        
        # Plot some example vectors
        vectors = [(1, 0), (0, 1), (0.5, 0.5), (-0.3, 0.7)]
        colors = ['r', 'g', 'purple', 'orange']
        
        for vec, color in zip(vectors, colors):
            # Scale vectors to have L1 norm = 1
            scaled_vec = vec / l1_norm(vec)
            plt.arrow(0, 0, scaled_vec[0], scaled_vec[1], head_width=0.05, head_length=0.08, fc=color, ec=color)
            plt.text(scaled_vec[0]*1.1, scaled_vec[1]*1.1, f"||v||_1 = {l1_norm(scaled_vec):.4f}", color=color)
            
        plt.show()
    
    elif dim == 3:
        # Create a unit L1 ball (octahedron)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the octahedron vertices
        vertices = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        
        # Plot faces
        faces = [
            [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
            [1, 2, 4], [1, 4, 3], [1, 3, 5], [1, 5, 2]
        ]
        
        for face in faces:
            x = [vertices[i][0] for i in face]
            y = [vertices[i][1] for i in face]
            z = [vertices[i][2] for i in face]
            # Close the triangle
            x.append(x[0])
            y.append(y[0])
            z.append(z[0])
            ax.plot(x, y, z, 'b-', alpha=0.5)
            
        # Plot some example vectors
        vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), 
                  (0.333, 0.333, 0.333), (-0.2, 0.3, 0.5)]
        colors = ['r', 'g', 'm', 'orange', 'purple']
        
        for vec, color in zip(vectors, colors):
            # Scale vectors to have L1 norm = 1
            scaled_vec = vec / l1_norm(vec)
            ax.quiver(0, 0, 0, scaled_vec[0], scaled_vec[1], scaled_vec[2], color=color, arrow_length_ratio=0.1)
            ax.text(scaled_vec[0]*1.1, scaled_vec[1]*1.1, scaled_vec[2]*1.1, f"||v||_1 = {l1_norm(scaled_vec):.4f}", color=color)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Vectors with L1 Norm = 1 (Octahedron)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example vectors
    vectors = [
        np.array([3, 4]),  # L1 norm should be 7
        np.array([1, 1, 1, 1]),  # L1 norm should be 4
        np.array([-2, 3, -1, 5, 2])  # mixed positive/negative values
    ]
    
    print("L1 Norm (Manhattan Distance) Examples:")
    print("-----------------------")
    
    for i, vec in enumerate(vectors):
        numpy_norm = np.linalg.norm(vec, ord=1)  # NumPy's built-in L1 norm
        our_norm = l1_norm(vec)    # Our implementation
        manual_norm = manual_l1_norm(vec)  # Manual implementation
        
        print(f"Vector {i+1}: {vec}")
        print(f"  NumPy L1 norm:   {numpy_norm:.6f}")
        print(f"  Our L1 norm:     {our_norm:.6f}")
        print(f"  Manual L1 norm:  {manual_norm:.6f}")
        
        # Normalize the vector using L1 norm
        normalized = normalize_vector_l1(vec)
        normalized_norm = l1_norm(normalized)
        print(f"  L1-Normalized vector: {normalized}")
        print(f"  L1 norm of normalized vector: {normalized_norm:.6f}")
        print()
    
    # Visualize the unit L1 ball in 2D and 3D
    print("Visualizing unit L1 ball in 2D and 3D...")
    visualize_l1_ball(2)
    visualize_l1_ball(3)
    
    # Application in AI/ML: Sparse feature selection example
    print("\nAI/ML Application: L1 Regularization (Lasso) Example")
    print("-----------------------------------------")
    
    # Simulated weights of a model
    weights = np.array([0.5, -0.3, 0.02, 0.7, -0.05])
    l1_penalty = 0.1  # Regularization strength
    
    # Calculate L1 regularization term
    l1_reg_term = l1_penalty * l1_norm(weights)
    
    print(f"Model weights: {weights}")
    print(f"L1 norm of weights: {l1_norm(weights):.6f}")
    print(f"L1 regularization term (added to loss): {l1_reg_term:.6f}")
    
    # Effect on sparsity
    small_threshold = 0.1
    original_zero_count = np.sum(np.abs(weights) < small_threshold)
    
    # Simulate the effect of L1 regularization (soft thresholding)
    updated_weights = np.sign(weights) * np.maximum(np.abs(weights) - l1_penalty, 0)
    updated_zero_count = np.sum(np.abs(updated_weights) < small_threshold)
    
    print(f"Number of near-zero weights (original): {original_zero_count}")
    print(f"Number of near-zero weights (after L1 regularization): {updated_zero_count}")
    print(f"Updated weights: {updated_weights}")
    print("This demonstrates how L1 regularization promotes sparsity by pushing small weights to zero.")
    
    # L1 distance example (in embedding space)
    print("\nL1 Distance Example in LLM Embedding Space:")
    print("------------------------------------------")
    embedding1 = np.array([0.2, 0.5, -0.1, 0.3, -0.4])
    embedding2 = np.array([0.3, 0.4, 0.1, 0.2, -0.3])
    
    l1_distance = l1_norm(embedding1 - embedding2)
    
    print(f"Embedding 1: {embedding1}")
    print(f"Embedding 2: {embedding2}")
    print(f"L1 Distance between embeddings: {l1_distance:.6f}")