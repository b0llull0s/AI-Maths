import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def frobenius_norm(matrix):
    """
    Calculate the Frobenius norm of a matrix.
    
    Parameters:
        matrix (array-like): Input matrix
        
    Returns:
        float: Frobenius norm of the matrix
    """
    return np.sqrt(np.sum(np.square(matrix)))

def manual_frobenius_norm(matrix):
    """
    Calculate the Frobenius norm of a matrix without using numpy's functions.
    
    Parameters:
        matrix (array-like): Input matrix
        
    Returns:
        float: Frobenius norm of the matrix
    """
    squared_sum = 0
    for row in matrix:
        for element in row:
            squared_sum += element ** 2
    return squared_sum ** 0.5

def normalize_matrix(matrix):
    """
    Normalize a matrix using the Frobenius norm.
    
    Parameters:
        matrix (array-like): Input matrix
        
    Returns:
        array: Normalized matrix with Frobenius norm = 1
    """
    norm = frobenius_norm(matrix)
    if norm == 0:
        return matrix
    return matrix / norm

def visualize_matrix_norms(matrices):
    """
    Visualize the Frobenius norms of matrices.
    
    Parameters:
        matrices (list): List of matrices with their descriptions
    """
    names = [desc for _, desc in matrices]
    norms = [frobenius_norm(mat) for mat, _ in matrices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, norms, color='skyblue')
    plt.xlabel('Matrix Type')
    plt.ylabel('Frobenius Norm')
    plt.title('Comparison of Frobenius Norms for Different Matrices')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def visualize_matrix_heatmap(matrix, title):
    """
    Visualize a matrix as a heatmap.
    
    Parameters:
        matrix (array-like): Matrix to visualize
        title (str): Title for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title(title)
    
    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f'{matrix[i, j]:.2f}', 
                     ha='center', va='center', 
                     color='white' if matrix[i, j] > 0.5 else 'black')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example matrices
    matrices = [
        (np.array([[1, 2], [3, 4]]), "2x2 Matrix"),
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), "Identity Matrix"),
        (np.random.rand(3, 3), "Random 3x3 Matrix"),
        (np.array([[0.1, 0.2], [0.3, 0.4]]), "Small Values Matrix"),
        (np.array([[10, 20], [30, 40]]), "Large Values Matrix")
    ]
    
    print("Frobenius Norm Examples:")
    print("-----------------------")
    
    for i, (matrix, desc) in enumerate(matrices):
        numpy_norm = np.linalg.norm(matrix, 'fro')  # NumPy's built-in Frobenius norm
        our_norm = frobenius_norm(matrix)           # Our implementation
        manual_norm = manual_frobenius_norm(matrix) # Manual implementation
        
        print(f"Matrix {i+1} ({desc}):")
        print(matrix)
        print(f"  NumPy Frobenius norm:   {numpy_norm:.6f}")
        print(f"  Our Frobenius norm:     {our_norm:.6f}")
        print(f"  Manual Frobenius norm:  {manual_norm:.6f}")
        
        # Normalize the matrix
        normalized = normalize_matrix(matrix)
        normalized_norm = frobenius_norm(normalized)
        print(f"  Normalized matrix Frobenius norm: {normalized_norm:.6f}")
        print()
    
    # Visualize the matrix norms
    print("Visualizing Frobenius norms of different matrices...")
    visualize_matrix_norms(matrices)
    
    # Visualize matrix heatmaps
    print("Visualizing matrix heatmaps...")
    for matrix, desc in matrices[:3]:  # Show first 3 matrices
        visualize_matrix_heatmap(matrix, f"Heatmap of {desc}")
    
    # Application in AI/ML: Weight regularization for neural networks
    print("\nAI/ML Application: Weight Matrix Regularization Example")
    print("-----------------------------------------")
    
    # Simulated weight matrices for a neural network layer
    np.random.seed(42)
    weight_matrix1 = np.random.normal(0, 0.1, (5, 3))  # Well-initialized weights
    weight_matrix2 = np.random.normal(0, 1.0, (5, 3))  # Poorly-initialized weights
    
    # Calculate Frobenius norms
    norm1 = frobenius_norm(weight_matrix1)
    norm2 = frobenius_norm(weight_matrix2)
    
    print(f"Well-initialized weight matrix:")
    print(weight_matrix1)
    print(f"Frobenius norm: {norm1:.6f}")
    print()
    
    print(f"Poorly-initialized weight matrix:")
    print(weight_matrix2)
    print(f"Frobenius norm: {norm2:.6f}")
    print()
    
    # Simulate weight regularization
    lambda_reg = 0.01
    l2_reg_loss1 = lambda_reg * (norm1 ** 2)
    l2_reg_loss2 = lambda_reg * (norm2 ** 2)
    
    print(f"L2 regularization loss for well-initialized weights: {l2_reg_loss1:.6f}")
    print(f"L2 regularization loss for poorly-initialized weights: {l2_reg_loss2:.6f}")
    print("This shows how the Frobenius norm penalizes larger weight matrices more heavily.")
    
    # Application in LLM: Attention matrix analysis
    print("\nLLM Application: Attention Matrix Analysis")
    print("-----------------------------------------")
    
    # Simulated attention matrices for different heads in a transformer
    attn1 = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])  # Strong diagonal attention
    
    attn2 = np.array([
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]
    ])  # Diffuse attention
    
    attn3 = np.array([
        [0.1, 0.1, 0.8],
        [0.1, 0.1, 0.8],
        [0.1, 0.1, 0.8]
    ])  # Focused attention on the last token
    
    attention_matrices = [
        (attn1, "Strong Diagonal Attention"),
        (attn2, "Diffuse Attention"),
        (attn3, "Focused Attention")
    ]
    
    print("Analyzing different attention patterns using Frobenius norm:")
    for attn, desc in attention_matrices:
        norm = frobenius_norm(attn)
        print(f"{desc}:")
        print(attn)
        print(f"Frobenius norm: {norm:.6f}")
        
        # Display the attention pattern
        visualize_matrix_heatmap(attn, f"Attention Pattern: {desc}")

    # Distance between matrices using Frobenius norm
    print("\nMatrix Distance Example:")
    print("--------------------------")
    print("Distance between attention patterns using Frobenius norm:")
    
    # Create a dataframe to store distances
    attn_names = [desc for _, desc in attention_matrices]
    distances = np.zeros((len(attention_matrices), len(attention_matrices)))
    
    for i, (mat1, _) in enumerate(attention_matrices):
        for j, (mat2, _) in enumerate(attention_matrices):
            distances[i, j] = frobenius_norm(mat1 - mat2)
    
    distance_df = pd.DataFrame(distances, index=attn_names, columns=attn_names)
    print(distance_df)
    print("\nThis shows how Frobenius norm can be used to measure similarity between matrices.")