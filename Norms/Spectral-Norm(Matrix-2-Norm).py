import numpy as np
import matplotlib.pyplot as plt
import time

def spectral_norm(matrix):
    """
    Calculate the spectral norm (matrix 2-norm) of a matrix.
    This is equal to the largest singular value of the matrix.
    
    Parameters:
        matrix (array-like): Input matrix
        
    Returns:
        float: Spectral norm of the matrix
    """
    # Calculate singular values and return the maximum
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return np.max(singular_values)

def power_iteration_spectral_norm(matrix, max_iter=100, tol=1e-10):
    """
    Calculate the spectral norm using the power iteration method.
    
    Parameters:
        matrix (array-like): Input matrix
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
        
    Returns:
        float: Spectral norm approximation
    """
    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape
    
    # Start with a random vector
    x = np.random.rand(n)
    
    # Normalize the vector
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        # Compute matrix-vector product
        mx = matrix.T @ (matrix @ x)
        
        # Compute the new vector
        x_new = mx / np.linalg.norm(mx)
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
    
    # Return the square root of the Rayleigh quotient
    return np.sqrt(x.T @ (matrix.T @ (matrix @ x)) / (x.T @ x))

def normalize_matrix_spectral(matrix):
    """
    Normalize a matrix using the spectral norm.
    
    Parameters:
        matrix (array-like): Input matrix
        
    Returns:
        array: Normalized matrix with spectral norm = 1
    """
    norm = spectral_norm(matrix)
    if norm == 0:
        return matrix
    return matrix / norm

def generate_ellipse_points(a, b, num_points=100):
    """
    Generate points on an ellipse with semi-major axis a and semi-minor axis b.
    
    Parameters:
        a (float): Semi-major axis
        b (float): Semi-minor axis
        num_points (int): Number of points to generate
        
    Returns:
        tuple: x and y coordinates of points on the ellipse
    """
    theta = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return x, y

def visualize_matrix_transformation(matrix):
    """
    Visualize how a 2x2 matrix transforms the unit circle.
    
    Parameters:
        matrix (array-like): 2x2 transformation matrix
    """
    # Check if matrix is 2x2
    if matrix.shape != (2, 2):
        print("Visualization only works for 2x2 matrices")
        return
    
    # Generate points on the unit circle
    circle_x, circle_y = generate_ellipse_points(1, 1)
    
    # Create points as column vectors and apply the transformation
    points = np.vstack((circle_x, circle_y))
    transformed_points = matrix @ points
    
    # Get the transformed x and y coordinates
    transformed_x = transformed_points[0, :]
    transformed_y = transformed_points[1, :]
    
    # Get singular values for the matrix
    u, s, vt = np.linalg.svd(matrix)
    
    # Plot the original unit circle and the transformed ellipse
    plt.figure(figsize=(12, 6))
    
    # Original circle
    plt.subplot(1, 2, 1)
    plt.plot(circle_x, circle_y, 'b-')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('Unit Circle')
    
    # Transformed ellipse
    plt.subplot(1, 2, 2)
    plt.plot(transformed_x, transformed_y, 'r-')
    
    # Draw the semi-major and semi-minor axes
    plt.arrow(0, 0, s[0]*u[0, 0], s[0]*u[1, 0], head_width=0.1, head_length=0.1, fc='green', ec='green')
    plt.arrow(0, 0, s[1]*u[0, 1], s[1]*u[1, 1], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    plt.grid(True)
    plt.axis('equal')
    max_val = max(np.max(np.abs(transformed_x)), np.max(np.abs(transformed_y)))
    plt.xlim(-max_val*1.2, max_val*1.2)
    plt.ylim(-max_val*1.2, max_val*1.2)
    plt.title(f'Transformed Ellipse\nSpectral Norm = {spectral_norm(matrix):.4f}')
    
    plt.tight_layout()
    plt.show()

def compare_norm_methods(matrices):
    """
    Compare SVD-based and power iteration methods for computing spectral norm.
    
    Parameters:
        matrices (list): List of matrices with their descriptions
    """
    results = []
    
    for matrix, desc in matrices:
        # Time SVD method
        start_time = time.time()
        svd_norm = spectral_norm(matrix)
        svd_time = time.time() - start_time
        
        # Time power iteration method
        start_time = time.time()
        power_norm = power_iteration_spectral_norm(matrix)
        power_time = time.time() - start_time
        
        results.append({
            'Matrix': desc,
            'SVD Norm': svd_norm,
            'Power Norm': power_norm,
            'Difference': abs(svd_norm - power_norm),
            'SVD Time (s)': svd_time,
            'Power Time (s)': power_time
        })
    
    # Print results as a table
    print("\nComparison of Spectral Norm Computation Methods:")
    print("-" * 80)
    print(f"{'Matrix':<20} {'SVD Norm':<10} {'Power Norm':<12} {'Difference':<12} {'SVD Time (s)':<12} {'Power Time (s)':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['Matrix']:<20} {result['SVD Norm']:<10.6f} {result['Power Norm']:<12.6f} "
              f"{result['Difference']:<12.6e} {result['SVD Time (s)']:<12.6f} {result['Power Time (s)']:<12.6f}")

if __name__ == "__main__":
    # Example matrices
    matrices = [
        (np.array([[1, 0], [0, 2]]), "Diagonal Matrix"),
        (np.array([[3, 1], [1, 3]]), "Symmetric Matrix"),
        (np.array([[0, 2], [-1, 0]]), "Skew-Symmetric Matrix"),
        (np.array([[1, 2], [3, 4]]), "General Matrix"),
        (np.array([[0.1, 0.2], [0.3, 0.4]]), "Small Values Matrix"),
        (np.random.rand(5, 5), "Random 5x5 Matrix"),
        (np.random.rand(10, 10), "Random 10x10 Matrix"),
        (np.random.rand(20, 20), "Random 20x20 Matrix")
    ]
    
    print("Spectral Norm Examples:")
    print("-----------------------")
    
    for i, (matrix, desc) in enumerate(matrices[:5]):  # First 5 matrices only
        numpy_norm = np.linalg.norm(matrix, ord=2)  # NumPy's built-in spectral norm
        our_norm = spectral_norm(matrix)           # Our SVD-based implementation
        power_norm = power_iteration_spectral_norm(matrix) # Power iteration method
        
        print(f"Matrix {i+1} ({desc}):")
        print(matrix)
        print(f"  NumPy spectral norm:        {numpy_norm:.6f}")
        print(f"  Our spectral norm (SVD):    {our_norm:.6f}")
        print(f"  Power iteration method:     {power_norm:.6f}")
        
        # Normalize the matrix
        normalized = normalize_matrix_spectral(matrix)
        normalized_norm = spectral_norm(normalized)
        print(f"  Normalized matrix spectral norm: {normalized_norm:.6f}")
        print()
        
        # For 2x2 matrices, visualize the transformation
        if matrix.shape == (2, 2):
            print(f"Visualizing transformation by {desc}...")
            visualize_matrix_transformation(matrix)
    
    # Compare norm computation methods
    print("\nComparing norm computation methods...")
    compare_norm_methods(matrices)
    
    # Application in AI/ML: Weight initialization
    print("\nAI/ML Application: Weight Initialization for Stable Training")
    print("-----------------------------------------")
    
    # Create matrices with different initializations
    np.random.seed(42)
    n_in, n_out = 100, 100
    
    # Standard normal initialization
    W_normal = np.random.normal(0, 1, (n_out, n_in))
    
    # Xavier/Glorot initialization
    W_xavier = np.random.normal(0, np.sqrt(2/(n_in + n_out)), (n_out, n_in))
    
    # He initialization
    W_he = np.random.normal(0, np.sqrt(2/n_in), (n_out, n_in))
    
    # Orthogonal initialization
    X = np.random.normal(0, 1, (n_out, n_in))
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    W_ortho = u @ vt
    
    # Compare spectral norms
    init_matrices = [
        (W_normal, "Standard Normal"),
        (W_xavier, "Xavier/Glorot"),
        (W_he, "He"),
        (W_ortho, "Orthogonal")
    ]
    
    print("Comparing spectral norms of different weight initializations:")
    for matrix, desc in init_matrices:
        norm = spectral_norm(matrix)
        print(f"{desc} initialization:")
        print(f"  Shape: {matrix.shape}")
        print(f"  Spectral norm: {norm:.6f}")
        print()
    
    # Application in LLM training: Lipschitz constrained transformations
    print("\nLLM Application: Lipschitz Constrained Transformations")
    print("-----------------------------------------")
    
    # Create a weight matrix
    W = np.random.normal(0, 1, (50, 50))
    
    # Original spectral norm
    orig_norm = spectral_norm(W)
    
    # Target Lipschitz constant
    target_lipschitz = 1.0
    
    # Scale the matrix to have the desired Lipschitz constant
    W_constrained = W * (target_lipschitz / orig_norm)
    
    print(f"Original matrix spectral norm: {orig_norm:.6f}")
    print(f"Target Lipschitz constant: {target_lipschitz}")
    print(f"Constrained matrix spectral norm: {spectral_norm(W_constrained):.6f}")
    print("This demonstrates how to constrain the Lipschitz constant of a transformation,")
    print("which is important for stable training and adversarial robustness in neural networks.")