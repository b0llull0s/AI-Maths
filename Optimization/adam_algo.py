import numpy as np

def adam_optimizer(gradient, start, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                   epsilon=1e-8, n_iterations=1000, tolerance=1e-6):
    """
    Adam optimization algorithm.
    
    :param gradient: The gradient function of the objective function.
    :param start: The starting point for the optimization.
    :param learning_rate: The learning rate (step size).
    :param beta1: Exponential decay rate for first moment estimates.
    :param beta2: Exponential decay rate for second moment estimates.
    :param epsilon: Small constant to prevent division by zero.
    :param n_iterations: The maximum number of iterations.
    :param tolerance: The tolerance for stopping early if the change is small.
    :return: The optimized value and history of values.
    """
    x = start
    m = np.zeros_like(x)  # First moment estimate
    v = np.zeros_like(x)  # Second moment estimate
    history = [x.copy()]
    
    for t in range(1, n_iterations + 1):
        grad = gradient(x)
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {t} iterations.")
            break
            
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * np.square(grad)
        
        # Compute bias-corrected moment estimates
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x.copy())
        
    return x, np.array(history)

# Example usage
if __name__ == "__main__":
    # Example with a more complex function
    def rosenbrock(x):
        """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def gradient_rosenbrock(x):
        """Gradient of Rosenbrock function"""
        dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dy = 200 * (x[1] - x[0]**2)
        return np.array([dx, dy])
    
    # Starting from a point away from the minimum [1,1]
    start_point = np.array([-1.0, 2.0])
    
    # Run Adam optimizer
    optimized_point, history = adam_optimizer(gradient_rosenbrock, start_point,
                                              learning_rate=0.01, n_iterations=2000)
    
    print(f"Starting point: {start_point}")
    print(f"Optimized point: {optimized_point}")
    print(f"Optimized value: {rosenbrock(optimized_point)}")
    print(f"Expected minimum at [1,1], value: {rosenbrock(np.array([1.0, 1.0]))}")
    print(f"Number of iterations: {len(history)}")