import numpy as np

def gradient_descent_with_momentum(gradient, start, learning_rate=0.01, momentum=0.9, 
                                   n_iterations=1000, tolerance=1e-6):
    """
    Gradient descent with momentum to accelerate convergence.
    
    :param gradient: The gradient function of the objective function.
    :param start: The starting point for the optimization.
    :param learning_rate: The learning rate (step size).
    :param momentum: The momentum coefficient (between 0 and 1).
    :param n_iterations: The maximum number of iterations.
    :param tolerance: The tolerance for stopping early if the change is small.
    :return: The optimized value and history of values.
    """
    x = start
    velocity = np.zeros_like(x)
    history = [x.copy()]
    
    for i in range(n_iterations):
        grad = gradient(x)
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i} iterations.")
            break
            
        # Update with momentum
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity
        history.append(x.copy())
        
    return x, np.array(history)

def adaptive_gradient_descent(gradient, obj_func, start, learning_rate=0.1, 
                              decay=0.95, n_iterations=1000, tolerance=1e-6):
    """
    Gradient descent with adaptive learning rate.
    
    :param gradient: The gradient function of the objective function.
    :param obj_func: The objective function to minimize.
    :param start: The starting point for the optimization.
    :param learning_rate: The initial learning rate.
    :param decay: The decay factor for adapting the learning rate.
    :param n_iterations: The maximum number of iterations.
    :param tolerance: The tolerance for stopping early.
    :return: The optimized value and history of values.
    """
    x = start
    history = [x.copy()]
    current_value = obj_func(x)
    
    for i in range(n_iterations):
        grad = gradient(x)
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged based on gradient after {i} iterations.")
            break
            
        # Try the step
        new_x = x - learning_rate * grad
        new_value = obj_func(new_x)
        
        # If we didn't improve, reduce learning rate and try again
        while new_value > current_value and learning_rate > 1e-10:
            learning_rate *= decay
            new_x = x - learning_rate * grad
            new_value = obj_func(new_x)
            
        if abs(new_value - current_value) < tolerance:
            print(f"Converged based on function value after {i} iterations.")
            break
            
        x = new_x
        current_value = new_value
        history.append(x.copy())
        
    return x, np.array(history)

# Example usage
if __name__ == "__main__":
    # Define a simple function: f(x) = x^2 + 5x + 4
    def f(x):
        return x**2 + 5*x + 4
    
    # Define its gradient: f'(x) = 2x + 5
    def grad_f(x):
        return 2*x + 5
    
    # Compare methods
    start = 10.0
    
    # Standard gradient descent
    from gradient_descent import gradient_descent
    result_standard = gradient_descent(grad_f, start, 0.1, 100)
    
    # With momentum
    result_momentum, history_momentum = gradient_descent_with_momentum(grad_f, start)
    
    # With adaptive learning rate
    result_adaptive, history_adaptive = adaptive_gradient_descent(grad_f, f, start)
    
    print(f"Standard GD result: {result_standard}, value: {f(result_standard)}")
    print(f"Momentum GD result: {result_momentum}, value: {f(result_momentum)}")
    print(f"Adaptive GD result: {result_adaptive}, value: {f(result_adaptive)}")