import numpy as np

def line_search(func, x, direction, max_alpha=1.0, c1=1e-4, c2=0.9, max_iterations=20):
    """
    Backtracking line search with Wolfe conditions.
    
    :param func: Function to minimize (returns function value and gradient).
    :param x: Current position.
    :param direction: Search direction.
    :param max_alpha: Maximum step size.
    :param c1: Parameter for Armijo condition (sufficient decrease).
    :param c2: Parameter for curvature condition.
    :param max_iterations: Maximum number of line search iterations.
    :return: Step size alpha.
    """
    f0, grad0 = func(x)
    directional_derivative = np.dot(grad0, direction)
    
    if directional_derivative >= 0:
        return 0.0
    
    alpha = max_alpha
    
    for i in range(max_iterations):
        f_new, grad_new = func(x + alpha * direction)
        
        if f_new <= f0 + c1 * alpha * directional_derivative:
            if np.dot(grad_new, direction) >= c2 * directional_derivative:
                return alpha
        
        alpha *= 0.5
    
    return alpha