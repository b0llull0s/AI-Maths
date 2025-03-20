import numpy as np
from scipy.optimize import approx_fprime

def newton_method(func, gradient, hessian, start, 
                  max_iterations=100, tolerance=1e-6):
    """
    Newton's method for optimization.
    
    :param func: The objective function to minimize.
    :param gradient: The gradient function of the objective function.
    :param hessian: The Hessian function of the objective function.
    :param start: The starting point for the optimization.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Convergence tolerance.
    :return: The optimized value and history of values.
    """
    x = np.asarray(start)
    history = [x.copy()]
    
    for i in range(max_iterations):
        grad = gradient(x)
        hess = hessian(x)
        
        # Check for convergence
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i} iterations.")
            break
            
        # Compute Newton direction
        try:
            # For more numerical stability, solve the linear system instead of inverting
            direction = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            # If Hessian is singular or poorly conditioned, use a regularized version
            print(f"Warning: Hessian is poorly conditioned at iteration {i}.")
            # Add small values to diagonal for regularization
            reg_hess = hess + np.eye(len(x)) * 1e-4
            direction = np.linalg.solve(reg_hess, -grad)
        
        # Update x
        x = x + direction
        history.append(x.copy())
        
        # Optional: check function value convergence
        if i > 0 and abs(func(history[-1]) - func(history[-2])) < tolerance:
            print(f"Function value converged after {i} iterations.")
            break
    
    return x, np.array(history)

def quasi_newton_bfgs(func, gradient, start, max_iterations=100, tolerance=1e-6):
    """
    BFGS Quasi-Newton method for optimization.
    
    :param func: The objective function to minimize.
    :param gradient: The gradient function of the objective function.
    :param start: The starting point for the optimization.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Convergence tolerance.
    :return: The optimized value and history of values.
    """
    x = np.asarray(start)
    n = len(x)
    history = [x.copy()]
    
    # Initialize approximate Hessian inverse as identity matrix
    H = np.eye(n)
    
    for i in range(max_iterations):
        # Compute gradient
        grad = gradient(x)
        
        # Check for convergence
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i} iterations.")
            break
            
        # Compute search direction
        direction = -H.dot(grad)
        
        # Line search (simplified backtracking)
        alpha = 1.0
        fx = func(x)
        while alpha > 1e-10:
            new_x = x + alpha * direction
            if func(new_x) < fx:
                break
            alpha *= 0.5
        
        # Update position
        new_x = x + alpha * direction
        history.append(new_x.copy())
        
        # Compute difference vectors
        s = new_x - x
        new_grad = gradient(new_x)
        y = new_grad - grad
        
        # Update Hessian approximation using BFGS formula
        if np.dot(y, s) > 1e-10:  # Safe update condition
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n)
            term1 = I - rho * np.outer(s, y)
            term2 = I - rho * np.outer(y, s)
            term3 = rho * np.outer(s, s)
            H = term1.dot(H).dot(term2) + term3
        
        # Update x for next iteration
        x = new_x
    
    return x, np.array(history)

# Example usage with numerical approximation of Hessian
def numerical_hessian(func, x, epsilon=1e-5):
    """
    Compute the Hessian matrix using finite differences.
    """
    n = len(x)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Perturb x in both the i and j directions
            x_plus_i_j = x.copy()
            x_plus_i_j[i] += epsilon
            x_plus_i_j[j] += epsilon
            
            x_plus_i = x.copy()
            x_plus_i[i] += epsilon
            
            x_plus_j = x.copy()
            x_plus_j[j] += epsilon
            
            # Use central difference formula for second derivative
            hessian[i, j] = (func(x_plus_i_j) - func(x_plus_i) - func(x_plus_j) + func(x)) / (epsilon**2)
            
    return hessian

if __name__ == "__main__":
    # Define test function: Rosenbrock function
    def rosenbrock(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def rosenbrock_grad(x):
        return np.array([
            -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0]**2)
        ])
    
    def rosenbrock_hess(x):
        return np.array([
            [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])
    
    # Starting point
    x0 = np.array([-1.2, 1.0])
    
    # Run Newton's method
    newton_result, newton_history = newton_method(
        rosenbrock, rosenbrock_grad, rosenbrock_hess, x0
    )
    
    # Run BFGS Quasi-Newton
    bfgs_result, bfgs_history = quasi_newton_bfgs(
        rosenbrock, rosenbrock_grad, x0
    )
    
    print("Newton's method result:", newton_result)
    print(f"Newton's method value: {rosenbrock(newton_result)}")
    print(f"Newton's method iterations: {len(newton_history)}")
    
    print("\nBFGS Quasi-Newton result:", bfgs_result)
    print(f"BFGS value: {rosenbrock(bfgs_result)}")
    print(f"BFGS iterations: {len(bfgs_history)}")
    
    # Compare with true minimum at [1, 1]
    print("\nTrue minimum at [1, 1], value:", rosenbrock(np.array([1, 1])))