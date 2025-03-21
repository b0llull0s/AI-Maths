from Math import abs

struct L1Regularizer:
    """
    L1 Regularization (Lasso) implementation in Mojo.
    
    L1 regularization adds a penalty equal to the sum of the absolute values
    of parameter values multiplied by a regularization strength (lambda).
    This promotes sparsity in the model parameters.
    """
    var lambda_param: Float64
    
    fn __init__(inout self, lambda_param: Float64 = 0.01):
        """
        Initialize L1 regularizer with regularization strength.
        
        Args:
            lambda_param: Regularization strength parameter (default: 0.01)
        """
        self.lambda_param = lambda_param
    
    fn compute_penalty(self, params: DTypePointer[DType.float64], param_count: Int) -> Float64:
        """
        Compute the L1 regularization penalty.
        
        Args:
            params: Pointer to model parameters
            param_count: Number of parameters
            
        Returns:
            The L1 regularization penalty term
        """
        var penalty: Float64 = 0.0
        
        for i in range(param_count):
            penalty += abs(params[i])
        
        return self.lambda_param * penalty
    
    fn compute_gradients(self, params: DTypePointer[DType.float64], 
                         inout gradients: DTypePointer[DType.float64], 
                         param_count: Int):
        """
        Add L1 regularization gradients to existing gradients.
        
        Args:
            params: Pointer to model parameters
            gradients: Pointer to gradients to be updated
            param_count: Number of parameters
        """
        for i in range(param_count):
            if params[i] > 0:
                gradients[i] += self.lambda_param
            elif params[i] < 0:
                gradients[i] -= self.lambda_param
            # For params[i] == 0, the gradient is technically undefined
            # In practice, we can use a subgradient in [−λ, λ], often just 0
        
# Example usage
fn main():
    print("L1 Regularization (Lasso) Example")
    
    # Create parameters and gradients
    let param_count = 5
    var params = DTypePointer[DType.float64].alloc(param_count)
    var gradients = DTypePointer[DType.float64].alloc(param_count)
    
    # Initialize parameters with some values
    params[0] = 2.0
    params[1] = -1.5
    params[2] = 0.0  # Zero parameter (demonstrates sparsity)
    params[3] = 3.0
    params[4] = -2.5
    
    # Initialize gradients from some loss function
    gradients[0] = 0.1
    gradients[1] = -0.2
    gradients[2] = 0.3
    gradients[3] = -0.4
    gradients[4] = 0.5
    
    print("Parameters before regularization:")
    for i in range(param_count):
        print("  params[", i, "] =", params[i])
    
    print("\nGradients before regularization:")
    for i in range(param_count):
        print("  grads[", i, "] =", gradients[i])
    
    # Create and apply L1 regularizer
    var l1_reg = L1Regularizer(0.1)  # lambda = 0.1
    
    # Compute penalty (for loss function)
    let penalty = l1_reg.compute_penalty(params, param_count)
    print("\nL1 Regularization penalty:", penalty)
    
    # Update gradients
    l1_reg.compute_gradients(params, gradients, param_count)
    
    print("\nGradients after adding L1 regularization:")
    for i in range(param_count):
        print("  grads[", i, "] =", gradients[i])
    
    # Free memory
    params.free()
    gradients.free()