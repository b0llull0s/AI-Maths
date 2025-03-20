import numpy as np

def stochastic_gradient_descent(gradient_func, X, y, start, 
                               learning_rate=0.01, n_epochs=50):
    """
    Stochastic Gradient Descent - updates parameters after each sample.
    
    :param gradient_func: Function that computes gradient for a single sample.
    :param X: Training data features.
    :param y: Training data targets.
    :param start: Initial parameters.
    :param learning_rate: The learning rate (step size).
    :param n_epochs: Number of passes through the entire dataset.
    :return: Optimized parameters and history of parameters.
    """
    theta = start.copy()
    n_samples = X.shape[0]
    history = [theta.copy()]
    
    for epoch in range(n_epochs):
        # Shuffle the data for each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(n_samples):
            xi = X_shuffled[i:i+1]  # Single sample (keeping dimensions)
            yi = y_shuffled[i:i+1]
            
            # Compute gradient for this sample
            grad = gradient_func(theta, xi, yi)
            
            # Update parameters
            theta = theta - learning_rate * grad
            
        # Store history after each epoch
        history.append(theta.copy())
        
        # Optional: print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} completed")
    
    return theta, np.array(history)

def mini_batch_gradient_descent(gradient_func, X, y, start, 
                               batch_size=32, learning_rate=0.01, n_epochs=50):
    """
    Mini-Batch Gradient Descent - updates parameters after each mini-batch.
    
    :param gradient_func: Function that computes gradient for a batch of samples.
    :param X: Training data features.
    :param y: Training data targets.
    :param start: Initial parameters.
    :param batch_size: Size of mini-batches.
    :param learning_rate: The learning rate (step size).
    :param n_epochs: Number of passes through the entire dataset.
    :return: Optimized parameters and history of parameters.
    """
    theta = start.copy()
    n_samples = X.shape[0]
    history = [theta.copy()]
    
    for epoch in range(n_epochs):
        # Shuffle the data for each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:end]
            y_batch = y_shuffled[i:end]
            
            # Compute gradient for this batch
            grad = gradient_func(theta, X_batch, y_batch)
            
            # Update parameters
            theta = theta - learning_rate * grad
        
        # Store history after each epoch
        history.append(theta.copy())
        
        # Optional: print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} completed")
    
    return theta, np.array(history)

# Example usage
if __name__ == "__main__":
    # Generate some synthetic linear regression data
    np.random.seed(42)
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    theta_true = np.array([0.5, -0.2, 0.3])
    y = X.dot(theta_true) + 0.1 * np.random.randn(100)  # Add some noise
    
    # Define gradient function for linear regression (MSE loss)
    def compute_gradient_sample(theta, X_sample, y_sample):
        """Compute gradient for a single sample"""
        prediction = X_sample.dot(theta)
        error = prediction - y_sample
        return X_sample.T.dot(error) / len(y_sample)
    
    def compute_gradient_batch(theta, X_batch, y_batch):
        """Compute gradient for a batch of samples"""
        predictions = X_batch.dot(theta)
        errors = predictions - y_batch
        return X_batch.T.dot(errors) / len(y_batch)
    
    # Initialize parameters
    initial_theta = np.zeros(3)
    
    # Run SGD
    theta_sgd, history_sgd = stochastic_gradient_descent(
        compute_gradient_sample, X, y, initial_theta, learning_rate=0.1, n_epochs=30
    )
    
    # Run Mini-Batch GD
    theta_mini, history_mini = mini_batch_gradient_descent(
        compute_gradient_batch, X, y, initial_theta, 
        batch_size=10, learning_rate=0.1, n_epochs=30
    )
    
    print("True parameters:", theta_true)
    print("SGD result:", theta_sgd)
    print("Mini-Batch GD result:", theta_mini)