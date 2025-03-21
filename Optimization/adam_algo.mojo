from Math import sqrt, exp
from Random import rand
from Time import now

struct Adam:
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var learning_rate: Float64
    var params: DTypePointer[DType.float64]
    var m: DTypePointer[DType.float64]
    var v: DTypePointer[DType.float64]
    var param_count: Int
    var t: Int
    
    fn __init__(inout self, param_count: Int, learning_rate: Float64 = 0.001, 
                beta1: Float64 = 0.9, beta2: Float64 = 0.999, epsilon: Float64 = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.param_count = param_count
        self.t = 0
        
        # Allocate memory for parameters and momentums
        self.params = DTypePointer[DType.float64].alloc(param_count)
        self.m = DTypePointer[DType.float64].alloc(param_count)
        self.v = DTypePointer[DType.float64].alloc(param_count)
        
        # Initialize parameters with small random values
        for i in range(param_count):
            self.params[i] = (rand(2.0) - 1.0) * 0.01
            self.m[i] = 0.0
            self.v[i] = 0.0
    
    fn __del__(owned self):
        self.params.free()
        self.m.free()
        self.v.free()
    
    fn step(inout self, gradients: DTypePointer[DType.float64]):
        self.t += 1
        
        for i in range(self.param_count):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients[i] * gradients[i])
            
            # Compute bias-corrected first moment estimate
            let m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            let v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.params[i] -= self.learning_rate * m_hat / (sqrt(v_hat) + self.epsilon)
    
    fn print_params(self):
        print("Parameters after optimization:")
        for i in range(self.param_count):
            print("param[", i, "]: ", self.params[i])

# A simple quadratic function to minimize: f(x,y) = x^2 + 2y^2
fn compute_loss(x: Float64, y: Float64) -> Float64:
    return x*x + 2.0*y*y

fn compute_gradients(x: Float64, y: Float64, inout grads: DTypePointer[DType.float64]):
    grads[0] = 2.0 * x  # df/dx = 2x
    grads[1] = 4.0 * y  # df/dy = 4y

fn main():
    # Create an Adam optimizer for 2 parameters (x and y)
    var optimizer = Adam(2, 0.1)
    var gradients = DTypePointer[DType.float64].alloc(2)
    
    print("Initial parameters:")
    print("x =", optimizer.params[0], ", y =", optimizer.params[1])
    print("Initial loss:", compute_loss(optimizer.params[0], optimizer.params[1]))
    
    let start = now()
    
    # Optimization loop
    for epoch in range(100):
        # Compute loss and gradients
        let x = optimizer.params[0]
        let y = optimizer.params[1]
        
        compute_gradients(x, y, gradients)
        
        # Update parameters
        optimizer.step(gradients)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 99:
            let current_loss = compute_loss(optimizer.params[0], optimizer.params[1])
            print("Epoch", epoch, ": x =", optimizer.params[0], 
                  ", y =", optimizer.params[1], ", loss =", current_loss)
    
    let end = now()
    print("Optimization took", (end - start) / 1_000_000, "ms")
    
    # Clean up
    gradients.free()