import numpy as np

def particle_swarm_optimization(objective_func, bounds, num_particles=30, 
                               max_iterations=100, w=0.5, c1=1, c2=2):
    """
    Particle Swarm Optimization (PSO) algorithm.
    
    :param objective_func: The objective function to minimize.
    :param bounds: Bounds for each dimension [(x1_min, x1_max), (x2_min, x2_max), ...].
    :param num_particles: Number of particles in the swarm.
    :param max_iterations: Maximum number of iterations.
    :param w: Inertia weight.
    :param c1: Cognitive parameter.
    :param c2: Social parameter.
    :return: Best position found and history of global best values.
    """
    # Problem dimensions
    dimensions = len(bounds)
    
    # Initialize particles
    particles = np.random.rand(num_particles, dimensions)
    velocities = np.zeros((num_particles, dimensions))
    
    # Scale particles to the given bounds
    for i in range(dimensions):
        particles[:, i] = particles[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    
    # Initialize personal best positions and values
    personal_best_pos = particles.copy()
    personal_best_val = np.array([objective_func(p) for p in particles])
    
    # Initialize global best
    global_best_idx = np.argmin(personal_best_val)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_val = personal_best_val[global_best_idx]
    
    # History of global best values
    history = [global_best_val]
    
    # Main PSO loop
    for iteration in range(max_iterations):
        # Update velocities and positions
        for i in range(num_particles):
            # Generate random vectors
            r1 = np.random.rand(dimensions)
            r2 = np.random.rand(dimensions)
            
            # Velocity update
            cognitive_component = c1 * r1 * (personal_best_pos[i] - particles[i])
            social_component = c2 * r2 * (global_best_pos - particles[i])
            velocities[i] = w * velocities[i] + cognitive_component + social_component
            
            # Position update
            particles[i] = particles[i] + velocities[i]
            
            # Apply bounds
            for j in range(dimensions):
                if particles[i, j] < bounds[j][0]:
                    particles[i, j] = bounds[j][0]
                    velocities[i, j] *= -0.5  # Bounce off boundary with reduced velocity
                elif particles[i, j] > bounds[j][1]:
                    particles[i, j] = bounds[j][1]
                    velocities[i, j] *= -0.5  # Bounce off boundary with reduced velocity
            
            # Evaluate objective function
            val = objective_func(particles[i])
            
            # Update personal best
            if val < personal_best_val[i]:
                personal_best_val[i] = val
                personal_best_pos[i] = particles[i].copy()
                
                # Update global best
                if val < global_best_val:
                    global_best_val = val
                    global_best_pos = particles[i].copy()
        
        # Record history
        history.append(global_best_val)
        
        # Optional: print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: Best value = {global_best_val}")
    
    return global_best_pos, np.array(history)

# Example usage
if __name__ == "__main__":
    # Test with Rastrigin function (a challenging multimodal function)
    def rastrigin(x, A=10):
        return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])
    
    # 2D Rastrigin function bounds
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    
    # Run PSO
    best_pos, history = particle_swarm_optimization(
        rastrigin, bounds, num_particles=50, max_iterations=200
    )
    
    print("Best position found:", best_pos)
    print("Best value:", rastrigin(best_pos))
    print("Expected global minimum at [0, 0], value:", rastrigin(np.array([0, 0])))
    
    # Additional test with a more complex function
    def ackley(x):
        a, b, c = 20, 0.2, 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.mean(np.square(x))))
        term2 = -np.exp(np.mean(np.cos(c * x)))
        return term1 + term2 + a + np.exp(1)
    
    # Ackley function bounds
    bounds_ackley = [(-5, 5), (-5, 5)]
    
    # Run PSO on Ackley function
    best_pos_ackley, history_ackley = particle_swarm_optimization(
        ackley, bounds_ackley, num_particles=50, max_iterations=200
    )
    
    print("\nAckley function:")
    print("Best position found:", best_pos_ackley)
    print("Best value:", ackley(best_pos_ackley))
    print("Expected global minimum at [0, 0], value:", ackley(np.array([0, 0])))