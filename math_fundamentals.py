import math
import numpy as np
import sympy as sp
from scipy.stats import norm

"""
Basic Arithmetic Operations
"""

# Multiplication (*)
def multiply(a, b):
    return a * b

# Division (/)
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Addition (+)
def add(a, b):
    return a + b

# Subtraction (-)
def subtract(a, b):
    return a - b

# Example:
if __name__ == "__main__":
    a = 10
    b = 5
    print(f"Addition: {add(a, b)}")
    print(f"Subtraction: {subtract(a, b)}")
    print(f"Multiplication: {multiply(a, b)}")
    print(f"Division: {divide(a, b)}")

"""
Subscript Notation (x_t)
Often used to represent variables that are indexed by another variable.
The notation x_t typically means that x is a variable that depends on the index t.
This is commonly used in sequences, time series, or states in a process.
"""

# Example sequence
x = [1, 3, 5, 7, 9, 11] 

# Explanation of subscript notation
print("Original sequence (x):", x)
print("\nExplanation of subscript notation:")
print("x_t represents the value of x at index t.")
print("For example:")
print("- x_0 is the first element:", x[0])
print("- x_1 is the second element:", x[1])
print("- x_2 is the third element:", x[2])

# Example:
print("\nExample: Calculate x_t + x_{t-2} for t = 2 to 5")
for t in range(2, len(x)):
    x_t = x[t]          
    x_t_minus_2 = x[t-2]  
    result = x_t + x_t_minus_2 
    print(f"At t = {t}: x_t = {x_t}, x_{{t-2}} = {x_t_minus_2}, x_t + x_{{t-2}} = {result}")

print("\nFinal results:", [x[t] + x[t-2] for t in range(2, len(x))])

"""
Superscript Notation (x^n)
Used to denote exponents or powers.
In Python, the exponentiation operator is ** (double asterisk).
"""

# Example: Calculate x^2 (x squared)
x = 5
x_squared = x ** 2 
print(f"{x}^2 = {x_squared}")

# Example: Calculate x^3 (x cubed)
x_cubed = x ** 3 
print(f"{x}^3 = {x_cubed}")

# Example: Calculate x^n for any exponent n
def power(x, n):
    return x ** n
x = 2
n = 4
result = power(x, n)
print(f"{x}^{n} = {result}")

"""
Norm (||...||)
The norm measures the size or length of a vector. The most common norms are:
- Euclidean norm (L2 norm): sqrt(v_1^2 + v_2^2 + ... + v_n^2)
- Manhattan norm (L1 norm): |v_1| + |v_2| + ... + |v_n|
- Maximum norm (L∞ norm): max(|v_1|, |v_2|, ..., |v_n|)
"""

# Euclidean Norm (L2 Norm)
def euclidean_norm(v):
    return math.sqrt(sum(x ** 2 for x in v))

# Manhattan Norm (L1 Norm)
def manhattan_norm(v):
    return sum(abs(x) for x in v)

# Maximum Norm (L∞ Norm)
def max_norm(v):
    return max(abs(x) for x in v)

# Example usage
if __name__ == "__main__":
    # Define a vector
    v = [3, -4, 5]

    print(f"Vector: {v}")
    print(f"Euclidean norm (L2): {euclidean_norm(v)}")
    print(f"Manhattan norm (L1): {manhattan_norm(v)}")
    print(f"Maximum norm (L∞): {max_norm(v)}")

"""
Summation Symbol (Σ)
The summation symbol (Σ) represents the sum of a sequence of terms.
"""

# Example 1: Summing a list of numbers using a loop
def summation_loop(sequence):
    total = 0
    for number in sequence:
        total += number
    return total

# Example 2: Summing a list of numbers using Python's built-in sum() function
def summation_builtin(sequence):
    return sum(sequence)

# Example 3: Sum of squares (Σ_{i=1}^{n} i^2)
def sum_of_squares(n):
    return sum(i ** 2 for i in range(1, n + 1))

# Example usage
if __name__ == "__main__":
    sequence = [1, 2, 3, 4, 5]
    n = 5
    result = sum_of_squares(n)

    sum_loop = summation_loop(sequence)
    sum_builtin = summation_builtin(sequence)

    print(f"Sequence: {sequence}")
    print(f"Sum using loop: {sum_loop}")
    print(f"Sum using built-in sum(): {sum_builtin}")
    print(f"Sum of squares from 1 to {n}: {result}") 

"""
Logarithm Base 2 (log2(x))
The logarithm base 2 of a number x is the power to which 2 must be raised to obtain x.
"""

# Example: Calculate log2(x)
def log2(x):
    return math.log2(x)

"""
Natural Logarithm (ln(x))
The natural logarithm of a number x is the logarithm with base e (Euler's number).
"""

# Example: Calculate ln(x)
def ln(x):
    return math.log(x)

# Example: Change of Base Formula
def log_base(x, base):
    return math.log(x) / math.log(base)    

# Example usage
if __name__ == "__main__":
    # Logarithm Base 2
    x = 8
    print(f"log2({x}) = {log2(x)}")  # Output: log2(8) = 3.0

    # Natural Logarithm
    x = math.e ** 2  # e^2
    print(f"ln({x:.2f}) = {ln(x)}")  # Output: ln(7.39) = 2.0

    # Change of Base Formula
    x = 100
    base = 10
    print(f"log_{base}({x}) = {log_base(x, base)}")  # Output: log_10(100) = 2.0

# Logarithm of a Product
    x = 5
    y = 10
    print(f"ln({x} * {y}) = {ln(x * y)}")  # Output: ln(5 * 10) = 4.007333185232471
    print(f"ln({x}) + ln({y}) = {ln(x) + ln(y)}")  # Output: ln(5) + ln(10) = 4.007333185232471

"""
Exponential Functions
The exponential function e^x represents Euler's number e raised to the power of x.
"""

# Exponential Function (Base e, e^x)
def exp_e(x):
    return math.exp(x)

# Exponential Function (Base 2, 2^x)
def exp_2(x):
    return 2 ** x

# Exponential Growth Model
def exponential_growth(P0, r, t):
    return P0 * math.exp(r * t)

# Exponential Decay Model
def exponential_decay(P0, r, t):
    return P0 * math.exp(-r * t)

# Example usage
if __name__ == "__main__":
    # Exponential Function (Base e)
    x = 2
    print(f"e^{x} ≈ {exp_e(x):.3f}")  # Output: e^2 ≈ 7.389

    # Exponential Function (Base 2)
    x = 3
    print(f"2^{x} = {exp_2(x)}")  # Output: 2^3 = 8

    # Exponential Growth and Decay
    P0 = 100  # Initial population/quantity
    r = 0.1   # Growth/decay rate
    t = 5     # Time

    # Exponential Growth
    population = exponential_growth(P0, r, t)
    print(f"Population after {t} years: {population:.2f}")  # Output: Population after 5 years: 164.87

    # Exponential Decay
    remaining = exponential_decay(P0, r, t)
    print(f"Remaining quantity after {t} years: {remaining:.2f}")  # Output: Remaining quantity after 5 years: 60.65

"""
Linear Algebra Concepts
"""

# Matrix-Vector Multiplication (A * v)
def matrix_vector_multiplication(A, v):
    return np.dot(A, v)

# Matrix-Matrix Multiplication (A * B)
def matrix_matrix_multiplication(A, B):
    return np.dot(A, B)

# Transpose (A^T)
def transpose(A):
    return np.transpose(A)

# Inverse (A^{-1})
def inverse(A):
    return np.linalg.inv(A)

# Determinant (det(A))
def determinant(A):
    return np.linalg.det(A)

# Trace (tr(A))
def trace(A):
    return np.trace(A)

# Example usage
if __name__ == "__main__":
    # Define a matrix A and vector v
    A = np.array([[1, 2], [3, 4]])
    v = np.array([5, 6])

    # Matrix-Vector Multiplication
    result_vector = matrix_vector_multiplication(A, v)
    print(f"Matrix-Vector Multiplication (A * v):\n{result_vector}")

    # Matrix-Matrix Multiplication
    B = np.array([[5, 6], [7, 8]])
    result_matrix = matrix_matrix_multiplication(A, B)
    print(f"\nMatrix-Matrix Multiplication (A * B):\n{result_matrix}")

    # Transpose
    A_transpose = transpose(A)
    print(f"\nTranspose of A (A^T):\n{A_transpose}")

    # Inverse
    A_inverse = inverse(A)
    print(f"\nInverse of A (A^{-1}):\n{A_inverse}")

    # Determinant
    det_A = determinant(A)
    print(f"\nDeterminant of A (det(A)): {det_A}")

    # Trace
    tr_A = trace(A)
    print(f"\nTrace of A (tr(A)): {tr_A}")

"""
Set Theory Concepts
"""

# Cardinality (|S|)
def cardinality(S):
    return len(S)

# Union (A ∪ B)
def union(A, B):
    return A.union(B)

# Intersection (A ∩ B)
def intersection(A, B):
    return A.intersection(B)

# Complement (A^c)
def complement(A, U):
    return U - A

# Example usage
if __name__ == "__main__":
    # Define sets
    S = {1, 2, 3, 4, 5}
    A = {1, 2, 3}
    B = {3, 4, 5}
    U = {1, 2, 3, 4, 5, 6, 7}  # Universal set

    # Cardinality
    print(f"Cardinality of S (|S|): {cardinality(S)}")

    # Union
    A_union_B = union(A, B)
    print(f"Union of A and B (A ∪ B): {A_union_B}")

    # Intersection
    A_intersection_B = intersection(A, B)
    print(f"Intersection of A and B (A ∩ B): {A_intersection_B}")

    # Complement
    A_complement = complement(A, U)
    print(f"Complement of A (A^c): {A_complement}")

"""
Eigenvalues and Eigenvectors
"""

# Define a matrix A
A = np.array([[4, 1], [2, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Example usage
if __name__ == "__main__":
    print("Matrix A:")
    print(A)

    # Eigenvalues
    print("\nEigenvalues (λ):")
    for i, lambda_ in enumerate(eigenvalues):
        print(f"λ_{i+1} = {lambda_:.2f}")

    # Eigenvectors
    print("\nEigenvectors:")
    for i, eigenvector in enumerate(eigenvectors.T):  # Transpose to get column vectors
        print(f"v_{i+1} = {eigenvector}")

    # Verification of the eigenvalue-eigenvector relationship (A * v = λ * v)
    print("\nVerification of A * v = λ * v:")
    for i, (lambda_, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        A_times_v = np.dot(A, eigenvector)  # Compute A * v
        lambda_times_v = lambda_ * eigenvector  # Compute λ * v

        print(f"\nFor λ_{i+1} = {lambda_:.2f} and v_{i+1} = {eigenvector}:")
        print(f"A * v_{i+1}: {A_times_v}")
        print(f"λ_{i+1} * v_{i+1}: {lambda_times_v}")

        # Check if A * v and λ * v are approximately equal
        if np.allclose(A_times_v, lambda_times_v):
            print("Verification successful: A * v = λ * v")
        else:
            print("Verification failed: A * v ≠ λ * v")

"""
Functions
"""

# Maximum Function (max(...))
def maximum_function(values):
    return max(values)

# Minimum Function (min(...))
def minimum_function(values):
    return min(values)

# Function Notation (f(x))
def f(x):
    return x**2 + 2*x + 1

if __name__ == "__main__":
    print("=== Functions ===")
    values = [4, 7, 2]
    print(f"Maximum of {values}: {maximum_function(values)}")
    print(f"Minimum of {values}: {minimum_function(values)}")
    x = 3
    print(f"f({x}) = {f(x)}")

"""
Operators
"""

# Reciprocal (1 / ...)
def reciprocal(x):
    return 1 / x

# Ellipsis (...)
def ellipsis_example():
    return "a_1 + a_2 + ... + a_n"

if __name__ == "__main__":
    print("\n=== Operators ===")
    x = 5
    print(f"Reciprocal of {x}: {reciprocal(x)}")
    print(f"Ellipsis example: {ellipsis_example()}")

"""
Probability Concepts
"""

# Conditional Probability Distribution (P(x | y))
def conditional_probability(x, y):
    # Placeholder: Assume P(x | y) = P(x) for simplicity
    return norm.pdf(x) / norm.pdf(y) if norm.pdf(y) != 0 else 0

# Expectation Operator (E[...])
def expectation(values, probabilities):
    return np.sum(values * probabilities)

# Variance (Var(X))
def variance(values, probabilities):
    mean = expectation(values, probabilities)
    return np.sum(probabilities * (values - mean) ** 2)

# Standard Deviation (σ(X))
def standard_deviation(values, probabilities):
    return np.sqrt(variance(values, probabilities))

# Covariance (Cov(X, Y))
def covariance(x_values, y_values, x_probabilities, y_probabilities):
    mean_x = expectation(x_values, x_probabilities)
    mean_y = expectation(y_values, y_probabilities)
    return np.sum(x_probabilities * y_probabilities * (x_values - mean_x) * (y_values - mean_y))

# Correlation (ρ(X, Y))
def correlation(x_values, y_values, x_probabilities, y_probabilities):
    cov = covariance(x_values, y_values, x_probabilities, y_probabilities)
    std_x = standard_deviation(x_values, x_probabilities)
    std_y = standard_deviation(y_values, y_probabilities)
    return cov / (std_x * std_y)

if __name__ == "__main__":
    print("\n=== Probability Concepts ===")
    x = 1
    y = 0
    print(f"P({x} | {y}) ≈ {conditional_probability(x, y):.4f}")

    values = np.array([1, 2, 3, 4])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"E[X] = {expectation(values, probabilities):.2f}")
    print(f"Var(X) = {variance(values, probabilities):.2f}")
    print(f"σ(X) = {standard_deviation(values, probabilities):.2f}")

    y_values = np.array([2, 3, 4, 5])
    y_probabilities = np.array([0.2, 0.3, 0.3, 0.2])
    print(f"Cov(X, Y) = {covariance(values, y_values, probabilities, y_probabilities):.2f}")
    print(f"ρ(X, Y) = {correlation(values, y_values, probabilities, y_probabilities):.2f}")


"""
Calculus Concepts
"""

# Derivatives
def derivative(func, var):
    return sp.diff(func, var)

# Integrals
def integral(func, var):
    return sp.integrate(func, var)

# Limits
def limit(func, var, point):
    return sp.limit(func, var, point)

# Taylor Series Expansion
def taylor_series(func, var, point, degree):
    return sp.series(func, var, point, degree).removeO()

# Example usage
if __name__ == "__main__":
    # Define a symbolic variable
    x = sp.symbols('x')
    
    # Define a function: f(x) = x^2 + 3x + 2
    f = x**2 + 3*x + 2
    
    # Compute the derivative
    df_dx = derivative(f, x)
    print(f"Derivative of f(x) = {f} with respect to x: {df_dx}")
    
    # Compute the integral
    integral_f = integral(f, x)
    print(f"Integral of f(x) = {f} with respect to x: {integral_f}")

    # Compute the limit as x approaches 2
    limit_f = limit(f, x, 2)
    print(f"Limit of f(x) = {f} as x approaches 2: {limit_f}")

    # Compute the Taylor series expansion around x = 0, degree 3
    taylor_f = taylor_series(f, x, 0, 3)
    print(f"Taylor series expansion of f(x) = {f} around x = 0 (degree 3): {taylor_f}")    