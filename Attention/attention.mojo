from sys import argv
from memory import memset_zero
from memory.unsafe import DTypePointer
from random import rand, random_float64
from algorithm import vectorize, parallelize
from time import now

alias DType = DType.float32

struct Matrix:
    """
    A simple 2D matrix structure with row-major storage.
    """
    var data: DTypePointer[DType]
    var rows: Int
    var cols: Int
    var allocated: Bool

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[DType].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.allocated = True

    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[DType]):
        self.rows = rows
        self.cols = cols
        self.data = data
        self.allocated = False

    fn __del__(owned self):
        if self.allocated:
            self.data.free()

    @always_inline
    fn __getitem__(self, i: Int, j: Int) -> Float32:
        return self.load[1](i, j)

    @always_inline
    fn __setitem__(self, i: Int, j: Int, val: Float32):
        self.store[1](i, j, val)

    @always_inline
    fn load[nelts: Int](self, i: Int, j: Int) -> SIMD[DType, nelts]:
        return self.data.load[nelts](i * self.cols + j)

    @always_inline
    fn store[nelts: Int](self, i: Int, j: Int, val: SIMD[DType, nelts]):
        self.data.store[nelts](i * self.cols + j, val)

    fn print(self):
        print("Matrix [", self.rows, "x", self.cols, "]")
        for i in range(self.rows):
            print_no_newline("[")
            for j in range(self.cols):
                print_no_newline(self[i, j])
                if j < self.cols - 1:
                    print_no_newline(", ")
            print("]")

    fn fill_random(self, min_val: Float32 = -1.0, max_val: Float32 = 1.0):
        """Fill the matrix with random values."""
        let range_val = max_val - min_val
        for i in range(self.rows):
            for j in range(self.cols):
                self[i, j] = min_val + Float32(random_float64() * Float64(range_val))


# Optimized matrix multiplication with vectorization and parallelization
fn matmul(Q: Matrix, K: Matrix, inout result: Matrix):
    """
    Matrix multiplication C = Q * K^T with SIMD optimizations.
    For attention: Q is [batch_size*num_heads, seq_len, head_dim]
                   K is [batch_size*num_heads, seq_len, head_dim]
                   result is [batch_size*num_heads, seq_len, seq_len]
    """
    # Ensure dimensions match
    if Q.cols != K.cols:
        print("Error: Inner dimensions must match for matrix multiplication")
        return
    if result.rows != Q.rows or result.cols != K.rows:
        print("Error: Result matrix has incorrect dimensions")
        return

    # Outer parallelization over rows of Q
    @parameter
    fn compute_row(i: Int):
        # Vectorized computation for each result element
        @parameter
        fn compute_element[nelts: Int](j: Int):
            var sum = SIMD[DType, nelts](0.0)
            
            # Process vectors of K elements at a time
            for k in range(0, Q.cols, nelts):
                let vec_size = min(nelts, Q.cols - k)
                if vec_size < nelts:
                    # Handle remaining elements (non-multiple of nelts)
                    var partial_sum = Float32(0.0)
                    for k_offset in range(vec_size):
                        partial_sum += Q[i, k + k_offset] * K[j, k + k_offset]
                    result[i, j] += partial_sum
                else:
                    # Full vector processing
                    let q_vec = Q.load[nelts](i, k)
                    let k_vec = K.load[nelts](j, k)
                    sum += q_vec * k_vec
            
            # Store the accumulated dot product
            if nelts > 1:
                result[i, j] = sum.reduce_add()

        # Process each column of the result
        vectorize[compute_element, 8](K.rows)

    # Parallelize over rows
    parallelize[compute_row](Q.rows)


fn scaled_dot_product_attention(
    Q: Matrix, K: Matrix, V: Matrix, inout output: Matrix, 
    scale_factor: Float32
):
    """
    Computes scaled dot-product attention: softmax(Q*K^T/sqrt(d_k))*V
    
    Args:
        Q: Query matrix [batch_size*num_heads, seq_len, head_dim]
        K: Key matrix [batch_size*num_heads, seq_len, head_dim]
        V: Value matrix [batch_size*num_heads, seq_len, head_dim]
        output: Output matrix [batch_size*num_heads, seq_len, head_dim]
        scale_factor: Scaling factor (typically 1/sqrt(head_dim))
    """
    # Step 1: Compute attention scores Q*K^T
    var attention_scores = Matrix(Q.rows, K.rows)
    matmul(Q, K, attention_scores)
    
    # Step 2: Apply scaling factor
    for i in range(attention_scores.rows):
        for j in range(attention_scores.cols):
            attention_scores[i, j] *= scale_factor
    
    # Step 3: Apply softmax to get attention weights
    var attention_weights = Matrix(attention_scores.rows, attention_scores.cols)
    for i in range(attention_scores.rows):
        # Find max for numerical stability
        var row_max = attention_scores[i, 0]
        for j in range(1, attention_scores.cols):
            if attention_scores[i, j] > row_max:
                row_max = attention_scores[i, j]
        
        # Compute exp(x - max) for each element
        var row_sum = Float32(0.0)
        for j in range(attention_scores.cols):
            attention_weights[i, j] = math.exp(attention_scores[i, j] - row_max)
            row_sum += attention_weights[i, j]
        
        # Normalize by sum
        for j in range(attention_weights.cols):
            attention_weights[i, j] /= row_sum
    
    # Step 4: Apply attention weights to values
    # Result: [batch_size*num_heads, seq_len, head_dim]
    matmul(attention_weights, V, output)


fn benchmark_attention(seq_len: Int, head_dim: Int):
    """
    Benchmark the scaled dot-product attention implementation.
    """
    print("Benchmarking scaled dot-product attention:")
    print("  Sequence length:", seq_len)
    print("  Head dimension:", head_dim)
    
    # Create sample matrices (we'll just use one batch with one head)
    var Q = Matrix(seq_len, head_dim)
    var K = Matrix(seq_len, head_dim)
    var V = Matrix(seq_len, head_dim)
    var output = Matrix(seq_len, head_dim)
    
    # Fill with random data
    Q.fill_random(-0.1, 0.1)
    K.fill_random(-0.1, 0.1)
    V.fill_random(-0.1, 0.1)
    
    # Compute scaling factor
    let scale_factor = 1.0 / math.sqrt(Float32(head_dim))
    
    # Benchmark
    let start_time = now()
    scaled_dot_product_attention(Q, K, V, output, scale_factor)
    let end_time = now()
    
    let duration_ms = (end_time - start_time) / 1_000_000  # Convert ns to ms
    print("  Execution time:", duration_ms, "ms")
    
    # Print a small sample of the output
    print("Output sample:")
    let sample_size = min(5, seq_len)
    for i in range(sample_size):
        print_no_newline("  Row ", i, ": [")
        for j in range(min(5, head_dim)):
            print_no_newline(output[i, j])
            if j < min(4, head_dim - 1):
                print_no_newline(", ")
        if head_dim > 5:
            print_no_newline(", ...")
        print("]")


fn main():
    # Use command line args if provided, otherwise use default values
    var seq_len = 32
    var head_dim = 64
    var batch_size = 1
    
    if len(argv) > 1:
        seq_len = atol(argv[1])
    if len(argv) > 2:
        head_dim = atol(argv[2])
    if len(argv) > 3:
        batch_size = atol(argv[3])
    
    benchmark_attention(seq_len, head_dim)