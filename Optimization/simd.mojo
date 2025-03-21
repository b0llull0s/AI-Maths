from Sys import print_no_newline
from Time import now
from SIMD import SIMD, simd_width
from Math import exp, sqrt

struct ParallelProcessor:
    fn process_vector[datatype: DType](
        data: DTypePointer[datatype],
        size: Int
    ):
        let vec_width = simd_width[datatype]()
        let simd_iters = size // vec_width
        let remainder = size % vec_width
        
        @parameter
        fn simd_op[width: Int](idx: Int):
            let vec_idx = idx * width
            var vec_data = SIMD[datatype, width].load(data.offset(vec_idx))
            
            # Apply function (here: multiply by 2 and add 1)
            vec_data = vec_data * 2 + 1
            
            # Calculate sqrt(exp(x))
            vec_data = sqrt(exp(vec_data))
            
            vec_data.store(data.offset(vec_idx))
        
        @parameter
        fn scalar_op(idx: Int):
            var val = data[idx]
            val = val * 2 + 1
            val = sqrt(exp(val))
            data[idx] = val
        
        # Process bulk of data with SIMD
        for i in range(simd_iters):
            simd_op[vec_width](i)
        
        # Process remainder
        let remainder_start = simd_iters * vec_width
        for i in range(remainder):
            scalar_op(remainder_start + i)

fn main():
    let size = 10_000_000
    let data = DTypePointer[DType.float32].alloc(size)
    
    # Initialize data
    for i in range(size):
        data[i] = i * 0.01
    
    print("Processing " + String(size) + " elements using SIMD parallelism")
    
    let start = now()
    ParallelProcessor.process_vector[DType.float32](data, size)
    let end = now()
    
    let time_ms = (end - start) / 1_000_000
    print("Time taken: " + String(time_ms) + " ms")
    
    # Print a small sample of the results
    print("First 5 results:")
    for i in range(5):
        print(data[i])
    
    data.free()