from Sys import print_no_newline
from Math import sqrt
from Time import now

struct Matrix:
    var data: DTypePointer[DType.float64]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[DType.float64].alloc(rows * cols)
        
    fn __del__(owned self):
        self.data.free()
    
    fn fill_random(inout self):
        # Simple random number generator
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i * self.cols + j] = (i * 0.3 + j * 0.7) % 10
    
    fn get(self, i: Int, j: Int) -> Float64:
        return self.data[i * self.cols + j]
        
    fn set(inout self, i: Int, j: Int, val: Float64):
        self.data[i * self.cols + j] = val
    
    fn print(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print_no_newline(self.get(i, j))
                print_no_newline(" ")
            print("")

fn multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    if a.cols != b.rows:
        print("Error: Matrix dimensions do not match for multiplication")
        return Matrix(0, 0)
    
    var result = Matrix(a.rows, b.cols)
    
    for i in range(a.rows):
        for j in range(b.cols):
            var sum: Float64 = 0
            for k in range(a.cols):
                sum += a.get(i, k) * b.get(k, j)
            result.set(i, j, sum)
    
    return result

fn main():
    let size = 100  # Try with different sizes
    
    var a = Matrix(size, size)
    var b = Matrix(size, size)
    
    a.fill_random()
    b.fill_random()
    
    print("Matrix multiplication " + String(size) + "x" + String(size))
    
    let start = now()
    var result = multiply_matrices(a, b)
    let end = now()
    
    print("Time taken: " + String((end - start) / 1_000_000) + " ms")
    
    # Print small sample of the result
    if size <= 5:
        print("Matrix A:")
        a.print()
        print("Matrix B:")
        b.print()
        print("Result:")
        result.print()
    else:
        print("Result is too large to print")