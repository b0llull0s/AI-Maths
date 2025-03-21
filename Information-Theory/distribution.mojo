from Math import log, exp
from Sys import print_no_newline

struct Distribution:
    var data: DTypePointer[DType.float64]
    var size: Int
    var is_normalized: Bool
    
    fn __init__(inout self, size: Int):
        self.size = size
        self.data = DTypePointer[DType.float64].alloc(size)
        self.is_normalized = False
        
        # Default: uniform initialization
        for i in range(size):
            self.data[i] = 1.0 / size
        
        self.is_normalized = True
    
    fn __init__(inout self, data: List[Float64]):
        self.size = len(data)
        self.data = DTypePointer[DType.float64].alloc(self.size)
        self.is_normalized = False
        
        var sum: Float64 = 0.0
        for i in range(self.size):
            self.data[i] = data[i]
            sum += data[i]
        
        # Check if already normalized
        if abs(sum - 1.0) < 1e-8:
            self.is_normalized = True
        
    fn __del__(owned self):
        self.data.free()
    
    fn normalize(inout self):
        if self.is_normalized:
            return
            
        var sum: Float64 = 0.0
        for i in range(self.size):
            sum += self.data[i]
            
        if sum <= 0:
            print("Error: Cannot normalize distribution with sum <= 0")
            return
            
        for i in range(self.size):
            self.data[i] /= sum
            
        self.is_normalized = True
    
    fn get(self, idx: Int) -> Float64:
        if idx >= 0 and idx < self.size:
            return self.data[idx]
        return 0.0
    
    fn set(inout self, idx: Int, value: Float64):
        if idx >= 0 and idx < self.size:
            self.data[idx] = value
            self.is_normalized = False  # Setting a value invalidates normalization
    
    fn print(self):
        print("[", end="")
        for i in range(self.size):
            print_no_newline(self.data[i])
            if i < self.size - 1:
                print_no_newline(", ")
        print("]")