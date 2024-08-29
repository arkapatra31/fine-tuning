import numpy as np
from numba import cuda

@cuda.jit
def gpu_add(x, y, out):
  idx = cuda.grid(1)
  out[idx] = x[idx] + y[idx]

# Create arrays
n = 100000
x = np.arange(n).astype(np.float32)
y = np.arange(n).astype(np.float32)
out = np.empty_like(x)

# GPU execution
threadsperblock = 128
blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
gpu_add[blockspergrid, threadsperblock](x, y, out)

# CPU execution
def cpu_add(x, y):
  out = np.empty_like(x)
  for i in range(len(x)):
    out[i] = x[i] + y[i]
  return out

import time

# Measure CPU execution time
start_time = time.time()
cpu_result = cpu_add(x, y)
end_time = time.time()
cpu_time = end_time - start_time
print(f"CPU execution time: {cpu_time} seconds")

# Measure GPU execution time
start_time = time.time()
gpu_add[blockspergrid, threadsperblock](x, y, out)
end_time = time.time()
gpu_time = end_time - start_time
print(f"GPU execution time: {gpu_time} seconds")