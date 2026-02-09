#!/usr/bin/env python3
import mlx.core as mx
import numpy as np
import math

# Test sin function
a = mx.array([0, math.pi / 4, math.pi / 2, math.pi])
result = mx.sin(a)
expected = np.sin(a, dtype=np.float32)

print("Input:", a)
print("MLX result:", result)
print("Expected (numpy):", expected)
print("Match:", np.allclose(result, expected))
print("Difference:", np.array(result) - expected)
