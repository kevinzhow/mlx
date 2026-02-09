#!/usr/bin/env python3
import mlx.core as mx
import numpy as np
import math

print("=" * 60)
print("Testing Sin precision: float32 vs float16")
print("=" * 60)

# Test values
test_values = [0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 4, 2 * math.pi]

print("\n### Float32 Test ###")
a_f32 = mx.array(test_values, dtype=mx.float32)
result_f32 = mx.sin(a_f32)
expected_f32 = np.sin(np.array(test_values), dtype=np.float32)

print(f"Input:    {a_f32}")
print(f"MLX:      {result_f32}")
print(f"NumPy:    {expected_f32}")
print(f"Diff:     {np.array(result_f32) - expected_f32}")
print(f"Max diff: {np.max(np.abs(np.array(result_f32) - expected_f32)):.2e}")
print(f"Match (default tol): {np.allclose(result_f32, expected_f32)}")
print(f"Match (rtol=1e-5):   {np.allclose(result_f32, expected_f32, rtol=1e-5, atol=1e-5)}")

print("\n### Float16 Test ###")
a_f16 = mx.array(test_values, dtype=mx.float16)
result_f16 = mx.sin(a_f16)
expected_f16 = np.sin(np.array(test_values), dtype=np.float16)

print(f"Input:    {a_f16}")
print(f"MLX:      {result_f16}")
print(f"NumPy:    {expected_f16}")
print(f"Diff:     {np.array(result_f16, dtype=np.float32) - expected_f16.astype(np.float32)}")
print(f"Max diff: {np.max(np.abs(np.array(result_f16, dtype=np.float32) - expected_f16.astype(np.float32))):.2e}")
print(f"Match (default tol): {np.allclose(result_f16, expected_f16)}")
print(f"Match (rtol=1e-3):   {np.allclose(result_f16, expected_f16, rtol=1e-3, atol=1e-3)}")

print("\n### Comparison ###")
print(f"Float32 max error: {np.max(np.abs(np.array(result_f32) - expected_f32)):.2e}")
print(f"Float16 max error: {np.max(np.abs(np.array(result_f16, dtype=np.float32) - expected_f16.astype(np.float32))):.2e}")
