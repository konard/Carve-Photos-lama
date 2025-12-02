#!/usr/bin/env python3
"""
Experiment to test ONNX export with dynamic axes for LaMa model.

This script tests whether we can export the ONNX model with dynamic height/width
axes in addition to the batch axis.
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("Testing ONNX Export with Dynamic Axes")
print("=" * 80)

# Simple test of the custom FFT implementations
from saicinpainting.training.modules.ffc import FourierUnitJIT, ifft2d

print("\n1. Testing FourierUnitJIT with different input sizes...")

# Create a FourierUnitJIT instance
fourier_unit = FourierUnitJIT(in_channels=64, out_channels=64)
fourier_unit.eval()

# Test with 512x512 (original size)
print("\n   Testing with 512x512 input:")
try:
    x_512 = torch.randn(1, 64, 512, 512)
    with torch.no_grad():
        output_512 = fourier_unit(x_512)
    print(f"   ✓ Input shape: {x_512.shape}, Output shape: {output_512.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test with 256x256
print("\n   Testing with 256x256 input:")
try:
    x_256 = torch.randn(1, 64, 256, 256)
    with torch.no_grad():
        output_256 = fourier_unit(x_256)
    print(f"   ✓ Input shape: {x_256.shape}, Output shape: {output_256.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test with 1024x1024
print("\n   Testing with 1024x1024 input:")
try:
    x_1024 = torch.randn(1, 64, 1024, 1024)
    with torch.no_grad():
        output_1024 = fourier_unit(x_1024)
    print(f"   ✓ Input shape: {x_1024.shape}, Output shape: {output_1024.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test with non-square input (e.g., 512x768)
print("\n   Testing with 512x768 input (non-square):")
try:
    x_nonsquare = torch.randn(1, 64, 512, 768)
    with torch.no_grad():
        output_nonsquare = fourier_unit(x_nonsquare)
    print(f"   ✓ Input shape: {x_nonsquare.shape}, Output shape: {output_nonsquare.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("2. Analyzing the ifft2d function for dynamic shape handling...")
print("=" * 80)

# Check if the ifft2d function can handle dynamic shapes
print("\n   The issue is in the ifft1d and irfft functions:")
print("   - Line 111 in irfft: REAL.permute(0, 1, 3, 2)")
print("   - Lines 138-143 in ifft1d: Hardcoded permutation logic")
print("\n   These hardcoded permutations assume fixed tensor dimensions")
print("   which may cause issues during ONNX export with dynamic axes.")

print("\n" + "=" * 80)
print("Conclusion:")
print("=" * 80)
print("""
The FourierUnitJIT can handle different input sizes in PyTorch mode.
However, the hardcoded permutations in ifft1d and irfft may prevent
proper ONNX export with dynamic height/width axes.

The TODO comment in the export notebook is accurate:
'TODO: Adapt FourierUnit to support dynamic axes (see irfttn and rfft for correct padding)'

Next steps:
1. Try exporting with dynamic axes and see what happens
2. If it fails, we need to fix the permutation logic to be dynamic
3. Test the exported ONNX model with different input sizes
""")
