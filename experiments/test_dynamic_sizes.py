#!/usr/bin/env python3
"""
Test script to verify that FourierUnitJIT now works with dynamic input sizes.

This validates that the fixes to ifft1d and irfft permutations allow
the model to process different resolutions correctly.
"""

import sys
import os
import torch
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saicinpainting.training.modules.ffc import FourierUnitJIT

print("=" * 80)
print("Testing FourierUnitJIT with Dynamic Input Sizes")
print("=" * 80)

# Test configuration
test_sizes = [
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 1024),
    (512, 768),  # Non-square
    (640, 480),  # Non-square
]

# Create a FourierUnitJIT instance
print("\nInitializing FourierUnitJIT...")
fourier_unit = FourierUnitJIT(in_channels=64, out_channels=64)
fourier_unit.eval()
print("✓ Initialized successfully")

# Test each size
results = []
for height, width in test_sizes:
    print(f"\nTesting size: {height}x{width}")
    try:
        x = torch.randn(1, 64, height, width)
        with torch.no_grad():
            output = fourier_unit(x)

        # Verify output shape
        expected_shape = (1, 64, height, width)
        if output.shape == expected_shape:
            print(f"  ✓ PASS: Input {x.shape} -> Output {output.shape}")
            results.append((height, width, "PASS", None))
        else:
            print(f"  ✗ FAIL: Expected {expected_shape}, got {output.shape}")
            results.append((height, width, "FAIL", f"Shape mismatch: expected {expected_shape}, got {output.shape}"))
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        results.append((height, width, "ERROR", str(e)))
        traceback.print_exc()

# Print summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

passed = sum(1 for _, _, status, _ in results if status == "PASS")
failed = sum(1 for _, _, status, _ in results if status == "FAIL")
errors = sum(1 for _, _, status, _ in results if status == "ERROR")

print(f"\nTotal tests: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Errors: {errors}")

if failed > 0 or errors > 0:
    print("\nFailed/Error details:")
    for h, w, status, msg in results:
        if status != "PASS":
            print(f"  - {h}x{w}: {status} - {msg}")

print("\n" + "=" * 80)
if passed == len(results):
    print("✓ ALL TESTS PASSED! Dynamic size support is working correctly.")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED! Dynamic size support needs more work.")
    sys.exit(1)
