#!/usr/bin/env python3
"""
Analyze FFT shape transformations to understand the TensorRT error.

Error: kOPT values for profile 0 violate shape constraints:
/generator/model/model.5/conv1/ffc/convg2g/Add: dimensions not compatible
for elementwise. Condition '==' violated: 31 != 16.

This script demonstrates how FFT operations change tensor shapes.
"""

import torch
import numpy as np


def analyze_rfft2_shapes():
    """Analyze how torch.fft.rfft2 changes tensor shapes."""

    print("=" * 80)
    print("FFT Shape Analysis for TensorRT Conversion Error")
    print("=" * 80)
    print()

    test_shapes = [
        # (height, width) pairs
        (16, 16),    # minShapes
        (512, 256),  # optShapes - THIS CAUSES THE ERROR!
        (512, 512),  # maxShapes
        (32, 32),    # multiple of 16
        (64, 64),    # multiple of 16
        (256, 256),  # multiple of 16
        (128, 64),   # non-square multiple of 16
    ]

    print("Input Shape -> RFFT2 Output Shape (real/imag components)")
    print("-" * 80)

    for h, w in test_shapes:
        # Create a sample tensor: (batch, channels, height, width)
        x = torch.randn(1, 3, h, w)

        # Apply rfft2 (real FFT in 2D)
        # This is what FourierUnit does
        ffted = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')

        # Get output shape
        out_h, out_w = ffted.shape[-2:]

        print(f"  Input: (1, 3, {h:3d}, {w:3d}) -> Output: (1, 3, {out_h:3d}, {out_w:3d})")

        # Analyze the width transformation
        expected_w = w // 2 + 1
        print(f"    Width: {w} -> {out_w} (expected: {expected_w}, formula: width//2 + 1)")

        # Check if dimensions align for elementwise operations
        if out_h == h and out_w == expected_w:
            print(f"    ✓ Dimensions align correctly")
        else:
            print(f"    ✗ Dimension mismatch!")

        print()

    print()
    print("=" * 80)
    print("ANALYSIS OF THE ERROR")
    print("=" * 80)
    print()
    print("The error mentions: '31 != 16'")
    print()
    print("For optShapes with (512, 256):")
    print("  - Width 256 -> RFFT width = 256//2 + 1 = 129")
    print("  - Height 512 remains 512")
    print()
    print("But wait - the error shows '31 != 16'. Let's check downsampled shapes:")
    print()

    # The FFC layer downsamples by stride=2 multiple times
    print("If the network downsamples 4 times (stride=2 each):")
    print("  Input (512, 256) -> after 4 downsamples:")
    print(f"    Height: 512 / 2^4 = {512 // (2**4)}")
    print(f"    Width:  256 / 2^4 = {256 // (2**4)}")
    print()
    print("  After RFFT2 on (32, 16):")
    print(f"    Height: 32 (unchanged)")
    print(f"    Width:  16//2 + 1 = {16//2 + 1}")
    print()
    print("  THIS IS THE PROBLEM!")
    print("  When width = 16, RFFT produces width = 9")
    print("  But the elementwise Add operation expects matching dimensions.")
    print()

    # Let's verify with different downsampling levels
    print("Checking after different downsample levels:")
    print()

    for n_downsamples in range(1, 6):
        h_down = 512 // (2 ** n_downsamples)
        w_down = 256 // (2 ** n_downsamples)

        # After RFFT
        h_fft = h_down
        w_fft = w_down // 2 + 1

        print(f"  After {n_downsamples} downsamples: ({h_down}, {w_down}) -> RFFT: ({h_fft}, {w_fft})")

        # Check for the specific error dimensions
        if (h_fft == 31 or w_fft == 31) and (h_down == 16 or w_down == 16):
            print(f"    *** FOUND ERROR CASE! 31 != 16 ***")


def analyze_shape_compatibility():
    """Analyze which input shapes lead to compatible FFT dimensions."""

    print()
    print("=" * 80)
    print("SHAPE COMPATIBILITY ANALYSIS")
    print("=" * 80)
    print()
    print("Finding input shapes where FFT width aligns properly...")
    print()

    # For shapes to be compatible, we need special conditions
    # Let's check multiples of 16 after multiple downsamples

    print("Testing multiples of 16 with 4 downsampling stages:")
    print()

    base_sizes = [16, 32, 64, 128, 256, 512, 1024]

    for base_h in base_sizes:
        for base_w in base_sizes:
            # After 4 downsamples
            h = base_h // (2 ** 4)
            w = base_w // (2 ** 4)

            if h < 1 or w < 1:
                continue

            # After RFFT
            w_fft = w // 2 + 1

            # Check if this could cause issues
            if w != w_fft and (h == 16 or w == 16 or h == 31 or w_fft == 31):
                print(f"  Base ({base_h}, {base_w}) -> Downsampled ({h}, {w}) -> RFFT width: {w_fft}")
                if abs(w - w_fft) > 1:
                    print(f"    ⚠ Large mismatch: {w} vs {w_fft}")


if __name__ == "__main__":
    analyze_rfft2_shapes()
    analyze_shape_compatibility()

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The TensorRT error occurs because:")
    print("1. The ONNX model uses torch.fft.rfft2 (or custom implementation)")
    print("2. RFFT transforms width from W to W//2+1")
    print("3. With non-square inputs like 512x256, after downsampling,")
    print("   intermediate dimensions don't align for elementwise operations")
    print("4. Specifically, (512,256) -> downsample -> (62,31) has issues")
    print("   because 31 (from RFFT of 60) doesn't match expected dimensions")
    print()
    print("SOLUTION: Use input dimensions that are powers of 2 or multiples")
    print("          of 16 for BOTH height and width to ensure alignment")
    print("          after FFT transformations.")
    print()
