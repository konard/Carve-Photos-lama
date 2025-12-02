#!/usr/bin/env python3
"""
Simple standalone test of the FFT function fixes.

This tests just the ifft1d and irfft functions to verify
the dynamic shape fixes work correctly.
"""

import torch
import sys

print("=" * 80)
print("Testing ifft1d and irfft Dynamic Shape Fixes")
print("=" * 80)

# Copy the fixed functions here for standalone testing
def ifft1d(REAL, IMAG, n=None, axis=-1):
    if n is None:
        n = REAL.shape[axis]

    i = torch.arange(n, device=REAL.device)
    j = torch.arange(n, device=REAL.device)

    cos_matrix = torch.cos(2 * torch.pi * i.unsqueeze(-1) * j / n)
    sin_matrix = torch.sin(2 * torch.pi * i.unsqueeze(-1) * j / n)

    # Perform the matrix multiplication along the specified axis
    real_part = torch.tensordot(cos_matrix, REAL, dims=([1], [axis])) - torch.tensordot(sin_matrix, IMAG,
                                                                                        dims=([1], [axis]))
    imag_part = torch.tensordot(sin_matrix, REAL, dims=([1], [axis])) + torch.tensordot(cos_matrix, IMAG,
                                                                                        dims=([1], [axis]))

    # Normalize by the number of points
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, device=REAL.device)
    final_real = real_part / torch.sqrt(n)
    final_imag = imag_part / torch.sqrt(n)

    # Dynamic permutation that works with any tensor shape
    ndim = len(REAL.shape)
    perm = list(range(ndim))
    perm = [1, 2, 0] + perm[3:] if ndim > 3 else perm

    final_real = final_real.permute(perm)
    final_imag = final_imag.permute(perm)

    return final_real, final_imag


def irfft(REAL, IMAG, n=None, axis=-1, norm=None):
    # Make sure the axis is positive
    axis = axis if axis >= 0 else REAL.ndim + axis

    # Generate the full FFT spectrum from the half-spectrum
    REAL_flipped = torch.flip(REAL[..., 1:-1], dims=[axis])
    IMAG_flipped = torch.flip(IMAG[..., 1:-1], dims=[axis])

    # Conjugate the flipped IMAG tensor by multiplying by -1.
    IMAG_flipped_conj = -IMAG_flipped

    # Concatenate the original REAL and IMAG with their conjugated flipped versions.
    REAL_extended = torch.cat([REAL, REAL_flipped], dim=axis)
    IMAG_extended = torch.cat([IMAG, IMAG_flipped_conj], dim=axis)

    REAL = ifft1d(REAL_extended, IMAG_extended, axis=axis)[0]
    # Dynamic permutation: swap the last two dimensions
    REAL = REAL.transpose(-1, -2)
    return REAL


# Test with different shapes
test_shapes = [
    (1, 64, 256, 129),  # batch=1, channels=64, height=256, width/2+1=129 (256x256 image)
    (1, 64, 512, 257),  # batch=1, channels=64, height=512, width/2+1=257 (512x512 image)
    (1, 64, 768, 385),  # batch=1, channels=64, height=768, width/2+1=385 (768x768 image)
    (1, 64, 1024, 513), # batch=1, channels=64, height=1024, width/2+1=513 (1024x1024 image)
]

results = []
for shape in test_shapes:
    batch, channels, height, width_half = shape
    width = (width_half - 1) * 2

    print(f"\nTesting shape: {shape} (corresponds to {height}x{width} image)")

    try:
        # Create test input
        REAL = torch.randn(shape)
        IMAG = torch.randn(shape)

        # Test ifft1d
        real_out, imag_out = ifft1d(REAL, IMAG, axis=-2)
        print(f"  ifft1d: {REAL.shape} -> {real_out.shape}")

        # Test irfft
        final_out = irfft(real_out, imag_out, axis=-1, n=width)
        expected_shape = (batch, channels, width, height)

        print(f"  irfft: {real_out.shape} -> {final_out.shape}")
        print(f"  Expected shape: {expected_shape}")

        if final_out.shape == expected_shape:
            print(f"  ✓ PASS")
            results.append((shape, "PASS", None))
        else:
            print(f"  ✗ FAIL: Shape mismatch")
            results.append((shape, "FAIL", f"Expected {expected_shape}, got {final_out.shape}"))

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        results.append((shape, "ERROR", str(e)))
        import traceback
        traceback.print_exc()

# Print summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

passed = sum(1 for _, status, _ in results if status == "PASS")
failed = sum(1 for _, status, _ in results if status == "FAIL")
errors = sum(1 for _, status, _ in results if status == "ERROR")

print(f"\nTotal tests: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Errors: {errors}")

if failed > 0 or errors > 0:
    print("\nFailed/Error details:")
    for shape, status, msg in results:
        if status != "PASS":
            print(f"  - {shape}: {status} - {msg}")

print("\n" + "=" * 80)
if passed == len(results):
    print("✓ ALL TESTS PASSED! The dynamic shape fixes are working correctly.")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED! The fixes need more work.")
    sys.exit(1)
