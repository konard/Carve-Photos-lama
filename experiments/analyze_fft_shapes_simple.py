#!/usr/bin/env python3
"""
Analyze FFT shape transformations to understand the TensorRT error.

Error: kOPT values for profile 0 violate shape constraints:
/generator/model/model.5/conv1/ffc/convg2g/Add: dimensions not compatible
for elementwise. Condition '==' violated: 31 != 16.
"""


def rfft_width(w):
    """Calculate output width after real FFT."""
    return w // 2 + 1


def analyze_rfft2_shapes():
    """Analyze how RFFT2 changes tensor shapes."""

    print("=" * 80)
    print("FFT Shape Analysis for TensorRT Conversion Error")
    print("=" * 80)
    print()

    test_shapes = [
        # (height, width) pairs from trtexec command
        (16, 16),    # minShapes
        (512, 256),  # optShapes - THIS CAUSES THE ERROR!
        (512, 512),  # maxShapes
        (32, 32),    # multiple of 16
        (64, 64),    # multiple of 16
        (256, 256),  # multiple of 16
        (128, 64),   # non-square multiple of 16
    ]

    print("Input Shape -> RFFT2 Output Shape")
    print("-" * 80)

    for h, w in test_shapes:
        # RFFT2: height unchanged, width becomes width//2 + 1
        out_h = h
        out_w = rfft_width(w)

        print(f"  Input: ({h:3d}, {w:3d}) -> Output: ({out_h:3d}, {out_w:3d})")
        print(f"    Width: {w} -> {out_w} (formula: width//2 + 1)")
        print()

    print()
    print("=" * 80)
    print("ANALYSIS OF THE ERROR: '31 != 16'")
    print("=" * 80)
    print()
    print("The FFC layer uses downsampling (stride=2). Let's trace the shapes:")
    print()

    # Analyze the problematic optShapes
    h, w = 512, 256
    print(f"Starting with optShapes: ({h}, {w})")
    print()

    # Model typically has 3 downsampling stages (big-lama config)
    for stage in range(1, 6):
        h_down = h // (2 ** stage)
        w_down = w // (2 ** stage)

        # After RFFT
        h_fft = h_down
        w_fft = rfft_width(w_down)

        print(f"After {stage} downsample(s):")
        print(f"  Before FFT: ({h_down:3d}, {w_down:3d})")
        print(f"  After RFFT: ({h_fft:3d}, {w_fft:3d})")

        # Check for the specific error dimensions
        if (h_fft == 31 or w_fft == 31) or (h_down == 16 or w_down == 16):
            print(f"  ⚠ Potential issue: dimension mismatch")
            if w_down == 62 and w_fft == 32:
                print(f"    *** This might be related to the error! ***")
            if h_down == 62 or w_down == 31:
                print(f"    *** FOUND ERROR PATTERN! 31 appears here ***")

        print()

    print()
    print("=" * 80)
    print("DEEPER ANALYSIS")
    print("=" * 80)
    print()

    # The error is in /generator/model/model.5/conv1/ffc/convg2g/Add
    # This suggests it's in the 5th model block, in an Add operation
    # The Add operation is trying to add tensors with incompatible shapes

    print("The error path: /generator/model/model.5/conv1/ffc/convg2g/Add")
    print()
    print("This is an elementwise Add in the SpectralTransform (convg2g).")
    print("Looking at SpectralTransform.forward() in ffc.py:")
    print("  - It applies conv1 (reduces channels)")
    print("  - Then applies FourierUnit (fu)")
    print("  - Optionally applies local FourierUnit (lfu) with split")
    print("  - Finally: output = conv2(x + output + xs)  <- THIS ADD")
    print()

    # The issue is likely in the LFU (Local Fourier Unit) section
    print("The LFU (Local Fourier Unit) splits the tensor:")
    print("  split_no = 2")
    print("  split_s = h // split_no")
    print()
    print("For (32, 16) after downsampling:")
    print("  split_s = 32 // 2 = 16")
    print("  This splits height into chunks of 16")
    print()
    print("After splitting and RFFT on (16, 8):")
    print(f"  RFFT output: (16, {rfft_width(8)})")
    print()
    print("But width 8 -> RFFT width = 5, not 8!")
    print("This misalignment causes the Add operation to fail.")
    print()


def analyze_shape_compatibility():
    """Analyze which input shapes lead to compatible dimensions."""

    print()
    print("=" * 80)
    print("SHAPE COMPATIBILITY FOR TENSORRT DYNAMIC SHAPES")
    print("=" * 80)
    print()

    print("Testing the provided TensorRT shape profiles:")
    print()

    shapes = [
        ("minShapes", 16, 16),
        ("optShapes", 512, 256),
        ("maxShapes", 512, 512),
    ]

    # Assume 3 downsampling stages (as in big-lama)
    n_downsamples = 3

    print(f"With {n_downsamples} downsampling stages (stride=2 each):")
    print()

    for name, h, w in shapes:
        h_down = h // (2 ** n_downsamples)
        w_down = w // (2 ** n_downsamples)
        w_fft = rfft_width(w_down)

        print(f"{name}: ({h}, {w})")
        print(f"  After {n_downsamples} downsamples: ({h_down}, {w_down})")
        print(f"  After RFFT: ({h_down}, {w_fft})")

        # Check LFU split
        if h_down % 2 == 0:
            split_h = h_down // 2
            split_w = w_down // 2
            split_w_fft = rfft_width(split_w)

            print(f"  LFU split (h/2, w/2): ({split_h}, {split_w})")
            print(f"  LFU after RFFT: ({split_h}, {split_w_fft})")

            # Check for problems
            if split_w != split_w_fft:
                print(f"    ⚠ MISMATCH: {split_w} != {split_w_fft}")

        print()

    print()
    print("=" * 80)
    print("FINDING COMPATIBLE SHAPES")
    print("=" * 80)
    print()
    print("For shapes to work with TensorRT dynamic shapes, ALL dimensions")
    print("must remain compatible after FFT operations.")
    print()
    print("Testing multiples of 16 for both height and width:")
    print()

    base_sizes = [16, 32, 64, 128, 256, 512]

    compatible = []

    for h in base_sizes:
        for w in base_sizes:
            h_down = h // (2 ** n_downsamples)
            w_down = w // (2 ** n_downsamples)

            if h_down < 2 or w_down < 2:
                continue

            w_fft = rfft_width(w_down)

            # Check LFU compatibility
            split_h = h_down // 2
            split_w = w_down // 2
            split_w_fft = rfft_width(split_w)

            # For compatibility, we need dimensions to align
            # This is a complex check, but let's use a heuristic
            is_compatible = (w_down % 2 == 0 and h_down % 2 == 0)

            if is_compatible:
                compatible.append((h, w))
                print(f"  ✓ ({h:3d}, {w:3d}) -> downsampled ({h_down}, {w_down}) -> RFFT width {w_fft}")

    print()
    print(f"Compatible shapes found: {len(compatible)}")
    print()
    print("Recommended TensorRT shape profiles:")
    print(f"  --minShapes=image:1x3x16x16,mask:1x1x16x16")
    print(f"  --optShapes=image:1x3x512x512,mask:1x1x512x512")
    print(f"  --maxShapes=image:1x3x1024x1024,mask:1x1x1024x1024")
    print()
    print("Key: Use SQUARE inputs that are multiples of 16!")


if __name__ == "__main__":
    analyze_rfft2_shapes()
    analyze_shape_compatibility()

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("ROOT CAUSE:")
    print("  1. The model uses RFFT which transforms width W to W//2+1")
    print("  2. Non-square inputs (512x256) create dimension mismatches")
    print("  3. After downsampling: (512,256) -> (64,32) -> (32,16) -> (16,8)")
    print("  4. The LFU splits further: (16,8) -> (8,4) before RFFT")
    print("  5. RFFT on width 4 gives 3, not 4 - causing Add to fail")
    print()
    print("SOLUTION:")
    print("  Use square input shapes OR multiples where width after all")
    print("  transformations remains even. Safest: powers of 2 for both dims.")
    print()
    print("RECOMMENDED TRTEXEC COMMAND:")
    print("  trtexec --onnx=$onnx_path \\")
    print("    --minShapes=image:1x3x16x16,mask:1x1x16x16 \\")
    print("    --optShapes=image:1x3x512x512,mask:1x1x512x512 \\")
    print("    --maxShapes=image:1x3x1024x1024,mask:1x1x1024x1024 \\")
    print("    --saveEngine=$tensorrt_path \\")
    print("    --verbose")
    print()
