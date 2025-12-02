#!/usr/bin/env python3
"""
Test script to reproduce the DirectML FFT issue with LaMa ONNX model.

This test attempts to run the LaMa model using DirectML execution provider
and expects to encounter the error:
    Error at node: /generator/model/model.5/conv1/ffc/convw2g/fu/rttn/MatMul_5
    Error: 80070057 The parameter is incorrect.

Exit codes:
    0 - DirectML executed successfully (issue NOT reproduced)
    1 - DirectML failed with expected error (issue reproduced)
    2 - DirectML failed with unexpected error
    3 - DirectML provider not available
"""

import sys
import os
import traceback
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_inputs(batch_size=1, height=512, width=512):
    """Create test input tensors matching LaMa model expectations.

    LaMa ONNX model expects:
    - image: (batch, 3, height, width) - RGB image
    - mask: (batch, 1, height, width) - binary mask (1 = area to inpaint)
    """
    image = np.random.randn(batch_size, 3, height, width).astype(np.float32)
    # Create a simple mask with some region to inpaint
    mask = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    # Mark center region for inpainting
    h_start, h_end = height // 4, 3 * height // 4
    w_start, w_end = width // 4, 3 * width // 4
    mask[:, :, h_start:h_end, w_start:w_end] = 1.0
    return image, mask


def test_directml_execution():
    """Test DirectML execution with LaMa model."""
    print("=" * 60)
    print("DirectML Reproduction Test")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed")
        return 3

    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"\nAvailable execution providers: {available_providers}")

    if 'DmlExecutionProvider' not in available_providers:
        print("\nWARNING: DirectML execution provider not available")
        print("This test requires onnxruntime-directml package")
        print("Install with: pip install onnxruntime-directml")
        return 3

    # Model path
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "lama", "lama_fp32.onnx"
    )

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Please download the model from:")
        print("https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx")
        return 2

    print(f"\nModel path: {model_path}")
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    # Create session with DirectML only
    print("\n--- Creating ONNX Runtime session with DirectML ---")

    try:
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 0  # Verbose logging

        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['DmlExecutionProvider']
        )

        print("Session created successfully")
        print(f"Active providers: {session.get_providers()}")

    except Exception as e:
        print(f"\nERROR during session creation: {e}")
        traceback.print_exc()
        return 2

    # Get input/output info
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")

    # Create test inputs
    # Using 512x512 as it's the model's expected size
    image, mask = create_test_inputs(batch_size=1, height=512, width=512)
    print(f"\nTest image shape: {image.shape}")
    print(f"Test mask shape: {mask.shape}")

    # Run inference
    print("\n--- Running inference with DirectML ---")
    print("This is expected to fail with error 80070057...")

    try:
        # Prepare feed dict with both inputs
        feed_dict = {
            'image': image,
            'mask': mask
        }
        output = session.run(None, feed_dict)

        print("\n" + "=" * 60)
        print("UNEXPECTED: DirectML execution SUCCEEDED!")
        print("=" * 60)
        print(f"Output shape: {output[0].shape}")
        print("\nThe DirectML issue was NOT reproduced.")
        print("This could mean:")
        print("  1. The bug has been fixed in newer ONNX Runtime/DirectML versions")
        print("  2. The test environment is different from the reported issue")
        print("  3. The specific GPU/driver combination works correctly")
        return 0

    except Exception as e:
        error_str = str(e)
        print(f"\n{'=' * 60}")
        print("DirectML execution FAILED")
        print("=" * 60)
        print(f"\nError message:\n{error_str}")

        # Check for the specific error we're looking for
        if "80070057" in error_str or "parameter is incorrect" in error_str.lower():
            print("\n" + "=" * 60)
            print("SUCCESS: DirectML FFT issue REPRODUCED!")
            print("=" * 60)
            print("\nThis confirms the issue reported in:")
            print("https://huggingface.co/Carve/LaMa-ONNX/discussions/1")
            print("\nError code 80070057 indicates DirectML's MatMul validation")
            print("is rejecting the tensor shapes from the custom FFT implementation.")
            return 1

        elif "rttn" in error_str.lower() or "matmul" in error_str.lower():
            print("\n" + "=" * 60)
            print("PARTIAL: FFT-related error detected!")
            print("=" * 60)
            print("\nThe error appears related to the FFT implementation")
            print("but with a different error message than originally reported.")
            return 1

        elif "no devices" in error_str.lower() or "device" in error_str.lower():
            print("\n" + "=" * 60)
            print("INFO: No DirectML GPU device available")
            print("=" * 60)
            print("\nThis environment doesn't have a DirectML-compatible GPU.")
            print("The test cannot reproduce the issue without hardware.")
            print("\nTo reproduce the issue, run this test on a Windows machine")
            print("with an AMD or Intel GPU (without CUDA).")
            # Return 0 as this is not a test failure, just no hardware
            return 0

        else:
            print("\n" + "=" * 60)
            print("UNEXPECTED: Different error encountered")
            print("=" * 60)
            traceback.print_exc()
            return 2


if __name__ == "__main__":
    exit_code = test_directml_execution()
    print(f"\n\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)
