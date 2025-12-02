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


def create_test_input(batch_size=1, channels=4, height=256, width=256):
    """Create a test input tensor matching LaMa model expectations."""
    # LaMa expects NCHW format with 4 channels (RGB + mask)
    return np.random.randn(batch_size, channels, height, width).astype(np.float32)


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
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"\nInput: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    print(f"Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")

    # Create test input
    # The model expects dynamic batch size and spatial dimensions
    # Using a small size for faster testing
    test_input = create_test_input(batch_size=1, channels=4, height=256, width=256)
    print(f"\nTest input shape: {test_input.shape}")

    # Run inference
    print("\n--- Running inference with DirectML ---")
    print("This is expected to fail with error 80070057...")

    try:
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: test_input})

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
        print("DirectML execution FAILED (as expected)")
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
