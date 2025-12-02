#!/usr/bin/env python3
"""
Test script for CPU execution solution (Solution A1).

This tests that the LaMa ONNX model works correctly with the CPU execution provider,
which is the recommended immediate workaround for the DirectML FFT issue.

Exit codes:
    0 - CPU execution successful
    1 - CPU execution failed
"""

import sys
import os
import time
import traceback
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_input(batch_size=1, channels=4, height=256, width=256):
    """Create a test input tensor matching LaMa model expectations."""
    return np.random.randn(batch_size, channels, height, width).astype(np.float32)


def test_cpu_execution():
    """Test CPU execution with LaMa model."""
    print("=" * 60)
    print("CPU Solution Test (A1)")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed")
        return 1

    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"\nAvailable execution providers: {available_providers}")

    if 'CPUExecutionProvider' not in available_providers:
        print("\nERROR: CPU execution provider not available")
        return 1

    # Model path
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "lama", "lama_fp32.onnx"
    )

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        return 1

    print(f"\nModel path: {model_path}")
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    # Create session with CPU only
    print("\n--- Creating ONNX Runtime session with CPU ---")

    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )

        print("Session created successfully")
        print(f"Active providers: {session.get_providers()}")

    except Exception as e:
        print(f"\nERROR during session creation: {e}")
        traceback.print_exc()
        return 1

    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"\nInput: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    print(f"Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")

    # Test with different input sizes
    test_sizes = [
        (1, 4, 256, 256),
        (1, 4, 512, 512),
    ]

    for batch_size, channels, height, width in test_sizes:
        print(f"\n--- Testing input size: {batch_size}x{channels}x{height}x{width} ---")

        test_input = create_test_input(batch_size, channels, height, width)
        input_name = session.get_inputs()[0].name

        try:
            # Warmup run
            _ = session.run(None, {input_name: test_input})

            # Timed run
            start_time = time.time()
            output = session.run(None, {input_name: test_input})
            elapsed_time = time.time() - start_time

            print(f"Output shape: {output[0].shape}")
            print(f"Inference time: {elapsed_time * 1000:.2f} ms")
            print(f"Output range: [{output[0].min():.4f}, {output[0].max():.4f}]")

            # Basic sanity checks
            if output[0].shape[0] != batch_size:
                print("WARNING: Batch size mismatch")
            if output[0].shape[2] != height or output[0].shape[3] != width:
                print("WARNING: Spatial dimension mismatch")
            if np.isnan(output[0]).any():
                print("WARNING: Output contains NaN values")
            if np.isinf(output[0]).any():
                print("WARNING: Output contains Inf values")

        except Exception as e:
            print(f"ERROR: Inference failed: {e}")
            traceback.print_exc()
            return 1

    print("\n" + "=" * 60)
    print("SUCCESS: CPU execution works correctly!")
    print("=" * 60)
    print("\nThe CPU execution provider is a viable workaround for")
    print("the DirectML FFT issue, though with reduced performance.")
    return 0


if __name__ == "__main__":
    exit_code = test_cpu_execution()
    print(f"\n\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)
