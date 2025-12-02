#!/usr/bin/env python3
"""
Test script for Hybrid execution solution (Solution A2).

This tests using DirectML as primary provider with CPU fallback.
The idea is that compatible operations run on GPU while incompatible
FFT operations automatically fall back to CPU.

Exit codes:
    0 - Hybrid execution successful
    1 - Hybrid execution failed
    2 - DirectML not available (skipped)
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


def test_hybrid_execution():
    """Test hybrid DirectML + CPU execution with LaMa model."""
    print("=" * 60)
    print("Hybrid Solution Test (A2)")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed")
        return 1

    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"\nAvailable execution providers: {available_providers}")

    if 'DmlExecutionProvider' not in available_providers:
        print("\nWARNING: DirectML execution provider not available")
        print("This test requires onnxruntime-directml package")
        print("Skipping hybrid test...")
        return 2

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

    # Create session with DirectML + CPU fallback
    print("\n--- Creating ONNX Runtime session with DirectML + CPU fallback ---")

    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # DirectML as primary, CPU as fallback
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers
        )

        print("Session created successfully")
        print(f"Requested providers: {providers}")
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

    # Test inference
    test_input = create_test_input(batch_size=1, channels=4, height=256, width=256)
    print(f"\nTest input shape: {test_input.shape}")

    print("\n--- Running inference with Hybrid execution ---")

    try:
        input_name = session.get_inputs()[0].name

        # Warmup run
        _ = session.run(None, {input_name: test_input})

        # Timed run
        start_time = time.time()
        output = session.run(None, {input_name: test_input})
        elapsed_time = time.time() - start_time

        print(f"\nOutput shape: {output[0].shape}")
        print(f"Inference time: {elapsed_time * 1000:.2f} ms")
        print(f"Output range: [{output[0].min():.4f}, {output[0].max():.4f}]")

        # Basic sanity checks
        if np.isnan(output[0]).any():
            print("WARNING: Output contains NaN values")
        if np.isinf(output[0]).any():
            print("WARNING: Output contains Inf values")

        print("\n" + "=" * 60)
        print("SUCCESS: Hybrid execution works!")
        print("=" * 60)
        print("\nThe hybrid approach (DirectML + CPU fallback) successfully")
        print("executed the model. This suggests ONNX Runtime may be")
        print("automatically falling back to CPU for incompatible operations.")
        return 0

    except Exception as e:
        error_str = str(e)
        print(f"\n{'=' * 60}")
        print("Hybrid execution FAILED")
        print("=" * 60)
        print(f"\nError message:\n{error_str}")

        if "80070057" in error_str or "parameter is incorrect" in error_str.lower():
            print("\nThe DirectML FFT issue still occurs even with CPU fallback.")
            print("This suggests the fallback mechanism doesn't work for this error.")
        else:
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    exit_code = test_hybrid_execution()
    print(f"\n\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)
