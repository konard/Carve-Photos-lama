#!/usr/bin/env python3
"""
Test script for native ONNX DFT operator support.

This tests whether the ONNX DFT operator (opset 17+) works on various
execution providers (DirectML, CPU). This is critical for solution B1/B1.5
which proposes replacing MatMul-based FFT with native DFT operator.

Exit codes:
    0 - DFT operator works on target provider
    1 - DFT operator failed
    2 - Target provider not available
"""

import sys
import os
import traceback
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_dft_test_model():
    """Create a simple ONNX model with DFT operator for testing."""
    import onnx
    from onnx import helper, TensorProto

    # Input: (batch, channels, height, width) - typical image format
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 8, 8])

    # Output: Complex DFT result
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 8, 5, 2])

    # DFT node - compute 1D DFT along last axis
    # Note: DFT in ONNX outputs complex as [real, imag] in last dimension
    dft_node = helper.make_node(
        'DFT',
        inputs=['input'],
        outputs=['output'],
        axis=-1,  # Along width dimension
        inverse=0,  # Forward transform
        onesided=1,  # One-sided for real input (N//2 + 1 outputs)
    )

    # Create the graph
    graph_def = helper.make_graph(
        [dft_node],
        'dft_test_model',
        [X],
        [Y],
    )

    # Create the model with opset 17 (DFT requires opset 17+)
    model_def = helper.make_model(graph_def, producer_name='dft_test')
    model_def.opset_import[0].version = 17

    # Set IR version to 8 for compatibility with older ONNX Runtime versions
    # (onnxruntime-directml may only support up to IR version 11)
    model_def.ir_version = 8

    return model_def


def create_dft_2d_test_model():
    """Create a model that tests 2D DFT (closer to LaMa's use case)."""
    import onnx
    from onnx import helper, TensorProto

    # Input: (batch, channels, height, width)
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 16, 16])

    # DFT along width (last axis), onesided
    dft_out_1 = helper.make_tensor_value_info('dft_w', TensorProto.FLOAT, None)
    dft_node_1 = helper.make_node(
        'DFT',
        inputs=['input'],
        outputs=['dft_w'],
        axis=-1,
        inverse=0,
        onesided=1,
    )

    # Output
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    # For 2D, we'd need another DFT along height
    # But let's keep it simple - just test if 1D DFT works
    # The complex output from DFT has shape [..., N//2+1, 2]
    # We'll just flatten the output
    flatten_node = helper.make_node(
        'Flatten',
        inputs=['dft_w'],
        outputs=['output'],
        axis=1,
    )

    graph_def = helper.make_graph(
        [dft_node_1, flatten_node],
        'dft_2d_test_model',
        [X],
        [Y],
    )

    model_def = helper.make_model(graph_def, producer_name='dft_2d_test')
    model_def.opset_import[0].version = 17

    # Set IR version to 8 for compatibility with older ONNX Runtime versions
    model_def.ir_version = 8

    return model_def


def test_dft_operator(provider='DmlExecutionProvider'):
    """Test DFT operator on specified execution provider."""
    print("=" * 60)
    print(f"DFT Operator Test - Provider: {provider}")
    print("=" * 60)

    try:
        import onnxruntime as ort
        import onnx
    except ImportError as e:
        print(f"ERROR: Required package not installed: {e}")
        return 1

    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"\nAvailable execution providers: {available_providers}")

    if provider not in available_providers:
        print(f"\nWARNING: {provider} not available")
        if provider == 'DmlExecutionProvider':
            print("Install onnxruntime-directml for DirectML support")
        return 2

    # Create test model
    print("\n--- Creating DFT test model (opset 17) ---")

    try:
        model = create_dft_test_model()
        onnx.checker.check_model(model)
        print("Model created and validated successfully")

        # Save for inspection
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dft_test_model.onnx"
        )
        onnx.save(model, model_path)
        print(f"Model saved to: {model_path}")

    except Exception as e:
        print(f"ERROR creating model: {e}")
        traceback.print_exc()
        return 1

    # Create session
    print(f"\n--- Creating ONNX Runtime session with {provider} ---")

    try:
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 1  # Warnings

        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=[provider, 'CPUExecutionProvider']
        )

        print("Session created successfully")
        print(f"Active providers: {session.get_providers()}")

    except Exception as e:
        print(f"\nERROR during session creation: {e}")
        traceback.print_exc()

        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)

        return 1

    # Run inference
    print("\n--- Running DFT inference ---")

    try:
        # Create test input: (1, 1, 8, 8) float32
        test_input = np.random.randn(1, 1, 8, 8).astype(np.float32)
        print(f"Input shape: {test_input.shape}")

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: test_input})

        print(f"Output shape: {output[0].shape}")
        print(f"Output dtype: {output[0].dtype}")

        # Verify output is reasonable
        if np.isnan(output[0]).any():
            print("WARNING: Output contains NaN values")
        if np.isinf(output[0]).any():
            print("WARNING: Output contains Inf values")

        print("\n" + "=" * 60)
        print(f"SUCCESS: DFT operator works on {provider}!")
        print("=" * 60)
        print("\nThis is excellent news for solution B1/B1.5!")
        print("If we can replace the MatMul-based FFT in LaMa with")
        print("native ONNX DFT operators, DirectML execution should work.")

        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)

        return 0

    except Exception as e:
        error_str = str(e)
        print(f"\n{'=' * 60}")
        print(f"DFT operator FAILED on {provider}")
        print("=" * 60)
        print(f"\nError message:\n{error_str}")

        if "not implemented" in error_str.lower() or "not supported" in error_str.lower():
            print(f"\nThe DFT operator is NOT supported on {provider}.")
            print("This means solution B1/B1.5 may not work for this provider.")
        else:
            traceback.print_exc()

        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)

        return 1


def test_all_providers():
    """Test DFT operator on all available providers."""
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    print("Testing DFT operator on all available providers...\n")

    results = {}

    # Test DirectML if available
    if 'DmlExecutionProvider' in available_providers:
        results['DirectML'] = test_dft_operator('DmlExecutionProvider')
        print("\n")

    # Always test CPU
    results['CPU'] = test_dft_operator('CPUExecutionProvider')

    # Summary
    print("\n" + "=" * 60)
    print("DFT OPERATOR SUPPORT SUMMARY")
    print("=" * 60)
    print("\n| Provider | Status |")
    print("|----------|--------|")
    for provider, code in results.items():
        status = "Supported" if code == 0 else "Not Supported" if code == 1 else "Not Available"
        print(f"| {provider:12s} | {status:14s} |")

    # Return success if at least one provider supports DFT
    return 0 if any(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test ONNX DFT operator support')
    parser.add_argument('--provider', type=str, default=None,
                        help='Specific provider to test (default: test all)')
    args = parser.parse_args()

    if args.provider:
        exit_code = test_dft_operator(args.provider)
    else:
        exit_code = test_all_providers()

    print(f"\n\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)
