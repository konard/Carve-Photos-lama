#!/usr/bin/env python3
"""
Experiment script to analyze the LaMa ONNX model graph structure.

This script examines the ONNX graph to:
1. Identify MatMul operations used in FFT computation
2. Understand the tensor shapes and patterns
3. Prepare for potential graph surgery to replace MatMul FFT with native DFT

Usage:
    python experiments/analyze_lama_graph.py [--model PATH]
"""

import sys
import os
import argparse
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_model(model_path):
    """Analyze the ONNX model graph structure."""
    import onnx
    from onnx import helper, numpy_helper

    print("=" * 70)
    print("LaMa ONNX Model Graph Analysis")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    print(f"Model producer: {model.producer_name}")
    print(f"Opset version: {model.opset_import[0].version}")
    print(f"Number of nodes: {len(graph.node)}")
    print(f"Number of initializers: {len(graph.initializer)}")

    # Count operations by type
    print("\n--- Operation Type Counts ---")
    op_counts = defaultdict(int)
    for node in graph.node:
        op_counts[node.op_type] += 1

    for op_type, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op_type:25s}: {count:4d}")

    # Find MatMul nodes related to FFT
    print("\n--- MatMul Nodes in FFT Path ---")
    fft_matmuls = []
    for node in graph.node:
        if node.op_type == "MatMul":
            # Check if node name suggests FFT computation
            if "rttn" in node.name.lower() or "fft" in node.name.lower():
                fft_matmuls.append(node)
                print(f"\nNode: {node.name}")
                print(f"  Inputs: {list(node.input)}")
                print(f"  Outputs: {list(node.output)}")

    print(f"\nTotal FFT-related MatMul nodes: {len(fft_matmuls)}")

    # Find nodes with 'fu' (FourierUnit) or 'rttn' in name
    print("\n--- Fourier Unit Related Nodes ---")
    fourier_nodes = []
    for node in graph.node:
        if "fu" in node.name.lower() or "rttn" in node.name.lower() or "fourier" in node.name.lower():
            fourier_nodes.append(node)

    print(f"Found {len(fourier_nodes)} Fourier-related nodes:")
    for node in fourier_nodes[:20]:  # Limit output
        print(f"  {node.op_type:15s}: {node.name}")

    if len(fourier_nodes) > 20:
        print(f"  ... and {len(fourier_nodes) - 20} more")

    # Analyze initializers (weights, constants)
    print("\n--- FFT-related Initializers ---")
    fft_initializers = []
    for init in graph.initializer:
        if "cos" in init.name.lower() or "sin" in init.name.lower() or "dft" in init.name.lower():
            fft_initializers.append(init)
            arr = numpy_helper.to_array(init)
            print(f"\n{init.name}")
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")
            print(f"  Range: [{arr.min():.6f}, {arr.max():.6f}]")

    # Find input/output nodes for FourierUnit
    print("\n--- FourierUnit Subgraph Boundaries ---")

    # Look for the pattern: input -> rfft -> conv -> bn -> relu -> irfft -> output
    # In the custom implementation, rfft uses MatMul with cos/sin matrices

    # Find all Conv nodes that might be in the Fourier path
    conv_in_fourier = []
    for node in graph.node:
        if node.op_type == "Conv" and "fu" in node.name.lower():
            conv_in_fourier.append(node)
            print(f"\nConv in FourierUnit: {node.name}")
            print(f"  Inputs: {list(node.input)}")
            print(f"  Outputs: {list(node.output)}")

    # Summary for graph surgery
    print("\n" + "=" * 70)
    print("SUMMARY FOR GRAPH SURGERY (Solution B1.5)")
    print("=" * 70)

    print(f"""
Analysis Results:
-----------------
1. Total MatMul operations: {op_counts.get('MatMul', 0)}
2. FFT-related MatMul operations: {len(fft_matmuls)}
3. Fourier-related nodes: {len(fourier_nodes)}
4. FFT initializers (cos/sin matrices): {len(fft_initializers)}

The MatMul-based FFT implementation uses precomputed cos/sin matrices
to perform DFT via matrix multiplication. To replace with native DFT:

1. Identify the subgraph boundaries:
   - Inputs to FFT computation
   - Outputs from FFT computation

2. Replace the MatMul subgraph with a single DFT node:
   - ONNX DFT operator (opset 17+)
   - Attributes: axis, inverse, onesided

3. Handle the 2D case:
   - Original: rfft2d = fft_dim(rfft_dim(x, -1), -2)
   - May need two DFT nodes for 2D transform

Note: Graph surgery is complex due to the custom implementation
mixing real/imaginary parts through separate tensors.
""")

    return model, fft_matmuls, fourier_nodes


def find_fft_pattern(graph):
    """Attempt to identify the FFT pattern in the graph."""
    print("\n--- Attempting to identify FFT pattern ---")

    # The custom FFT implementation follows this pattern:
    # 1. cos_part = torch.cos(-2*pi*n*k/N)
    # 2. sin_part = torch.sin(-2*pi*n*k/N)
    # 3. real_part = matmul(x, cos_part)
    # 4. imag_part = matmul(x, sin_part)

    # Look for this pattern by finding consecutive MatMuls
    # with Cos/Sin initializers

    # This is a placeholder for more sophisticated pattern matching
    print("Pattern matching for graph surgery would require:")
    print("  1. Building a dependency graph")
    print("  2. Identifying MatMul nodes with cos/sin weights")
    print("  3. Tracing back to find common input")
    print("  4. Tracing forward to find where outputs combine")
    print("  5. Replacing the entire subgraph")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze LaMa ONNX model graph')
    parser.add_argument('--model', type=str,
                        default='models/lama/lama_fp32.onnx',
                        help='Path to ONNX model')
    args = parser.parse_args()

    # Try to import onnx
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("ERROR: onnx package not installed")
        print("Install with: pip install onnx")
        sys.exit(1)

    analyze_model(args.model)
    print("\nAnalysis complete!")
