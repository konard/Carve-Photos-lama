# Root Cause Analysis: torch.dynamo_export Limitations

## Executive Summary

The `torch.dynamo_export` function was introduced as PyTorch's next-generation ONNX exporter with promises of native FFT support. However, the Carve.Photos team discovered that models exported with dynamo_export exhibit critical limitations that make them unsuitable for production use. This document analyzes the root causes of these limitations.

## Problem Statement

Models exported using `torch.dynamo_export` with FFT operations suffer from:
1. Low runtime compatibility - doesn't work with ONNX converters
2. Poor performance - significantly slower than traditionally exported models
3. No GPU support - unable to utilize GPU acceleration
4. Architecture differences - structural differences visible in model inspection tools

## Root Cause #1: Low Runtime Compatibility

### Symptoms
- Exported ONNX models cannot be converted to other formats (TensorRT, CoreML, MNN, etc.)
- ONNX converters fail or produce invalid models
- Limited to specific ONNX runtime versions

### Technical Root Cause

**Hypothesis: Non-standard ONNX operator usage**

dynamo_export likely uses one or more of the following approaches that break compatibility:

1. **Custom Operators**: Exports FFT as custom ONNX operators that other runtimes don't recognize
   ```
   # Instead of standard ONNX DFT op
   domain: "torch.onnx.dynamo"
   op_type: "aten_fft_rfftn"  # Non-standard!
   ```

2. **Opset Version Mismatch**: Uses bleeding-edge ONNX opset features not widely supported
   - ONNX opset 17+ DFT operators may not be implemented in all runtimes
   - Converters may target older opset versions

3. **Graph Structure**: Produces complex subgraph patterns that converters can't optimize or translate
   - Nested control flow
   - Dynamic shapes in incompatible ways
   - Excessive use of symbolic shape inference

### Evidence
- OPHoperHPO: "lack of support in other ONNX converters"
- PyTorch Issue #133785: "Failed to export the model to ONNX"
- Requires "modifying internal torch and onnxscript code"

### Why Traditional Export Works
Traditional export with custom FFT uses only basic ONNX operators:
- MatMul (widely supported since opset 1)
- Sin, Cos (mathematical functions, opset 7+)
- Concat, Transpose (fundamental operations)
- All converters understand these primitives

## Root Cause #2: Poor Performance (Low Speed)

### Symptoms
- Exported models run significantly slower than traditionally exported models
- Performance degradation observed even when models run

### Technical Root Cause

**Hypothesis: Unoptimized operator decomposition**

dynamo_export appears to decompose FFT operations inefficiently:

1. **No Runtime Optimization**: ONNX runtime can't recognize the pattern as FFT
   ```
   Traditional Custom FFT:
   Input → MatMul (cos basis) → ... → MatMul (IFFT) → Output
   ↓
   Runtime sees: "generic matrix operations"
   ✓ Optimizes with BLAS libraries (cuBLAS, MKL)

   dynamo_export:
   Input → [complex graph of primitives] → Output
   ↓
   Runtime sees: "unfamiliar pattern"
   ✗ Cannot apply FFT-specific optimizations
   ✗ Cannot fuse operations efficiently
   ```

2. **Excessive Graph Fragmentation**: FFT decomposed into many small operations
   - Each operation has kernel launch overhead
   - Prevents fusion and vectorization
   - Memory bandwidth limited by excessive intermediate tensors

3. **Missing Constant Folding**: Dynamic computation of FFT basis functions
   - Traditional approach precomputes sin/cos matrices
   - dynamo might recompute these at runtime

4. **Suboptimal Memory Layout**: Data transformations between operations
   - Extra transpose/reshape operations
   - Cache-unfriendly memory access patterns

### Why Traditional Export Performs Better

Custom FFT implementation:
- Uses large matrix multiplications → optimized by BLAS
- Precomputes basis functions (cos/sin matrices)
- Minimizes graph fragmentation
- Predictable memory access patterns

Even though custom FFT is O(N³) vs O(N log N):
- BLAS-optimized MatMul is highly efficient
- Runs on optimized hardware (TensorCores on GPU)
- Better than unoptimized O(N log N) decomposition

## Root Cause #3: No GPU Support

### Symptoms
- Inability to use the model on GPU
- Models may fail to run on GPU or fall back to CPU

### Technical Root Cause

**Hypothesis: Operator placement and kernel availability**

1. **Missing GPU Kernels**: Custom/experimental operators lack GPU implementations
   ```
   torch.onnx.dynamo exports → "aten::fft_rfftn" custom op
   ↓
   ONNX Runtime searches for GPU kernel
   ↓
   Kernel not found → fallback to CPU or fail
   ```

2. **Mixed Device Execution**: Some operators on GPU, FFT on CPU
   - Excessive CPU↔GPU transfers
   - Synchronization overhead
   - Appears as "no GPU support" to user

3. **Runtime Provider Limitations**:
   - ONNX Runtime's CUDA execution provider doesn't support the exported operations
   - TensorRT execution provider can't parse the graph

### Why Traditional Export Has GPU Support

- Uses only standard ONNX operators
- All operators (MatMul, Sin, Cos, etc.) have GPU implementations
- ONNX Runtime can execute entire graph on GPU
- No CPU fallback required

## Root Cause #4: Architecture Differences

### Symptoms
- Netron visualization shows "significant differences in architecture"
- Graph structure doesn't match expected pattern

### Technical Root Cause

**Hypothesis: Over-eager graph transformation**

dynamo_export performs aggressive graph transformations:

1. **Decomposition vs Preservation**:
   ```
   Traditional Export:
   FourierUnitJIT.forward()
     ├─ MatMul (cos)
     ├─ MatMul (sin)
     └─ [clean, recognizable structure]

   dynamo_export:
   [Heavily transformed graph]
     ├─ Aten ops decomposed
     ├─ Additional nodes from tracing
     ├─ Symbolic shape computation nodes
     └─ [unrecognizable structure]
   ```

2. **Trace Artifacts**: Dynamo traces through internal PyTorch operations
   - Captures implementation details not visible to traditional export
   - Includes optimization artifacts (autograd, JIT compilation)
   - Results in "correct but different" graph

3. **Incomplete Simplification**: Graph transformations don't fully simplify
   - Constant folding incomplete
   - Dead code not eliminated
   - Redundant operations remain

### Impact

While architecturally different graphs can be functionally equivalent:
- Harder to debug and validate
- More difficult for runtimes to optimize
- May trigger bugs in downstream tools
- Reduces user confidence in correctness

## Underlying System Causes

### 1. Maturity Gap

**Traditional torch.onnx.export**:
- Years of production use and bug fixes
- Extensive operator coverage
- Well-tested with various runtimes

**torch.dynamo_export**:
- Relatively new (introduced ~2022-2023)
- Still experimental for complex operations
- Limited production validation

### 2. Design Philosophy Mismatch

**Traditional Export Philosophy**:
- "Export what you see" - high-level operations
- Manual operator mapping
- Predictable output

**Dynamo Export Philosophy**:
- "Trace and transform" - capture everything
- Automatic decomposition
- Optimizes for PyTorch execution graph

The dynamo approach creates a gap: PyTorch's internal representations don't map cleanly to ONNX runtime expectations.

### 3. Ecosystem Fragmentation

ONNX ecosystem has multiple players:
- ONNX spec (standards body)
- ONNX Runtime (Microsoft)
- TensorRT (NVIDIA)
- CoreML (Apple)
- Various converters

dynamo_export optimizes for ONNX spec compliance but doesn't account for:
- Runtime-specific quirks
- Converter limitations
- Performance characteristics

Traditional export has been battle-tested across this ecosystem.

## Synthesis: Why Traditional Export Works Better

| Aspect | Traditional + Custom FFT | dynamo_export |
|--------|-------------------------|---------------|
| **Operators** | Standard ONNX ops only | Custom/experimental ops |
| **Optimization** | BLAS-optimized MatMul | Unoptimized decomposition |
| **GPU Support** | All ops have GPU kernels | Missing GPU kernels |
| **Compatibility** | Works with all converters | Limited converter support |
| **Maturity** | Battle-tested (2018+) | Experimental (2023+) |
| **Graph Structure** | Clean, recognizable | Complex, transformed |

## Recommendations

### For LaMa Project (Short Term)
1. ✅ Continue using traditional export with FourierUnitJIT
2. ✅ Document the reasons for this approach
3. ✅ Monitor PyTorch releases but don't switch prematurely

### For PyTorch Team
1. Add GPU kernel implementations for FFT in dynamo_export path
2. Improve operator fusion to match performance of traditional export
3. Ensure output is compatible with major ONNX converters (TensorRT, CoreML)
4. Provide migration guide from traditional to dynamo export

### For ML Engineers
1. Don't assume newer PyTorch features are production-ready
2. Test exported models across target deployment platforms
3. Benchmark performance before switching export methods
4. Keep traditional export as fallback option

## Validation of Analysis

### Evidence Supporting This Analysis

1. **Primary Source**: OPHoperHPO's direct experience testing dynamo_export
2. **Community Reports**: Multiple PyTorch issues describing similar problems
3. **Technical Plausibility**: Root causes align with known ONNX ecosystem challenges
4. **Outcome**: Traditional export demonstrably works in production

### Confidence Levels

- **Runtime Compatibility Issues**: High confidence (directly reported)
- **Performance Issues**: High confidence (directly reported)
- **GPU Support Issues**: High confidence (directly reported)
- **Specific Technical Mechanisms**: Medium confidence (inferred from symptoms)

### What We Don't Know

- Exact ONNX operators used by dynamo_export for FFT
- Specific runtime errors encountered
- Detailed performance metrics (quantified slowdown)
- PyTorch team's current progress on fixes

## Conclusion

The root causes of torch.dynamo_export limitations stem from:
1. **Immaturity**: New exporter not yet production-ready
2. **Design Tradeoffs**: Optimization for PyTorch ↔ ONNX mismatch
3. **Ecosystem Gaps**: Incomplete support across ONNX runtimes and converters
4. **Performance Regression**: Graph transformations prevent optimization

The traditional export + custom FFT approach succeeds because it uses battle-tested, widely-supported ONNX operators that run efficiently across the ecosystem. While less elegant than native FFT support, it provides reliability, compatibility, and acceptable performance for production use.
