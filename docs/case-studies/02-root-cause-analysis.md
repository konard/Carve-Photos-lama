# Root Cause Analysis: LaMa ONNX DirectML GPU Inference Failure

## Executive Summary

The LaMa ONNX model fails to execute on Windows using the DirectML execution provider due to incompatible tensor operations in the custom FFT (Fast Fourier Transform) implementation. The root cause is a combination of:

1. **Custom FFT implementation using MatMul operations** that produce tensor shapes incompatible with DirectML's strict validation
2. **DirectML's stricter parameter validation** compared to CPU and CUDA providers
3. **Known ONNX Runtime DirectML bugs** with dynamically-shaped tensors and certain operator configurations

## Technical Analysis

### 1. The LaMa Model Architecture

LaMa (Large Mask Inpainting) uses a unique architecture called **Fast Fourier Convolutions (FFC)** for image inpainting. The key component is the `FourierUnit` which performs frequency-domain processing.

**Architecture Path of Failure:**
```
Generator Model
  -> FFCResnetBlock (model.5)
     -> FFC_BN_ACT (conv1)
        -> FFC
           -> SpectralTransform (convg2g)
              -> FourierUnitJIT (fu)
                 -> RFFTTN_REAL_ONLY (rttn)
                    -> MatMul operations <- FAILURE POINT
```

### 2. Custom FFT Implementation

The ONNX export uses a custom JIT-compatible FFT implementation (`FourierUnitJIT`) that replaces PyTorch's native `torch.fft.rfftn()` and `torch.fft.irfftn()` with matrix multiplication-based equivalents.

**Why this custom implementation exists:**
- PyTorch's native FFT operations (`torch.fft.*`) are not directly exportable to ONNX with full compatibility
- The custom implementation (`ffc.py` lines 23-146) uses `torch.matmul()` operations to compute DFT/IDFT

**Custom FFT Code Flow:**
```python
# rfft function (lines 23-40)
def rfft(x):
    cos_part = torch.cos(-2 * torch.pi * n[:, None] * k / N)
    sin_part = torch.sin(-2 * torch.pi * n[:, None] * k / N)
    real_part = torch.matmul(x, cos_part)  # <- MatMul operation
    imag_part = torch.matmul(x, sin_part)  # <- MatMul operation
    return (real_part / torch.sqrt(N), imag_part / torch.sqrt(N))
```

### 3. DirectML MatMul Incompatibility

The error occurs at node `/generator/model/model.5/conv1/ffc/convw2g/fu/rttn/MatMul_5`:

```
80070057 The parameter is incorrect.
```

**Root causes identified:**

#### A. Tensor Shape Broadcasting Issues

The custom FFT implementation performs MatMul between:
- Input tensor `x`: shape `[batch, channels, height, width]`
- DFT matrix `cos_part`/`sin_part`: shape `[N, N//2+1]`

DirectML has **stricter validation** for broadcasting rules compared to CPU/CUDA providers. The specific shapes produced during FFT computation may violate DirectML's expected parameter constraints.

#### B. Known ONNX Runtime DirectML Bugs

Research into Microsoft's ONNX Runtime issues revealed multiple related problems:

1. **Issue #20575**: DirectML Exception 80070057 with TorchDynamo-exported models
   - Similar error pattern with graph fusion validation failures
   - Closed without resolution

2. **Issue #6075**: DirectML MatMul failure with variable-sized inputs
   - The model had incorrect output shape specifications
   - DirectML's strict validation exposed the flaw
   - CPU/CUDA providers were more lenient

3. **Issue #12677**: InstanceNormalization parameter incorrect error
   - Similar 80070057 error code
   - Related to shape validation in DirectML

#### C. Dynamic vs Static Shape Handling

The LaMa ONNX export includes dynamic axes for batch dimension:
```python
dynamic_axes={
    "image": {0: "batch"},
    "mask": {0: "batch"},
    "output": {0: "batch"}
}
```

However, the internal FFT operations compute DFT matrices based on fixed tensor sizes. DirectML may have issues with:
- Mixed static/dynamic dimensions
- Operations where intermediate shapes depend on runtime values

### 4. Why CPU and CUDA Providers Work

| Provider | Behavior |
|----------|----------|
| CPU | More lenient shape validation, handles edge cases gracefully |
| CUDA | Native FFT support via cuFFT, doesn't use MatMul-based implementation |
| DirectML | Strict validation, limited FFT support, fails on non-standard MatMul configurations |

### 5. The Two Model Versions

| Model | Export Method | Issue |
|-------|---------------|-------|
| `lama.onnx` | `torch.onnx.dynamo_export` (opset 18) | Cannot be fixed for GPU execution |
| `lama_fp32.onnx` | `torch.onnx.export` (opset 17) | Uses FourierUnitJIT, still fails on DirectML |

Both models use the custom MatMul-based FFT implementation, which is fundamentally incompatible with DirectML's execution requirements.

## Contributing Factors

1. **ONNX FFT operator limitations**: ONNX DFT operator (opset 17+) exists but has limited support across execution providers
2. **DirectML development status**: Microsoft has moved new feature development to WinML, leaving DirectML in "sustained engineering mode"
3. **Custom operator workarounds**: The MatMul-based FFT is a workaround that works on CPU but not on all GPU providers
4. **Lack of DirectML testing**: The Carve team acknowledged they don't test on Windows/DirectML

## Evidence Summary

| Evidence | Source | Relevance |
|----------|--------|-----------|
| Error screenshot | HuggingFace Discussion #1 | Shows exact error location and code |
| MatMul node path | Error message | Confirms FFT operation is failing |
| Error code 80070057 | Windows HRESULT | "The parameter is incorrect" |
| CPU works, GPU fails | User report | Confirms provider-specific issue |
| Both models fail | User testing | Rules out simple model version fix |

## Conclusion

The root cause is a **fundamental architectural incompatibility** between:
1. LaMa's custom MatMul-based FFT implementation (required for ONNX export)
2. DirectML's strict tensor operation validation

This is not a simple configuration issue but requires either:
- Modifications to the ONNX model structure
- DirectML provider updates to support these operations
- Alternative execution strategies (see Solutions document)

## References

- ONNX Runtime Issue #20575: https://github.com/microsoft/onnxruntime/issues/20575
- ONNX Runtime Issue #6075: https://github.com/microsoft/onnxruntime/issues/6075
- ONNX Runtime Issue #12677: https://github.com/microsoft/onnxruntime/issues/12677
- DirectML Execution Provider Docs: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- ONNX DFT Operator Spec: https://onnx.ai/onnx/operators/onnx__DFT.html
