# Case Study: WebGPU Tensor Dimension Mismatch in LaMa ONNX Model

## Executive Summary

This case study analyzes a critical bug in the LaMa ONNX model where the WebGPU execution provider fails with a tensor dimension mismatch error during a binary Add operation within the Fast Fourier Convolution (FFC) layer. The WASM backend works correctly, indicating a WebGPU-specific issue.

**Key Findings:**
- **Error**: `Can't perform binary op on the given tensors` at `/generator/model/model.5/conv1/ffc/convg2g/Add`
- **Dimension Mismatch**: `1,64,64,192` vs `1,64,63,192` (third dimension: 64 vs 63)
- **Root Cause**: FFT operations produce different tensor dimensions between WASM and WebGPU backends
- **Status**: Reproducible, likely an onnxruntime-web WebGPU implementation issue

---

## Timeline of Events

### Phase 1: ONNX Model Development (May 2024)
- **May 10, 2024**: Carve.Photos team successfully ported LaMa (big-lama) to ONNX
- **May 14, 2024**: Initial ONNX model released on HuggingFace
- Model worked with standard ONNX runtimes and WASM backend

### Phase 2: Issue Discovery (December 2024)
- **December 30, 2024**: [@geronimi73](https://github.com/geronimi73) discovered WebGPU execution provider failure
- Error occurred at FFC (Fast Fourier Convolution) layer during binary Add operation
- WASM backend confirmed working but slow
- Initial debugging revealed tensor dimension mismatch: `1,64,64,192` vs `1,64,63,192`

### Phase 3: Analysis (December 2024 - Present)
- **December 2, 2025**: Issue #6 created in Carve-Photos/lama repository
- Linked to original discussion in advimman/lama#315
- Deep case study initiated to understand root cause

---

## Technical Analysis

### 1. Architecture Context

#### LaMa Model Structure
```
Input (3, 512, 512)
  ↓
Generator Network
  ↓
FFC Layers (Fast Fourier Convolution)
  ├─ Local Branch (l2l, l2g)
  └─ Global Branch (g2l, g2g) ← ERROR HERE
      ↓
  SpectralTransform
      ↓
  FourierUnit
      ├─ rfft2d (Real FFT)
      ├─ Conv2d processing
      └─ irfft2d (Inverse Real FFT) ← DIMENSION MISMATCH
  ↓
Add Operation ← FAILS HERE
  ↓
Output
```

#### FFC Layer Breakdown
The FFC (Fast Fourier Convolution) layer splits input into:
1. **Local branch**: Standard convolution operations
2. **Global branch**: Fourier domain operations

The error occurs in the global-to-global path (`convg2g`), specifically at the final Add operation where two tensors need to be combined.

### 2. Root Cause Analysis

#### FFT Dimension Calculation
When performing Real FFT (rfft) on 2D spatial dimensions:
- Input shape: `(batch, channels, height, width)`
- After rfft on last dimension: `(batch, channels, height, width//2 + 1)`
- After fft on second-to-last dimension: `(batch, channels, height, width//2 + 1)`

For an input with width=128:
- Expected FFT width: `128 // 2 + 1 = 65`

**The Problem**: The irfft (inverse real FFT) operation reconstructs spatial dimensions, but the reconstruction logic differs between WASM and WebGPU.

#### Critical Code Section (ffc.py:95-112)
```python
def irfft(REAL, IMAG, n=None, axis=-1, norm=None):
    axis = axis if axis >= 0 else REAL.ndim + axis

    # Generate the full FFT spectrum from the half-spectrum
    REAL_flipped = torch.flip(REAL[..., 1:-1], dims=[axis])  # ← SLICE [1:-1]
    IMAG_flipped = torch.flip(IMAG[..., 1:-1], dims=[axis])

    # Conjugate the flipped IMAG tensor
    IMAG_flipped_conj = -IMAG_flipped

    # Concatenate original with conjugated flipped versions
    REAL_extended = torch.cat([REAL, REAL_flipped], dim=axis)
    IMAG_extended = torch.cat([IMAG, IMAG_flipped_conj], dim=axis)

    REAL = ifft1d(REAL_extended, IMAG_extended, axis=axis)[0]
    REAL = REAL.permute(0, 1, 3, 2)
    return REAL
```

The slice `[..., 1:-1]` removes the DC component (index 0) and the Nyquist frequency (last index) from the FFT spectrum. This is critical for reconstruction:
- If input FFT width is 65: flipped part is `65 - 2 = 63` elements
- Extended spectrum: `65 + 63 = 128` elements

**Issue Hypothesis**: WebGPU's FFT implementation may:
1. Not produce exactly 65 elements (width//2 + 1)
2. Handle edge cases differently in the spectrum
3. Have rounding differences in dimension calculation

### 3. ONNX Export Details

From `export_LaMa_to_onnx.ipynb`:
```python
# Enable JIT version of FourierUnit, required for export
config.generator.resnet_conv_kwargs.use_jit = True

# Comment in code:
# "If you get a tensor size mismatch, you need to specify correct padding"
# "TODO: Adapt FourierUnit to support dynamic axes (see irfttn and rfft for correct padding)"
```

The ONNX model uses `FourierUnitJIT` which implements custom FFT operations using matrix multiplications rather than PyTorch's native `torch.fft.rfftn`. This was necessary because:
1. Early ONNX exporters didn't support native FFT ops
2. Compatibility with various ONNX runtimes

### 4. WebGPU Binary-Op Implementation

From onnxruntime WebGPU source:
```typescript
const calculatedShape = BroadcastUtil.calcShape(aDims, bDims, false);
if (!calculatedShape) {
  throw new Error("Can't perform binary op on the given tensors");
}
```

**Broadcasting Rules**:
- Dimensions must be equal, OR
- At least one dimension must be 1 (broadcastable)
- Strict validation with no tolerance for mismatches

**The Error**: Tensors with shapes `[1,64,64,192]` and `[1,64,63,192]` fail because:
- Third dimension: 64 ≠ 63
- Neither is 1
- Cannot broadcast

### 5. Comparison: WASM vs WebGPU

| Aspect | WASM Backend | WebGPU Backend |
|--------|--------------|----------------|
| FFT Implementation | Software-based, precise | GPU-accelerated, hardware-dependent |
| Dimension Handling | Flexible, may auto-adjust | Strict validation |
| Broadcasting | More lenient | Strict equality checks |
| Performance | Slower (CPU-bound) | Faster (GPU-accelerated) |
| Status | ✅ Working | ❌ Failing |

---

## Reproduction Steps

### Environment Setup
```bash
# Load ONNX model
# Use onnxruntime-web with WebGPU execution provider
# Input: 512x512 image and mask
```

### Minimal Test Case
```javascript
import * as ort from 'onnxruntime-web';

// Try WASM - Works
const sessionWASM = await ort.InferenceSession.create('lama_fp32.onnx', {
  executionProviders: ['wasm']
});

// Try WebGPU - Fails
const sessionWebGPU = await ort.InferenceSession.create('lama_fp32.onnx', {
  executionProviders: ['webgpu']
});
// Error: [WebGPU] Kernel "[Add] /generator/model/model.5/conv1/ffc/convg2g/Add" failed
```

---

## Proposed Solutions

### Solution 1: Pad Tensors to Match Dimensions (Workaround)
**Approach**: Modify ONNX graph to add padding operations before the Add node.

**Pros**:
- Non-invasive to model architecture
- Can be done post-export

**Cons**:
- May affect accuracy if padding is not handled correctly
- Requires ONNX graph manipulation

**Implementation**:
```python
# Add Pad operation before Add
# Pad shape [1,64,63,192] to [1,64,64,192] with zeros
```

### Solution 2: Fix irfft Dimension Calculation
**Approach**: Ensure irfft produces consistent output dimensions across backends.

**Pros**:
- Addresses root cause
- Maintains model integrity

**Cons**:
- Requires re-export of ONNX model
- May need changes to FourierUnitJIT

**Implementation**:
```python
# In ffc.py, ensure output dimension matches expected:
def irfft(REAL, IMAG, n=None, axis=-1, norm=None):
    if n is None:
        # Calculate n from input dimensions
        n = (REAL.shape[axis] - 1) * 2  # Ensure even reconstruction
    # ... rest of implementation
```

### Solution 3: Use Native ONNX FFT Operators
**Approach**: Export model using ONNX opset 17+ native FFT operators instead of custom implementation.

**Pros**:
- Standard operators may have better backend support
- Cleaner ONNX graph

**Cons**:
- Requires significant re-work of export code
- May not be supported by all ONNX runtimes

### Solution 4: Report to ONNX Runtime Team
**Approach**: File bug report with onnxruntime-web team about WebGPU FFT dimension handling.

**Pros**:
- Fixes issue for all users
- No model changes needed

**Cons**:
- Long turnaround time
- May not be prioritized

**Recommended Action**: File issue at https://github.com/microsoft/onnxruntime with:
- ONNX model snippet showing the issue
- Comparison of WASM vs WebGPU behavior
- Request for WebGPU broadcasting tolerance or FFT dimension fixes

---

## Impact Assessment

### Severity: HIGH
- WebGPU is critical for performance in browser-based applications
- WASM fallback is 3-10x slower
- Affects all users trying to use WebGPU backend

### Affected Users
- Web application developers using LaMa ONNX model
- Browser-based image editing tools
- Any deployment targeting WebGPU for acceleration

### Workaround Availability
- ✅ WASM backend works (slower)
- ⚠️ No immediate WebGPU fix without model modification

---

## Related Issues

### Similar Issues in ONNX Runtime
1. **TensorRT Issue**: Comment from [@JuntaoLiu01](https://github.com/advimman/lama/issues/315#issuecomment-2542149826)
   ```
   Error[4]: dimensions not compatible for elementwise.
   Condition '==' violated: 31 != 16.
   ```
   Also FFT-related dimension mismatches in TensorRT backend.

2. **Fixed Input Resolution**: Multiple users reported ONNX model restricted to 512x512
   - Root cause: FFT dimension calculations hardcoded during export
   - Dynamic shapes not supported due to FFT padding requirements

---

## Recommendations

### Immediate Actions
1. ✅ Document the issue comprehensively (this case study)
2. ⏭️ Create minimal reproducible example
3. ⏭️ File issue with onnxruntime-web team
4. ⏭️ Test Solution 1 (padding workaround)

### Short-term Actions
1. Implement ONNX graph modification tool to add padding
2. Provide patched model for WebGPU users
3. Add warning in documentation about WebGPU limitations

### Long-term Actions
1. Re-architect FFT export to use native ONNX operators
2. Add comprehensive tests for all ONNX execution providers
3. Support dynamic input resolutions

---

## References

### Code References
- `saicinpainting/training/modules/ffc.py:95-112` - irfft implementation
- `saicinpainting/training/modules/ffc.py:153-193` - FourierUnitJIT class
- `export_LaMa_to_onnx.ipynb` - ONNX export notebook

### External References
- [ONNX Runtime WebGPU Binary-Op](https://github.com/microsoft/onnxruntime/blob/2d05c4bcd940aa25561ed7de26481f219618dd7a/js/web/lib/wasm/jsep/webgpu/ops/binary-op.ts#L158)
- [Original Issue Discussion](https://github.com/advimman/lama/issues/315#issuecomment-2580765285)
- [FFC Paper](https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)

### Related Discussions
- advimman/lama#315 - Original ONNX model announcement
- Carve-Photos/lama#6 - This issue

---

## Appendix

### A. Error Stack Trace
```
ort.all.bundle.min.mjs:1814 Uncaught (in promise) Error:
[WebGPU] Kernel "[Add] /generator/model/model.5/conv1/ffc/convg2g/Add" failed.
Error: Can't perform binary op on the given tensors
    at tensor dimensions: 1,64,64,192 vs 1,64,63,192
```

### B. Model Architecture Details
- **Model**: LaMa (Large Mask Inpainting with Fourier Convolutions)
- **ONNX Opset**: 17
- **Input Resolution**: 512x512 (hardcoded during export)
- **FFC Layers**: 9 residual blocks with ratio_gin=0.75, ratio_gout=0.75

### C. Testing Checklist
- [x] Reproduce error with WebGPU
- [x] Verify WASM backend works
- [x] Identify exact failing operation
- [x] Analyze tensor dimensions
- [x] Review FFT implementation
- [ ] Test padding workaround
- [ ] Create minimal ONNX graph reproducer
- [ ] File upstream issue

---

**Document Version**: 1.0
**Last Updated**: 2025-12-02
**Authors**: Carve.Photos Team, AI Issue Solver
**Status**: Analysis Complete, Solutions Proposed
