# Technical Analysis: LaMa ONNX Export Implementation

## Current Implementation Overview

The LaMa project uses a **traditional ONNX export approach** with a custom FFT implementation to work around PyTorch's lack of native FFT ONNX support.

### Key Implementation Files

**File**: `saicinpainting/training/modules/ffc.py`
- Contains two versions of Fourier Unit implementations:
  1. `FourierUnit` (lines 228-292): Uses native PyTorch FFT (`torch.fft.rfftn` and `torch.fft.irfftn`)
  2. `FourierUnitJIT` (lines 153-193): Uses custom FFT implementation for ONNX export

**File**: `export_LaMa_to_onnx.ipynb`
- Export script that enables JIT version: `config.generator.resnet_conv_kwargs.use_jit = True`
- Uses traditional `torch.onnx.export` with opset_version=17

## Custom FFT Implementation Details

### Why Custom FFT Was Necessary

The custom FFT implementation was created because:
1. PyTorch's native FFT operations (`torch.fft.rfftn`, `torch.fft.irfftn`) were not supported in traditional ONNX export
2. ONNX opset 17 added DFT operations, but PyTorch's traditional exporter couldn't utilize them
3. The team needed a working solution that could run on various ONNX runtimes

### Custom Implementation Components

**FourierUnitJIT** uses manual FFT computation:

1. **Forward FFT** (rfft function, lines 23-40):
   - Implements real FFT using matrix multiplication with cosine/sine basis
   - Formula: Real part = x @ cos(-2πnk/N), Imag part = x @ sin(-2πnk/N)
   - Returns separate real and imaginary tensors

2. **Complex FFT** (fft function, lines 43-67):
   - Full FFT for complex inputs
   - Used after initial real FFT to process second dimension

3. **Inverse FFT** (irfft/ifft2d functions, lines 95-150):
   - Reconstructs full spectrum from half-spectrum (exploiting Hermitian symmetry)
   - Performs inverse transform to recover spatial domain

4. **RFFTTN_REAL_ONLY class** (lines 70-92):
   - Wraps the custom FFT operations
   - Provides 2D real FFT functionality

## Comparison: Traditional vs Native FFT

### FourierUnit (Native - Lines 228-292)
```python
# Uses PyTorch's native FFT
ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
```

**Advantages:**
- Fast GPU execution
- Optimized by PyTorch team
- Uses efficient FFT algorithms (e.g., Cooley-Tukey)

**Disadvantages:**
- Not exportable to ONNX with traditional exporter

### FourierUnitJIT (Custom - Lines 153-193)
```python
# Uses custom matrix-based FFT
ffted_real, ffted_imag = self.rttn(x, dim=(-2, -1), norm=self.fft_norm)
output = ifft2d(ffted_real, ffted_imag, shape=ifft_shape_slice)
```

**Advantages:**
- Exportable to ONNX
- Works with traditional torch.onnx.export
- No special ONNX runtime requirements

**Disadvantages:**
- Slower than native FFT (O(N³) matrix operations vs O(N log N) FFT)
- Higher memory usage
- Potential accuracy differences due to different numerical implementations

## Export Configuration

From `export_LaMa_to_onnx.ipynb`:

```python
# Enable JIT version of FourierUnit
config.generator.resnet_conv_kwargs.use_jit = True

# Export with traditional exporter
torch.onnx.export(
    exported_model,
    (image_tensor, mask_tensor),
    "/content/lama_fp32.onnx",
    opset_version=17,
    dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}, "output": {0: "batch"}}
)
```

**Note**: The TODO comment at line 239 indicates a known limitation:
> "TODO: Adapt FourierUnit to support dynamic axes (see irfttn and rfft for correct padding)"

This suggests the custom FFT implementation has constraints around dynamic input sizes.

## Performance Implications

### Computational Complexity

**Native FFT (torch.fft.rfftn)**:
- Time: O(N² log N) for 2D FFT
- Space: O(N²)
- Uses highly optimized libraries (cuFFT on GPU)

**Custom Matrix-Based FFT**:
- Time: O(N³) - matrix multiplication dominates
- Space: O(N²) for intermediate matrices
- Runs as generic matrix operations on ONNX runtime

### Real-World Impact

For a 512×512 image:
- Native FFT: ~0.5-2ms on GPU
- Custom FFT: Likely 10-50x slower (estimated, depends on hardware)

This performance difference is acceptable for inference workloads where:
1. FFT is only a portion of total model computation
2. Portability and runtime compatibility are priorities
3. CPU inference is acceptable

## Accuracy Considerations

The custom FFT implementation should theoretically produce identical results to native FFT, but:
1. Different floating-point operation ordering can cause minor numerical differences
2. Matrix multiplication accumulates rounding errors differently than FFT algorithms
3. The OPHoperHPO comment mentioned accuracy losses with the custom approach

## Summary

The current LaMa ONNX export uses a **pragmatic workaround**:
- Implements FFT as matrix operations for ONNX compatibility
- Sacrifices performance for portability
- Successfully produces working ONNX models that match PyTorch results (when preprocessing is correct)
- Avoids the experimental and problematic `torch.dynamo_export` approach
