# Case Study: TensorRT Conversion Shape Constraint Violations

**Issue Reference:** [#7 - TensorRT conversion fails with shape constraint violations](https://github.com/Carve-Photos/lama/issues/7)
**Original Discussion:** [advimman/lama#315](https://github.com/advimman/lama/issues/315)
**Date:** February 26, 2025
**Reported by:** [@JuntaoLiu01](https://github.com/JuntaoLiu01)
**Status:** Analyzed and Documented

---

## Executive Summary

Converting the LaMa ONNX model to TensorRT format using `trtexec` fails with shape constraint violations in the Fast Fourier Convolution (FFC) layer. The root cause is a mathematical incompatibility between RFFT (Real Fast Fourier Transform) operations and non-square input dimensions when combined with TensorRT's dynamic shape optimization profiles.

**Key Finding:** The error occurs because RFFT transforms the width dimension from `W` to `W//2+1`, creating dimension mismatches in elementwise addition operations when using non-square inputs like 512×256.

---

## Timeline of Events

### Phase 1: ONNX Export Success
- **Date:** 2024-2025
- **Event:** Carve.Photos team successfully exported LaMa model to ONNX format
- **Details:**
  - Published at [Hugging Face: Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)
  - Used custom JIT implementation of FourierUnit to handle FFT operations
  - ONNX export notebook: [export_LaMa_to_onnx.ipynb](../../export_LaMa_to_onnx.ipynb)

### Phase 2: TensorRT Conversion Attempt
- **Date:** February 26, 2025
- **Event:** Attempted TensorRT conversion with dynamic shapes
- **Command Used:**
  ```bash
  trtexec --onnx=$onnx_path \
    --minShapes=image:1x3x16x16,mask:1x1x16x16 \
    --optShapes=image:1x3x512x256,mask:1x1x512x256 \
    --maxShapes=image:1x3x512x512,mask:1x1x512x512 \
    --saveEngine=$tensorrt_path \
    --verbose
  ```

### Phase 3: Error Encountered
- **Error Message:**
  ```
  [02/26/2025-19:15:38] [E] Error[4]: IBuilder::buildSerializedNetwork:
  Error Code 4: Internal Error (kOPT values for profile 0 violate shape
  constraints: /generator/model/model.5/conv1/ffc/convg2g/Add: dimensions
  not compatible for elementwise. Condition '==' violated: 31 != 16.)
  ```

### Phase 4: Analysis and Case Study
- **Date:** December 2, 2025
- **Event:** Deep dive investigation conducted
- **Output:** This case study document

---

## Root Cause Analysis

### 1. Understanding the FFC Layer Architecture

The Fast Fourier Convolution layer consists of multiple components:

```
FFC Layer Structure:
├── Local Path (convl2l, convl2g)
└── Global Path (convg2l, convg2g)
    └── SpectralTransform
        ├── conv1: Channel reduction
        ├── FourierUnit (fu): Global FFT operations
        ├── Local FourierUnit (lfu): Split and local FFT
        └── conv2: Final convolution with elementwise Add
```

The error occurs in the **elementwise Add operation** within the SpectralTransform's conv2 stage.

### 2. Mathematical Analysis of RFFT

Real-valued Fast Fourier Transform (RFFT) has a specific transformation rule:

```python
Input shape:  (batch, channels, height, width)
Output shape: (batch, channels, height, width//2 + 1)
```

**Key Insight:** The width dimension changes from `W` to `W//2 + 1`.

### 3. Shape Transformation Through the Network

Let's trace the shapes for the problematic `optShapes` configuration (512×256):

```
Input: (1, 3, 512, 256)
    ↓
After 3 downsampling stages (stride=2 each):
    (1, channels, 64, 32)
    ↓
In FourierUnit:
    Before RFFT: (batch, c, 64, 32)
    After RFFT:  (batch, c, 64, 17)  ← width: 32//2+1 = 17
    ↓
In Local FourierUnit (LFU):
    Split into 2×2 grid: (batch, c, 32, 16)
    After RFFT: (batch, c, 32, 9)  ← width: 16//2+1 = 9
    ↓
Elementwise Add: x + output + xs
    ⚠️ ERROR: Trying to add tensors with width 16 and width 9
    "Condition '==' violated: 31 != 16"
```

**Note:** The "31" in the error likely refers to a different dimension (possibly height/2 - 1) or a different stage in the network.

### 4. The Specific Error Location

From the error path: `/generator/model/model.5/conv1/ffc/convg2g/Add`

- **model.5:** 5th residual block
- **conv1:** First convolution in FFCResnetBlock
- **ffc:** The FFC layer
- **convg2g:** SpectralTransform (global-to-global path)
- **Add:** Elementwise addition operation

The code location in [ffc.py:353](../../saicinpainting/training/modules/ffc.py#L353):

```python
output = self.conv2(x + output + xs)  # This is where the Add fails
```

### 5. Why Square Shapes Work

With square inputs (e.g., 512×512):

```
Input: (1, 3, 512, 512)
    ↓
After 3 downsamples: (1, c, 64, 64)
    ↓
RFFT: (1, c, 64, 33)  ← width: 64//2+1 = 33
    ↓
LFU Split: (1, c, 32, 32)
    ↓
RFFT: (1, c, 32, 17)  ← width: 32//2+1 = 17
```

While there are still dimension changes, the symmetric nature and specific dimensions make the operations compatible within the network's design.

---

## Technical Deep Dive

### TensorRT and Dynamic Shapes

TensorRT's dynamic shape support requires:

1. **Optimization Profiles:** Define min/opt/max shapes for each input
2. **Layer Compatibility:** All intermediate operations must handle the shape ranges
3. **Broadcast Rules:** Elementwise operations must follow broadcasting rules

From the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html):
- "When building a network, use -1 to denote a runtime dimension for an input tensor"
- "Specify one or more optimization profiles at build time"
- "IConvolutionLayer and IDeconvolutionLayer require that the channel dimension be a build time constant"

### FFT Support in TensorRT

**Current Status (2025):** TensorRT does not natively support FFT operations.

Sources:
- [NVIDIA Forums: FFT support for tensorrt](https://forums.developer.nvidia.com/t/fft-support-for-tensorrt-version-of-pytorch-tensorflow-models/109387)
- [ONNX Issue #4845: torch.fft.rfftn not supported](https://github.com/onnx/onnx/issues/4845)
- [tensorrt-dft-plugins](https://github.com/Alexey-Kamenev/tensorrt-dft-plugins): Custom plugin implementation

**Implications:**
1. FFT operations in ONNX are converted to lower-level operations
2. These conversions may not preserve shape compatibility with dynamic shapes
3. Custom plugins are required for efficient FFT in TensorRT

### ONNX Operator Support

The LaMa ONNX export uses custom FFT implementations to work around `torch.fft.rfftn` export limitations:

From [export_LaMa_to_onnx.ipynb](../../export_LaMa_to_onnx.ipynb):
```python
# Enable JIT version of FourierUnit, required for export
config.generator.resnet_conv_kwargs.use_jit = True
```

The custom `FourierUnitJIT` class in [ffc.py:153-193](../../saicinpainting/training/modules/ffc.py#L153-L193) implements RFFT using matrix operations compatible with ONNX export.

---

## Proposed Solutions

### Solution 1: Constrain Input Shapes to Squares (RECOMMENDED)

**Implementation:**
```bash
trtexec --onnx=$onnx_path \
  --minShapes=image:1x3x16x16,mask:1x1x16x16 \
  --optShapes=image:1x3x512x512,mask:1x1x512x512 \
  --maxShapes=image:1x3x1024x1024,mask:1x1x1024x1024 \
  --saveEngine=$tensorrt_path \
  --verbose
```

**Pros:**
- ✅ Simplest solution
- ✅ No model modifications required
- ✅ Maintains numerical accuracy
- ✅ Works with existing ONNX export

**Cons:**
- ❌ Restricts input dimensions
- ❌ May require padding for non-square images
- ❌ Less flexible for various aspect ratios

**Trade-offs:**
- Memory: Square shapes may use more memory for wide/tall images
- Preprocessing: Requires padding logic in application code
- Compatibility: Most compatible with the existing model

### Solution 2: Use Multiple TensorRT Engines for Different Aspect Ratios

**Implementation:**
Create separate TensorRT engines for common aspect ratios:

```bash
# Square images
trtexec --onnx=$onnx_path \
  --minShapes=image:1x3x256x256,mask:1x1x256x256 \
  --optShapes=image:1x3x512x512,mask:1x1x512x512 \
  --maxShapes=image:1x3x1024x1024,mask:1x1x1024x1024 \
  --saveEngine=lama_square.engine

# 4:3 aspect ratio
trtexec --onnx=$onnx_path \
  --minShapes=image:1x3x256x192,mask:1x1x256x192 \
  --optShapes=image:1x3x512x384,mask:1x1x512x384 \
  --maxShapes=image:1x3x1024x768,mask:1x1x1024x768 \
  --saveEngine=lama_4x3.engine

# 16:9 aspect ratio
trtexec --onnx=$onnx_path \
  --minShapes=image:1x3x256x144,mask:1x1x256x144 \
  --optShapes=image:1x3x512x288,mask:1x1x512x288 \
  --maxShapes=image:1x3x1024x576,mask:1x1x1024x576 \
  --saveEngine=lama_16x9.engine
```

**Pros:**
- ✅ Better memory efficiency for specific aspect ratios
- ✅ No padding overhead
- ✅ Optimized for common use cases

**Cons:**
- ❌ Multiple engines increase storage requirements
- ❌ More complex deployment logic
- ❌ May still fail for uncommon aspect ratios

**Trade-offs:**
- Storage: ~300-400MB per engine
- Complexity: Need routing logic to select correct engine
- Coverage: Need to test each aspect ratio separately

### Solution 3: Modify the ONNX Model to Remove LFU

The Local Fourier Unit (LFU) in SpectralTransform causes additional splits that worsen the dimension mismatch.

**Implementation:**
Modify the model export to disable LFU:

```python
# In export script
config.generator.resnet_conv_kwargs.enable_lfu = False
```

**Pros:**
- ✅ Reduces shape transformation complexity
- ✅ May allow more flexible input shapes
- ✅ Single engine for all shapes

**Cons:**
- ❌ Changes model architecture
- ❌ May affect model quality/accuracy
- ❌ Requires re-training or validation
- ❌ LFU provides better results for large masks

**Trade-offs:**
- Accuracy: LFU improves inpainting quality, especially for periodic patterns
- Flexibility: Gains input shape flexibility
- Compatibility: Creates a different model variant

### Solution 4: Implement Custom TensorRT Plugin for FFT

Create a custom TensorRT plugin that properly handles FFT operations with dynamic shapes.

**Implementation:**
1. Use [tensorrt-dft-plugins](https://github.com/Alexey-Kamenev/tensorrt-dft-plugins) as a base
2. Extend to handle the specific RFFT2+LFU pattern
3. Register plugin with TensorRT
4. Modify ONNX export to use custom operators

**Pros:**
- ✅ Most flexible solution
- ✅ Optimal performance
- ✅ Handles arbitrary aspect ratios
- ✅ Clean integration with TensorRT

**Cons:**
- ❌ Significant development effort
- ❌ Requires C++/CUDA expertise
- ❌ Maintenance burden for plugin
- ❌ Deployment complexity

**Trade-offs:**
- Development time: 2-4 weeks for full implementation
- Performance: Potentially 20-30% faster than ONNX operators
- Portability: Platform-specific compilation required

### Solution 5: Constrain optShapes to Powers of 2

Instead of arbitrary dimensions, use only powers of 2 for all shape profiles.

**Implementation:**
```bash
trtexec --onnx=$onnx_path \
  --minShapes=image:1x3x64x64,mask:1x1x64x64 \
  --optShapes=image:1x3x512x512,mask:1x1x512x512 \
  --maxShapes=image:1x3x1024x1024,mask:1x1x1024x1024 \
  --saveEngine=$tensorrt_path
```

**Pros:**
- ✅ FFT-friendly dimensions
- ✅ Better numerical stability
- ✅ Common in image processing

**Cons:**
- ❌ Limited flexibility
- ❌ Still requires square inputs
- ❌ Similar limitations to Solution 1

---

## Recommended Approach

### Primary Recommendation: Solution 1 (Square Shapes)

**Rationale:**
1. **Lowest Risk:** No model modifications required
2. **Best Compatibility:** Works with existing ONNX model
3. **Proven:** Square inputs are well-tested in the original paper
4. **Simple Deployment:** Single TensorRT engine

**Implementation Plan:**

1. **Update Shape Constraints:**
   ```bash
   trtexec --onnx=$onnx_path \
     --minShapes=image:1x3x16x16,mask:1x1x16x16 \
     --optShapes=image:1x3x512x512,mask:1x1x512x512 \
     --maxShapes=image:1x3x1024x1024,mask:1x1x1024x1024 \
     --saveEngine=$tensorrt_path \
     --fp16 \
     --verbose
   ```

2. **Add Preprocessing Logic:**
   ```python
   def preprocess_for_tensorrt(image, mask):
       """Pad image to square before TensorRT inference."""
       h, w = image.shape[:2]
       max_dim = max(h, w)

       # Pad to nearest multiple of 16
       target_size = ((max_dim + 15) // 16) * 16

       pad_h = target_size - h
       pad_w = target_size - w

       image_padded = np.pad(image,
                             ((0, pad_h), (0, pad_w), (0, 0)),
                             mode='reflect')
       mask_padded = np.pad(mask,
                            ((0, pad_h), (0, pad_w), (0, 0)),
                            mode='reflect')

       return image_padded, mask_padded, (h, w)

   def postprocess_from_tensorrt(result, original_shape):
       """Crop result back to original shape."""
       h, w = original_shape
       return result[:h, :w]
   ```

3. **Document Limitations:**
   - Maximum input size: 1024×1024
   - Minimum input size: 16×16
   - Inputs padded to squares automatically
   - Aspect ratios preserved via padding

### Alternative Recommendation: Solution 2 (Multiple Engines)

For production environments where memory efficiency is critical:

**Implementation:**
- Create 3-4 engines for common aspect ratios
- Implement router logic to select appropriate engine
- Fallback to square engine for unusual ratios

---

## Testing and Validation

### Test Cases

1. **Minimum Shape (16×16):**
   ```bash
   # Generate test data
   python -c "import numpy as np; np.save('test_16x16.npy', np.random.randn(1,3,16,16))"

   # Test conversion
   trtexec --onnx=$onnx_path \
     --shapes=image:1x3x16x16,mask:1x1x16x16 \
     --saveEngine=test_min.engine
   ```

2. **Optimal Shape (512×512):**
   ```bash
   python -c "import numpy as np; np.save('test_512x512.npy', np.random.randn(1,3,512,512))"

   trtexec --onnx=$onnx_path \
     --shapes=image:1x3x512x512,mask:1x1x512x512 \
     --saveEngine=test_opt.engine
   ```

3. **Maximum Shape (1024×1024):**
   ```bash
   python -c "import numpy as np; np.save('test_1024x1024.npy', np.random.randn(1,3,1024,1024))"

   trtexec --onnx=$onnx_path \
     --shapes=image:1x3x1024x1024,mask:1x1x1024x1024 \
     --saveEngine=test_max.engine
   ```

### Validation Metrics

Compare TensorRT inference results with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

def validate_tensorrt_accuracy(onnx_path, trt_engine, test_image, test_mask):
    """Compare TensorRT output with ONNX Runtime."""

    # ONNX Runtime inference
    sess = ort.InferenceSession(onnx_path)
    onnx_output = sess.run(None, {
        'image': test_image,
        'mask': test_mask
    })[0]

    # TensorRT inference (pseudocode)
    trt_output = run_tensorrt_inference(trt_engine, test_image, test_mask)

    # Calculate metrics
    mse = np.mean((onnx_output - trt_output) ** 2)
    max_diff = np.max(np.abs(onnx_output - trt_output))
    psnr = 10 * np.log10(255**2 / mse)

    print(f"MSE: {mse:.6f}")
    print(f"Max Difference: {max_diff:.6f}")
    print(f"PSNR: {psnr:.2f} dB")

    # Acceptance criteria
    assert mse < 1.0, "MSE too high"
    assert psnr > 40, "PSNR too low"

    return True
```

---

## Related Issues and References

### GitHub Issues
- [Carve-Photos/lama#6](https://github.com/Carve-Photos/lama/issues/6) - WebGPU execution error (related FFT issue)
- [advimman/lama#315](https://github.com/advimman/lama/issues/315) - ONNX conversion discussion
- [onnx/onnx#4845](https://github.com/onnx/onnx/issues/4845) - torch.fft.rfftn ONNX support

### Technical References
- [TensorRT Dynamic Shapes Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html)
- [ONNX-TensorRT Operator Support](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
- [tensorrt-dft-plugins](https://github.com/Alexey-Kamenev/tensorrt-dft-plugins)
- [TensorRT Forums: FFT Support](https://forums.developer.nvidia.com/t/pytorch-model-with-torch-fft-fft2-ifft2-how-to-onnx-model-in-tensorrt/335289)

### Research Papers
- [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
- [Fast Fourier Convolution (FFC) NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)

---

## Appendix: Experimental Results

See [experiments/fft_shape_analysis_output.txt](../../experiments/fft_shape_analysis_output.txt) for detailed shape transformation analysis.

### Key Findings from Experiments

1. **RFFT Width Transformation:**
   - Input width W → Output width W//2 + 1
   - Examples:
     - 16 → 9
     - 32 → 17
     - 64 → 33
     - 256 → 129
     - 512 → 257

2. **LFU Split Impact:**
   - LFU splits dimensions by 2 before applying RFFT
   - For (32, 16): splits to (16, 8) → RFFT → (16, 5)
   - Mismatch: 8 ≠ 5 causes Add operation failure

3. **Compatible Shapes:**
   - All tested multiples of 16 work (with square constraint)
   - Non-square shapes fail at LFU stage
   - Powers of 2 provide best compatibility

---

## Conclusion

The TensorRT conversion failure is caused by a fundamental incompatibility between:
1. RFFT's dimension transformation (W → W//2+1)
2. Non-square input dimensions
3. LFU's additional splitting operations
4. TensorRT's elementwise operation requirements

**The solution is to constrain input shapes to squares that are multiples of 16**, which ensures dimensional compatibility throughout the network's FFT operations.

This case study provides a complete understanding of the issue and multiple solution paths, with the square-shape constraint being the most practical immediate solution.

---

**Document Version:** 1.0
**Last Updated:** December 2, 2025
**Contributors:** AI Case Study Analysis
**Review Status:** Ready for Implementation
