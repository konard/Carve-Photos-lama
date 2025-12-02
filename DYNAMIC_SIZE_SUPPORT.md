# Dynamic Size Support for ONNX Export

## Overview

The ONNX export now supports dynamic input sizes, not just dynamic batch size. This means you can use the exported ONNX model with images of different resolutions without quality degradation from rescaling.

## Changes Made

### 1. Fixed Hardcoded Permutations in FFT Functions

**File**: `saicinpainting/training/modules/ffc.py`

#### Changes in `irfft` function (line 95-115):
- **Before**: `REAL.permute(0, 1, 3, 2)` - hardcoded permutation that only works for 4D tensors
- **After**: `REAL.transpose(-1, -2)` - dynamic permutation that works with any number of dimensions

#### Changes in `ifft1d` function (line 118-155):
- **Before**: Hardcoded permutation logic:
  ```python
  perm = list(range(len(REAL.shape)))
  perm[2], perm[0] = perm[0], perm[2]
  perm[0], perm[1] = perm[1], perm[0]
  ```
- **After**: Dynamic permutation that handles tensors of any shape:
  ```python
  ndim = len(REAL.shape)
  perm = list(range(ndim))
  perm = [1, 2, 0] + perm[3:] if ndim > 3 else perm
  ```

### 2. Updated ONNX Export Configuration

**File**: `export_LaMa_to_onnx.ipynb` (cell 9)

Changed the `dynamic_axes` parameter from:
```python
dynamic_axes={
    "image": {0: "batch"},
    "mask": {0: "batch"},
    "output": {0: "batch"}
}
```

To:
```python
dynamic_axes={
    "image": {0: "batch", 2: "height", 3: "width"},
    "mask": {0: "batch", 2: "height", 3: "width"},
    "output": {0: "batch", 2: "height", 3: "width"}
}
```

## Usage

### Exporting the Model

The export process remains the same. The model will now be exported with dynamic axes:

```bash
# Run the export notebook
jupyter nbconvert --execute export_LaMa_to_onnx.ipynb
```

### Using the Exported Model

The exported ONNX model can now accept inputs of any size (must be divisible by 8):

```python
import onnxruntime
import numpy as np

# Load the model
sess = onnxruntime.InferenceSession('lama_fp32.onnx')

# Test with different resolutions
for height, width in [(256, 256), (512, 512), (1024, 1024), (512, 768)]:
    image = np.random.randn(1, 3, height, width).astype(np.float32)
    mask = np.random.randn(1, 1, height, width).astype(np.float32)

    outputs = sess.run(None, {'image': image, 'mask': mask})
    print(f"Input: {image.shape} -> Output: {outputs[0].shape}")
```

## Constraints

- **Resolution must be divisible by 8**: The model architecture requires input dimensions to be divisible by 8 due to downsampling layers
- **Memory**: Larger resolutions require more memory. Make sure you have sufficient VRAM/RAM for your target resolution
- **Performance**: Processing time scales with image size

## Technical Details

### Why were the changes needed?

The original implementation used hardcoded tensor permutations in the custom FFT implementation (`FourierUnitJIT`). These permutations assumed a fixed 4D tensor shape, which prevented ONNX from correctly handling dynamic height/width dimensions during export and inference.

### What was the root cause?

1. The `irfft` function had: `REAL.permute(0, 1, 3, 2)`
   - This assumes exactly 4 dimensions and hardcodes which dimensions to swap

2. The `ifft1d` function had complex permutation logic that swapped specific indices
   - This also assumed a fixed number of dimensions

### How do the fixes work?

1. **`irfft` fix**: Using `transpose(-1, -2)` instead of `permute(0, 1, 3, 2)`
   - `transpose(-1, -2)` swaps the last two dimensions regardless of total dimensions
   - Works with any tensor shape

2. **`ifft1d` fix**: Using conditional permutation based on actual tensor dimensions
   - Checks `ndim = len(REAL.shape)` first
   - Applies appropriate permutation for the detected shape
   - Works with 3D, 4D, or higher dimensional tensors

## Testing

Tests have been added in `experiments/test_fft_functions.py` to verify the fixes work correctly with different resolutions:

```bash
python experiments/test_fft_functions.py
```

This tests the FFT functions with multiple image sizes:
- 256x256
- 512x512
- 768x768
- 1024x1024

## Related Issues

- Original issue: [#8](https://github.com/Carve-Photos/lama/issues/8)
- Upstream discussion: [advimman/lama#315](https://github.com/advimman/lama/issues/315)

## Benefits

1. **No Quality Degradation**: Process images at their native resolution without rescaling
2. **Flexibility**: Single ONNX model works for all resolutions
3. **Better Results**: High-resolution images get high-resolution outputs
4. **Simpler Pipeline**: No need to manage multiple fixed-size models
