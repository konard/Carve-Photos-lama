# Technical Deep Dive: FFT Dimension Mismatch in WebGPU

## Overview

This document provides a detailed technical analysis of the tensor dimension mismatch that occurs when using the WebGPU execution provider with the LaMa ONNX model.

---

## 1. FFT Mathematics and Dimension Calculations

### 1.1 Real FFT (rfft) Output Dimensions

For a real-valued input signal of length N:
- **DFT**: Produces N complex values
- **RFFT**: Exploits Hermitian symmetry, produces only `N/2 + 1` complex values

**Why N/2 + 1?**
```
For N = 128 (even):
- Positive frequencies: 0, 1, 2, ..., 64  (65 values)
- DC component (0 Hz): Real only
- Nyquist frequency (N/2): Real only
- Negative frequencies: Redundant (conjugate symmetry)

Total unique values: 65 = 128/2 + 1
```

For N = 127 (odd):
```
- Positive frequencies: 0, 1, 2, ..., 63  (64 values)
- DC component: Real only
- No Nyquist frequency (N is odd)
- Total unique values: 64 = (127+1)/2 = 128/2
```

### 1.2 Inverse Real FFT (irfft) Reconstruction

To reconstruct N real values from `N/2 + 1` complex values:

1. **Extract**: `[DC, f1, f2, ..., fN/2]`
2. **Mirror**: Create conjugate for negative frequencies
3. **Inverse DFT**: Apply inverse FFT

**Implementation in ffc.py**:
```python
def irfft(REAL, IMAG, n=None, axis=-1, norm=None):
    # Input: REAL, IMAG with shape [..., N/2+1]

    # Step 1: Flip middle portion (exclude DC and Nyquist)
    REAL_flipped = torch.flip(REAL[..., 1:-1], dims=[axis])  # [N/2-1 elements]
    IMAG_flipped = torch.flip(IMAG[..., 1:-1], dims=[axis])

    # Step 2: Conjugate (negate imaginary part)
    IMAG_flipped_conj = -IMAG_flipped

    # Step 3: Concatenate
    # [DC, f1, ..., fN/2] + [fN/2-1, ..., f1] = N elements
    REAL_extended = torch.cat([REAL, REAL_flipped], dim=axis)
    IMAG_extended = torch.cat([IMAG, IMAG_flipped_conj], dim=axis)

    # Step 4: Inverse FFT
    REAL = ifft1d(REAL_extended, IMAG_extended, axis=axis)[0]

    return REAL
```

**Critical Observation**:
- Input length: `N/2 + 1` (e.g., 65 for N=128)
- Flipped length: `N/2 + 1 - 2 = N/2 - 1` (e.g., 63 for N=128)
- Extended length: `(N/2 + 1) + (N/2 - 1) = N` (e.g., 128)

---

## 2. The Dimension Mismatch Problem

### 2.1 Expected Flow (WASM Backend)

```
Input: [1, 64, H, W] where H=128, W=128

↓ rfft2d (on H and W dimensions)

FFT Output: [1, 64, H, W/2+1] = [1, 64, 128, 65]

↓ Conv2d processing in frequency domain

Processed: [1, 64, 128, 65]

↓ irfft2d (reconstruct H and W)

Step 1 - irfft on W dimension:
  - Flip [..., 1:-1]: Shape changes from [65] to [63]
  - Concatenate: [65] + [63] = [128]
  - ifft1d: Output [128]
  - Result: [1, 64, 128, 128]

Step 2 - irfft on H dimension:
  - Similar process
  - Result: [1, 64, 128, 128]

Output: [1, 64, 128, 128] ✅
```

### 2.2 Actual Flow (WebGPU Backend)

```
Input: [1, 64, H, W] where H=128, W=128

↓ rfft2d (on H and W dimensions)

FFT Output: [1, 64, H, W/2+1] = [1, 64, 128, 65] or [1, 64, 128, 64]? ⚠️

↓ Conv2d processing

Processed: [1, 64, 128, ?]

↓ irfft2d (reconstruct)

irfft on W dimension:
  - IF input is [65]: Flipped is [63], concatenated is [128] ✅
  - IF input is [64]: Flipped is [62], concatenated is [126] ❌

Result: Dimension mismatch at Add operation
  Expected: [1, 64, 64, 192]
  Actual:   [1, 64, 63, 192]
```

**Hypothesis**: WebGPU's FFT implementation may be:
1. Producing 64 instead of 65 elements (off-by-one error)
2. Treating even/odd lengths differently
3. Rounding down instead of using ceiling

---

## 3. ONNX Graph Analysis

### 3.1 FFC Layer in ONNX

```
/generator/model/model.5/conv1/ffc/convg2g/
├── SpectralTransform
│   ├── Conv1 (reduce channels)
│   ├── FourierUnit
│   │   ├── rfft2d_operations (custom matrix mult)
│   │   │   ├── MatMul (height dimension)
│   │   │   └── MatMul (width dimension)
│   │   ├── Reshape & Permute
│   │   ├── Conv2d (frequency domain)
│   │   ├── BatchNorm2d
│   │   ├── ReLU
│   │   ├── Reshape & Permute back
│   │   └── irfft2d_operations
│   │       ├── Slice [1:-1]  ← POTENTIAL ISSUE
│   │       ├── Flip
│   │       ├── Concat
│   │       └── MatMul (inverse)
│   └── Conv2 (expand channels)
└── Add ← FAILS HERE
```

### 3.2 Critical ONNX Operations

**Slice Operation** (in irfft):
```
ONNX Slice node:
  Input: tensor with shape [..., N/2+1]
  Starts: [1]
  Ends: [-1]
  Axes: [axis]
  Output: [..., N/2-1]
```

**Issue**: Different backends may compute `N/2+1` differently:
- WASM: Precise integer division
- WebGPU: May use floating point intermediate, then round

---

## 4. Backend Implementation Differences

### 4.1 WASM Backend Characteristics

```javascript
// Likely implementation (simplified)
function rfftLength(n) {
  return Math.floor(n / 2) + 1;  // Integer arithmetic
}

// For n = 128: 128 / 2 + 1 = 64 + 1 = 65 ✅
```

**Advantages**:
- CPU-based, deterministic
- Standard floating-point operations
- Mature implementation

### 4.2 WebGPU Backend Characteristics

```javascript
// Possible implementation
function rfftLength(n) {
  // GPU shader may use different calculation
  return (n >> 1) + 1;  // Bit shift
  // OR
  return Math.ceil(n / 2);  // Ceiling instead of floor
}

// For n = 128:
// Bit shift: 128 >> 1 + 1 = 64 + 1 = 65 ✅
// Ceiling: ceil(128 / 2) = ceil(64) = 64 ❌
```

**Potential Issues**:
- GPU shaders use different precision
- Intermediate calculations in shader may round differently
- Shape inference at compile time vs runtime

---

## 5. WebGPU Binary-Op Validation

### 5.1 Broadcasting Rules Implementation

From `onnxruntime/js/web/lib/wasm/jsep/webgpu/ops/binary-op.ts`:

```typescript
export const createBinaryOpProgramInfo = (
  name: string,
  cacheKey: string,
  a: TensorView,
  b: TensorView,
  // ...
): ProgramInfo => {
  const aDims = a.dims.map((x) => Number(x) ?? 1);
  const bDims = b.dims.map((x) => Number(x) ?? 1);

  const isBroadcast = !ShapeUtil.areEqual(aDims, bDims);

  if (isBroadcast) {
    const calculatedShape = BroadcastUtil.calcShape(aDims, bDims, false);
    if (!calculatedShape) {
      throw new Error("Can't perform binary op on the given tensors");
    }
  }
};
```

### 5.2 BroadcastUtil.calcShape Logic

```typescript
static calcShape(
  adims: readonly number[],
  bdims: readonly number[],
  isMatMul = false,
): readonly number[] | undefined {
  // Align dimensions from the right
  const outputRank = Math.max(adims.length, bdims.length);
  const outputShape = new Array(outputRank);

  for (let i = 0; i < outputRank; i++) {
    const a_dim = adims[adims.length - 1 - i] ?? 1;
    const b_dim = bdims[bdims.length - 1 - i] ?? 1;

    // Both dimensions must be equal OR one must be 1
    if (a_dim !== b_dim && a_dim !== 1 && b_dim !== 1) {
      return undefined;  // ← ERROR TRIGGERED HERE
    }

    outputShape[outputRank - 1 - i] = Math.max(a_dim, b_dim);
  }

  return outputShape;
}
```

**For our case**:
```
Tensor A: [1, 64, 64, 192]
Tensor B: [1, 64, 63, 192]

Dimension comparison (right to left):
  i=0: 192 === 192 ✅
  i=1: 64 !== 63 AND 64 !== 1 AND 63 !== 1 ❌
  → return undefined
  → throw Error
```

**Strict Validation**: WebGPU backend does not allow broadcasting when dimensions differ and neither is 1.

---

## 6. Debugging Approach

### 6.1 Instrumentation Points

To debug this issue, we need to log tensor shapes at:

```python
# In FourierUnitJIT.forward()
def forward(self, x):
    print(f"FourierUnit input: {x.shape}")

    # After rfft
    ffted_real, ffted_imag = self.rttn(x, dim=(-2, -1), norm=self.fft_norm)
    print(f"After rfft - Real: {ffted_real.shape}, Imag: {ffted_imag.shape}")

    # After conv processing
    ffted = self.conv_layer(ffted)
    print(f"After conv: {ffted.shape}")

    # Before irfft
    print(f"Before irfft - Real: {ffted_real.shape}, Imag: {ffted_imag.shape}")

    # After irfft
    output = ifft2d(ffted_real, ffted_imag, shape=ifft_shape_slice)
    print(f"After irfft: {output.shape}")

    return output
```

### 6.2 ONNX Graph Inspection

```python
import onnx
from onnx import helper, shape_inference

# Load model
model = onnx.load("lama_fp32.onnx")

# Find the problematic Add node
for node in model.graph.node:
    if "convg2g/Add" in node.name:
        print(f"Node: {node.name}")
        print(f"Inputs: {node.input}")
        print(f"Outputs: {node.output}")

        # Get input shapes
        for inp in node.input:
            # Trace back to find shape
            pass
```

---

## 7. Proposed Fix Strategies

### 7.1 Strategy 1: Dynamic Padding in ONNX

```python
def add_dynamic_padding(model, node_name):
    """Add padding to ensure dimension alignment before Add operation"""
    # Find the Add node
    add_node = find_node(model, node_name)

    # Get input shapes
    input_shapes = get_input_shapes(model, add_node)

    # Determine which input needs padding
    target_shape = max(input_shapes, key=lambda s: s[2])  # Largest dim 2

    # Add Pad node
    for i, (inp, shape) in enumerate(zip(add_node.input, input_shapes)):
        if shape[2] < target_shape[2]:
            pad_node = helper.make_node(
                'Pad',
                inputs=[inp],
                outputs=[f'{inp}_padded'],
                mode='constant',
                pads=[0, 0, 0, 0, 0, 0, 1, 0],  # Pad dim 2 by 1
                constant_value=0.0
            )
            # Insert pad node
            add_node.input[i] = f'{inp}_padded'
```

### 7.2 Strategy 2: Fix irfft to Force Correct Output Size

```python
def irfft_fixed(REAL, IMAG, n=None, axis=-1, norm=None):
    """Fixed irfft that ensures output dimension matches n"""
    axis = axis if axis >= 0 else REAL.ndim + axis

    # CRITICAL: Explicitly specify target length
    if n is None:
        # Calculate from input assuming proper rfft output
        n = (REAL.shape[axis] - 1) * 2

    # Verify input length matches expected rfft output
    expected_fft_len = n // 2 + 1
    actual_fft_len = REAL.shape[axis]

    if actual_fft_len != expected_fft_len:
        print(f"WARNING: FFT length mismatch. Expected {expected_fft_len}, got {actual_fft_len}")
        # Pad or trim to expected length
        if actual_fft_len < expected_fft_len:
            # Pad with zeros
            pad_width = [(0, 0)] * REAL.ndim
            pad_width[axis] = (0, expected_fft_len - actual_fft_len)
            REAL = torch.nn.functional.pad(REAL, pad_width)
            IMAG = torch.nn.functional.pad(IMAG, pad_width)
        else:
            # Trim
            indices = [slice(None)] * REAL.ndim
            indices[axis] = slice(0, expected_fft_len)
            REAL = REAL[tuple(indices)]
            IMAG = IMAG[tuple(indices)]

    # Now proceed with standard irfft
    REAL_flipped = torch.flip(REAL[..., 1:-1], dims=[axis])
    IMAG_flipped = torch.flip(IMAG[..., 1:-1], dims=[axis])
    IMAG_flipped_conj = -IMAG_flipped

    REAL_extended = torch.cat([REAL, REAL_flipped], dim=axis)
    IMAG_extended = torch.cat([IMAG, IMAG_flipped_conj], dim=axis)

    REAL = ifft1d(REAL_extended, IMAG_extended, axis=axis)[0]

    # Verify output length
    assert REAL.shape[axis] == n, f"Output length {REAL.shape[axis]} != expected {n}"

    REAL = REAL.permute(0, 1, 3, 2)
    return REAL
```

### 7.3 Strategy 3: Use ONNX Native FFT Ops

```python
# In export script
# Instead of use_jit=True, export with native FFT ops
config.generator.resnet_conv_kwargs.use_jit = False

# Use torch.onnx.export with newer opset that supports DFT
torch.onnx.export(
    model,
    inputs,
    "lama_native_fft.onnx",
    opset_version=18,  # Supports DFT operator
    # ...
)
```

**ONNX DFT Operator** (opset 18+):
- Explicitly specifies output length
- Standard implementation across backends
- Better backend support

---

## 8. Testing Plan

### 8.1 Unit Tests

```python
def test_rfft_dimensions():
    """Test that rfft produces correct output dimensions"""
    for n in [64, 127, 128, 129, 256, 512]:
        x = torch.randn(1, 64, n, n)

        # RFFT
        real, imag = rfft2d(x)

        # Expected dimensions
        expected_w = n // 2 + 1
        expected_h = n // 2 + 1  # After 2D FFT

        assert real.shape == (1, 64, n, expected_w), \
            f"RFFT width mismatch for n={n}: {real.shape} vs expected (1, 64, {n}, {expected_w})"

        # IRFFT
        output = irfft2d(real, imag, shape=(n, n))

        assert output.shape == (1, 64, n, n), \
            f"IRFFT output mismatch for n={n}: {output.shape} vs expected (1, 64, {n}, {n})"

def test_onnx_fft_consistency():
    """Test ONNX model FFT consistency across backends"""
    import onnxruntime as ort

    # Load model
    sess_wasm = ort.InferenceSession('lama.onnx', providers=['CPUExecutionProvider'])

    # Test input
    image = np.random.rand(1, 3, 512, 512).astype(np.float32)
    mask = np.random.rand(1, 1, 512, 512).astype(np.float32)

    # Run with WASM
    output_wasm = sess_wasm.run(None, {'image': image, 'mask': mask})

    # Compare shapes at intermediate nodes
    # (Requires access to intermediate outputs)
```

### 8.2 Integration Tests

```javascript
// test_webgpu.js
import * as ort from 'onnxruntime-web';

async function testWebGPU() {
  try {
    const session = await ort.InferenceSession.create('lama_fp32.onnx', {
      executionProviders: ['webgpu']
    });

    const image = new ort.Tensor('float32',
      new Float32Array(1 * 3 * 512 * 512),
      [1, 3, 512, 512]);
    const mask = new ort.Tensor('float32',
      new Float32Array(1 * 1 * 512 * 512),
      [1, 1, 512, 512]);

    const output = await session.run({ image, mask });

    console.log('WebGPU test passed!', output);
  } catch (error) {
    console.error('WebGPU test failed:', error);
  }
}
```

---

## 9. Conclusion

The tensor dimension mismatch in WebGPU is caused by subtle differences in how FFT operations compute output dimensions. The issue manifests in the `irfft` reconstruction where the slice operation `[1:-1]` depends on the exact input length.

**Key Findings**:
1. WASM correctly produces `N/2 + 1` elements from rfft
2. WebGPU may produce `N/2` elements (off-by-one)
3. This causes the reconstructed dimension to be 63 instead of 64
4. The Add operation fails strict dimension checking in WebGPU binary-op

**Recommended Path Forward**:
1. Implement Strategy 2 (fix irfft with validation)
2. Re-export ONNX model with fixed code
3. Add comprehensive tests for all dimensions
4. File issue with onnxruntime-web team

---

**Last Updated**: 2025-12-02
**Version**: 1.0
