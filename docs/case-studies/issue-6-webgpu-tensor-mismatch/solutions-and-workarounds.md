# Solutions and Workarounds for WebGPU Tensor Mismatch

## Quick Reference

| Solution | Difficulty | Impact | Timeline | Recommended |
|----------|------------|--------|----------|-------------|
| 1. ONNX Graph Padding | Medium | Low-Medium | 1-2 days | ⭐ Short-term |
| 2. Fix irfft Implementation | Hard | High | 3-5 days | ⭐⭐ Long-term |
| 3. Use Native ONNX FFT | Very Hard | High | 1-2 weeks | ⭐⭐⭐ Future |
| 4. Upstream Bug Report | Easy | Variable | Weeks-Months | ⭐ Parallel track |

---

## Solution 1: ONNX Graph Padding (Workaround)

### Description
Modify the ONNX graph post-export to add padding operations before problematic Add nodes, ensuring dimension alignment.

### Implementation

#### Step 1: Identify Problematic Nodes
```python
import onnx
from onnx import helper, numpy_helper
import numpy as np

def find_dimension_mismatches(model_path):
    """Scan ONNX model for potential dimension mismatches in Add operations"""
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)

    mismatches = []

    for node in model.graph.node:
        if node.op_type == 'Add':
            # Get input shapes
            input_shapes = []
            for inp in node.input:
                shape = get_tensor_shape(model, inp)
                input_shapes.append(shape)

            # Check if dimensions match
            if len(input_shapes) == 2:
                shape_a, shape_b = input_shapes
                if shape_a != shape_b:
                    mismatches.append({
                        'node': node.name,
                        'shape_a': shape_a,
                        'shape_b': shape_b
                    })

    return mismatches
```

#### Step 2: Insert Pad Operations
```python
def insert_padding_before_add(model_path, output_path, add_node_name):
    """Insert Pad node before specific Add operation"""
    model = onnx.load(model_path)

    # Find the Add node
    add_node = None
    for node in model.graph.node:
        if add_node_name in node.name:
            add_node = node
            break

    if not add_node:
        raise ValueError(f"Node {add_node_name} not found")

    # Analyze which input needs padding
    # Assuming input[1] needs padding from 63 to 64
    input_to_pad = add_node.input[1]

    # Create Pad node
    pad_node = helper.make_node(
        'Pad',
        inputs=[input_to_pad, 'pad_constant'],
        outputs=[f'{input_to_pad}_padded'],
        mode='constant'
    )

    # Create padding constant: [0, 0, 0, 0, 0, 0, 1, 0]
    # This pads dimension 2 by 1 at the end
    pad_tensor = helper.make_tensor(
        name='pad_constant',
        data_type=onnx.TensorProto.INT64,
        dims=[8],
        vals=[0, 0, 0, 0, 0, 0, 1, 0]  # [begin_0, begin_1, ..., end_0, end_1, ...]
    )

    # Add constant to initializers
    model.graph.initializer.append(pad_tensor)

    # Insert pad node before add node
    node_index = list(model.graph.node).index(add_node)
    model.graph.node.insert(node_index, pad_node)

    # Update Add node input
    add_node.input[1] = f'{input_to_pad}_padded'

    # Save modified model
    onnx.save(model, output_path)

    print(f"Modified model saved to {output_path}")
    return model
```

#### Step 3: Verification Script
```python
def verify_padding_fix(original_model, patched_model):
    """Verify that padding fix maintains model accuracy"""
    import onnxruntime as ort
    import numpy as np

    # Test data
    image = np.random.rand(1, 3, 512, 512).astype(np.float32)
    mask = np.random.rand(1, 1, 512, 512).astype(np.float32)

    # Run original with WASM
    sess_orig = ort.InferenceSession(original_model, providers=['CPUExecutionProvider'])
    output_orig = sess_orig.run(None, {'image': image, 'mask': mask})

    # Run patched with WebGPU
    sess_patched = ort.InferenceSession(patched_model, providers=['WebGPUExecutionProvider'])
    output_patched = sess_patched.run(None, {'image': image, 'mask': mask})

    # Compare outputs
    diff = np.abs(output_orig[0] - output_patched[0])
    print(f"Max difference: {diff.max()}")
    print(f"Mean difference: {diff.mean()}")

    # Check if within tolerance
    assert diff.max() < 1.0, "Output difference too large"
```

### Pros
- ✅ Non-invasive (doesn't require re-training)
- ✅ Quick to implement
- ✅ Can be automated

### Cons
- ❌ May introduce small numerical differences
- ❌ Increases model size slightly
- ❌ Doesn't fix root cause

### Testing Checklist
- [ ] Identify all Add nodes with mismatches
- [ ] Insert padding for each mismatch
- [ ] Verify model loads in WebGPU
- [ ] Compare output with WASM baseline
- [ ] Test with various input sizes

---

## Solution 2: Fix irfft Implementation and Re-export

### Description
Modify the `irfft` function in `ffc.py` to explicitly handle dimension calculations and ensure consistent output across backends.

### Implementation

#### Step 1: Enhanced irfft with Dimension Validation
```python
# File: saicinpainting/training/modules/ffc.py

def irfft_fixed(REAL, IMAG, n=None, axis=-1, norm=None):
    """
    Enhanced irfft with explicit dimension validation and correction.

    Args:
        REAL: Real part of FFT (shape: [..., N/2+1])
        IMAG: Imaginary part of FFT (shape: [..., N/2+1])
        n: Target output length (if None, inferred from input)
        axis: Axis to perform inverse FFT on
        norm: Normalization mode

    Returns:
        Real-valued tensor of shape [..., n]
    """
    axis = axis if axis >= 0 else REAL.ndim + axis

    # CRITICAL: Explicitly calculate target length
    if n is None:
        # Standard: n = (fft_length - 1) * 2
        # For fft_length = 65: n = (65 - 1) * 2 = 128
        fft_length = REAL.shape[axis]
        n = (fft_length - 1) * 2

        # Handle edge case: if fft_length suggests odd original length
        # For fft_length = 64: could be n = 126 or n = 127
        # We assume even lengths for simplicity
        print(f"[DEBUG] irfft: Inferred n={n} from fft_length={fft_length}")

    # Validate input dimensions
    expected_fft_length = n // 2 + 1
    actual_fft_length = REAL.shape[axis]

    if actual_fft_length != expected_fft_length:
        import logging
        logging.warning(
            f"irfft dimension mismatch: expected FFT length {expected_fft_length} "
            f"for output length {n}, but got {actual_fft_length}"
        )

        # Correction strategy: pad or trim
        if actual_fft_length < expected_fft_length:
            # Pad with zeros at the end (high frequencies)
            pad_length = expected_fft_length - actual_fft_length
            pad_shape = list(REAL.shape)
            pad_shape[axis] = pad_length

            REAL_pad = torch.zeros(pad_shape, dtype=REAL.dtype, device=REAL.device)
            IMAG_pad = torch.zeros(pad_shape, dtype=IMAG.dtype, device=IMAG.device)

            REAL = torch.cat([REAL, REAL_pad], dim=axis)
            IMAG = torch.cat([IMAG, IMAG_pad], dim=axis)

            logging.warning(f"Padded FFT from {actual_fft_length} to {expected_fft_length}")

        else:
            # Trim excess frequencies
            indices = [slice(None)] * REAL.ndim
            indices[axis] = slice(0, expected_fft_length)
            REAL = REAL[tuple(indices)]
            IMAG = IMAG[tuple(indices)]

            logging.warning(f"Trimmed FFT from {actual_fft_length} to {expected_fft_length}")

    # Standard irfft procedure
    # Exclude DC (index 0) and Nyquist (index -1), then flip
    REAL_flipped = torch.flip(REAL[..., 1:-1], dims=[axis])
    IMAG_flipped = torch.flip(IMAG[..., 1:-1], dims=[axis])

    # Conjugate
    IMAG_flipped_conj = -IMAG_flipped

    # Concatenate: [DC, f1, ..., fN/2] + [fN/2-1, ..., f1]
    REAL_extended = torch.cat([REAL, REAL_flipped], dim=axis)
    IMAG_extended = torch.cat([IMAG, IMAG_flipped_conj], dim=axis)

    # Verify concatenated length matches target
    assert REAL_extended.shape[axis] == n, \
        f"Concatenated length {REAL_extended.shape[axis]} != target {n}"

    # Perform inverse FFT
    REAL_output = ifft1d(REAL_extended, IMAG_extended, axis=axis)[0]

    # Verify output length
    assert REAL_output.shape[axis] == n, \
        f"Output length {REAL_output.shape[axis]} != target {n}"

    # Permute back to expected format
    REAL_output = REAL_output.permute(0, 1, 3, 2)

    return REAL_output


# Replace the original irfft
irfft = irfft_fixed
```

#### Step 2: Update ifft2d
```python
def ifft2d_fixed(REAL, IMAG, shape=None):
    """
    2D inverse FFT with explicit shape specification

    Args:
        REAL, IMAG: Real and imaginary parts
        shape: Target output shape (H, W). If None, inferred from input
    """
    # Get current shape
    current_h = REAL.shape[-2]
    current_w = REAL.shape[-1]  # This is W/2+1

    if shape is None:
        # Infer from current dimensions
        target_h = current_h
        target_w = (current_w - 1) * 2
        shape = (target_h, target_w)
    else:
        target_h, target_w = shape

    # Inverse FFT on height dimension first
    REAL, IMAG = ifft1d(REAL, IMAG, n=target_h, axis=-2)

    # Inverse FFT on width dimension
    REAL = irfft_fixed(REAL, IMAG, n=target_w, axis=-1)

    return REAL

# Replace
ifft2d = ifft2d_fixed
```

#### Step 3: Re-export ONNX Model
```python
# In export notebook
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable dimension debugging
torch.set_printoptions(profile="full")

# Export with fixed FFT
torch.onnx.export(
    exported_model,
    (torch.rand(1, 3, 512, 512).float(), torch.rand(1, 1, 512, 512).float()),
    "lama_fixed_fft.onnx",
    input_names=["image", "mask"],
    output_names=["output"],
    dynamic_axes={
        "image": {0: "batch"},
        "mask": {0: "batch"},
        "output": {0: "batch"}
    },
    opset_version=17,
    verbose=True  # Enable verbose logging
)
```

### Pros
- ✅ Fixes root cause
- ✅ Maintains model accuracy
- ✅ Adds robustness with dimension validation

### Cons
- ❌ Requires re-export of model
- ❌ More complex implementation
- ❌ Needs thorough testing

### Testing Checklist
- [ ] Test irfft_fixed with various input lengths (64, 127, 128, 256, 512)
- [ ] Verify ifft2d produces correct output shapes
- [ ] Export new ONNX model
- [ ] Test with WASM backend (baseline)
- [ ] Test with WebGPU backend
- [ ] Compare outputs for accuracy
- [ ] Test with different input resolutions

---

## Solution 3: Use Native ONNX FFT Operators

### Description
Export the model using ONNX's native DFT (Discrete Fourier Transform) operator introduced in opset 18, eliminating custom FFT implementations.

### Implementation

#### Step 1: Update FourierUnit for Native Export
```python
class FourierUnitNative(nn.Module):
    """FourierUnit using PyTorch native FFT for ONNX export"""

    def __init__(self, in_channels, out_channels, groups=1, **kwargs):
        super(FourierUnitNative, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0,
            groups=self.groups, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fft_norm = kwargs.get('fft_norm', 'ortho')

    def forward(self, x):
        batch = x.shape[0]

        # Use PyTorch native FFT (exports to ONNX DFT operator)
        ffted = torch.fft.rfft2(x, norm=self.fft_norm)

        # Convert complex to real representation
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # Process in frequency domain
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # Convert back to complex
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # Inverse FFT with explicit output size
        output_size = x.shape[-2:]
        output = torch.fft.irfft2(ffted, s=output_size, norm=self.fft_norm)

        return output
```

#### Step 2: Export with Opset 18
```python
# Export configuration
torch.onnx.export(
    model,
    (image, mask),
    "lama_native_fft.onnx",
    opset_version=18,  # Required for DFT operator
    input_names=["image", "mask"],
    output_names=["output"],
    dynamic_axes={
        "image": {0: "batch", 2: "height", 3: "width"},
        "mask": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"}
    },
    do_constant_folding=True,
    verbose=True
)
```

### Pros
- ✅ Standard ONNX operators
- ✅ Better backend support
- ✅ Cleaner implementation
- ✅ Supports dynamic shapes

### Cons
- ❌ Requires opset 18 (not all runtimes support it yet)
- ❌ Significant re-work needed
- ❌ May have performance implications

### Testing Checklist
- [ ] Verify ONNX Runtime supports opset 18 DFT
- [ ] Export model with native FFT
- [ ] Test with CPUExecutionProvider
- [ ] Test with WebGPU
- [ ] Compare performance vs custom FFT
- [ ] Verify accuracy matches original

---

## Solution 4: Report Upstream to ONNX Runtime

### Description
File a detailed bug report with the onnxruntime-web team about WebGPU FFT dimension handling inconsistencies.

### Implementation

#### Bug Report Template
```markdown
## Title
WebGPU execution provider: Tensor dimension mismatch in FFT-based operations

## Environment
- onnxruntime-web version: [version]
- Browser: [Chrome/Edge version]
- WebGPU support: Yes
- Model: LaMa inpainting (ONNX)

## Description
When using the WebGPU execution provider, a binary Add operation fails with dimension mismatch error in models containing FFT operations. The WASM backend works correctly.

## Error Message
```
[WebGPU] Kernel "[Add] /generator/model/model.5/conv1/ffc/convg2g/Add" failed.
Error: Can't perform binary op on the given tensors
```

## Reproduction
Tensor shapes at Add operation:
- Input A: [1, 64, 64, 192]
- Input B: [1, 64, 63, 192]

The tensors come from FFT operations where irfft produces different dimensions in WebGPU vs WASM.

## Expected Behavior
Both backends should produce identical tensor dimensions after FFT/iFFT operations.

## Minimal Reproducible Example
[Attach simplified ONNX model showing the issue]

## Additional Context
- Related to rfft output dimension calculation (N/2+1)
- irfft reconstruction uses slicing [1:-1] which depends on exact FFT output length
- WASM produces expected dimensions, WebGPU off-by-one

## Logs
[Attach WebGPU debug logs]
```

### Follow-up Actions
1. Monitor issue for responses
2. Provide additional information if requested
3. Test any proposed patches
4. Update case study with findings

---

## Recommended Implementation Strategy

### Phase 1: Immediate (Week 1)
1. ✅ Complete case study documentation
2. ⏭️ Implement Solution 1 (ONNX graph padding)
3. ⏭️ Test padded model with WebGPU
4. ⏭️ File upstream bug report (Solution 4)

### Phase 2: Short-term (Weeks 2-3)
1. ⏭️ Implement Solution 2 (fix irfft)
2. ⏭️ Re-export ONNX model with fixes
3. ⏭️ Comprehensive testing across backends
4. ⏭️ Performance benchmarking

### Phase 3: Long-term (Month 2+)
1. ⏭️ Explore Solution 3 (native ONNX FFT)
2. ⏭️ Wait for upstream fixes
3. ⏭️ Integrate improvements
4. ⏭️ Add dynamic shape support

---

## Success Criteria

### Must Have
- [ ] WebGPU backend runs without errors
- [ ] Output accuracy within 1% of WASM baseline
- [ ] No performance regression
- [ ] Works with 512x512 input

### Should Have
- [ ] Automated tests for all backends
- [ ] Documentation updated
- [ ] Example code provided
- [ ] CI/CD integration

### Nice to Have
- [ ] Dynamic shape support
- [ ] Multiple resolution support
- [ ] FP16 optimization
- [ ] Mobile deployment support

---

**Last Updated**: 2025-12-02
**Version**: 1.0
**Status**: Ready for Implementation
