# Proposed Solutions: FP16 Black Output Issue

## Overview

This document presents multiple solution approaches to address the FP16 numerical overflow issue in the LaMa ONNX model when using MNN. Solutions are ordered by effectiveness and feasibility.

---

## Solution 1: Mixed Precision - Keep FFT Operations in FP32 (RECOMMENDED)

### Description
Selectively maintain FP32 precision for FFT-sensitive operations while using FP16 for other layers.

### Implementation Strategy

#### A. Identify Critical Operations
Mark the following operations to remain in FP32:
1. FFT forward transforms (rfft, fft functions)
2. FFT inverse transforms (irfft, ifft2d functions)
3. Complex number operations in frequency domain
4. Normalization divisions (sqrt operations)

#### B. ONNX Model Modification
```python
# During ONNX export, mark FFT nodes to preserve FP32
import onnx
from onnx import helper, TensorProto

def mark_fft_ops_fp32(model_path, output_path):
    model = onnx.load(model_path)

    # List of operation names to keep in FP32
    fp32_ops = [
        'MatMul',      # FFT matrix multiplications
        'Div',         # Normalization divisions
        'Sqrt',        # Square root operations
        'Sin', 'Cos',  # Trigonometric functions
    ]

    # Add precision hints to nodes
    for node in model.graph.node:
        if node.op_type in fp32_ops:
            # Check if node is in FFT-related subgraph
            if is_fft_operation(node, model):
                attr = helper.make_attribute("precision", "FP32")
                node.attribute.append(attr)

    onnx.save(model, output_path)
```

#### C. MNN Conversion Configuration
```python
# MNN conversion tool configuration
config = {
    "version": "2.0",
    "precision": "low",  # FP16 default
    "fp32_ops": [
        # Specify layers by name pattern
        "FourierUnit*",
        "SpectralTransform*",
        "*rfft*",
        "*ifft*",
        "*fft*"
    ]
}
```

### Advantages
✅ **Most effective**: Directly addresses root cause
✅ **Minimal accuracy loss**: Critical operations maintain precision
✅ **Reasonable performance**: Only ~10-20% of operations use FP32
✅ **Proven approach**: Widely used in production ML systems

### Disadvantages
⚠️ **Requires conversion tool support**: MNN must support mixed precision
⚠️ **Slightly larger model**: FP32 weights are 2× size of FP16
⚠️ **Implementation complexity**: Requires careful operator identification

### Estimated Impact
- **Model size increase**: +15-20% (vs pure FP16)
- **Inference speed**: 5-10% slower than pure FP16
- **Accuracy**: Equivalent to FP32
- **Success rate**: Should fix 100% of black output cases

---

## Solution 2: Input Normalization and Output Scaling

### Description
Preprocess inputs and scale intermediate values to keep them within FP16 safe range.

### Implementation Strategy

#### A. Input Preprocessing
```python
def preprocess_for_fp16(image):
    """Normalize input to safe FP16 range."""
    # Convert to [0, 1] range
    image = image.astype(np.float32) / 255.0

    # Scale to smaller range to prevent accumulation overflow
    # Use 0.9 safety margin
    image = image * 0.5  # Now in [0, 0.5]

    return image
```

#### B. Model Output Scaling
```python
def postprocess_from_fp16(output):
    """Scale output back to original range."""
    # Reverse the input scaling
    output = output * 2.0  # Reverse 0.5 scaling

    # Convert back to [0, 255]
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return output
```

#### C. Internal Layer Scaling
Modify ONNX model to add scaling layers:
```python
# After each FFT operation, add scaling
scale_factor = 0.1  # Reduce magnitude by 10×

# Insert scale operation
scaled_output = fft_output * scale_factor

# Before inverse FFT, reverse scaling
unscaled_input = scaled_input / scale_factor
```

### Advantages
✅ **Simple to implement**: No special conversion tool features needed
✅ **Universally applicable**: Works with any backend (CPU/GPU)
✅ **Full FP16 benefits**: Maintains speed and size advantages
✅ **No model architecture changes**: Only data preprocessing

### Disadvantages
⚠️ **Reduces dynamic range**: May lose detail in subtle textures
⚠️ **Requires careful tuning**: Scale factors need optimization
⚠️ **Not guaranteed**: Extreme images may still overflow
⚠️ **User workflow change**: Requires specific preprocessing pipeline

### Estimated Impact
- **Success rate**: 90-95% of previously failing images
- **Accuracy loss**: Minor (1-3% quality degradation)
- **Inference speed**: No change
- **Implementation effort**: Low-Medium

---

## Solution 3: Dynamic Range Quantization with Calibration

### Description
Use quantization calibration to learn optimal scale factors for each layer based on representative data.

### Implementation Strategy

#### A. Collect Calibration Dataset
```python
# Gather diverse images that trigger overflow
calibration_images = [
    bright_images,      # High values
    high_contrast,      # Large range
    uniform_regions,    # Accumulation risk
    normal_images       # Baseline
]
```

#### B. Calibration Process
```python
def calibrate_fp16_conversion(model, calibration_data):
    """
    Run calibration to find per-layer scale factors.
    """
    layer_stats = {}

    # Run inference and collect statistics
    for image in calibration_data:
        activations = collect_all_activations(model, image)

        for layer_name, values in activations.items():
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    'min': float('inf'),
                    'max': float('-inf')
                }

            layer_stats[layer_name]['min'] = min(
                layer_stats[layer_name]['min'],
                values.min()
            )
            layer_stats[layer_name]['max'] = max(
                layer_stats[layer_name]['max'],
                values.max()
            )

    # Calculate scale factors
    scale_factors = {}
    FP16_MAX = 65504.0

    for layer_name, stats in layer_stats.items():
        max_abs = max(abs(stats['min']), abs(stats['max']))
        # Add 10% safety margin
        scale_factors[layer_name] = FP16_MAX * 0.9 / max_abs

    return scale_factors
```

#### C. Apply to ONNX Model
```python
def insert_scaling_layers(onnx_model, scale_factors):
    """Insert scale/unscale operations around sensitive layers."""
    for layer_name, scale in scale_factors.items():
        # Add scaling after layer
        # Add unscaling before next layer
        pass
```

### Advantages
✅ **Data-driven**: Optimized for actual usage patterns
✅ **Adaptive**: Different scales for different layers
✅ **Maintains FP16**: Full performance benefits
✅ **Scientifically sound**: Standard quantization approach

### Disadvantages
⚠️ **Requires tooling**: Need quantization-aware framework
⚠️ **Calibration overhead**: One-time setup cost
⚠️ **Distribution-dependent**: May fail on out-of-distribution images
⚠️ **Complex implementation**: Requires ONNX graph modification

### Estimated Impact
- **Success rate**: 95-98% of images
- **Accuracy**: Minimal loss with good calibration
- **Model size**: +5-10% (scale parameters)
- **Implementation effort**: High

---

## Solution 4: Backend Switch (Workaround)

### Description
Use MNN's OpenCL backend instead of CPU backend, as reported to work in MNN issue #2977.

### Implementation

```python
# MNN configuration
config = MNN.Config()
config.backend = MNN.Backend.OPENCL  # Instead of CPU
config.precision = MNN.Precision.LOW  # FP16

# Create session
session = MNN.Session(model, config)
```

### Advantages
✅ **Immediate solution**: No model changes needed
✅ **Proven to work**: Confirmed by MNN user
✅ **Simple**: One-line configuration change
✅ **Full FP16 benefits**: Maintains speed/size

### Disadvantages
⚠️ **Platform-limited**: Requires OpenCL support
⚠️ **Not universal**: Doesn't address root cause
⚠️ **Compatibility**: May not work on all devices
⚠️ **Masked issue**: Problem could resurface on other backends

### Estimated Impact
- **Success rate**: 100% on OpenCL-compatible devices
- **Platform coverage**: ~70-80% of target devices
- **Implementation effort**: Very low

---

## Solution 5: Precision Fallback Detection

### Description
Detect when FP16 produces invalid results and automatically fallback to FP32.

### Implementation Strategy

#### A. Output Validation
```python
def detect_fp16_failure(output):
    """Detect if FP16 inference failed."""
    # Check for all-black output
    if np.allclose(output, 0, atol=1e-6):
        return True

    # Check for NaN/Inf
    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
        return True

    # Check for abnormally low variance
    if np.std(output) < 0.01:
        return True

    return False

def inference_with_fallback(image, fp16_model, fp32_model):
    """Try FP16, fallback to FP32 if needed."""
    result_fp16 = fp16_model.infer(image)

    if detect_fp16_failure(result_fp16):
        # Fallback to FP32
        result = fp32_model.infer(image)
    else:
        result = result_fp16

    return result
```

### Advantages
✅ **Robust**: Always produces valid output
✅ **Best of both worlds**: FP16 speed when safe, FP32 accuracy when needed
✅ **Transparent**: User doesn't need to know which path was used
✅ **No model changes**: Works with existing models

### Disadvantages
⚠️ **Requires both models**: 2× storage
⚠️ **Detection overhead**: Extra computation per inference
⚠️ **Unpredictable latency**: Varies by image
⚠️ **Waste of failed inference**: FP16 computation discarded

### Estimated Impact
- **Reliability**: 100% (always valid output)
- **Average speed**: 10-20% slower than pure FP16
- **Storage requirement**: 2× (both models)
- **Implementation effort**: Low

---

## Solution 6: Model Architecture Modification (Long-term)

### Description
Redesign model to be inherently FP16-friendly by reducing FFT dependency.

### Potential Approaches

#### A. Replace FFT with Depthwise Convolutions
- Use large receptive fields with efficient depthwise separable convolutions
- Approximate global context without frequency domain

#### B. Use Attention Mechanisms
- Replace FFC modules with self-attention
- More numerically stable in FP16

#### C. Hybrid Architecture
- Keep FFT for final layers only
- Use standard convolutions for early layers

### Advantages
✅ **Fundamental solution**: Eliminates root cause
✅ **Future-proof**: Compatible with all quantization schemes
✅ **Potentially better**: Opportunity for architecture improvements

### Disadvantages
⚠️ **Requires retraining**: Entire model must be retrained
⚠️ **Research effort**: Significant development time
⚠️ **Validation needed**: May affect quality
⚠️ **Not backward compatible**: Different model architecture

### Estimated Impact
- **Timeline**: 3-6 months research + development
- **Success**: Uncertain without experimentation
- **Compatibility**: Breaks existing deployments

---

## Recommendation Matrix

| Solution | Effectiveness | Effort | Compatibility | Timeline |
|----------|--------------|--------|---------------|----------|
| 1. Mixed Precision | ⭐⭐⭐⭐⭐ | Medium | Good | 1-2 weeks |
| 2. Input Scaling | ⭐⭐⭐⭐ | Low | Excellent | 2-3 days |
| 3. Calibration | ⭐⭐⭐⭐⭐ | High | Good | 2-3 weeks |
| 4. Backend Switch | ⭐⭐⭐ | Very Low | Limited | Immediate |
| 5. Fallback Detection | ⭐⭐⭐⭐ | Low | Excellent | 1 week |
| 6. Architecture Change | ⭐⭐⭐⭐⭐ | Very High | Breaking | 3-6 months |

---

## Recommended Implementation Path

### Phase 1: Immediate (Days 1-3)
1. **Implement Solution 4**: Switch to OpenCL backend as temporary workaround
2. **Implement Solution 2**: Add input normalization for broader compatibility
3. **Document**: Update user guide with preprocessing requirements

### Phase 2: Short-term (Weeks 1-2)
1. **Implement Solution 1**: Mixed precision conversion
2. **Test extensively**: Validate on diverse image dataset
3. **Benchmark**: Measure performance impact

### Phase 3: Medium-term (Weeks 3-4)
1. **Implement Solution 5**: Fallback detection for robustness
2. **Optimize**: Fine-tune scale factors and thresholds
3. **Release**: Production-ready FP16 model with safeguards

### Phase 4: Long-term (Optional, Months 3-6)
1. **Research Solution 6**: Explore architecture improvements
2. **Evaluate trade-offs**: Quality vs efficiency
3. **Next generation**: Deploy improved architecture if beneficial

---

## Testing & Validation Plan

### Test Dataset Requirements
1. **Bright images**: Values near 255
2. **High contrast**: Large dynamic range
3. **Uniform regions**: Solid colors
4. **Large dimensions**: 512×512 and above
5. **Normal images**: Typical use cases

### Success Criteria
- ✅ 0% black output failures
- ✅ <2% quality degradation vs FP32
- ✅ >50% speedup vs FP32 (maintain FP16 benefits)
- ✅ Works across MNN CPU and GPU backends

### Metrics to Monitor
- Peak memory usage
- Inference latency (p50, p95, p99)
- Model file size
- Output quality (SSIM, PSNR vs FP32 baseline)
- Failure rate on diverse image set (10k+ images)
