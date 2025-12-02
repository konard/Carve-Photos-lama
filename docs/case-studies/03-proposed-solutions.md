# Proposed Solutions: LaMa ONNX DirectML GPU Inference

## Overview

This document outlines potential solutions to enable GPU inference for the LaMa ONNX model on Windows, addressing the DirectML compatibility issues identified in the root cause analysis.

## Solution Categories

### Category A: Workarounds (No Model Changes Required)

#### A1. CPU Execution Provider (Current Recommendation)

**Description:** Use the CPU execution provider instead of DirectML for inference.

**Implementation:**
```csharp
// C# ONNX Runtime
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider_CPU(0);
var session = new InferenceSession("lama_fp32.onnx", sessionOptions);
```

```python
# Python ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("lama_fp32.onnx", providers=['CPUExecutionProvider'])
```

**Pros:**
- Works immediately, no model changes
- Cross-platform compatibility
- Stable and well-tested

**Cons:**
- Significantly slower than GPU execution
- No hardware acceleration

**Estimated Performance:** 2-10x slower than GPU execution

---

#### A2. Hybrid Execution (Partial GPU Offload)

**Description:** Configure ONNX Runtime to use DirectML for compatible operations and fall back to CPU for incompatible ones.

**Implementation:**
```python
import onnxruntime as ort

# DirectML as primary, CPU as fallback
providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("lama_fp32.onnx", providers=providers)
```

**Pros:**
- Some GPU acceleration for compatible operations
- Automatic fallback for problematic nodes
- No model modification required

**Cons:**
- May still crash on some configurations
- Performance benefit limited by CPU-GPU data transfers
- Unpredictable behavior

**Status:** Needs testing - may or may not work depending on ONNX Runtime version

---

#### A3. Use CUDA Instead of DirectML (NVIDIA GPUs Only)

**Description:** For users with NVIDIA GPUs, use CUDAExecutionProvider instead of DirectML.

**Implementation:**
```python
import onnxruntime as ort
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("lama_fp32.onnx", providers=providers)
```

**Pros:**
- Native FFT support via cuFFT
- Better tested with this model
- Full GPU acceleration

**Cons:**
- Requires NVIDIA GPU hardware
- Requires CUDA toolkit installation
- Not available for AMD/Intel GPUs

---

### Category B: Model Modifications

#### B1. Re-export with Native ONNX DFT Operators

**Description:** Modify the export process to use ONNX's native DFT operator instead of MatMul-based FFT.

**Technical Approach:**
1. Update `FourierUnitJIT` to use ONNX opset 17+ DFT operator
2. Re-export model with the new implementation
3. Test with DirectML

**Implementation Sketch:**
```python
# In ffc.py, replace custom rfft/ifft with ONNX DFT approach
# This requires changes to the export process

class FourierUnitONNX(nn.Module):
    def forward(self, x):
        # Use torch operations that map to ONNX DFT
        # opset_version=17 includes DFT operator
        pass
```

**Pros:**
- Uses standard ONNX operators
- Better execution provider compatibility
- Cleaner model graph

**Cons:**
- Significant development effort required
- DFT operator support may still be limited in DirectML
- Requires testing across multiple platforms

**Complexity:** High

---

#### B2. Graph Optimization and Operator Substitution

**Description:** Post-process the ONNX model to replace problematic MatMul operations with DirectML-compatible alternatives.

**Technical Approach:**
1. Load the ONNX model
2. Identify problematic MatMul nodes in FFT path
3. Replace with functionally equivalent operations
4. Validate output correctness

**Tools:**
- ONNX GraphSurgeon
- ONNX Optimizer
- Custom transformation scripts

**Implementation Sketch:**
```python
import onnx
from onnx import helper, numpy_helper
import onnx_graphsurgeon as gs

# Load and modify graph
graph = gs.import_onnx(onnx.load("lama_fp32.onnx"))

# Find and replace problematic nodes
for node in graph.nodes:
    if "rttn/MatMul" in node.name:
        # Apply transformation
        pass

# Export modified model
onnx.save(gs.export_onnx(graph), "lama_fp32_directml.onnx")
```

**Pros:**
- No retraining required
- Can target specific problematic operations
- Preserves model weights

**Cons:**
- Complex to implement correctly
- Risk of numerical accuracy issues
- May require deep understanding of model internals

**Complexity:** High

---

#### B3. Model Splitting and Piecewise Execution

**Description:** Split the model into DirectML-compatible and incompatible sections, execute separately.

**Technical Approach:**
1. Identify the boundary where FFT operations begin
2. Export two models: pre-FFT and FFT-onwards
3. Execute pre-FFT on GPU (DirectML)
4. Execute FFT section on CPU
5. Continue remaining operations on GPU

**Pros:**
- Maximizes GPU utilization for compatible parts
- Maintains model accuracy
- Allows incremental improvements

**Cons:**
- Complex orchestration required
- Multiple data transfers between GPU and CPU
- Maintenance overhead

**Complexity:** Medium-High

---

### Category C: Alternative Approaches

#### C1. WebNN/WinML Migration

**Description:** Microsoft recommends WinML for new Windows deployments. Migrate the execution to WinML API.

**Background:**
> "DirectML is in sustained engineering mode. DirectML continues to be supported, but new feature development has moved to WinML for Windows-based ONNX Runtime deployments."

**Implementation:**
```csharp
// Windows ML API (UWP/WinRT)
using Windows.AI.MachineLearning;

var model = await LearningModel.LoadFromFilePath("lama_fp32.onnx");
var device = new LearningModelDevice(LearningModelDeviceKind.DirectXHighPerformance);
var session = new LearningModelSession(model, device);
```

**Pros:**
- Actively developed by Microsoft
- Better Windows integration
- May have improved operator support

**Cons:**
- Different API, requires code changes
- Windows-only
- May have same underlying DirectML limitations

---

#### C2. TensorRT Conversion (NVIDIA Only)

**Description:** Convert the ONNX model to TensorRT format for optimal NVIDIA GPU performance.

**Implementation:**
```bash
# Using trtexec
trtexec --onnx=lama_fp32.onnx --saveEngine=lama_fp32.trt
```

**Pros:**
- Best performance on NVIDIA hardware
- Officially supported by Carve team
- Includes optimizations like layer fusion

**Cons:**
- NVIDIA-only
- Requires TensorRT installation
- Different deployment requirements

---

#### C3. OpenVINO Conversion (Intel GPUs)

**Description:** Convert to OpenVINO format for Intel GPU acceleration.

**Implementation:**
```bash
# OpenVINO Model Optimizer
mo --input_model lama_fp32.onnx --output_dir ./openvino_model
```

**Pros:**
- Good Intel GPU support
- CPU optimization included
- Cross-platform on Intel hardware

**Cons:**
- Intel-specific
- Additional framework dependency
- May have similar FFT limitations

---

## Recommended Solution Path

### For Immediate Use:
1. **A1 (CPU Execution)** - Works now, no changes needed

### For NVIDIA GPU Users:
2. **A3 (CUDA Provider)** - Best performance with available hardware

### For Long-term Fix:
3. **B1 (Native ONNX DFT)** - Requires development effort but provides proper solution

### For Windows-Specific Deployment:
4. **C1 (WinML Migration)** - Aligns with Microsoft's direction

## Solution Comparison Matrix

| Solution | Effort | Performance | Compatibility | Risk |
|----------|--------|-------------|---------------|------|
| A1: CPU | None | Low | High | None |
| A2: Hybrid | Low | Medium | Medium | Medium |
| A3: CUDA | Low | High | NVIDIA only | Low |
| B1: ONNX DFT | High | High | High | Medium |
| B2: Graph Opt | High | High | Unknown | High |
| B3: Split Model | Medium | Medium | High | Medium |
| C1: WinML | Medium | High | Windows only | Medium |
| C2: TensorRT | Low | Highest | NVIDIA only | Low |
| C3: OpenVINO | Low | High | Intel only | Low |

## Next Steps

1. **Community Testing:** Test A2 (Hybrid Execution) to determine if it provides any improvement
2. **Documentation:** Document the CPU fallback as the recommended approach for DirectML users
3. **Feature Request:** File an issue with ONNX Runtime team regarding DirectML FFT/MatMul support
4. **Long-term:** Investigate B1 (Native ONNX DFT) as the proper architectural fix

## References

- ONNX Runtime DirectML Documentation: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- Windows ML Documentation: https://docs.microsoft.com/en-us/windows/ai/windows-ml/
- ONNX DFT Operator: https://onnx.ai/onnx/operators/onnx__DFT.html
- TensorRT Documentation: https://developer.nvidia.com/tensorrt
