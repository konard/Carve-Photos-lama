# Proposed Solutions: LaMa ONNX DirectML GPU Inference

## Overview

This document outlines potential solutions to enable GPU inference for the LaMa ONNX model on Windows, addressing the DirectML compatibility issues identified in the root cause analysis.

## Key Research Findings

Before diving into solutions, here are the critical findings from our investigation:

### DirectML DFT Support Status

| Component | DFT Support | Notes |
|-----------|-------------|-------|
| DirectML Native API | ❌ No | No DFT/FFT operator in DML_OPERATOR_TYPE enumeration |
| ONNX Runtime DirectML EP | ✅ Yes (since ~v1.13) | PR #12710 adds DFT kernel via Stockham FFT shader |
| ONNX DFT Operator | ✅ Yes (opset 17+) | Standard operator, v20 is current |

### Why PyTorch FFT → ONNX is Problematic

The root issue is that **PyTorch's `torch.fft.*` functions cannot be directly exported to ONNX**:

1. **Legacy exporter**: Does not support `aten::fft_rfft`, `aten::fft_rfftn`, etc.
2. **Dynamo exporter**: Has ongoing issues with FFT operators (GitHub issues #128324, #133785, #142458)
3. **Complex tensor support**: ONNX/DirectML complex number handling is limited

This is why the LaMa model uses a custom MatMul-based FFT implementation - it was the only way to export to ONNX at the time.

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

**Critical Finding:** ONNX Runtime DirectML **does support DFT** via a Stockham FFT shader implementation added in PR #12710. This makes B1 a **viable and promising solution**.

**Technical Approach:**
1. Create a custom export function that maps FFT operations to ONNX DFT nodes
2. Use `torch.onnx.export` with custom operator registration
3. Target opset 17+ which includes the DFT operator
4. Test specifically with DirectML execution provider

**Implementation Sketch:**
```python
import torch
import torch.onnx
from torch.onnx import register_custom_op_symbolic

# Register custom symbolic for rfftn that maps to ONNX DFT
def custom_rfftn_symbolic(g, input, s, dim, norm):
    # Map to ONNX DFT operator (opset 17+)
    # DFT inputs: input, dft_length (optional), axis (optional)
    return g.op("DFT", input,
                axis_i=dim[-1],  # Last dimension for rfft
                inverse_i=0,     # Forward transform
                onesided_i=1)    # One-sided for real input

register_custom_op_symbolic('aten::fft_rfftn', custom_rfftn_symbolic, 17)

# Export with custom symbolics
torch.onnx.export(model, dummy_input, "lama_dft.onnx",
                  opset_version=17,
                  custom_opsets={"": 17})
```

**Pros:**
- Uses standard ONNX DFT operator with DirectML support (Stockham shader)
- Better execution provider compatibility
- Cleaner model graph
- DirectML has proven DFT implementation

**Cons:**
- Requires careful mapping of 2D RFFT to ONNX DFT semantics
- May need multiple DFT nodes for multi-dimensional transforms
- Testing required to verify numerical accuracy

**Complexity:** High (but feasible)

**Key Reference:** [ONNX Runtime PR #12710](https://github.com/microsoft/onnxruntime/pull/12710) - DFT on DirectML

---

#### B1.5. ONNX Graph Surgery: Replace MatMul FFT with DFT Nodes

**Description:** Post-process the existing ONNX model to replace the custom MatMul-based FFT subgraph with native ONNX DFT operators.

**Technical Approach:**
1. Identify MatMul subgraphs that implement DFT (pattern matching)
2. Replace entire subgraph with single DFT node
3. Verify numerical equivalence

**Implementation Sketch:**
```python
import onnx
from onnx import helper, TensorProto
import onnx_graphsurgeon as gs

# Load model
graph = gs.import_onnx(onnx.load("lama_fp32.onnx"))

# Find RFFTTN pattern (cos_part/sin_part matmuls)
for node in graph.nodes:
    if is_fft_pattern(node):  # Custom pattern matching
        # Create DFT node
        dft_node = gs.Node(
            op="DFT",
            inputs=[node.inputs[0]],  # Original input
            outputs=[node.outputs[0]],
            attrs={"inverse": 0, "onesided": 1}
        )
        # Replace in graph
        graph.replace_node(node, dft_node)

# Update opset to 17+
model = gs.export_onnx(graph)
model.opset_import[0].version = 17
onnx.save(model, "lama_dft_directml.onnx")
```

**Pros:**
- No PyTorch/training code changes needed
- Works with existing exported model
- Can be automated for other models with same pattern

**Cons:**
- Complex pattern matching required
- Must handle all FFT variants (rfft, fft, irfft, ifft)
- Risk of missing edge cases

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

#### C4. ROCm on Linux (AMD GPUs)

**Description:** For AMD GPU users, use ROCm on Linux instead of DirectML on Windows.

**Implementation:**
```bash
# Install ROCm-enabled PyTorch
pip install torch --extra-index-url https://download.pytorch.org/whl/rocm5.2/

# For unsupported AMD GPUs (e.g., Navi 23 gfx1032)
HSA_OVERRIDE_GFX_VERSION=10.3.0 python infer.py
```

**Pros:**
- Native AMD GPU support on Linux
- Uses PyTorch directly (no ONNX needed)
- Active development and community support

**Cons:**
- Linux-only (not available on Windows)
- Some AMD GPU models need workarounds
- Different deployment environment

---

#### C5. WebNN Backend (Browser/Emerging)

**Description:** Use WebNN for cross-platform GPU acceleration in web contexts.

**Background:** WebNN is emerging as a standard for ML acceleration across browsers and operating systems. ONNX Runtime Web supports WebNN as an execution provider.

**Implementation:**
```javascript
// ONNX Runtime Web with WebNN
const session = await ort.InferenceSession.create('lama_fp32.onnx', {
  executionProviders: ['webnn']
});
```

**Pros:**
- Cross-platform (browser-based)
- Hardware agnostic
- No installation required for end users

**Cons:**
- Still maturing
- May have limited operator support
- Performance may vary by browser/hardware

---

### Category D: Architecture-Level Solutions

#### D1. Replace FFT with Learned Spatial Operations

**Description:** Research suggests FFT in inpainting can be replaced or augmented with learned spatial operations that are fully ONNX/DirectML compatible.

**Background:** The ICCV 2023 paper "Rethinking Fast Fourier Convolution in Image Inpainting" (UFFC) identifies key issues with vanilla FFC and proposes modifications that could be more hardware-friendly.

**Technical Approach:**
1. Study UFFC (Unbiased Fast Fourier Convolution) modifications
2. Implement alternative frequency-domain processing using standard operators
3. Fine-tune or retrain model with new architecture

**Considerations:**
- UFFC uses "range transform and inverse transform" before/after activation
- Adds "adaptive clipping" to prevent extreme values
- May require model retraining but produces cleaner ONNX graphs

**Pros:**
- Addresses fundamental architectural issues
- Better numerical stability
- Standard operator compatibility

**Cons:**
- Requires model retraining
- Significant research and development effort
- May affect output quality

**Complexity:** Very High

**Reference:** [UFFC Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.pdf)

---

#### D2. Spatial-Domain Only Model Variant

**Description:** Create a LaMa variant that uses only spatial-domain convolutions, sacrificing some global context capability for full GPU compatibility.

**Technical Approach:**
1. Replace FourierUnit with dilated convolutions or attention mechanisms
2. Use large receptive field alternatives (e.g., Swin Transformer blocks)
3. Fine-tune on inpainting datasets

**Pros:**
- Full DirectML/GPU compatibility
- No FFT-related issues
- Standard deep learning operators only

**Cons:**
- May reduce inpainting quality for large masks
- Requires substantial retraining
- Different model characteristics

**Complexity:** Very High

---

## Recommended Solution Path

### For Immediate Use:
1. **A1 (CPU Execution)** - Works now, no changes needed

### For NVIDIA GPU Users:
2. **A3 (CUDA Provider)** - Best performance with available hardware

### For AMD GPU Users on Linux:
3. **C4 (ROCm)** - Native PyTorch support, no ONNX needed

### For Long-term Fix (Most Promising):
4. **B1 (Native ONNX DFT)** or **B1.5 (Graph Surgery)** - ONNX Runtime DirectML **does support DFT** via Stockham FFT shader. This is the most promising path forward.

### For Windows-Specific Deployment:
5. **C1 (WinML Migration)** - Aligns with Microsoft's direction

## Solution Comparison Matrix

| Solution | Effort | Performance | Compatibility | Risk | Priority |
|----------|--------|-------------|---------------|------|----------|
| A1: CPU | None | Low | High | None | ⭐ Immediate |
| A2: Hybrid | Low | Medium | Medium | Medium | Test |
| A3: CUDA | Low | High | NVIDIA only | Low | ⭐ NVIDIA users |
| **B1: ONNX DFT** | High | High | **High** | Medium | ⭐⭐ **Best long-term** |
| B1.5: Graph Surgery | High | High | High | Medium | Alternative to B1 |
| B2: Graph Opt | High | High | Unknown | High | Research |
| B3: Split Model | Medium | Medium | High | Medium | Fallback |
| C1: WinML | Medium | High | Windows only | Medium | MS-aligned |
| C2: TensorRT | Low | Highest | NVIDIA only | Low | ⭐ NVIDIA best |
| C3: OpenVINO | Low | High | Intel only | Low | Intel users |
| C4: ROCm | Low | High | AMD+Linux | Low | ⭐ AMD users |
| C5: WebNN | Medium | Medium | Browser | Medium | Emerging |
| D1: UFFC | Very High | High | High | High | Research |
| D2: Spatial-only | Very High | High | High | High | Last resort |

## Key Insight: DFT Support in DirectML

**The most important finding from this research is that ONNX Runtime DirectML DOES support the DFT operator** via a Stockham FFT shader (PR #12710, merged September 2022). This means:

1. The problem is NOT that DirectML cannot do FFT
2. The problem IS that the model uses MatMul-based FFT instead of the native DFT operator
3. **Solution B1/B1.5 can work** if we replace the MatMul FFT with native DFT nodes

## Next Steps

### Immediate Actions:
1. **Test A2 (Hybrid):** Verify if automatic fallback helps
2. **Verify DFT Support:** Test a simple ONNX model with DFT node on DirectML

### Development Actions:
3. **Implement B1 or B1.5:** Create a proof-of-concept that uses ONNX DFT operator
4. **Create test script:** Validate numerical accuracy of DFT-based vs MatMul-based FFT

### Community Actions:
5. **File ONNX Runtime issue:** Request guidance on MatMul→DFT conversion
6. **Contribute back:** If B1/B1.5 works, contribute DirectML-compatible model

## References

### Official Documentation
- ONNX Runtime DirectML: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- ONNX DFT Operator: https://onnx.ai/onnx/operators/onnx__DFT.html
- Windows ML: https://docs.microsoft.com/en-us/windows/ai/windows-ml/
- TensorRT: https://developer.nvidia.com/tensorrt

### Key GitHub Resources
- **DFT on DirectML PR #12710**: https://github.com/microsoft/onnxruntime/pull/12710
- DirectML Repository: https://github.com/microsoft/DirectML
- PyTorch FFT Export Issues: https://github.com/pytorch/pytorch/issues/112382

### Research Papers
- UFFC Paper (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/papers/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.pdf
- Original LaMa Paper: https://arxiv.org/abs/2109.07161

### Community Resources
- AMD GPU Guide for lama-cleaner: https://github.com/Sanster/lama-cleaner/issues/106
- LaMa ONNX on HuggingFace: https://huggingface.co/Carve/LaMa-ONNX
