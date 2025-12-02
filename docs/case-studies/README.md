# Case Study: LaMa ONNX Model DirectML GPU Inference Failure

## Abstract

This case study documents and analyzes an issue where the LaMa (Large Mask Inpainting) ONNX model fails to execute on Windows systems using the DirectML execution provider for GPU acceleration. The investigation reveals that the root cause is a fundamental incompatibility between the model's custom FFT (Fast Fourier Transform) implementation and DirectML's strict tensor operation validation.

## Table of Contents

1. [Background](#background)
2. [Problem Statement](#problem-statement)
3. [Investigation](#investigation)
4. [Root Cause](#root-cause)
5. [Impact Assessment](#impact-assessment)
6. [Proposed Solutions](#proposed-solutions)
7. [Recommendations](#recommendations)
8. [Appendix](#appendix)

## Background

### LaMa Model Overview

LaMa (Large Mask Inpainting) is a state-of-the-art image inpainting model developed by Samsung AI Center Moscow. The model uses a novel architecture called **Fast Fourier Convolutions (FFC)** that processes images in both spatial and frequency domains simultaneously.

**Key Innovation:** The FFC architecture enables the model to:
- Handle large mask regions (up to 90% of the image)
- Maintain structural coherence across inpainted regions
- Generalize to resolutions much higher than training data (256x256 -> 2K+)

### ONNX Export

The Carve.Photos team exported the LaMa model to ONNX format to enable cross-platform deployment. Two versions were created:

| Model | Export Method | Opset | Status |
|-------|---------------|-------|--------|
| `lama.onnx` | `torch.onnx.dynamo_export` | 18 | Not recommended |
| `lama_fp32.onnx` | `torch.onnx.export` | 17 | Recommended |

Both models include a custom `FourierUnitJIT` implementation that replaces PyTorch's native FFT operations with ONNX-compatible MatMul-based alternatives.

### DirectML Context

DirectML is Microsoft's hardware-accelerated machine learning library for Windows, providing GPU acceleration across AMD, Intel, NVIDIA, and Qualcomm hardware. It serves as an execution provider for ONNX Runtime.

**Current Status (as of 2024):**
> "DirectML is in sustained engineering mode. DirectML continues to be supported, but new feature development has moved to WinML for Windows-based ONNX Runtime deployments."

## Problem Statement

**Issue:** The LaMa ONNX model crashes when attempting GPU inference on Windows using the DirectML execution provider, while CPU inference works correctly.

**Original Report:** HuggingFace Discussion #1 (May 15, 2024)
- User: Crowlley
- Platform: Windows with DirectML
- Symptom: Inference crash with error code 80070057

## Investigation

### Evidence Collected

1. **Error Screenshot** (`screenshots/directml-error-screenshot.png`)
   - Shows ONNX Runtime warnings and fatal error
   - Timestamps indicate ~47 seconds between session creation and crash

2. **Error Log** (`logs/directml-error-log.txt`)
   - Warning: Nodes not assigned to preferred execution providers
   - Fatal: MatMul node failure with parameter validation error

3. **Source Code Analysis**
   - `saicinpainting/training/modules/ffc.py`: FourierUnit implementations
   - `export_LaMa_to_onnx.ipynb`: Export process with JIT configuration

### Key Error Message

```
[E:onnxruntime:, sequential_executor.cc:514 onnxruntime::ExecuteKernel]
Non-zero status code returned while running MatMul node.
Name:'/generator/model/model.5/conv1/ffc/convw2g/fu/rttn/MatMul_5'
Status Message: [...]\MLOperatorAuthorImpl.cpp(2482)\onnxruntime.DLL
Exception(3) tid(3428) 80070057 The parameter is incorrect.
```

### Related ONNX Runtime Issues

| Issue # | Description | Relevance |
|---------|-------------|-----------|
| #20575 | DirectML Exception 80070057 with TorchDynamo models | Same error pattern |
| #6075 | DirectML MatMul failure with variable inputs | Shape validation |
| #12677 | InstanceNormalization parameter error | Similar 80070057 |
| #14268 | Stable Diffusion DirectML issues | Complex model failure |

## Root Cause

The failure stems from **three interrelated factors**:

### 1. Custom FFT Implementation

The LaMa ONNX model uses a MatMul-based FFT implementation because PyTorch's native `torch.fft.*` operations cannot be directly exported to ONNX with full compatibility.

**Code Location:** `ffc.py`, lines 23-146

```python
def rfft(x):
    # Uses matmul to compute DFT
    real_part = torch.matmul(x, cos_part)  # <- Problem area
    imag_part = torch.matmul(x, sin_part)  # <- Problem area
```

### 2. DirectML's Strict Validation

DirectML implements stricter parameter validation than CPU or CUDA providers. The specific tensor shapes and broadcasting patterns produced by the custom FFT implementation violate DirectML's constraints.

**Error Code:** `80070057` = Windows HRESULT for "The parameter is incorrect"

### 3. Model Architecture Complexity

The failing node path reveals the deep nesting:
```
generator/model/model.5/conv1/ffc/convw2g/fu/rttn/MatMul_5
```

This is the 5th MatMul operation within the RFFTTN (Real FFT N-dimensional) module, inside the FourierUnit, within a Fast Fourier Convolution layer.

## Impact Assessment

### Affected Users

- Windows users without NVIDIA GPUs who want GPU acceleration
- Developers targeting AMD or Intel integrated graphics
- Applications requiring broad hardware compatibility on Windows

### Scope

| Platform | CPU Provider | CUDA Provider | DirectML Provider |
|----------|-------------|---------------|-------------------|
| Windows | Works | Works (NVIDIA only) | **FAILS** |
| Linux | Works | Works (NVIDIA only) | N/A |
| macOS | Works | N/A | N/A |

### Severity

**Medium-High:** The model remains usable via CPU execution, but performance is significantly impacted for Windows users without NVIDIA hardware.

## Proposed Solutions

### Immediate (Low Effort)

1. **CPU Execution Provider** - Use CPU inference as fallback
2. **CUDA Provider** - For NVIDIA GPU users, use CUDA instead of DirectML

### Medium-Term (Medium Effort)

3. **Hybrid Execution** - Configure DirectML with CPU fallback for problematic nodes
4. **WinML Migration** - Move to Windows ML API for better Windows support

### Long-Term (High Effort)

5. **Native ONNX DFT Operators** - Re-export model using ONNX's built-in DFT operator
6. **Graph Optimization** - Post-process ONNX graph to replace problematic operations

*See `03-proposed-solutions.md` for detailed implementation guidance.*

## Recommendations

### For End Users

1. **Immediate:** Use CPU execution provider for reliable inference
2. **If NVIDIA GPU available:** Use CUDA execution provider instead of DirectML
3. **Monitor:** Watch for ONNX Runtime updates that may address DirectML compatibility

### For Carve.Photos Team

1. **Documentation:** Update HuggingFace model card with DirectML limitation warning
2. **Testing:** Establish DirectML testing in CI/CD pipeline
3. **Development:** Investigate native ONNX DFT operator support for future exports

### For ONNX Runtime Team

1. **Bug Report:** File issue documenting this specific failure pattern
2. **Validation:** Improve error messages for DirectML parameter validation failures
3. **Compatibility:** Consider relaxing validation for common FFT computation patterns

## Appendix

### Files in This Case Study

```
docs/case-studies/
    README.md                      # This document
    01-timeline.md                 # Chronological event reconstruction
    02-root-cause-analysis.md      # Technical deep-dive
    03-proposed-solutions.md       # Solution options and comparison
    screenshots/
        directml-error-screenshot.png  # Error screenshot from HuggingFace
    logs/
        directml-error-log.txt     # Extracted error messages
```

### References

#### Primary Sources
- HuggingFace Discussion: https://huggingface.co/Carve/LaMa-ONNX/discussions/1
- LaMa ONNX Model: https://huggingface.co/Carve/LaMa-ONNX
- Original LaMa Paper: https://arxiv.org/abs/2109.07161

#### ONNX Runtime Issues
- Issue #20575: https://github.com/microsoft/onnxruntime/issues/20575
- Issue #6075: https://github.com/microsoft/onnxruntime/issues/6075
- Issue #12677: https://github.com/microsoft/onnxruntime/issues/12677

#### Technical Documentation
- DirectML Execution Provider: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- ONNX DFT Operator: https://onnx.ai/onnx/operators/onnx__DFT.html
- Fast Fourier Convolutions Paper: https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

### Glossary

| Term | Definition |
|------|------------|
| DFT | Discrete Fourier Transform |
| DirectML | DirectX Machine Learning library by Microsoft |
| FFC | Fast Fourier Convolution |
| FFT | Fast Fourier Transform |
| HRESULT | Windows error code format |
| LaMa | Large Mask Inpainting model |
| MatMul | Matrix Multiplication operation |
| ONNX | Open Neural Network Exchange format |
| RFFT | Real-valued Fast Fourier Transform |

---

*Case study compiled: December 2024*
*Author: AI Issue Solver*
*GitHub Issue: https://github.com/Carve-Photos/lama/issues/1*
