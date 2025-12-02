# Timeline of Events: LaMa ONNX DirectML GPU Inference Issue

## Overview

This document reconstructs the timeline and sequence of events related to the DirectML GPU inference issue with the LaMa ONNX model on Windows.

## Chronological Timeline

### 2024-05-15: Issue Discovery

**10:XX (approx)** - User `Crowlley` opens HuggingFace Discussion #1
- Reports implementing LaMa ONNX model in C# application
- CPU inference works correctly
- GPU inference via DirectML provider causes crash
- Initial error messages mention nodes not assigned to preferred execution providers

**16:45:08.182** - First warning logged (from error screenshot):
```
[W:onnxruntime:, session_state.cc:1166 onnxruntime::VerifyEachNodeIsAssignedToAnEp]
Some nodes were not assigned to the preferred execution providers which may or may
not have a negative impact on performance.
```

**16:45:08.193** - Second warning logged:
```
[W:onnxruntime:, session_state.cc:1168 onnxruntime::VerifyEachNodeIsAssignedToAnEp]
Rerunning with verbose output on a non-minimal build will show node assignments.
```

**16:45:55.303** - Fatal error during MatMul operation:
```
[E:onnxruntime:, sequential_executor.cc:514 onnxruntime::ExecuteKernel]
Non-zero status code returned while running MatMul node.
Name:'/generator/model/model.5/conv1/ffc/convw2g/fu/rttn/MatMul_5'
Status Message: [...] 80070057 The parameter is incorrect.
```

### 2024-05-15: Carve Team Response

**Time not specified** - User `anodev` (Carve organization) responds:
- Recommends using `lama_fp32.onnx` instead of older `lama.onnx`
- Explains that the older model "cannot be fixed to run on GPU"
- Notes newer model can export to TensorRT for NVIDIA GPU support
- Mentions partial GPU support on Linux via CUDAExecutionProvider
- Acknowledges DirectML is untested on Windows

### 2024-05-15: User Clarification

**Time not specified** - `Crowlley` clarifies:
- Has tested both model versions (`lama.onnx` and `lama_fp32.onnx`)
- Does not have CUDA-capable GPU hardware
- Suspects DirectML-specific limitation

### 2024-05-15: Error Evidence Shared

**Time not specified (edited)** - `Crowlley` shares:
- Screenshot of DirectML runtime error output
- Detailed error message showing MatMul operation failure
- Windows error code 80070057 ("The parameter is incorrect")

### 2024-05-16: Discussion Closed

**Time not specified** - `Crowlley` closes the discussion
- No resolution was reached
- Issue remains unresolved for DirectML execution provider

## Key Observations

1. **~47 seconds elapsed** between session creation and crash (16:45:08 to 16:45:55)
2. **Warnings appeared first**, indicating some nodes couldn't be assigned to DirectML
3. **Fatal error occurred specifically** in the FourierUnit's MatMul operation
4. **The affected node path**: `/generator/model/model.5/conv1/ffc/convw2g/fu/rttn/MatMul_5`
   - `ffc` = Fast Fourier Convolution layer
   - `fu` = FourierUnit module
   - `rttn` = RFFTTN (Real FFT N-dimensional) operation
   - `MatMul_5` = Matrix multiplication for FFT computation

## Source References

- HuggingFace Discussion: https://huggingface.co/Carve/LaMa-ONNX/discussions/1
- Screenshot: `./screenshots/directml-error-screenshot.png`
- Detailed error log: `./logs/directml-error-log.txt`
