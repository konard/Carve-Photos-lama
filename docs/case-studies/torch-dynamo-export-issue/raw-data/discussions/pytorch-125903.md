# PyTorch Issue #125903: FFT ONNX Export Shape Issue

**Source**: https://github.com/pytorch/pytorch/issues/125903

## Problem Description
When converting PyTorch models using `torch.fft.rfftn` and `torch.fft.irfftn` to ONNX format via `torch.compile` with the "onnxrt" backend, the output tensor shape becomes incorrect. The issue manifests as a shape mismatch in the inverse FFT operation.

## Specific Issue
A FourierUnit model expected to produce output shape `[1, 192, 64, 64]` instead generates `[1, 192, 64, 33]` when exported to ONNX. The error log reveals:

> "Error merging shape info for output. '_fft_c2r' source:{1,192,64,33} target:{1,192,64,64}. Falling back to lenient merge."

This indicates the ONNX runtime detects conflicting shape information between the computed output and the target shape.

## Technical Context
The problem occurs in the inverse FFT transformation step where `torch.fft.irfftn` is called with shape parameter `s=ifft_shape_slice`. The real-to-complex and complex-to-real FFT operations handle frequency domain representation differently—real FFT transforms produce asymmetric frequency bins (width/2+1 for the last dimension), which the inverse operation should restore to the original dimensions when provided the correct shape parameter.

## Scope
The issue persists across:
- PyTorch 2.3.0 with CUDA 12.1
- Both `torch.compile` and `torch.onnx.dynamo_export` approaches
- Nightly PyTorch builds

No documented workarounds or solutions were presented in the issue discussion.
