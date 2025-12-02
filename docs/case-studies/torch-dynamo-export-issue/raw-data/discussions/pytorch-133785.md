# PyTorch Issue #133785: LAMA Inpainting Model FFT ONNX Export Issue

**Source**: https://github.com/pytorch/pytorch/issues/133785
**Reporter**: K-prog

## Problem Description

User K-prog reported difficulty converting the LAMA Inpainting model to ONNX format using PyTorch's `torch.onnx.dynamo_export`. The model contains `fft_rfftn` layers that were previously unsupported by ONNX.

## Key Technical Details

**Initial Challenge:**
The user noted that earlier versions lacked support for FFT operations in ONNX. A workaround existed using a custom FourierUnitJIT class, but this approach "has accuracy losses."

**Current Issue:**
Despite ONNX now supporting these layers, attempting to export using dynamo_export generates: `"OnnxExporterError: Failed to export the model to ONNX. Generating SARIF report..."`

**Testing Approaches:**
- Attempted dynamo_export with dummy tensors (1×3×512×512 image, 1×1×512×512 mask)
- Tried legacy `torch.onnx.export` with opset_version=18
- Old exporter explicitly rejected `'aten::fft_rfftn'` as unsupported

## Status and Response

**Issue Management:**
The issue received "triaged" and "module: onnx" labels, indicating PyTorch team acknowledgment. User @justinchuby was assigned, responding: "I will be working on this in the coming week or so." The issue was tracked across multiple release milestones (2.5.0 through 2.8.0) before eventual removal from scheduling.

## Related Context

The user referenced issue #107588 regarding FFT layer limitations and cross-referenced related problem #128324 about dynamic model exports with copy/roll/fftn operations.
