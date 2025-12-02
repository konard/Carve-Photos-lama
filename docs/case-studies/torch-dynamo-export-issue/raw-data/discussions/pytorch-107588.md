# PyTorch Issue #107588: FFT ONNX Export Support Discussion

**Source**: https://github.com/pytorch/pytorch/issues/107588

## Issue Overview
This GitHub issue (#107588) requests native ONNX export support for FFT-related operations in PyTorch using ONNX opset17.

## Key Technical Details

**Requested Feature:**
The user requests the ability to export `torch.fft.rfft` and related functions directly to ONNX without manual basis construction. ONNX opset17 now supports DFT operations natively.

**Current Status:**
According to assignee Justin Chu's comment: "FFT and STFT will be supported by the onnx.dynamo_export exporter"

This indicates that PyTorch's newer `dynamo_export` exporter will handle FFT operations, moving away from manual implementation approaches.

## Related Issues
- Issue #107446 (mentioned as related)
- Issue #98833 (FFT export to opset 18)
- Issue #59246 (complex type support in ONNX)
- Issue #106850 (STFT conversion failures)

## Community Impact
The issue notes potential importance to the torchaudio community, suggesting this feature affects audio processing workflows.

## Timeline
- Created: August 17, 2023
- Status: Reopened (as of May 12, 2025)
- Assigned to: Justin Chu
