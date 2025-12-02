# GitHub Issue #315: ONNX Model Conversion Discussion

**Source**: https://github.com/advimman/lama/issues/315

## Overview
Issue #315 documents the successful ONNX conversion of LaMa (big-lama) by Carve-Photos, with extensive technical discussion about export challenges and model performance.

## Initial Announcement
**OPHoperHPO** (May 10, 2024): The Carve-Photos team successfully ported LaMa to ONNX format with comparable results to the original. Resources shared include:
- HuggingFace model: https://huggingface.co/Carve/LaMa-ONNX
- Demo space: https://huggingface.co/spaces/Carve/LaMa-Demo-ONNX

## Key Technical Discussion

### K-prog's Concerns (August 17, 2024)
K-prog raised questions about ONNX performance quality versus PyTorch, noting: "it seems like it differs from the original model and doesn't perform that well." They inquired about using PyTorch's `torch.dynamo_export` function for direct FFT layer support.

### OPHoperHPO's Detailed Response
Regarding `dynamo_export`, OPHoperHPO explained: "this doesn't work without modifying internal torch and onnxscript code." Key points included:

- **Initial attempts**: The team implemented custom code converting `aten::fft_rfftn` operators to ONNX
- **Limitations identified**: "The model exported via torch.dynamo_export is not suitable for use due to lack of support in other ONNX converters, low speed, and inability to use the model on a GPU"
- **Architecture differences**: Comparing models via Netron revealed "significant differences in architecture" between approaches

### Resolution
The actual issue was **preprocessing/postprocessing**, not the ONNX model itself. OPHoperHPO identified: "You just need to divide the output by 255, as the ONNX model multiplies it."

## Additional Issues
- **FP16 conversion**: User ljdang reported black output results with FP16 format when using MNN conversion, suggesting potential numerical precision problems

## Conclusion
The ONNX conversion successfully matches PyTorch results when correct preprocessing protocols are applied. The project demonstrates practical limitations of newer PyTorch export mechanisms for complex FFT operations.
