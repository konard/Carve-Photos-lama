# GitHub Issue #315: ONNX Model Done

**Repository:** advimman/lama
**Issue Number:** 315
**Title:** ONNX Model done
**Author:** OPHoperHPO
**Status:** Open
**Created:** May 10, 2024
**URL:** https://github.com/advimman/lama/issues/315

## Original Description

The Carve-Photos team announced they successfully converted LaMa (big-lama) to ONNX format with results "closely resembling the original." They shared the model at Hugging Face and created a demonstration space for testing.

**Key Resources:**
- Model: https://huggingface.co/Carve/LaMa-ONNX
- Demo: https://huggingface.co/spaces/Carve/LaMa-Demo-ONNX

## Notable Discussion Points

### Performance Concerns (K-prog)

K-prog raised questions about ONNX model quality, noting it appeared to lack structure preservation compared to PyTorch. After investigation, OPHoperHPO identified a postprocessing error—the ONNX output required division by 255.0 that K-prog had overlooked, resolving the quality discrepancy.

### PyTorch Export Limitations (K-prog)

Discussion about using `torch.dynamo_export` revealed that direct FFT operator export remains problematic. OPHoperHPO explained their conversion required custom implementations, stating the new exporter has "limitations...low speed, and the inability to use the model on a GPU."

### FP16 Precision Issues (ljdang)

FP16 conversion via MNN occasionally produced completely black outputs, though K-prog reported no visible FP16 losses with proper conversion procedures.

## Community Engagement

The issue generated 18 comments with positive reactions (6 hooray, 3 heart, 3 rocket emojis), indicating strong community interest in the ONNX implementation.
