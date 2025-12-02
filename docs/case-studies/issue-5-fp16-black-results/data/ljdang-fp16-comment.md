# FP16 Black Output Issue - ljdang's Report

**Repository:** advimman/lama
**Issue:** #315
**Comment URL:** https://github.com/advimman/lama/issues/315#issuecomment-2369308152
**Author:** ljdang
**Date:** October 11, 2024

## Original Comment

> "It works well with fp32, but fp16 sometimes produces completely black results, possibly due to numerical overflow."

## Follow-up Discussion

**Response from K-prog** (October 18, 2024):
> "fp16 doesn't seem to have any visible losses, is your conversion correct?"

**Additional Context from ljdang** (October 18, 2024):
The user clarified they used MNN for the FP16 conversion and noted the issue occurs inconsistently across images.

## Technical Details

- **Framework**: MNN (Alibaba's neural network inference framework)
- **Precision**: FP16 (half-precision floating point)
- **Issue**: Sporadic black output images when using FP16 precision
- **Working condition**: FP32 produces correct results
- **Suspected cause**: Possible numerical overflow during half-precision computation
- **Behavior**: Inconsistent - only affects certain images
- **Status**: Unresolved at time of documentation

## Key Observations

1. The issue is **intermittent** - not all images produce black output
2. FP32 model works correctly for all images
3. The problem specifically occurs during MNN FP16 conversion
4. Another user (K-prog) did not experience visible losses with FP16, suggesting the issue may be conversion-method specific

## Related Technical Context

The discussion suggests potential issues with how MNN handles the FFT operations present in the LaMa model during FP16 quantization. FFT operations are known to be sensitive to numerical precision.
