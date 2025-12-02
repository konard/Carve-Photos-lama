# MNN Issue #2977: Precision_Low Backend Configuration

**Repository:** alibaba/MNN
**Issue Number:** 2977
**Title:** Backend设置Precision_Low，部分图片输出结果正常，部分图片输出全0
(Backend Precision_Low Setting: Some Images Output Correctly, Others Output All Zeros)
**Status:** Closed (Stale)
**Closed:** October 2024
**URL:** https://github.com/alibaba/MNN/issues/2977

## Problem Description

When setting `Precision_Low` in MNN's backend configuration on Android, the user reported **inconsistent results**: "some images produce normal output while others return all zeros."

With `Precision_Normal` or `Precision_High`, all images processed correctly.

## Technical Details

### Configuration

```cpp
MNN::BackendConfig backend_config;
backend_config.precision = MNN::BackendConfig::Precision_Low;
```

### Environment

- **MNN version:** 2.8.0 (later updated to 2.9.3)
- **Hardware:** Qualcomm Snapdragon 6450
- **Compilation flags:** `-DMNN_ARM82=ON`, `-DMNN_SUPPORT_BF16=ON`
- **Device capability:** Device logs showed "support fp16:1" indicating FP16 capability
- **Model:** LaMa inpainting model (related to MNN issue #2960)

### Initial Backend

`MNN_FORWARD_CPU`

## Investigation Process

### Recommendation from Collaborator

A collaborator suggested:
1. Update MNN to latest version
2. Use `ModuleBasic.out` testing with `mask=2` to identify which layer produces anomalies

### Resolution/Workaround

The user reported that **switching the backend from `MNN_FORWARD_CPU` to `MNN_FORWARD_OPENCL`** resolved the issue with `Precision_Low`.

This indicates a **CPU backend-specific problem** with low-precision (FP16) quantization.

## Key Findings

1. **Issue is backend-specific**: Problem only occurs with CPU backend, not OpenCL
2. **Precision-dependent**: Only affects `Precision_Low` (FP16), not higher precision modes
3. **Inconsistent behavior**: Some images work, others produce all zeros (black output)
4. **Hardware capable**: Device supports FP16, so issue is likely software/implementation related

## Relationship to LaMa FP16 Issue

This MNN issue directly relates to the FP16 black output problem reported in advimman/lama#315:

- Same model: LaMa inpainting
- Same framework: MNN
- Same precision: FP16/Precision_Low
- Same symptom: Black/zero output for some images
- Same inconsistency: Only affects certain images

## Status

Issue closed as stale (October 2024) without confirmed resolution for CPU backend. The workaround suggests potential quantization handling issues in MNN's CPU backend FP16 implementation remain unresolved.
