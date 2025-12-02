# Timeline: FP16 Black Output Issue

## Chronological Sequence of Events

### May 10, 2024 - ONNX Model Release
**Event:** OPHoperHPO announces successful LaMa ONNX conversion
- Created issue #315 in advimman/lama repository
- Published ONNX model on Hugging Face: https://huggingface.co/Carve/LaMa-ONNX
- Created demo space: https://huggingface.co/spaces/Carve/LaMa-Demo-ONNX
- Claimed results "closely resembling the original"

**Significance:** Establishes FP32 ONNX baseline that works correctly

---

### May-October 2024 - Community Testing & Feedback
**Event:** Various users test and validate the ONNX model
- K-prog initially questions quality, discovers postprocessing issue (division by 255.0)
- Discussion about PyTorch export limitations and FFT operators
- General positive reception (18 comments, multiple positive reactions)

**Significance:** Confirms FP32 ONNX model stability across multiple users

---

### ~September 2024 - MNN FP16 Issue First Encountered
**Event:** User reports MNN Precision_Low issues (MNN #2977)
- Testing LaMa model with MNN's `Precision_Low` setting
- Environment: Android, Qualcomm Snapdragon 6450
- Backend: MNN_FORWARD_CPU
- MNN version: 2.8.0

**Symptoms:**
- Some images output correctly
- Other images produce all zeros (black output)
- Issue only with Precision_Low (FP16)
- Higher precision modes (Normal/High) work correctly

**Significance:** First documented case of FP16-specific black output problem with LaMa model

---

### October 11, 2024 - ljdang Reports FP16 Black Results
**Event:** ljdang posts comment in advimman/lama#315
- URL: https://github.com/advimman/lama/issues/315#issuecomment-2369308152
- Reports: "It works well with fp32, but fp16 sometimes produces completely black results"
- Hypothesis: "possibly due to numerical overflow"

**Significance:** Independent confirmation of FP16 issue, establishes pattern of intermittent failures

---

### October 18, 2024 - K-prog Questions Conversion Method
**Event:** K-prog responds to ljdang
- States: "fp16 doesn't seem to have any visible losses"
- Questions: "is your conversion correct?"

**Clarification from ljdang:**
- Confirms using MNN for FP16 conversion
- Notes issue occurs inconsistently across images

**Significance:** Suggests issue may be conversion-method or implementation-specific, not inherent to FP16

---

### October 2024 - MNN Issue Closed as Stale
**Event:** MNN #2977 closed without resolution
- Workaround identified: Switch from MNN_FORWARD_CPU to MNN_FORWARD_OPENCL
- CPU backend FP16 issue remains unresolved

**Significance:** Establishes that problem is backend-specific in MNN implementation

---

### December 2, 2025 - Case Study Investigation Initiated
**Event:** Comprehensive analysis requested for Carve-Photos/lama#5
- Request to download all related data
- Compile comprehensive case study
- Reconstruct timeline
- Identify root causes
- Propose solutions

**Significance:** Formal investigation to understand and address the FP16 black output problem

---

## Timeline Summary

```
May 2024          Sept 2024         Oct 11, 2024      Oct 18, 2024      Oct 2024          Dec 2, 2025
    |                 |                    |                 |                |                 |
    v                 v                    v                 v                v                 v
ONNX Model      MNN FP16 Issue      ljdang Reports    K-prog Questions  MNN Issue      Case Study
Released        First Seen          Black Output      Conversion        Closed         Investigation
(FP32 works)    (MNN #2977)        (Intermittent)    (No issues        (No CPU fix)   Initiated
                                                      for some)
```

## Key Patterns Identified

1. **Temporal Pattern**: Issue emerged 4-5 months after ONNX release, during community adoption phase
2. **Consistency**: Multiple independent reports of same symptom (black output)
3. **Conditional**: Only affects FP16/Precision_Low, not FP32 or higher precision
4. **Intermittent**: Not all images affected, suggesting content-dependent trigger
5. **Platform-specific**: Evidence suggests backend/implementation differences matter (CPU vs OpenCL)
