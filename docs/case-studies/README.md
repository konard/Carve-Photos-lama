# LaMa Case Studies

This directory contains comprehensive case studies of issues, improvements, and technical investigations related to the LaMa inpainting model.

## Available Case Studies

### Issue #5: FP16 Model Conversion Black Results

**Status:** ✅ Investigation Complete
**Directory:** [issue-5-fp16-black-results/](issue-5-fp16-black-results/)

Deep analysis of numerical overflow issues when converting the LaMa ONNX model to FP16 precision using MNN framework.

**Key Findings:**
- Root cause: FFT operation overflow in FP16 (±65,504 limit)
- Affects 10-20% of images (bright, high-contrast content)
- Recommended solution: Mixed precision (FP32 for FFT, FP16 elsewhere)

**Contents:**
- Timeline reconstruction
- Root cause analysis
- Proposed solutions with implementation details
- Testing and validation guidelines

[→ Read full case study](issue-5-fp16-black-results/README.md)

---

## Contributing Case Studies

When adding new case studies, please follow this structure:

```
case-studies/
├── issue-{number}-{short-description}/
│   ├── README.md                 # Main case study document
│   ├── data/                     # Raw data, logs, screenshots
│   ├── timeline/                 # Event chronology
│   ├── analysis/                 # Technical analysis
│   └── solutions/                # Proposed solutions
```

### Case Study Template

Each case study should include:

1. **Executive Summary**
   - Problem statement
   - Key findings
   - Recommended actions

2. **Timeline**
   - Chronological sequence of events
   - When issue was discovered
   - Investigation milestones

3. **Data Collection**
   - Original issue reports
   - Related discussions
   - Supporting materials

4. **Root Cause Analysis**
   - Technical deep dive
   - Evidence and verification
   - Hypothesis testing

5. **Proposed Solutions**
   - Multiple approaches
   - Trade-off analysis
   - Implementation guidelines

6. **Testing & Validation**
   - Test criteria
   - Success metrics
   - Validation plan

7. **References**
   - Links to issues/PRs
   - Related documentation
   - Technical papers

---

## Index

| Case Study | Issue | Status | Date |
|------------|-------|--------|------|
| FP16 Black Results | [#5](https://github.com/Carve-Photos/lama/issues/5) | Complete | 2025-12-02 |

---

**Last Updated:** December 2, 2025
