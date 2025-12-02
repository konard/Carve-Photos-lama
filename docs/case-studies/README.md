# LaMa Case Studies

This directory contains in-depth technical analyses of issues, bugs, and challenges encountered in the LaMa project.

## Available Case Studies

### [TensorRT Conversion Shape Constraints](./tensorrt-conversion-shape-constraints.md)
**Issue:** [#7](https://github.com/Carve-Photos/lama/issues/7)
**Status:** Analyzed and Documented
**Date:** December 2, 2025

Deep dive into TensorRT conversion failures caused by shape constraint violations in Fast Fourier Convolution layers. Includes root cause analysis, mathematical explanations, and multiple solution approaches.

**Key Findings:**
- RFFT operations transform width dimension (W → W//2+1)
- Non-square inputs cause dimension mismatches in elementwise operations
- Recommended solution: Constrain input shapes to squares (multiples of 16)

**Artifacts:**
- Analysis script: [experiments/analyze_fft_shapes_simple.py](../../experiments/analyze_fft_shapes_simple.py)
- Output logs: [experiments/fft_shape_analysis_output.txt](../../experiments/fft_shape_analysis_output.txt)

---

## Case Study Structure

Each case study follows this structure:

1. **Executive Summary** - High-level overview of the issue
2. **Timeline of Events** - Chronological reconstruction
3. **Root Cause Analysis** - Technical deep dive
4. **Proposed Solutions** - Multiple approaches with trade-offs
5. **Recommended Approach** - Best solution with implementation details
6. **Testing and Validation** - How to verify the solution
7. **Related Issues** - Links to related problems
8. **Appendix** - Additional data and experiments

---

## Contributing Case Studies

When adding a new case study:

1. **Create a descriptive filename:** `issue-name-description.md`
2. **Follow the template structure** above
3. **Include all supporting artifacts** in `experiments/` or `docs/`
4. **Link to original issue** and related discussions
5. **Update this README** with the new case study

### Naming Convention

```
[issue-type]-[brief-description].md

Examples:
- tensorrt-conversion-shape-constraints.md
- webgpu-execution-fft-errors.md
- onnx-export-dynamic-shapes.md
```

### Supporting Materials

- **Scripts:** Place in `experiments/` directory
- **Logs:** Store as `.txt` or `.log` files
- **Images:** Save in `docs/images/case-studies/`
- **Data:** Keep test data in `experiments/data/`

---

## Index by Topic

### Model Conversion
- [TensorRT Conversion Shape Constraints](./tensorrt-conversion-shape-constraints.md)

### FFT Operations
- [TensorRT Conversion Shape Constraints](./tensorrt-conversion-shape-constraints.md) - RFFT dimension transformations

### Dynamic Shapes
- [TensorRT Conversion Shape Constraints](./tensorrt-conversion-shape-constraints.md) - TensorRT dynamic shape profiles

---

## Related Documentation

- [Main README](../../README.md) - Project overview
- [ONNX Export Guide](../../export_LaMa_to_onnx.ipynb) - How to export models
- [Contributing Guidelines](../../CONTRIBUTING.md) - If it exists

---

**Last Updated:** December 2, 2025
