# Timeline: torch.dynamo_export and FFT ONNX Support

## 2023

### August 17, 2023
**PyTorch Issue #107588 Created**
- Community requests native ONNX export support for FFT operations
- ONNX opset 17 has added DFT operations, but PyTorch can't export to them
- Justin Chu (PyTorch team) comments: "FFT and STFT will be supported by the onnx.dynamo_export exporter"
- **Significance**: This was the promise that dynamo_export would solve FFT export problems

## 2024

### May 10, 2024
**advimman/lama Issue #315 - Initial Success Announcement**
- OPHoperHPO (Carve.Photos team) announces successful ONNX conversion of LaMa
- Published model at HuggingFace: https://huggingface.co/Carve/LaMa-ONNX
- Used custom FFT implementation (FourierUnitJIT) with traditional torch.onnx.export
- **Key Decision**: Avoided dynamo_export, created custom FFT workaround instead

### August 17, 2024
**advimman/lama Issue #315 - K-prog Questions Quality**
- K-prog notices performance differences from original model
- Questions: "Why not use torch.dynamo_export which supports FFT layers directly?"
- Represents community interest in using the "official" dynamo_export solution
- **Significance**: This question sparked the investigation into dynamo_export viability

### August 17, 2024 (same day)
**OPHoperHPO's Response - Key Findings Revealed**
- Team had already attempted torch.dynamo_export approach
- Required "modifying internal torch and onnxscript code"
- Found critical limitations:
  1. "Lack of support in other ONNX converters" - can't convert to other formats
  2. "Low speed" - poor performance compared to traditional export
  3. "Inability to use the model on a GPU" - no GPU acceleration
  4. "Significant differences in architecture" when viewed in Netron
- **Verdict**: "The model exported via torch.dynamo_export is not suitable for use"
- Team also solved the quality issue: preprocessing problem, not ONNX export issue

### August/September 2024 (estimated)
**K-prog's PyTorch Issue #133785 Created**
- K-prog attempts to export LaMa using torch.onnx.dynamo_export
- Gets error: "OnnxExporterError: Failed to export the model to ONNX"
- Justin Chu responds: "I will be working on this in the coming week or so"
- Issue gets triaged, assigned, and added to milestones
- **Tracking**: Issue moved through milestones 2.5.0 → 2.6.0 → 2.7.0 → 2.8.0
- Eventually removed from all milestones without resolution
- **Significance**: Even when dynamo_export "works", it produces unusable models

### 2024 (date unclear)
**PyTorch Issue #125903 - Shape Mismatch Problem**
- Reports FFT ONNX export producing wrong output shapes
- FourierUnit expected [1, 192, 64, 64], got [1, 192, 64, 33]
- Error: "Error merging shape info for output. '_fft_c2r' source:{1,192,64,33} target:{1,192,64,64}"
- Affects both torch.compile and torch.onnx.dynamo_export
- **Significance**: Fundamental correctness issues with dynamo FFT export

## 2025

### 2025 (present)
**PyTorch Issue #107588 Status**
- Original issue still open and "Reopened" status
- Promise of dynamo_export FFT support from 2023 remains unfulfilled
- **Significance**: After 1.5+ years, native FFT ONNX export still not production-ready

## Key Milestones Summary

| Date | Event | Impact |
|------|-------|--------|
| Aug 2023 | PyTorch promises dynamo FFT support | Creates expectation |
| May 2024 | Carve.Photos ships working ONNX model | Proves custom approach works |
| Aug 2024 | Community questions why not dynamo | Triggers public documentation |
| Aug 2024 | Carve.Photos reveals dynamo_export testing | Documents critical limitations |
| Late 2024 | Multiple PyTorch issues remain unresolved | Confirms dynamo_export not ready |
| Present | Traditional export remains best practice | Custom FFT approach validated |

## Lessons Learned

### 1. Production Requirements vs Experimental Features
- dynamo_export promised FFT support but couldn't deliver production-quality exports
- Key gaps: runtime compatibility, GPU support, conversion to other formats

### 2. Pragmatic Engineering Wins
- Custom FFT implementation, while slower, provides:
  - Guaranteed ONNX compatibility
  - Cross-runtime support
  - GPU execution capability
  - Conversion to TensorRT, CoreML, etc.

### 3. Documentation Prevents Repeated Mistakes
- Without Issue #315 discussion, developers would waste time trying dynamo_export
- Public documentation of limitations saves community time and effort

### 4. PyTorch ONNX Export Maturity
- Traditional torch.onnx.export remains more reliable than dynamo_export (as of 2025)
- Cutting-edge features often require workarounds in production
- Community should monitor but not immediately adopt new export methods

## Future Outlook

### Short Term (2025-2026)
- Continue using traditional export with custom FFT
- Monitor PyTorch releases for dynamo_export improvements
- Watch for ONNX runtime native FFT optimization

### Medium Term (2027+)
- Re-evaluate dynamo_export when:
  1. GPU support is confirmed working
  2. Runtime compatibility issues are resolved
  3. Performance matches or exceeds traditional export
  4. Multiple production deployments validate stability

### Long Term
- Native FFT operations become standard in ONNX runtimes
- Custom FFT implementations become unnecessary
- Export process simplifies to single-step without workarounds
