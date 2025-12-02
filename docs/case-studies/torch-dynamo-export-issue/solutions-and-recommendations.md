# Solutions and Recommendations

## Current Situation Assessment

As of 2025, the LaMa project uses a **proven, production-ready solution**:
- Traditional `torch.onnx.export` with opset 17
- Custom FFT implementation (FourierUnitJIT) using matrix operations
- Successfully deployed at scale by Carve.Photos

**Status**: ✅ **No immediate action required** - current solution works well

## Recommended Actions by Stakeholder

### For LaMa Users (Model Consumers)

#### ✅ DO: Use the Official ONNX Export
```python
# Follow export_LaMa_to_onnx.ipynb
config.generator.resnet_conv_kwargs.use_jit = True

torch.onnx.export(
    model,
    (image, mask),
    "lama_fp32.onnx",
    opset_version=17,  # Use opset 17 for best compatibility
    dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}, "output": {0: "batch"}}
)
```

#### ❌ DON'T: Try torch.dynamo_export
```python
# AVOID THIS - known to produce unusable models
torch.onnx.dynamo_export(model, image, mask)  # ❌ Don't use
```

#### 📋 Deployment Checklist
- [ ] Use the pre-exported ONNX model from HuggingFace: https://huggingface.co/Carve/LaMa-ONNX
- [ ] Test on your target runtime (ONNX Runtime, TensorRT, CoreML, etc.)
- [ ] Verify preprocessing: input images as float32 in [0, 1] range
- [ ] Verify postprocessing: output is already in [0, 255] range
- [ ] Benchmark performance on your hardware
- [ ] Test with various image sizes (model supports 512×512 by default)

### For LaMa Maintainers (Carve.Photos Team)

#### Short Term (2025)
1. ✅ **Keep Current Implementation** - it works and is proven
2. 📝 **Document the Decision** - this case study serves that purpose
3. 🔍 **Monitor PyTorch Releases**
   - Subscribe to PyTorch ONNX export release notes
   - Watch issues #107588, #133785, #125903 for resolution
   - Test dynamo_export with each major PyTorch release (2.6, 2.7, etc.)

4. 🧪 **Create Validation Tests**
   ```python
   # tests/test_onnx_export.py
   def test_onnx_matches_pytorch():
       """Ensure ONNX output matches PyTorch within tolerance"""
       pytorch_output = pytorch_model(image, mask)
       onnx_output = onnx_model(image, mask)
       assert torch.allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-4)

   def test_gpu_support():
       """Verify ONNX model can run on GPU"""
       session = ort.InferenceSession("lama.onnx", providers=['CUDAExecutionProvider'])
       # ... test GPU execution ...

   def test_converter_compatibility():
       """Test conversion to TensorRT, CoreML, etc."""
       # ... test conversions ...
   ```

#### Medium Term (2026)
1. 🔬 **Evaluate dynamo_export Maturity**

   Create an evaluation protocol:
   ```python
   # scripts/evaluate_dynamo_export.py
   def evaluate_dynamo_export():
       """Test if dynamo_export is ready for production"""

       # Test 1: Export succeeds
       try:
           onnx_program = torch.onnx.dynamo_export(model, image, mask)
       except Exception as e:
           return False, f"Export failed: {e}"

       # Test 2: GPU support
       if not test_gpu_execution(onnx_program):
           return False, "No GPU support"

       # Test 3: Runtime compatibility
       if not test_converter_compatibility(onnx_program):
           return False, "Converter compatibility issues"

       # Test 4: Performance
       perf_ratio = benchmark_performance(onnx_program, baseline_onnx)
       if perf_ratio < 0.8:  # If >20% slower
           return False, f"Performance regression: {perf_ratio:.2%}"

       # Test 5: Accuracy
       if not test_accuracy_match(onnx_program):
           return False, "Accuracy mismatch"

       return True, "All tests passed"
   ```

2. 📊 **Benchmark and Compare**
   - If dynamo_export passes all tests, benchmark against current solution
   - Document tradeoffs (if any)
   - Consider gradual migration

#### Long Term (2027+)
1. 🎯 **Plan Migration** (if/when dynamo_export matures)
   - Maintain backward compatibility
   - Support both export methods during transition
   - Document migration path for users

2. 🔄 **Simplify Codebase**
   - Remove FourierUnitJIT custom FFT implementation
   - Use native `torch.fft.rfftn` everywhere
   - Simplify export script

### For PyTorch Contributors

#### Critical Issues to Fix

1. **GPU Support** (Highest Priority)
   ```
   Issue: Models exported with dynamo_export can't run on GPU
   Solution: Implement GPU kernels for all exported FFT operators
   Impact: Blocks all production use cases requiring GPU
   ```

2. **Runtime Compatibility** (High Priority)
   ```
   Issue: Exported models don't work with ONNX converters (TensorRT, CoreML)
   Solution: Ensure exported ops are standard ONNX ops or widely supported
   Impact: Limits deployment options
   ```

3. **Performance** (High Priority)
   ```
   Issue: Exported models run slower than traditionally exported models
   Solution: Improve operator fusion and graph optimization
   Impact: Makes dynamo_export uncompetitive
   ```

4. **Shape Inference** (Medium Priority)
   ```
   Issue: FFT operations export with incorrect output shapes
   Solution: Fix shape inference for irfftn operations
   Reference: PyTorch issue #125903
   ```

#### Recommended Improvements

1. **Create Comparison Tests**
   ```python
   # pytorch/test/onnx/test_fft_export.py
   def test_dynamo_vs_traditional_export():
       """Ensure dynamo_export matches traditional export quality"""
       # Compare performance, compatibility, accuracy
   ```

2. **Document Migration Path**
   - When is dynamo_export ready?
   - How to migrate from traditional export?
   - What are the tradeoffs?

3. **Provide Production Checklist**
   - Runtime compatibility matrix
   - Performance expectations
   - Known limitations

### For ML Engineers (General Guidance)

#### Decision Framework: Which Export Method?

```
┌─────────────────────────────────────────┐
│ Does your model use FFT operations?    │
└────────────┬────────────────────────────┘
             │ No
             ├─────────> Use torch.onnx.export (traditional)
             │           or torch.onnx.dynamo_export
             │
             │ Yes
             ▼
┌─────────────────────────────────────────┐
│ Need to convert to TensorRT/CoreML?    │
└────────────┬────────────────────────────┘
             │ Yes
             ├─────────> Use traditional export + custom FFT
             │           (dynamo_export won't work)
             │
             │ No
             ▼
┌─────────────────────────────────────────┐
│ Need GPU inference?                     │
└────────────┬────────────────────────────┘
             │ Yes
             ├─────────> Use traditional export + custom FFT
             │           (dynamo_export may not support GPU)
             │
             │ No (CPU only)
             ▼
┌─────────────────────────────────────────┐
│ Need best performance?                  │
└────────────┬────────────────────────────┘
             │ Yes
             ├─────────> Use traditional export + custom FFT
             │           (dynamo_export is slower)
             │
             │ No (performance acceptable)
             ▼
┌─────────────────────────────────────────┐
│ Test both approaches and measure!      │
└─────────────────────────────────────────┘
```

#### Best Practices

1. **Always Test on Target Platform**
   ```python
   # Don't assume export works - validate it!
   def validate_export(onnx_path, test_cases):
       session = onnxruntime.InferenceSession(onnx_path)

       for image, mask, expected_output in test_cases:
           actual_output = session.run(None, {'image': image, 'mask': mask})[0]
           assert np.allclose(actual_output, expected_output, rtol=1e-3)

       print("✅ Export validated!")
   ```

2. **Benchmark Performance**
   ```python
   import time

   def benchmark_model(session, image, mask, n_runs=100):
       # Warmup
       for _ in range(10):
           session.run(None, {'image': image, 'mask': mask})

       # Benchmark
       start = time.time()
       for _ in range(n_runs):
           session.run(None, {'image': image, 'mask': mask})

       avg_time = (time.time() - start) / n_runs
       print(f"Average inference time: {avg_time*1000:.2f}ms")
   ```

3. **Version Lock for Reproducibility**
   ```txt
   # requirements.txt
   torch==2.5.0  # Lock specific version
   onnx==1.16.0
   onnxruntime==1.18.0
   ```

4. **Document Your Export Process**
   ```python
   """
   ONNX Export Configuration
   ========================
   PyTorch Version: 2.5.0
   ONNX Opset: 17
   Export Method: torch.onnx.export (traditional)
   Custom FFT: Yes (FourierUnitJIT)

   Reason for Traditional Export:
   - GPU support required
   - TensorRT deployment needed
   - dynamo_export not production-ready as of 2025

   Validation:
   - Accuracy: PSNR > 35dB vs PyTorch
   - Performance: 15ms per image @ 512×512 on V100
   - Compatibility: Tested on ONNX Runtime 1.18, TensorRT 8.6
   """
   ```

## Alternative Solutions (If Current Approach Has Issues)

### Option 1: JIT Scripting (Instead of ONNX)
If ONNX export becomes too limiting:

```python
# Export to TorchScript instead
scripted_model = torch.jit.script(model)
scripted_model.save("lama.pt")

# Deploy with TorchScript Runtime
# Pros: Native FFT support, good performance
# Cons: Less portable than ONNX, larger ecosystem dependency
```

### Option 2: Custom ONNX Operator
Implement FFT as a custom ONNX operator:

```python
# Define custom FFT operator with optimized implementation
# Pros: Best performance, clean graph
# Cons: High implementation complexity, runtime support needed
```

### Option 3: Hybrid Approach
Split model into FFT and non-FFT parts:

```python
# Export non-FFT parts with dynamo_export
# Handle FFT parts separately
# Combine at runtime
# Pros: Use best tool for each part
# Cons: Complex deployment, multiple models
```

## Monitoring and Triggers for Re-evaluation

### Set Up Automated Monitoring

```python
# .github/workflows/monitor_dynamo_export.yml
name: Test Dynamo Export
on:
  schedule:
    - cron: '0 0 1 * *'  # Monthly

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Test dynamo_export
        run: python scripts/evaluate_dynamo_export.py

      - name: Report results
        if: success()
        run: |
          echo "🎉 dynamo_export passed all tests!"
          echo "Consider evaluating migration"
```

### Triggers to Revisit Decision

Re-evaluate dynamo_export when:

1. ✅ PyTorch announces "production-ready" dynamo_export for FFT
2. ✅ Community reports successful FFT model deployments with GPU
3. ✅ PyTorch issues #107588, #133785, #125903 are closed as resolved
4. ✅ Your automated tests pass consistently
5. ✅ Performance benchmarks show parity or improvement

### When to Migrate

Migrate to dynamo_export when ALL of these are true:

- [ ] Automated tests pass for 3+ consecutive PyTorch releases
- [ ] Performance is within 10% of current solution
- [ ] GPU support confirmed working
- [ ] At least 2 converter targets (e.g., TensorRT, CoreML) work
- [ ] Production deployment reports from community (not just us)
- [ ] Migration provides clear benefits (not just "newer is better")

## Summary: The Right Solution for Right Now

### Current Best Practice (2025)

```python
# ✅ RECOMMENDED APPROACH
config.generator.resnet_conv_kwargs.use_jit = True  # Enable custom FFT

torch.onnx.export(
    model,
    (image, mask),
    "lama.onnx",
    opset_version=17
)
```

**Why This Works:**
- ✅ Production-tested by Carve.Photos
- ✅ Works on GPU
- ✅ Compatible with converters (TensorRT, CoreML, etc.)
- ✅ Acceptable performance
- ✅ Reliable and predictable

**Tradeoffs Accepted:**
- ⚠️ Custom FFT implementation (complexity)
- ⚠️ Slower than native FFT (but still fast enough)
- ⚠️ Manual maintenance required (but stable)

### Future Best Practice (2027+, projected)

```python
# 🔮 FUTURE APPROACH (when ready)
torch.onnx.dynamo_export(model, image, mask)
```

**When This Will Work:**
- ⏳ After PyTorch team fixes GPU support
- ⏳ After converter compatibility is resolved
- ⏳ After performance is optimized
- ⏳ After production validation by community

**The Wait is Worth It:**
- Simpler code (no custom FFT)
- Better maintainability
- Native PyTorch support
- Potentially better performance long-term

## Conclusion

The LaMa project's approach is **correct and should be maintained**. The decision to use traditional export with custom FFT was:
- ✅ Evidence-based (tested both approaches)
- ✅ Pragmatic (chose what works over what's new)
- ✅ Documented (this case study explains why)
- ✅ Flexible (can migrate when dynamo_export matures)

**Main Takeaway**: Don't chase new features; validate they meet production requirements first.
