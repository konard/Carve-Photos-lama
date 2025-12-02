# LaMa ONNX Inference Examples

This directory contains example implementations for running LaMa ONNX model inference in different programming languages.

## ⚠️ Critical Information

All examples demonstrate the correct preprocessing and postprocessing for LaMa ONNX model:

1. **Input preprocessing**: Images and masks must be normalized to `[0, 1]` range (divide by 255)
2. **Output postprocessing**: **CRITICAL** - Output is already in `[0, 255]` range (DO NOT multiply by 255 again!)
3. **Tensor format**: All inputs must be in CHW (Channel, Height, Width) format with batch dimension

For detailed documentation, see the main [README's ONNX Preprocessing & Postprocessing Guide](../README.md#-onnx-preprocessing--postprocessing-guide).

## Available Examples

### Python (`onnx_inference_example.py`)

**Requirements:**
```bash
pip install onnxruntime pillow numpy opencv-python
```

**Usage:**
```bash
python onnx_inference_example.py model.onnx input.jpg mask.png output.jpg
```

**Features:**
- Complete preprocessing pipeline with detailed comments
- Proper padding to divisible by 8
- Correct output postprocessing (no extra scaling)
- Command-line interface

### JavaScript/Node.js (`onnx_inference_example.js`)

**Requirements:**
```bash
npm install onnxruntime-node sharp
```

**Usage:**
```bash
node onnx_inference_example.js model.onnx input.jpg mask.png output.jpg
```

**Features:**
- Browser and Node.js compatible patterns
- Uses Sharp for efficient image processing
- Full CHW/HWC tensor conversion examples
- Async/await patterns

### C# (`OnnxInferenceExample.cs`)

**Requirements:**
```bash
dotnet add package Microsoft.ML.OnnxRuntime
dotnet add package SixLabors.ImageSharp
```

**Usage:**
```bash
dotnet run model.onnx input.jpg mask.png output.jpg
```

**Features:**
- Uses ImageSharp for cross-platform image processing
- LINQ integration for tensor operations
- Proper memory management with using statements
- Type-safe tensor operations

## Common Implementation Pattern

All examples follow the same pattern:

```
1. Load ONNX model
2. Load and preprocess image:
   - Convert to CHW format
   - Normalize to [0, 1] by dividing by 255
   - Pad to dimensions divisible by 8
   - Add batch dimension → shape (1, 3, H, W)
3. Load and preprocess mask:
   - Convert to grayscale
   - Convert to CHW format
   - Normalize to [0, 1] and binarize (> 0.5 = 1)
   - Pad to same dimensions as image
   - Add batch dimension → shape (1, 1, H, W)
4. Run inference with inputs: {image, mask}
5. Postprocess output:
   - Remove batch dimension
   - Convert from CHW to HWC
   - Clamp to [0, 255] and convert to uint8
   - NO multiplication by 255 needed!
6. Save result
```

## Testing the Examples

You can download a test model and images from:
- Model: [Hugging Face - LaMa ONNX](https://huggingface.co/Carve/LaMa-ONNX)
- Test images: Available in the HuggingFace repository

## Common Pitfalls

❌ **Don't do this:**
```python
# WRONG: Multiplying output by 255
output = model.run(...)[0]
output = output * 255  # ❌ Output is already in [0, 255]!
```

✅ **Do this instead:**
```python
# CORRECT: Output is already scaled
output = model.run(...)[0]
output = output.astype(np.uint8)  # ✅ Just convert to uint8
```

## Debugging Checklist

If your results don't look right, verify:

- [ ] Input image values are in range [0, 1] (not [0, 255])
- [ ] Input mask values are binary (0 or 1) and in range [0, 1]
- [ ] Both inputs are in CHW format with batch dimension
- [ ] Images are padded to be divisible by 8
- [ ] Output is NOT multiplied by 255 (it's already scaled)
- [ ] Output is converted from CHW to HWC before saving
- [ ] Output values are directly cast to uint8 without extra scaling

## Additional Resources

- [Main README - ONNX Guide](../README.md#-onnx-preprocessing--postprocessing-guide)
- [ONNX Export Notebook](../export_LaMa_to_onnx.ipynb)
- [Original Discussion](https://github.com/advimman/lama/issues/315)
- [IOPaint Reference Implementation](https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py)
