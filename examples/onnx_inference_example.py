#!/usr/bin/env python3
"""
LaMa ONNX Inference Example

This script demonstrates the correct preprocessing and postprocessing for LaMa ONNX model.

CRITICAL INFORMATION:
- Input images and masks must be normalized to [0, 1] range
- Output is already in [0, 255] range (DO NOT multiply by 255 again!)
- All inputs must be in CHW (Channel, Height, Width) format with batch dimension

For more details, see the README's "ONNX Preprocessing & Postprocessing Guide" section.
"""

import argparse
import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from PIL import Image


def preprocess_image(image):
    """
    Convert PIL Image or numpy array to ONNX model input format.

    Args:
        image: PIL Image or numpy array (HWC format)

    Returns:
        numpy array in CHW format, normalized to [0, 1] range
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise ValueError("Input image should be either PIL Image or numpy array!")

    # Convert to CHW (Channel, Height, Width) format
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    elif img.ndim == 2:
        img = img[np.newaxis, ...]  # Add channel dimension for grayscale

    assert img.ndim == 3, f"Expected 3D array, got {img.ndim}D"

    # CRITICAL: Normalize to [0, 1] range
    # This is required for ONNX model input!
    img = img.astype(np.float32) / 255.0
    return img


def pad_to_modulo(img, mod=8):
    """
    Pad image to make dimensions divisible by mod.

    This is required because the model expects dimensions divisible by 8.
    Uses symmetric padding to avoid edge artifacts.

    Args:
        img: Image in CHW format
        mod: Modulo value (typically 8)

    Returns:
        Padded image in CHW format
    """
    channels, height, width = img.shape
    out_height = (height + mod - 1) // mod * mod
    out_width = (width + mod - 1) // mod * mod

    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode='symmetric'  # Symmetric padding to avoid edge artifacts
    )


def postprocess_output(output, original_height=None, original_width=None):
    """
    Convert ONNX model output to displayable image.

    Args:
        output: Model output in CHW format with batch dimension, shape (1, C, H, W)
        original_height: Optional, crop to this height
        original_width: Optional, crop to this width

    Returns:
        numpy array in HWC format, dtype uint8
    """
    # Remove batch dimension and convert CHW to HWC
    output = output[0].transpose(1, 2, 0)  # (1, C, H, W) -> (H, W, C)

    # Crop to original dimensions if specified
    if original_height and original_width:
        output = output[:original_height, :original_width]

    # CRITICAL: Output is already in [0, 255] range!
    # DO NOT multiply by 255 again!
    # Just clip and convert to uint8
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def run_inference(model_path, image_path, mask_path, output_path):
    """
    Run LaMa ONNX inference on an image and mask.

    Args:
        model_path: Path to ONNX model file
        image_path: Path to input image
        mask_path: Path to mask image (white = inpaint region)
        output_path: Path to save output image
    """
    print(f"Loading ONNX model from {model_path}...")
    session = onnxruntime.InferenceSession(str(model_path))

    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size

    print(f"Loading mask from {mask_path}...")
    mask = Image.open(mask_path).convert('L')  # Grayscale

    # Ensure mask has same dimensions as image
    if mask.size != image.size:
        print(f"Resizing mask from {mask.size} to {image.size}...")
        mask = mask.resize(image.size, Image.NEAREST)

    # Preprocess inputs
    print("Preprocessing inputs...")
    img_array = preprocess_image(image)
    mask_array = preprocess_image(mask)

    # Pad to required dimensions (divisible by 8)
    img_array = pad_to_modulo(img_array, mod=8)
    mask_array = pad_to_modulo(mask_array, mod=8)

    # Binarize mask (values > 0.5 become 1, else 0)
    mask_array = (mask_array > 0.5).astype(np.float32)

    # Add batch dimension
    img_array = img_array[np.newaxis, ...]  # (C, H, W) -> (1, C, H, W)
    mask_array = mask_array[np.newaxis, ...]  # (C, H, W) -> (1, C, H, W)

    # Run inference
    print("Running inference...")
    print(f"  Input image shape: {img_array.shape}, dtype: {img_array.dtype}, range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    print(f"  Input mask shape: {mask_array.shape}, dtype: {mask_array.dtype}, range: [{mask_array.min():.3f}, {mask_array.max():.3f}]")

    outputs = session.run(
        None,
        {
            'image': img_array.astype(np.float32),
            'mask': mask_array.astype(np.float32)
        }
    )

    output = outputs[0]
    print(f"  Output shape: {output.shape}, dtype: {output.dtype}, range: [{output.min():.3f}, {output.max():.3f}]")

    # Postprocess output
    print("Postprocessing output...")
    result = postprocess_output(output, original_height, original_width)

    # Save result
    print(f"Saving result to {output_path}...")
    Image.fromarray(result).save(output_path)

    print("✅ Done!")
    print(f"\nIMPORTANT NOTES:")
    print(f"  - Input was normalized to [0, 1] range")
    print(f"  - Output is already in [0, 255] range (NO additional scaling)")
    print(f"  - Result saved as uint8 image")


def main():
    parser = argparse.ArgumentParser(
        description='LaMa ONNX Inference Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python onnx_inference_example.py model.onnx input.jpg mask.png output.jpg

For more information, see the README's "ONNX Preprocessing & Postprocessing Guide" section.
        """
    )

    parser.add_argument('model', type=Path, help='Path to ONNX model file')
    parser.add_argument('image', type=Path, help='Path to input image')
    parser.add_argument('mask', type=Path, help='Path to mask image (white = inpaint)')
    parser.add_argument('output', type=Path, help='Path to save output image')

    args = parser.parse_args()

    # Validate inputs
    if not args.model.exists():
        parser.error(f"Model file not found: {args.model}")
    if not args.image.exists():
        parser.error(f"Image file not found: {args.image}")
    if not args.mask.exists():
        parser.error(f"Mask file not found: {args.mask}")

    # Run inference
    run_inference(args.model, args.image, args.mask, args.output)


if __name__ == '__main__':
    main()
