/**
 * LaMa ONNX Inference Example - JavaScript/Node.js
 *
 * This script demonstrates the correct preprocessing and postprocessing for LaMa ONNX model.
 *
 * CRITICAL INFORMATION:
 * - Input images and masks must be normalized to [0, 1] range
 * - Output is already in [0, 255] range (DO NOT multiply by 255 again!)
 * - All inputs must be in CHW (Channel, Height, Width) format with batch dimension
 *
 * Installation:
 *   npm install onnxruntime-node sharp
 *
 * Usage:
 *   node onnx_inference_example.js model.onnx input.jpg mask.png output.jpg
 *
 * For more details, see the README's "ONNX Preprocessing & Postprocessing Guide" section.
 */

const ort = require('onnxruntime-node');
const sharp = require('sharp');
const fs = require('fs').promises;

/**
 * Convert image buffer to CHW format tensor normalized to [0, 1]
 *
 * @param {Buffer} imageBuffer - Image buffer from sharp
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @param {number} channels - Number of channels (3 for RGB, 1 for grayscale)
 * @returns {Float32Array} - Flattened CHW tensor normalized to [0, 1]
 */
function preprocessImage(imageBuffer, width, height, channels) {
    // Input buffer is in HWC format (height, width, channels)
    // We need to convert to CHW format and normalize to [0, 1]

    const size = width * height * channels;
    const tensor = new Float32Array(size);

    // Convert HWC to CHW and normalize to [0, 1]
    for (let c = 0; c < channels; c++) {
        for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
                const hwcIndex = (h * width + w) * channels + c;
                const chwIndex = c * height * width + h * width + w;

                // CRITICAL: Divide by 255 to normalize to [0, 1]
                tensor[chwIndex] = imageBuffer[hwcIndex] / 255.0;
            }
        }
    }

    return tensor;
}

/**
 * Pad image dimensions to be divisible by mod
 *
 * @param {number} dimension - Original dimension
 * @param {number} mod - Modulo value (typically 8)
 * @returns {number} - Padded dimension
 */
function padToModulo(dimension, mod = 8) {
    return Math.ceil(dimension / mod) * mod;
}

/**
 * Convert CHW output tensor to HWC image buffer
 *
 * @param {Float32Array} output - Model output in CHW format
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @param {number} channels - Number of channels (3 for RGB)
 * @returns {Uint8Array} - Image buffer in HWC format
 */
function postprocessOutput(output, width, height, channels) {
    const size = width * height * channels;
    const imageBuffer = new Uint8Array(size);

    // Convert CHW to HWC
    for (let c = 0; c < channels; c++) {
        for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
                const chwIndex = c * height * width + h * width + w;
                const hwcIndex = (h * width + w) * channels + c;

                // CRITICAL: Output is already in [0, 255] range!
                // DO NOT multiply by 255 again!
                // Just clamp and convert to uint8
                const value = Math.max(0, Math.min(255, output[chwIndex]));
                imageBuffer[hwcIndex] = value;
            }
        }
    }

    return imageBuffer;
}

/**
 * Run LaMa ONNX inference
 *
 * @param {string} modelPath - Path to ONNX model
 * @param {string} imagePath - Path to input image
 * @param {string} maskPath - Path to mask image
 * @param {string} outputPath - Path to save output
 */
async function runInference(modelPath, imagePath, maskPath, outputPath) {
    console.log(`Loading ONNX model from ${modelPath}...`);
    const session = await ort.InferenceSession.create(modelPath);

    console.log(`Loading and preprocessing image from ${imagePath}...`);
    const imageMetadata = await sharp(imagePath).metadata();
    const originalWidth = imageMetadata.width;
    const originalHeight = imageMetadata.height;

    // Pad dimensions to be divisible by 8
    const paddedWidth = padToModulo(originalWidth, 8);
    const paddedHeight = padToModulo(originalHeight, 8);

    console.log(`  Original size: ${originalWidth}x${originalHeight}`);
    console.log(`  Padded size: ${paddedWidth}x${paddedHeight}`);

    // Load and preprocess image
    const imageBuffer = await sharp(imagePath)
        .resize(paddedWidth, paddedHeight, { fit: 'contain', background: { r: 0, g: 0, b: 0 } })
        .raw()
        .toBuffer();

    const imageArray = preprocessImage(imageBuffer, paddedWidth, paddedHeight, 3);

    console.log(`Loading and preprocessing mask from ${maskPath}...`);
    // Load and preprocess mask (convert to grayscale)
    const maskBuffer = await sharp(maskPath)
        .resize(paddedWidth, paddedHeight, { fit: 'contain', background: { r: 0, g: 0, b: 0 } })
        .grayscale()
        .raw()
        .toBuffer();

    const maskArray = preprocessImage(maskBuffer, paddedWidth, paddedHeight, 1);

    // Binarize mask (values > 0.5 become 1.0, else 0.0)
    for (let i = 0; i < maskArray.length; i++) {
        maskArray[i] = maskArray[i] > 0.5 ? 1.0 : 0.0;
    }

    // Create input tensors with batch dimension
    const imageTensor = new ort.Tensor('float32', imageArray, [1, 3, paddedHeight, paddedWidth]);
    const maskTensor = new ort.Tensor('float32', maskArray, [1, 1, paddedHeight, paddedWidth]);

    console.log(`Running inference...`);
    console.log(`  Image tensor shape: [${imageTensor.dims.join(', ')}]`);
    console.log(`  Mask tensor shape: [${maskTensor.dims.join(', ')}]`);

    // Run inference
    const feeds = { image: imageTensor, mask: maskTensor };
    const results = await session.run(feeds);
    const output = results.output;

    console.log(`  Output tensor shape: [${output.dims.join(', ')}]`);
    console.log(`  Output range: [${Math.min(...output.data)}, ${Math.max(...output.data)}]`);

    // Postprocess output
    console.log(`Postprocessing output...`);
    const outputBuffer = postprocessOutput(
        output.data,
        paddedWidth,
        paddedHeight,
        3
    );

    // Crop to original size and save
    console.log(`Saving result to ${outputPath}...`);
    await sharp(outputBuffer, {
        raw: {
            width: paddedWidth,
            height: paddedHeight,
            channels: 3
        }
    })
    .extract({ left: 0, top: 0, width: originalWidth, height: originalHeight })
    .toFile(outputPath);

    console.log('✅ Done!');
    console.log('\nIMPORTANT NOTES:');
    console.log('  - Input was normalized to [0, 1] range');
    console.log('  - Output is already in [0, 255] range (NO additional scaling)');
    console.log('  - Result saved as uint8 image');
}

// Main entry point
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.length !== 4) {
        console.error('Usage: node onnx_inference_example.js <model.onnx> <input.jpg> <mask.png> <output.jpg>');
        console.error('');
        console.error('For more information, see the README\'s "ONNX Preprocessing & Postprocessing Guide" section.');
        process.exit(1);
    }

    const [modelPath, imagePath, maskPath, outputPath] = args;

    runInference(modelPath, imagePath, maskPath, outputPath)
        .catch(err => {
            console.error('Error:', err);
            process.exit(1);
        });
}

module.exports = { runInference, preprocessImage, postprocessOutput, padToModulo };
