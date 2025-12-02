/**
 * LaMa ONNX Inference Example - C#
 *
 * This script demonstrates the correct preprocessing and postprocessing for LaMa ONNX model.
 *
 * CRITICAL INFORMATION:
 * - Input images and masks must be normalized to [0, 1] range
 * - Output is already in [0, 255] range (DO NOT multiply by 255 again!)
 * - All inputs must be in CHW (Channel, Height, Width) format with batch dimension
 *
 * Installation (NuGet packages):
 *   Microsoft.ML.OnnxRuntime
 *   SixLabors.ImageSharp
 *
 * Usage:
 *   dotnet run model.onnx input.jpg mask.png output.jpg
 *
 * For more details, see the README's "ONNX Preprocessing & Postprocessing Guide" section.
 */

using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LamaOnnxInference
{
    public class OnnxInferenceExample
    {
        /// <summary>
        /// Convert image to CHW format tensor normalized to [0, 1]
        /// </summary>
        /// <param name="image">Input image</param>
        /// <returns>Float tensor in CHW format, normalized to [0, 1]</returns>
        private static float[] PreprocessImage(Image<Rgb24> image)
        {
            int width = image.Width;
            int height = image.Height;
            float[] tensor = new float[3 * height * width];

            // Convert HWC to CHW and normalize to [0, 1]
            image.ProcessPixelRows(accessor =>
            {
                for (int h = 0; h < height; h++)
                {
                    Span<Rgb24> pixelRow = accessor.GetRowSpan(h);
                    for (int w = 0; w < width; w++)
                    {
                        var pixel = pixelRow[w];

                        // CHW format: all R values, then all G values, then all B values
                        // CRITICAL: Normalize to [0, 1] by dividing by 255
                        tensor[0 * height * width + h * width + w] = pixel.R / 255.0f;
                        tensor[1 * height * width + h * width + w] = pixel.G / 255.0f;
                        tensor[2 * height * width + h * width + w] = pixel.B / 255.0f;
                    }
                }
            });

            return tensor;
        }

        /// <summary>
        /// Convert grayscale image to CHW format tensor normalized to [0, 1]
        /// </summary>
        private static float[] PreprocessMask(Image<L8> mask)
        {
            int width = mask.Width;
            int height = mask.Height;
            float[] tensor = new float[height * width];

            // Convert to CHW and normalize to [0, 1]
            mask.ProcessPixelRows(accessor =>
            {
                for (int h = 0; h < height; h++)
                {
                    Span<L8> pixelRow = accessor.GetRowSpan(h);
                    for (int w = 0; w < width; w++)
                    {
                        // CRITICAL: Normalize to [0, 1] and binarize (> 0.5 = 1.0)
                        float value = pixelRow[w].PackedValue / 255.0f;
                        tensor[h * width + w] = value > 0.5f ? 1.0f : 0.0f;
                    }
                }
            });

            return tensor;
        }

        /// <summary>
        /// Pad dimension to be divisible by mod
        /// </summary>
        private static int PadToModulo(int dimension, int mod = 8)
        {
            return (int)Math.Ceiling((double)dimension / mod) * mod;
        }

        /// <summary>
        /// Convert CHW output tensor to image
        /// </summary>
        /// <param name="output">Model output in CHW format</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <returns>Output image</returns>
        private static Image<Rgb24> PostprocessOutput(float[] output, int width, int height)
        {
            var image = new Image<Rgb24>(width, height);

            image.ProcessPixelRows(accessor =>
            {
                for (int h = 0; h < height; h++)
                {
                    Span<Rgb24> pixelRow = accessor.GetRowSpan(h);
                    for (int w = 0; w < width; w++)
                    {
                        // Convert from CHW to HWC
                        int rIndex = 0 * height * width + h * width + w;
                        int gIndex = 1 * height * width + h * width + w;
                        int bIndex = 2 * height * width + h * width + w;

                        // CRITICAL: Output is already in [0, 255] range!
                        // DO NOT multiply by 255 again!
                        // Just clamp and convert to byte
                        byte r = (byte)Math.Clamp(output[rIndex], 0, 255);
                        byte g = (byte)Math.Clamp(output[gIndex], 0, 255);
                        byte b = (byte)Math.Clamp(output[bIndex], 0, 255);

                        pixelRow[w] = new Rgb24(r, g, b);
                    }
                }
            });

            return image;
        }

        /// <summary>
        /// Run LaMa ONNX inference
        /// </summary>
        public static void RunInference(string modelPath, string imagePath, string maskPath, string outputPath)
        {
            Console.WriteLine($"Loading ONNX model from {modelPath}...");
            using var session = new InferenceSession(modelPath);

            Console.WriteLine($"Loading image from {imagePath}...");
            using var originalImage = Image.Load<Rgb24>(imagePath);
            int originalWidth = originalImage.Width;
            int originalHeight = originalImage.Height;

            // Pad dimensions to be divisible by 8
            int paddedWidth = PadToModulo(originalWidth, 8);
            int paddedHeight = PadToModulo(originalHeight, 8);

            Console.WriteLine($"  Original size: {originalWidth}x{originalHeight}");
            Console.WriteLine($"  Padded size: {paddedWidth}x{paddedHeight}");

            // Resize to padded dimensions
            var image = originalImage.Clone(ctx => ctx.Resize(paddedWidth, paddedHeight));

            Console.WriteLine($"Loading mask from {maskPath}...");
            using var originalMask = Image.Load<L8>(maskPath);
            var mask = originalMask.Clone(ctx => ctx.Resize(paddedWidth, paddedHeight));

            // Preprocess inputs
            Console.WriteLine("Preprocessing inputs...");
            float[] imageArray = PreprocessImage(image);
            float[] maskArray = PreprocessMask(mask);

            // Create tensors with batch dimension
            var imageTensor = new DenseTensor<float>(imageArray, new[] { 1, 3, paddedHeight, paddedWidth });
            var maskTensor = new DenseTensor<float>(maskArray, new[] { 1, 1, paddedHeight, paddedWidth });

            Console.WriteLine($"Running inference...");
            Console.WriteLine($"  Image tensor shape: [{string.Join(", ", imageTensor.Dimensions)}]");
            Console.WriteLine($"  Mask tensor shape: [{string.Join(", ", maskTensor.Dimensions)}]");
            Console.WriteLine($"  Image range: [{imageArray.Min():F3}, {imageArray.Max():F3}]");
            Console.WriteLine($"  Mask range: [{maskArray.Min():F3}, {maskArray.Max():F3}]");

            // Run inference
            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("image", imageTensor),
                NamedOnnxValue.CreateFromTensor("mask", maskTensor)
            };

            using var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();

            Console.WriteLine($"  Output length: {output.Length}");
            Console.WriteLine($"  Output range: [{output.Min():F3}, {output.Max():F3}]");

            // Postprocess output
            Console.WriteLine("Postprocessing output...");
            using var resultImage = PostprocessOutput(output, paddedWidth, paddedHeight);

            // Crop to original size
            resultImage.Mutate(ctx => ctx.Crop(new Rectangle(0, 0, originalWidth, originalHeight)));

            // Save result
            Console.WriteLine($"Saving result to {outputPath}...");
            resultImage.Save(outputPath);

            Console.WriteLine("✅ Done!");
            Console.WriteLine("\nIMPORTANT NOTES:");
            Console.WriteLine("  - Input was normalized to [0, 1] range");
            Console.WriteLine("  - Output is already in [0, 255] range (NO additional scaling)");
            Console.WriteLine("  - Result saved as uint8 image");
        }

        public static void Main(string[] args)
        {
            if (args.Length != 4)
            {
                Console.Error.WriteLine("Usage: dotnet run <model.onnx> <input.jpg> <mask.png> <output.jpg>");
                Console.Error.WriteLine("");
                Console.Error.WriteLine("For more information, see the README's \"ONNX Preprocessing & Postprocessing Guide\" section.");
                Environment.Exit(1);
                return;
            }

            string modelPath = args[0];
            string imagePath = args[1];
            string maskPath = args[2];
            string outputPath = args[3];

            try
            {
                RunInference(modelPath, imagePath, maskPath, outputPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                Console.Error.WriteLine(ex.StackTrace);
                Environment.Exit(1);
            }
        }
    }
}
