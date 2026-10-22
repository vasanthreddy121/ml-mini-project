#!/usr/bin/env python3
"""Run DDColor pretrained deep-learning benchmark on runtime test images.

This script executes inference using the official DDColor ONNX model (DDColor-tiny FP16).
It serves as an external state-of-the-art benchmark to compare against the classical
SLIC + FFT + RBF-SVR + MRF colorization pipeline.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_MODEL_PATH = "pretrained_models/ddcolor/ddcolor-tiny-fp16.onnx"
DEFAULT_OUTPUT_ROOT = "outputs/review_demo/pretrained"


def colorize_with_ddcolor(image_path, model_path=DEFAULT_MODEL_PATH, input_size=512):
    """Run DDColor ONNX inference on a grayscale or RGB image.

    Args:
        image_path: Path to input image file.
        model_path: Path to DDColor ONNX model.
        input_size: Model input spatial resolution (default 512).

    Returns:
        tuple: (colorized_bgr, orig_gray_bgr, orig_l, pred_ab_resized, inference_time_sec)
    """
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image via OpenCV
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")
    orig_h, orig_w = img.shape[:2]

    # Convert to float32 [0.0, 1.0]
    img_float = (img / 255.0).astype(np.float32)

    # Extract original high-resolution L channel in CIE Lab space
    orig_lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
    orig_l = orig_lab[:, :, :1]  # shape (H, W, 1), L in [0, 100]

    # Generate grayscale BGR preview for visualization
    gray_lab = np.concatenate([orig_l, np.zeros_like(orig_l), np.zeros_like(orig_l)], axis=-1)
    orig_gray_bgr = np.clip(cv2.cvtColor(gray_lab, cv2.COLOR_Lab2BGR) * 255.0, 0, 255).astype(np.uint8)

    # Prepare input tensor: resize to 512x512, build neutral Lab, convert to RGB
    img_resize = cv2.resize(img_float, (input_size, input_size))
    img_l = cv2.cvtColor(img_resize, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    # Tensor format: NCHW, float32, batch size 1
    blob = np.transpose(img_gray_rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    # Initialize ONNX Runtime Session (CPU Execution Provider)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # Suppress warning noise
    sess = ort.InferenceSession(str(model_path), session_options, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    start_time = time.perf_counter()
    outputs = sess.run([out_name], {inp_name: blob})
    elapsed = time.perf_counter() - start_time

    # Output ab channels: shape (1, 2, 512, 512)
    pred_ab = outputs[0][0]  # (2, 512, 512)
    pred_ab_hw = np.transpose(pred_ab, (1, 2, 0))  # (512, 512, 2)

    # Resize predicted chrominance back to full image resolution
    pred_ab_resized = cv2.resize(pred_ab_hw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Combine original full-res L with predicted ab chrominance
    output_lab = np.concatenate([orig_l, pred_ab_resized], axis=-1)

    # Convert back to BGR and uint8
    output_bgr_float = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
    colorized_bgr = np.clip(output_bgr_float * 255.0, 0, 255).astype(np.uint8)

    return colorized_bgr, orig_gray_bgr, orig_l, pred_ab_resized, elapsed


def process_image(image_path, model_path, output_root):
    image_path = Path(image_path)
    stem = image_path.stem
    target_dir = Path(output_root) / stem
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {image_path.name} with DDColor pretrained benchmark...")
    colorized_bgr, orig_gray, orig_l, ab, elapsed = colorize_with_ddcolor(image_path, model_path)

    out_file = target_dir / "ddcolor_colorization.png"
    preview_file = target_dir / "input_preview.png"
    cv2.imwrite(str(out_file), colorized_bgr)
    cv2.imwrite(str(preview_file), orig_gray)

    # Also save top-level file for direct access
    top_file = Path(output_root) / f"{stem}_colorized.png"
    cv2.imwrite(str(top_file), colorized_bgr)

    # Compute diagnostics
    b_mean, g_mean, r_mean = colorized_bgr.mean(axis=(0, 1))
    hsv = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    print(f"  Inference time: {elapsed:.3f} s")
    print(f"  Saved output to: {out_file}")
    print(f"  Saved output to: {top_file}")
    print(f"  Output size: {colorized_bgr.shape[1]}x{colorized_bgr.shape[0]}")
    print(f"  Mean BGR: B={b_mean:.1f}, G={g_mean:.1f}, R={r_mean:.1f} (R - B = {r_mean - b_mean:+.1f})")
    print(f"  Chroma ab range: a=[{ab[:,:,0].min():.1f}, {ab[:,:,0].max():.1f}], b=[{ab[:,:,1].min():.1f}, {ab[:,:,1].max():.1f}]")
    print(f"  Mean Saturation: {sat.mean():.1f}, Saturated pixels (>15): {np.mean(sat > 15)*100:.1f}%")

    return {
        "colorized_path": str(out_file),
        "top_colorized_path": str(top_file),
        "preview_path": str(preview_file),
        "elapsed": elapsed,
        "r_minus_b": r_mean - b_mean,
        "mean_sat": sat.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run DDColor pretrained colorization benchmark")
    parser.add_argument("--input", nargs="*", default=["human_test.png", "tree.jpg"],
                        help="Path(s) to input images (default: human_test.png tree.jpg)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"Path to DDColor ONNX model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT})")
    args = parser.parse_args()

    model_file = Path(args.model_path)
    if not model_file.is_file():
        sys.exit(f"Error: Model file does not exist: {model_file}")

    for img_path in args.input:
        process_image(img_path, args.model_path, args.output_root)


if __name__ == "__main__":
    main()
