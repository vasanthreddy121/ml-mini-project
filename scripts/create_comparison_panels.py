#!/usr/bin/env python3
"""Generate 3-way and 4-way comparison panels for faculty review.

Compares:
1. Grayscale Input
2. Baseline Classical SVR (models/)
3. Best Candidate Classical SVR (models_experiment_comp0.2/)
4. External Pretrained Benchmark (DDColor-tiny ONNX)
"""

from pathlib import Path
import cv2
import numpy as np


def add_label_banner(image, title, subtitle=None, banner_height=65):
    """Add a dark banner at the top of an image with text label."""
    h, w = image.shape[:2]
    canvas = np.zeros((h + banner_height, w, 3), dtype=np.uint8)
    canvas[:banner_height, :] = (28, 30, 35)  # Dark charcoal header
    canvas[banner_height:, :] = image

    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (15, 26), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (15, 48), font, 0.45, (180, 200, 220), 1, cv2.LINE_AA)

    # Subtle bottom border for the banner
    cv2.line(canvas, (0, banner_height - 1), (w, banner_height - 1), (70, 75, 85), 1)
    return canvas


def build_3way_panel(gray_path, cand_path, pre_path, out_path, subject_name):
    """Build side-by-side: Grayscale | Best SVR Candidate | Pretrained Benchmark."""
    img_gray = cv2.imread(str(gray_path))
    img_cand = cv2.imread(str(cand_path))
    img_pre = cv2.imread(str(pre_path))

    target_h, target_w = img_cand.shape[:2]
    img_gray_res = cv2.resize(img_gray, (target_w, target_h))
    img_pre_res = cv2.resize(img_pre, (target_w, target_h))

    panel1 = add_label_banner(img_gray_res, "1. Grayscale Input", "Luminance (Y Channel)")
    panel2 = add_label_banner(img_cand_res := img_cand, "2. Best SVR Candidate", "SLIC comp=0.2, MRF gamma=0.1")
    panel3 = add_label_banner(img_pre_res, "3. Pretrained Benchmark", "DDColor Deep Learning (ICCV 2023)")

    border = np.full((panel1.shape[0], 4, 3), 50, dtype=np.uint8)
    combined = np.hstack([panel1, border, panel2, border, panel3])

    # Add master title header
    master_h = 50
    final_canvas = np.zeros((combined.shape[0] + master_h, combined.shape[1], 3), dtype=np.uint8)
    final_canvas[:master_h, :] = (18, 20, 24)
    final_canvas[master_h:, :] = combined

    cv2.putText(final_canvas, f"Automatic Image Colorization - 3-Way Faculty Review Comparison ({subject_name})",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 245), 2, cv2.LINE_AA)
    cv2.line(final_canvas, (0, master_h - 1), (final_canvas.shape[1], master_h - 1), (90, 100, 115), 1)

    cv2.imwrite(str(out_path), final_canvas)
    print(f"Saved 3-way comparison: {out_path} ({final_canvas.shape[1]}x{final_canvas.shape[0]})")


def build_4way_panel(gray_path, base_path, cand_path, pre_path, out_path, subject_name):
    """Build side-by-side: Grayscale | Baseline SVR | Best SVR Candidate | Pretrained Benchmark."""
    img_gray = cv2.imread(str(gray_path))
    img_base = cv2.imread(str(base_path))
    img_cand = cv2.imread(str(cand_path))
    img_pre = cv2.imread(str(pre_path))

    target_h, target_w = img_cand.shape[:2]
    img_gray_res = cv2.resize(img_gray, (target_w, target_h))
    img_base_res = cv2.resize(img_base, (target_w, target_h))
    img_pre_res = cv2.resize(img_pre, (target_w, target_h))

    panel1 = add_label_banner(img_gray_res, "1. Grayscale Input", "Original Luminance")
    panel2 = add_label_banner(img_base_res, "2. Baseline Model", "models/ (comp=10, gamma=1.0)")
    panel3 = add_label_banner(img_cand, "3. Candidate Model", "models_exp_0.2 (comp=0.2, gamma=0.1)")
    panel4 = add_label_banner(img_pre_res, "4. Pretrained Benchmark", "DDColor Deep Learning (ONNX)")

    border = np.full((panel1.shape[0], 4, 3), 50, dtype=np.uint8)
    combined = np.hstack([panel1, border, panel2, border, panel3, border, panel4])

    master_h = 50
    final_canvas = np.zeros((combined.shape[0] + master_h, combined.shape[1], 3), dtype=np.uint8)
    final_canvas[:master_h, :] = (18, 20, 24)
    final_canvas[master_h:, :] = combined

    cv2.putText(final_canvas, f"Colorization Evolution ({subject_name}): Baseline vs. Candidate vs. Deep Benchmark",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 245), 2, cv2.LINE_AA)
    cv2.line(final_canvas, (0, master_h - 1), (final_canvas.shape[1], master_h - 1), (90, 100, 115), 1)

    cv2.imwrite(str(out_path), final_canvas)
    print(f"Saved 4-way comparison: {out_path} ({final_canvas.shape[1]}x{final_canvas.shape[0]})")


def main():
    root = Path("outputs/review_demo")

    # 1. Human portrait comparisons
    build_3way_panel(
        gray_path=root / "candidate/human_test/input_preview.png",
        cand_path=root / "candidate/human_test/mrf_refined_colorization.png",
        pre_path=root / "pretrained/human_test/ddcolor_colorization.png",
        out_path=root / "portrait_3way_comparison.png",
        subject_name="Human Portrait"
    )

    build_4way_panel(
        gray_path=root / "baseline/human_test/input_preview.png",
        base_path=root / "baseline/human_test/mrf_refined_colorization.png",
        cand_path=root / "candidate/human_test/mrf_refined_colorization.png",
        pre_path=root / "pretrained/human_test/ddcolor_colorization.png",
        out_path=root / "portrait_4way_comparison.png",
        subject_name="Human Portrait"
    )

    # 2. Waterfall comparisons
    build_3way_panel(
        gray_path=root / "candidate/tree/input_preview.png",
        cand_path=root / "candidate/tree/mrf_refined_colorization.png",
        pre_path=root / "pretrained/tree/ddcolor_colorization.png",
        out_path=root / "waterfall_3way_comparison.png",
        subject_name="Waterfall Landscape"
    )

    build_4way_panel(
        gray_path=root / "baseline/tree/input_preview.png",
        base_path=root / "baseline/tree/mrf_refined_colorization.png",
        cand_path=root / "candidate/tree/mrf_refined_colorization.png",
        pre_path=root / "pretrained/tree/ddcolor_colorization.png",
        out_path=root / "waterfall_4way_comparison.png",
        subject_name="Waterfall Landscape"
    )


if __name__ == "__main__":
    main()

