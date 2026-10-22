#!/usr/bin/env python3
"""Generate all visual comparisons and final packaging for outputs/final_solution/."""

import json
import os
import shutil
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src import colorize, evaluation

FINAL_DIR = Path("outputs/final_solution")
VIS_DIR = FINAL_DIR / "visual_comparisons"
NATURE_VIS_DIR = VIS_DIR / "nature"
GENERAL_VIS_DIR = VIS_DIR / "portraits_general"


def ensure_dirs():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    NATURE_VIS_DIR.mkdir(parents=True, exist_ok=True)
    GENERAL_VIS_DIR.mkdir(parents=True, exist_ok=True)


def add_banner(image, title, subtitle=None, banner_h=60):
    """Add a dark header banner with clean typography."""
    h, w = image.shape[:2]
    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
    canvas[:banner_h, :] = (26, 28, 32)
    canvas[banner_h:, :] = image

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (12, 24), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (12, 46), font, 0.42, (175, 195, 215), 1, cv2.LINE_AA)

    cv2.line(canvas, (0, banner_h - 1), (w, banner_h - 1), (65, 70, 80), 1)
    return canvas


def build_comparison_panel(images_with_titles, master_title, output_path):
    """Combine multiple images into a side-by-side comparison panel."""
    # Find common target dimensions
    target_h = min(img.shape[0] for img, _, _ in images_with_titles)
    target_w = min(img.shape[1] for img, _, _ in images_with_titles)

    banners = []
    for img, title, subtitle in images_with_titles:
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        banners.append(add_banner(resized, title, subtitle))

    sep = np.full((banners[0].shape[0], 4, 3), 45, dtype=np.uint8)
    blocks = []
    for i, b in enumerate(banners):
        if i > 0:
            blocks.append(sep)
        blocks.append(b)
    combined = np.hstack(blocks)

    header_h = 44
    final_canvas = np.zeros((combined.shape[0] + header_h, combined.shape[1], 3), dtype=np.uint8)
    final_canvas[:header_h, :] = (16, 18, 22)
    final_canvas[header_h:, :] = combined

    cv2.putText(final_canvas, master_title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (240, 240, 245), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), final_canvas)
    print(f"Saved panel: {output_path} ({final_canvas.shape[1]}x{final_canvas.shape[0]})")


def generate_nature_panels():
    nature_tests = [
        ("tree.jpg", "Waterfall Landscape", "tree"),
        ("dataset/test/r1.jpg", "River Rapids", "r1"),
        ("dataset/test/l2.jpg", "Sunny Landscape", "l2"),
        ("dataset/test/m1.jpg", "Mountain Lake", "m1"),
        ("dataset/test/l1.jpg", "Dense Forest", "l1"),
        ("dataset/test/m2.jpg", "Rocky Mountain Crags", "m2"),
    ]

    for img_path_str, subject_name, stem in nature_tests:
        img_path = Path(img_path_str)
        if not img_path.is_file():
            continue

        print(f"\nProcessing nature test: {stem} ({subject_name})...")
        # 1. Baseline
        out_base = colorize.colorize_live_image(
            str(img_path), model_dir="models",
            output_root=str(FINAL_DIR / "cache_baseline"),
            compactness=10.0, gamma=2.0
        )
        # 2. SLIC 0.2 Candidate
        out_cand = colorize.colorize_live_image(
            str(img_path), model_dir="models_experiment_comp0.2",
            output_root=str(FINAL_DIR / "cache_cand0.2"),
            compactness=0.2, gamma=0.1
        )
        # 3. Nature-Aware ExtraTrees (Top model)
        out_et = colorize.colorize_live_image(
            str(img_path), model_dir="models_sprint_et_nature",
            output_root=str(FINAL_DIR / "cache_et"),
            compactness=0.2, gamma=0.2
        )
        # 4. Nature-Aware SVR
        out_svr = colorize.colorize_live_image(
            str(img_path), model_dir="models_sprint_svr_nature",
            output_root=str(FINAL_DIR / "cache_svr"),
            compactness=0.2, gamma=0.2
        )

        gray = cv2.imread(str(out_base["input_preview"]))
        base_img = cv2.imread(str(out_base["mrf_refined"]))
        cand_img = cv2.imread(str(out_cand["mrf_refined"]))
        et_img = cv2.imread(str(out_et["mrf_refined"]))
        svr_img = cv2.imread(str(out_svr["mrf_refined"]))

        # Build 4-way comparison panel (Input | Baseline | Nature-SVR | Nature-ExtraTrees)
        panels_4way = [
            (gray, "1. Grayscale Input", "Original Luminance"),
            (base_img, "2. Baseline Model", "models/ (comp=10, g=2.0)"),
            (svr_img, "3. Nature-Aware SVR", "Mixed 10k, comp=0.2, g=0.2"),
            (et_img, "4. Nature-Aware ExtraTrees", "Selected Best (comp=0.2, g=0.2)"),
        ]
        out_panel_4way = NATURE_VIS_DIR / f"{stem}_comparison.png"
        build_comparison_panel(panels_4way, f"Colorization Results - {subject_name} ({stem})", out_panel_4way)


def generate_portrait_panels():
    portrait_path = Path("human_test.png")
    if not portrait_path.is_file():
        return

    print("\nProcessing portrait test: human_test.png...")
    # Baseline
    out_base = colorize.colorize_live_image(
        str(portrait_path), model_dir="models",
        output_root=str(FINAL_DIR / "cache_baseline"),
        compactness=10.0, gamma=2.0
    )
    # Selected ExtraTrees
    out_et = colorize.colorize_live_image(
        str(portrait_path), model_dir="models_sprint_et_nature",
        output_root=str(FINAL_DIR / "cache_et"),
        compactness=0.2, gamma=0.2
    )

    # DDColor Deep Learning Pretrained Benchmark
    ddcolor_path = Path("outputs/review_demo/pretrained/human_test/ddcolor_colorization.png")
    if not ddcolor_path.is_file():
        from scripts.run_pretrained_benchmark import colorize_with_ddcolor
        dd_img, _, _, _, _ = colorize_with_ddcolor(portrait_path)
    else:
        dd_img = cv2.imread(str(ddcolor_path))

    gray = cv2.imread(str(out_base["input_preview"]))
    base_img = cv2.imread(str(out_base["mrf_refined"]))
    et_img = cv2.imread(str(out_et["mrf_refined"]))

    portrait_panels = [
        (gray, "1. Grayscale Input", "Original Luminance"),
        (base_img, "2. Classical Baseline", "Trained on 12 landscapes"),
        (et_img, "3. Nature-Aware Model", "Nature specialist (cyan cast)"),
        (dd_img, "4. External Benchmark", "DDColor Deep Learning (ONNX)"),
    ]
    out_panel_portrait = GENERAL_VIS_DIR / "human_portrait_cross_domain_comparison.png"
    build_comparison_panel(
        portrait_panels,
        "Cross-Domain Evaluation: Human Portrait (Nature Specialist vs Deep Benchmark)",
        out_panel_portrait
    )


def cleanup_cache():
    for cache_name in ("cache_baseline", "cache_cand0.2", "cache_et", "cache_svr"):
        p = FINAL_DIR / cache_name
        if p.is_dir():
            shutil.rmtree(p)


def main():
    ensure_dirs()
    generate_nature_panels()
    generate_portrait_panels()
    cleanup_cache()
    print("\nVisual comparison panels generated successfully in outputs/final_solution/visual_comparisons/.")


if __name__ == "__main__":
    main()
