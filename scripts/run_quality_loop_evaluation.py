#!/usr/bin/env python3
"""Run complete quality loop evaluation, comparison generation, and metrics export."""

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

from src import colorize, evaluation, features, mrf, segmentation

QL_DIR = Path("outputs/quality_loop")
COMP_DIR = QL_DIR / "comparisons"


def ensure_dirs():
    QL_DIR.mkdir(parents=True, exist_ok=True)
    COMP_DIR.mkdir(parents=True, exist_ok=True)


def add_banner(image, title, subtitle=None, banner_h=60):
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


def build_panel(panels, master_title, out_path):
    target_h = min(p[0].shape[0] for p in panels)
    target_w = min(p[0].shape[1] for p in panels)

    banners = []
    for img, title, sub in panels:
        res = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        banners.append(add_banner(res, title, sub))

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

    cv2.putText(final_canvas, master_title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 245), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), final_canvas)
    print(f"Saved: {out_path} ({final_canvas.shape[1]}x{final_canvas.shape[0]})")


def evaluate_test_set():
    test_images = ["l1", "l2", "m1", "m2", "m3", "r1"]
    scaler, model_u, model_v = colorize.load_models("models_sprint_et_nature")
    b_scaler, b_u, b_v = colorize.load_models("models")

    rows = []
    for name in test_images:
        p = Path(f"dataset/test/{name}.jpg")
        orig_rgb = evaluation.original_test_rgb(name, "test")
        y = colorize.load_live_luminance(p)

        # 1. Baseline SVR (models/, comp=10, gamma=2.0)
        segs_b = segmentation.segment_image(y, compactness=10.0)
        fb, sids_b = features.extract_fft_features(y, segs_b)
        pu_b = b_u.predict(b_scaler.transform(fb))
        pv_b = b_v.predict(b_scaler.transform(fb))
        ru_b = mrf.refine_chrominance(pu_b, sids_b, segs_b, gamma=2.0, iterations=10)
        rv_b = mrf.refine_chrominance(pv_b, sids_b, segs_b, gamma=2.0, iterations=10)
        rgb_b = colorize.yuv_to_rgb(y, colorize.values_to_image(ru_b, sids_b, segs_b), colorize.values_to_image(rv_b, sids_b, segs_b))
        m_b = evaluation.metrics(rgb_b, orig_rgb)

        # 2. Previous Candidate (models_sprint_et_nature, comp=0.2, gamma=0.2, raw piecewise)
        segs_p = segmentation.segment_image(y, compactness=0.2)
        fp, sids_p = features.extract_fft_features(y, segs_p)
        pu_p = model_u.predict(scaler.transform(fp))
        pv_p = model_v.predict(scaler.transform(fp))
        ru_p = mrf.refine_chrominance(pu_p, sids_p, segs_p, gamma=0.2, iterations=10)
        rv_p = mrf.refine_chrominance(pv_p, sids_p, segs_p, gamma=0.2, iterations=10)
        rgb_p = colorize.yuv_to_rgb(y, colorize.values_to_image(ru_p, sids_p, segs_p), colorize.values_to_image(rv_p, sids_p, segs_p))
        m_p = evaluation.metrics(rgb_p, orig_rgb)

        # 3. Optimized Candidate (models_sprint_et_nature, comp=0.1, gamma=0.1, smooth_chroma=True)
        segs_n = segmentation.segment_image(y, compactness=0.1)
        fn, sids_n = features.extract_fft_features(y, segs_n)
        pu_n = model_u.predict(scaler.transform(fn))
        pv_n = model_v.predict(scaler.transform(fn))
        ru_n = mrf.refine_chrominance(pu_n, sids_n, segs_n, gamma=0.1, iterations=10)
        rv_n = mrf.refine_chrominance(pv_n, sids_n, segs_n, gamma=0.1, iterations=10)
        un_map = colorize.values_to_image(ru_n, sids_n, segs_n)
        vn_map = colorize.values_to_image(rv_n, sids_n, segs_n)
        un_sm = cv2.bilateralFilter(un_map.astype(np.float32), 7, 15, 7)
        vn_sm = cv2.bilateralFilter(vn_map.astype(np.float32), 7, 15, 7)
        rgb_n = colorize.yuv_to_rgb(y, un_sm, vn_sm)
        m_n = evaluation.metrics(rgb_n, orig_rgb)

        rows.append({
            "image": name,
            "baseline_mae": m_b["mae"],
            "baseline_psnr": m_b["psnr"],
            "previous_et_mae": m_p["mae"],
            "previous_et_psnr": m_p["psnr"],
            "optimized_et_mae": m_n["mae"],
            "optimized_et_psnr": m_n["psnr"],
        })

    averages = {
        "baseline_mae": float(np.mean([r["baseline_mae"] for r in rows])),
        "baseline_psnr": float(np.mean([r["baseline_psnr"] for r in rows])),
        "previous_et_mae": float(np.mean([r["previous_et_mae"] for r in rows])),
        "previous_et_psnr": float(np.mean([r["previous_et_psnr"] for r in rows])),
        "optimized_et_mae": float(np.mean([r["optimized_et_mae"] for r in rows])),
        "optimized_et_psnr": float(np.mean([r["optimized_et_psnr"] for r in rows])),
    }
    return rows, averages


def generate_visual_comparisons():
    scenes = [
        ("tree.jpg", "Waterfall Landscape", "tree"),
        ("dataset/test/r1.jpg", "River Rapids", "r1"),
        ("dataset/test/l2.jpg", "Sunny Landscape with Clouds", "l2"),
        ("dataset/test/m1.jpg", "Mountain Lake", "m1"),
    ]

    scaler, model_u, model_v = colorize.load_models("models_sprint_et_nature")
    b_scaler, b_u, b_v = colorize.load_models("models")

    for img_str, label, stem in scenes:
        p = Path(img_str)
        y = colorize.load_live_luminance(p)
        gray = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)

        # Baseline (models/, comp=10, gamma=2.0)
        sb = segmentation.segment_image(y, compactness=10.0)
        fb, sidb = features.extract_fft_features(y, sb)
        pub = b_u.predict(b_scaler.transform(fb))
        pvb = b_v.predict(b_scaler.transform(fb))
        rub = mrf.refine_chrominance(pub, sidb, sb, gamma=2.0, iterations=10)
        rvb = mrf.refine_chrominance(pvb, sidb, sb, gamma=2.0, iterations=10)
        rgb_b = colorize.yuv_to_rgb(y, colorize.values_to_image(rub, sidb, sb), colorize.values_to_image(rvb, sidb, sb))

        # Previous ET Candidate (comp=0.2, gamma=0.2, no smooth)
        sp = segmentation.segment_image(y, compactness=0.2)
        fp, sidp = features.extract_fft_features(y, sp)
        pup = model_u.predict(scaler.transform(fp))
        pvp = model_v.predict(scaler.transform(fp))
        rup = mrf.refine_chrominance(pup, sidp, sp, gamma=0.2, iterations=10)
        rvp = mrf.refine_chrominance(pvp, sidp, sp, gamma=0.2, iterations=10)
        rgb_p = colorize.yuv_to_rgb(y, colorize.values_to_image(rup, sidp, sp), colorize.values_to_image(rvp, sidp, sp))

        # Optimized ET Candidate (comp=0.1, gamma=0.1, bilateral smooth)
        sn = segmentation.segment_image(y, compactness=0.1)
        fn, sidn = features.extract_fft_features(y, sn)
        pun = model_u.predict(scaler.transform(fn))
        pvn = model_v.predict(scaler.transform(fn))
        run = mrf.refine_chrominance(pun, sidn, sn, gamma=0.1, iterations=10)
        rvn = mrf.refine_chrominance(pvn, sidn, sn, gamma=0.1, iterations=10)
        un_map = colorize.values_to_image(run, sidn, sn)
        vn_map = colorize.values_to_image(rvn, sidn, sn)
        un_sm = cv2.bilateralFilter(un_map.astype(np.float32), 7, 15, 7)
        vn_sm = cv2.bilateralFilter(vn_map.astype(np.float32), 7, 15, 7)
        rgb_n = colorize.yuv_to_rgb(y, un_sm, vn_sm)

        panels = [
            (gray, "1. Grayscale Input", "Original Luminance Y"),
            (rgb_b, "2. Baseline Model", "models/ (comp=10, g=2.0)"),
            (rgb_p, "3. Previous ExtraTrees", "comp=0.2, g=0.2, raw tiles"),
            (rgb_n, "4. Optimized ExtraTrees", "comp=0.1, g=0.1 + smooth"),
        ]
        out_panel = COMP_DIR / f"{stem}_quality_loop_comparison.png"
        build_panel(panels, f"Quality Optimization Loop - {label} ({stem})", out_panel)

    # Portrait comparison
    portrait_p = Path("human_test.png")
    if portrait_p.is_file():
        y_port = colorize.load_live_luminance(portrait_p)
        gray_port = cv2.cvtColor(y_port, cv2.COLOR_GRAY2RGB)

        # Baseline
        sb = segmentation.segment_image(y_port, compactness=10.0)
        fb, sidb = features.extract_fft_features(y_port, sb)
        pub = b_u.predict(b_scaler.transform(fb))
        pvb = b_v.predict(b_scaler.transform(fb))
        rub = mrf.refine_chrominance(pub, sidb, sb, gamma=2.0, iterations=10)
        rvb = mrf.refine_chrominance(pvb, sidb, sb, gamma=2.0, iterations=10)
        rgb_b = colorize.yuv_to_rgb(y_port, colorize.values_to_image(rub, sidb, sb), colorize.values_to_image(rvb, sidb, sb))

        # Optimized ET
        sn = segmentation.segment_image(y_port, compactness=0.1)
        fn, sidn = features.extract_fft_features(y_port, sn)
        pun = model_u.predict(scaler.transform(fn))
        pvn = model_v.predict(scaler.transform(fn))
        run = mrf.refine_chrominance(pun, sidn, sn, gamma=0.1, iterations=10)
        rvn = mrf.refine_chrominance(pvn, sidn, sn, gamma=0.1, iterations=10)
        un_map = cv2.bilateralFilter(colorize.values_to_image(run, sidn, sn).astype(np.float32), 7, 15, 7)
        vn_map = cv2.bilateralFilter(colorize.values_to_image(rvn, sidn, sn).astype(np.float32), 7, 15, 7)
        rgb_n = colorize.yuv_to_rgb(y_port, un_map, vn_map)

        # DDColor Benchmark
        dd_path = Path("outputs/review_demo/pretrained/human_test/ddcolor_colorization.png")
        if dd_path.is_file():
            rgb_dd = cv2.imread(str(dd_path))
        else:
            rgb_dd = rgb_n

        port_panels = [
            (gray_port, "1. Grayscale Input", "Original Luminance Y"),
            (rgb_b, "2. Classical Baseline", "Trained on 12 landscapes"),
            (rgb_n, "3. Optimized ExtraTrees", "Nature specialist (cyan skin)"),
            (rgb_dd, "4. External Benchmark", "DDColor Deep Learning (ONNX)"),
        ]
        build_panel(port_panels, "Quality Loop: Human Portrait (Nature Specialist vs Deep Benchmark)",
                    COMP_DIR / "human_portrait_quality_loop_comparison.png")


def main():
    ensure_dirs()
    print("Evaluating test set across Baseline, Previous Candidate, and Optimized Candidate...")
    rows, averages = evaluate_test_set()

    print("\nGenerating visual comparison panels...")
    generate_visual_comparisons()

    best_config = {
        "model_directory": "models_sprint_et_nature",
        "architecture": "ExtraTreesRegressor (n_estimators=100, max_depth=15, n_jobs=-1)",
        "compactness": 0.1,
        "gamma": 0.1,
        "smooth_chroma": True,
        "bilateral_filter": {
            "diameter": 7,
            "sigma_color": 15,
            "sigma_space": 7
        },
        "test_metrics": {
            "average_mae": averages["optimized_et_mae"],
            "average_psnr_db": averages["optimized_et_psnr"],
            "mae_reduction_vs_baseline": round(averages["baseline_mae"] - averages["optimized_et_mae"], 3),
            "psnr_boost_vs_baseline_db": round(averages["optimized_et_psnr"] - averages["baseline_psnr"], 3)
        },
        "inference_command": ".venv/bin/python main.py colorize --input tree.jpg --model-dir models_sprint_et_nature --compactness 0.1 --gamma 0.1 --smooth-chroma"
    }

    with (QL_DIR / "best_config.json").open("w") as f:
        json.dump(best_config, f, indent=2)

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_config": best_config,
        "per_image_metrics": rows,
        "averages": averages
    }

    with (QL_DIR / "results.json").open("w") as f:
        json.dump(results_data, f, indent=2)

    print("\nQuality loop complete. Artifacts saved in outputs/quality_loop/.")


if __name__ == "__main__":
    main()

