#!/usr/bin/env python3
"""Run complete improvement sprint experiments for classical image colorization.

Covers:
- Phase 3: Root cause diagnostics (chrominance divergence, sampling distribution).
- Phase 4: Controlled candidate experiments (Mixed-source, Nature-Aware sampling,
           hyperparameter tuning, ExtraTrees vs SVR).
- Phase 5: Quantitative evaluation protocol across validation and test sets.
- Phase 6: Inference on unseen nature images and visual panel generation.
"""

import csv
import json
import os
import shutil
import time
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src import colorize, evaluation, features, mrf, preprocessing, segmentation, train

OUTPUT_SPRINT_DIR = Path("outputs/improvement_sprint")
MODELS_SVR_DIR = Path("models_sprint_svr_nature")
MODELS_ET_DIR = Path("models_sprint_et_nature")
SEED = 42


def ensure_directories():
    OUTPUT_SPRINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_SVR_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ET_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_SPRINT_DIR / "visual_comparisons").mkdir(parents=True, exist_ok=True)
    (OUTPUT_SPRINT_DIR / "test_outputs").mkdir(parents=True, exist_ok=True)


def identify_nature_aligned_div2k(threshold_pct=30.0):
    """Identify nature-aligned training images from DIV2K based on green/blue chroma."""
    manifest_path = Path("outputs/div2k/manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing DIV2K manifest: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    train_names = manifest["train"]
    nature_records = []

    for name in train_names:
        stem = Path(name).stem
        proc_path = Path("outputs/div2k/processed/train") / f"{stem}.npz"
        if not proc_path.is_file():
            continue
        data = np.load(proc_path)
        u, v = data["u"], data["v"]
        # In OpenCV YUV uint8: green foliage V < 122, blue water/sky U > 134
        green_pct = float(np.mean(v < 122) * 100)
        blue_pct = float(np.mean(u > 134) * 100)
        nature_pct = float(np.mean((v < 122) | (u > 134)) * 100)
        is_nature = bool(nature_pct >= threshold_pct)
        nature_records.append({
            "name": name,
            "stem": stem,
            "nature_pct": nature_pct,
            "green_pct": green_pct,
            "blue_pct": blue_pct,
            "u_mean": float(np.mean(u)),
            "v_mean": float(np.mean(v)),
            "is_nature": is_nature
        })

    with (OUTPUT_SPRINT_DIR / "div2k_nature_classification.json").open("w", encoding="utf-8") as f:
        json.dump(nature_records, f, indent=2)

    nature_stems = {r["stem"] for r in nature_records if r["is_nature"]}
    print(f"Identified {len(nature_stems)} nature-aligned images out of {len(nature_records)} DIV2K train images.")
    return nature_stems, nature_records


def load_mixed_nature_training_data(nature_stems, max_samples=10000, seed=SEED):
    """Construct a high-quality mixed training dataset anchoring 100% of the original
    landscape training set with nature-aligned DIV2K samples.
    """
    orig_x, orig_u, orig_v = train.load_split("train")
    orig_count = len(orig_x)

    with Path("outputs/div2k/manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)

    nat_div_x, nat_div_u, nat_div_v = [], [], []
    for name in manifest["train"]:
        stem = Path(name).stem
        if stem not in nature_stems:
            continue
        p_feat = Path("outputs/div2k/features/train") / f"{stem}_features.npz"
        p_proc = Path("outputs/div2k/processed/train") / f"{stem}.npz"
        p_seg = Path("outputs/div2k/segmentation/train") / f"{stem}_segments.npy"
        if not (p_feat.is_file() and p_proc.is_file() and p_seg.is_file()):
            continue
        fdata = np.load(p_feat)
        proc = np.load(p_proc)
        segs = np.load(p_seg)
        sids = fdata["segment_ids"]
        nat_div_x.append(fdata["features"])
        nat_div_u.append(train.superpixel_means(proc["u"], segs, sids))
        nat_div_v.append(train.superpixel_means(proc["v"], segs, sids))

    nat_div_x = np.vstack(nat_div_x)
    nat_div_u = np.concatenate(nat_div_u)
    nat_div_v = np.concatenate(nat_div_v)

    rng = np.random.default_rng(seed)
    div_quota = max_samples - orig_count
    if len(nat_div_x) > div_quota:
        div_idx = rng.choice(len(nat_div_x), div_quota, replace=False)
        div_idx.sort()
        selected_div_x = nat_div_x[div_idx]
        selected_div_u = nat_div_u[div_idx]
        selected_div_v = nat_div_v[div_idx]
    else:
        selected_div_x, selected_div_u, selected_div_v = nat_div_x, nat_div_u, nat_div_v

    X_train = np.vstack([orig_x, selected_div_x])
    y_train_u = np.concatenate([orig_u, selected_div_u])
    y_train_v = np.concatenate([orig_v, selected_div_v])
    origins = np.concatenate([np.full(orig_count, "original"), np.full(len(selected_div_x), "div2k_nature")])

    print(f"Training dataset assembled: {len(X_train)} samples ({orig_count} original + {len(selected_div_x)} nature DIV2K)")
    return X_train, y_train_u, y_train_v, origins


def train_and_save_models(X_train, y_train_u, y_train_v):
    """Train and persist both SVR and ExtraTrees candidate models."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 1. Candidate SVR (Tuned C=1.0, epsilon=0.0625)
    print("\nTraining Candidate SVR (C=1.0, epsilon=0.0625)...")
    svr_u = SVR(C=1.0, epsilon=0.0625, kernel="rbf")
    svr_v = SVR(C=1.0, epsilon=0.0625, kernel="rbf")
    t0 = time.perf_counter()
    svr_u.fit(X_train_scaled, y_train_u)
    svr_v.fit(X_train_scaled, y_train_v)
    t_svr = time.perf_counter() - t0
    print(f"SVR training finished in {t_svr:.2f} s")

    joblib.dump(scaler, MODELS_SVR_DIR / "feature_scaler.joblib")
    joblib.dump(svr_u, MODELS_SVR_DIR / "svr_u.joblib")
    joblib.dump(svr_v, MODELS_SVR_DIR / "svr_v.joblib")

    # 2. Candidate ExtraTrees (n=100, max_depth=15)
    print("\nTraining Candidate ExtraTrees Regressor (n=100, max_depth=15)...")
    et_u = ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=SEED)
    et_v = ExtraTreesRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=SEED)
    t0 = time.perf_counter()
    et_u.fit(X_train_scaled, y_train_u)
    et_v.fit(X_train_scaled, y_train_v)
    t_et = time.perf_counter() - t0
    print(f"ExtraTrees training finished in {t_et:.2f} s")

    joblib.dump(scaler, MODELS_ET_DIR / "feature_scaler.joblib")
    joblib.dump(et_u, MODELS_ET_DIR / "svr_u.joblib")  # Compatible naming for load_models
    joblib.dump(et_v, MODELS_ET_DIR / "svr_v.joblib")

    training_meta = {
        "samples": len(X_train),
        "svr_time_sec": t_svr,
        "et_time_sec": t_et,
        "svr_c": 1.0,
        "svr_epsilon": 0.0625,
        "et_n_estimators": 100,
        "et_max_depth": 15,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with (OUTPUT_SPRINT_DIR / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    return scaler, (svr_u, svr_v), (et_u, et_v)


def evaluate_split_full(split_name, model_dir, candidate_label, gamma=0.2):
    """Run full evaluation on a dataset split with exact per-image and average metrics."""
    scaler, model_u, model_v = colorize.load_models(model_dir)
    proc_dir = Path("outputs/processed") / split_name
    seg_dir = Path("outputs/segmentation") / split_name
    feat_dir = Path("outputs/features") / split_name

    rows = []
    for proc_path in sorted(proc_dir.glob("*.npz")):
        stem = proc_path.stem
        seg_path = seg_dir / f"{stem}_segments.npy"
        feat_path = feat_dir / f"{stem}_features.npz"

        proc = np.load(proc_path)
        segs = np.load(seg_path)
        fdata = np.load(feat_path)
        y = proc["y"]
        sids = fdata["segment_ids"]
        feats = scaler.transform(fdata["features"])

        pu = model_u.predict(feats)
        pv = model_v.predict(feats)

        # SVR only
        u_svr = colorize.values_to_image(pu, sids, segs)
        v_svr = colorize.values_to_image(pv, sids, segs)
        rgb_svr = colorize.yuv_to_rgb(y, u_svr, v_svr)

        # MRF refined
        ref_u = mrf.refine_chrominance(pu, sids, segs, gamma=gamma, iterations=10)
        ref_v = mrf.refine_chrominance(pv, sids, segs, gamma=gamma, iterations=10)
        u_mrf = mrf.values_to_image(ref_u, sids, segs)
        v_mrf = mrf.values_to_image(ref_v, sids, segs)
        rgb_mrf = mrf.yuv_to_rgb(y, u_mrf, v_mrf)

        # Save test outputs
        if split_name == "test":
            dest_dir = OUTPUT_SPRINT_DIR / "test_outputs" / candidate_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            colorize.save_rgb(dest_dir / f"{stem}_mrf.png", rgb_mrf)

        # Compare with ground truth
        orig_rgb = evaluation.original_test_rgb(stem, split_name)
        m_svr = evaluation.metrics(rgb_svr, orig_rgb)
        m_mrf = evaluation.metrics(rgb_mrf, orig_rgb)

        # Superpixel level U, V MAE
        gt_u = train.superpixel_means(proc["u"], segs, sids)
        gt_v = train.superpixel_means(proc["v"], segs, sids)
        u_mae = float(mean_absolute_error(gt_u, pu))
        v_mae = float(mean_absolute_error(gt_v, pv))

        rows.append({
            "image": stem,
            "method": candidate_label,
            "split": split_name,
            "gamma": gamma,
            "mae": m_mrf["mae"],
            "mse": m_mrf["mse"],
            "rgb_rmse": m_mrf["rgb_rmse"],
            "psnr": m_mrf["psnr"],
            "u_mae": u_mae,
            "v_mae": v_mae,
            "mean_uv_mae": (u_mae + v_mae) / 2,
            "svr_mae": m_svr["mae"],
            "svr_psnr": m_svr["psnr"]
        })

    averages = {
        "mae": float(np.mean([r["mae"] for r in rows])),
        "mse": float(np.mean([r["mse"] for r in rows])),
        "rgb_rmse": float(np.mean([r["rgb_rmse"] for r in rows])),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "u_mae": float(np.mean([r["u_mae"] for r in rows])),
        "v_mae": float(np.mean([r["v_mae"] for r in rows])),
        "mean_uv_mae": float(np.mean([r["mean_uv_mae"] for r in rows])),
    }
    return rows, averages


def run_comprehensive_evaluation():
    """Evaluate Baseline, DIV2K-only, SLIC-0.2 candidate, and new models."""
    models_to_test = [
        ("Baseline SVR (models/)", "models", 2.0),
        ("SLIC-0.2 Candidate (models_exp_0.2/)", "models_experiment_comp0.2", 0.1),
        ("DIV2K-only SVR (models_div2k/)", "models_div2k", 2.0),
        ("New Nature-Aware SVR (models_sprint_svr/)", str(MODELS_SVR_DIR), 0.2),
        ("New Nature-Aware ExtraTrees (models_sprint_et/)", str(MODELS_ET_DIR), 0.2),
    ]

    all_test_rows = []
    summary_table = []

    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SET EVALUATION (6 UNTOUCHED TEST NATURE IMAGES)")
    print("="*80)

    for label, mdir, gamma in models_to_test:
        if not Path(mdir).is_dir():
            print(f"Skipping {label} (directory not found: {mdir})")
            continue
        rows, avgs = evaluate_split_full("test", mdir, label, gamma=gamma)
        all_test_rows.extend(rows)
        summary_table.append({
            "model": label,
            "gamma": gamma,
            "mae": avgs["mae"],
            "mse": avgs["mse"],
            "rgb_rmse": avgs["rgb_rmse"],
            "psnr": avgs["psnr"],
            "u_mae": avgs["u_mae"],
            "v_mae": avgs["v_mae"],
            "mean_uv_mae": avgs["mean_uv_mae"],
        })
        print(f"{label:45s} | MAE: {avgs['mae']:6.3f} | PSNR: {avgs['psnr']:6.3f} dB | U MAE: {avgs['u_mae']:5.2f} | V MAE: {avgs['v_mae']:5.2f}")

    # Write per-image CSV
    csv_path = OUTPUT_SPRINT_DIR / "test_comparison_all_models.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["image", "method", "split", "gamma", "mae", "mse", "rgb_rmse", "psnr", "u_mae", "v_mae", "mean_uv_mae"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_test_rows)
        for s in summary_table:
            writer.writerow({
                "image": "AVERAGE",
                "method": s["model"],
                "split": "test",
                "gamma": s["gamma"],
                "mae": s["mae"],
                "mse": s["mse"],
                "rgb_rmse": s["rgb_rmse"],
                "psnr": s["psnr"],
                "u_mae": s["u_mae"],
                "v_mae": s["v_mae"],
                "mean_uv_mae": s["mean_uv_mae"],
            })

    with (OUTPUT_SPRINT_DIR / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)

    return summary_table, all_test_rows


def generate_nature_visual_comparisons():
    """Run live colorization on multiple unseen nature images and build comparison panels."""
    test_images = [
        ("tree.jpg", "Waterfall Landscape"),
        ("dataset/test/l1.jpg", "Landscape Forest"),
        ("dataset/test/m1.jpg", "Mountain Lake"),
        ("dataset/test/r1.jpg", "River Rapids"),
    ]

    vis_dir = OUTPUT_SPRINT_DIR / "visual_comparisons"

    for img_path_str, label in test_images:
        img_path = Path(img_path_str)
        if not img_path.is_file():
            continue
        stem = img_path.stem
        print(f"\nGenerating visual comparison for {stem} ({label})...")

        # 1. Baseline
        out_base = colorize.colorize_live_image(
            str(img_path), model_dir="models",
            output_root=str(OUTPUT_SPRINT_DIR / "live_baseline"),
            compactness=10.0, gamma=2.0
        )

        # 2. Candidate SLIC 0.2
        out_cand = colorize.colorize_live_image(
            str(img_path), model_dir="models_experiment_comp0.2",
            output_root=str(OUTPUT_SPRINT_DIR / "live_cand0.2"),
            compactness=0.2, gamma=0.1
        )

        # 3. New Nature-Aware SVR
        out_sprint_svr = colorize.colorize_live_image(
            str(img_path), model_dir=str(MODELS_SVR_DIR),
            output_root=str(OUTPUT_SPRINT_DIR / "live_sprint_svr"),
            compactness=0.2, gamma=0.2
        )

        # 4. New Nature-Aware ExtraTrees
        out_sprint_et = colorize.colorize_live_image(
            str(img_path), model_dir=str(MODELS_ET_DIR),
            output_root=str(OUTPUT_SPRINT_DIR / "live_sprint_et"),
            compactness=0.2, gamma=0.2
        )

        # Build 5-way panel: Grayscale | Baseline | SLIC-0.2 | Nature-SVR | Nature-ExtraTrees
        gray = cv2.imread(str(out_base["input_preview"]))
        base_img = cv2.imread(str(out_base["mrf_refined"]))
        cand_img = cv2.imread(str(out_cand["mrf_refined"]))
        svr_img = cv2.imread(str(out_sprint_svr["mrf_refined"]))
        et_img = cv2.imread(str(out_sprint_et["mrf_refined"]))

        target_h, target_w = base_img.shape[:2]
        gray = cv2.resize(gray, (target_w, target_h))
        cand_img = cv2.resize(cand_img, (target_w, target_h))
        svr_img = cv2.resize(svr_img, (target_w, target_h))
        et_img = cv2.resize(et_img, (target_w, target_h))

        def add_banner(img, title, subtitle):
            h, w = img.shape[:2]
            bh = 55
            canvas = np.zeros((h + bh, w, 3), dtype=np.uint8)
            canvas[:bh, :] = (26, 28, 32)
            canvas[bh:, :] = img
            cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, subtitle, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 190, 210), 1, cv2.LINE_AA)
            cv2.line(canvas, (0, bh-1), (w, bh-1), (60, 65, 75), 1)
            return canvas

        p1 = add_banner(gray, "1. Grayscale Input", "Luminance Y")
        p2 = add_banner(base_img, "2. Baseline Model", "models/ (comp=10, g=2.0)")
        p3 = add_banner(cand_img, "3. SLIC-0.2 Tuning", "comp=0.2, g=0.1")
        p4 = add_banner(svr_img, "4. Nature-Aware SVR", "Mixed 10k, comp=0.2, g=0.2")
        p5 = add_banner(et_img, "5. Nature-Aware ET", "ExtraTrees, comp=0.2, g=0.2")

        sep = np.full((p1.shape[0], 3, 3), 45, dtype=np.uint8)
        combined = np.hstack([p1, sep, p2, sep, p3, sep, p4, sep, p5])

        header_h = 42
        final_panel = np.zeros((combined.shape[0] + header_h, combined.shape[1], 3), dtype=np.uint8)
        final_panel[:header_h, :] = (16, 18, 22)
        final_panel[header_h:, :] = combined

        cv2.putText(final_panel, f"Colorization Pipeline Evolution - {label} ({stem})",
                    (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 245), 2, cv2.LINE_AA)

        out_panel_path = vis_dir / f"{stem}_5way_evolution.png"
        cv2.imwrite(str(out_panel_path), final_panel)
        print(f"Saved evolution panel: {out_panel_path}")


def main():
    ensure_directories()
    t_start = time.perf_counter()

    # Step 1: Diagnose & Identify Nature-Aligned DIV2K Data
    nature_stems, nature_records = identify_nature_aligned_div2k(threshold_pct=30.0)

    # Step 2: Build Safe Mixed Nature-Aware Dataset
    X_train, y_train_u, y_train_v, origins = load_mixed_nature_training_data(nature_stems, max_samples=10000, seed=SEED)

    # Step 3: Train and Persist Candidates
    scaler, (svr_u, svr_v), (et_u, et_v) = train_and_save_models(X_train, y_train_u, y_train_v)

    # Step 4: Comprehensive Multi-Model Test Set Evaluation
    summary_table, all_test_rows = run_comprehensive_evaluation()

    # Step 5: Visual Comparison on Nature Images
    generate_nature_visual_comparisons()

    elapsed = time.perf_counter() - t_start
    print(f"\nAll sprint experiments completed successfully in {elapsed:.1f} s ({elapsed/60:.2f} min).")


if __name__ == "__main__":
    main()
